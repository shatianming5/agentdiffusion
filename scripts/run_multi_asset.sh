#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "============================================================"
echo "  Multi-Asset Video DiT: 13 Crypto Currencies"
echo "============================================================"

.venv/bin/python3 -u << 'PYEOF'
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, logging
from pathlib import Path
from tqdm import tqdm
from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.data.binance_multi_asset import BinanceMultiAssetDataset, SYMBOLS
from agentdiffusion.infer.interactive_sim import InteractiveSimulator
from scipy.stats import kurtosis as scipy_kurtosis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("outputs/vdit_multi_asset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Dataset ---
dataset = BinanceMultiAssetDataset(
    "data/external/binance/aggTrades",
    total_frames=20, cond_frames=4,
    window_seconds=60.0,  # 1-minute windows
    max_months=3,
)
logger.info("Dataset: %d sequences", len(dataset))

if len(dataset) == 0:
    logger.error("No data! Check Binance downloads.")
    import sys; sys.exit(1)

loader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)

d_latent = dataset[0]["frames"].shape[-1]  # 8
logger.info("d_latent=%d, grid=4x4, %d symbols", d_latent, len(SYMBOLS))

# --- Model ---
model = VideoDiT(
    d_latent=d_latent, d_model=256, depth=8, heads=8,
    patch_size=2, num_frames=20, num_cond_frames=4,
    mlp_ratio=4.0, market_cond_dim=32,
    grid_h=4, grid_w=4,
    causal_temporal=True, alibi_temporal=True,
).to(device)
logger.info("Params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

scheduler = NoiseScheduler(1000, "cosine").to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
TOTAL_STEPS = 20000
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)

# EMA
ema_state = {k: v.clone() for k, v in model.state_dict().items()}
ema_decay = 0.999
def update_ema():
    with torch.no_grad():
        for k, v in model.state_dict().items():
            ema_state[k].lerp_(v, 1 - ema_decay)

# --- Training ---
K = 4
step = 0
pbar = tqdm(total=TOTAL_STEPS, desc="Multi-Asset DiT")

while step < TOTAL_STEPS:
    for batch in loader:
        if step >= TOTAL_STEPS:
            break
        frames = batch["frames"].to(device)
        B, T, H, W, C = frames.shape
        N = T - K
        z_cond, z_gen = frames[:, :K], frames[:, K:]
        t = torch.randint(0, 1000, (B,), device=device)
        noise = torch.randn_like(z_gen)
        t_exp = t.unsqueeze(1).expand(B, N).reshape(B * N)
        z_noisy = scheduler.q_sample(
            z_gen.reshape(B*N, H, W, C), noise.reshape(B*N, H, W, C), t_exp
        ).reshape(B, N, H, W, C)

        v_pred = model(z_cond, z_noisy, t)
        v_target = scheduler.v_target(
            z_gen.reshape(B*N, H, W, C), noise.reshape(B*N, H, W, C), t_exp
        ).reshape(B, N, H, W, C)
        loss = F.mse_loss(v_pred, v_target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        update_ema()

        step += 1
        pbar.update(1)
        if step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}", step=step)
        if step % 5000 == 0:
            torch.save({"model": model.state_dict(), "ema": ema_state, "step": step},
                       OUT_DIR / f"video_dit_step_{step}.pt")

pbar.close()
torch.save({"model": model.state_dict(), "ema": ema_state, "step": step},
           OUT_DIR / f"video_dit_step_{step}.pt")
logger.info("Training done.")

# --- Evaluation ---
logger.info("=== Multi-Asset Evaluation ===")
model.load_state_dict(ema_state)
model.eval()

sampler = VideoDDIMSampler(model, scheduler, "v_prediction", ddim_steps=20, eta=0.0)
sim = InteractiveSimulator(model, sampler, num_cond=4, num_gen=16, zero_sum_proj=False)

# Generate and analyze cross-asset correlations
seed = dataset[0]["frames"][:4]
sim.init(seed)
all_gen = []
for _ in range(10):
    gen = sim.step()
    all_gen.append(gen.cpu())
    sim.trim_buffer(keep_last=8)
all_gen = torch.cat(all_gen, dim=0)  # [160, 4, 4, 8]

# Extract per-asset return series (dim 0 = log_return)
print("\n" + "=" * 64)
print("  Multi-Asset Results")
print("=" * 64)

# Per-asset metrics
for i, sym in enumerate(SYMBOLS):
    r, c = i // 4, i % 4
    returns = all_gen[:, r, c, 0].numpy()
    kurt = scipy_kurtosis(returns, fisher=False)
    vol = returns.std()
    abs_ret = np.abs(returns)
    x = abs_ret - abs_ret.mean()
    ac1 = np.sum(x[:-1]*x[1:]) / (np.sum(x**2) + 1e-12)
    print(f"  {sym:<10s}: kurtosis={kurt:>6.2f}, vol={vol:.4f}, ACF1={ac1:.4f}")

# Cross-asset correlation matrix
print("\nCross-asset return correlation:")
corr_matrix = np.zeros((len(SYMBOLS), len(SYMBOLS)))
for i, sym_i in enumerate(SYMBOLS):
    ri, ci = i // 4, i % 4
    for j, sym_j in enumerate(SYMBOLS):
        rj, cj = j // 4, j % 4
        ret_i = all_gen[:, ri, ci, 0].numpy()
        ret_j = all_gen[:, rj, cj, 0].numpy()
        corr_matrix[i, j] = np.corrcoef(ret_i, ret_j)[0, 1]

# Print top 5 correlations (excluding diagonal)
pairs = []
for i in range(len(SYMBOLS)):
    for j in range(i+1, len(SYMBOLS)):
        pairs.append((corr_matrix[i, j], SYMBOLS[i], SYMBOLS[j]))
pairs.sort(key=lambda x: -abs(x[0]))

print("  Top 5 correlated pairs:")
for corr, s1, s2 in pairs[:5]:
    print(f"    {s1}-{s2}: {corr:.4f}")
print("  Top 5 anti-correlated:")
for corr, s1, s2 in pairs[-5:]:
    print(f"    {s1}-{s2}: {corr:.4f}")

# Shock test: crash BTC, see effect on others
print("\nShock test: BTC crash → cross-asset contagion")
sim2 = InteractiveSimulator(model, sampler, num_cond=4, num_gen=16, zero_sum_proj=False)
sim2.init(dataset[0]["frames"][:4])
pre_gen = sim2.step()  # normal generation

# Inject BTC crash
shock = torch.zeros(d_latent)
shock[0] = -5.0  # large negative return
sim2.intervene(frame_idx=-1, mask=None, delta=None,
               absolute=sim2.latest_frame.clone())
sim2.buffer[0, -1, 0, 0, 0] = -5.0  # BTC position (0,0)

post_gen = sim2.step()  # generation after shock

print("  Pre-shock vs Post-shock (mean return per asset):")
for i, sym in enumerate(SYMBOLS):
    r, c = i // 4, i % 4
    pre_ret = pre_gen[:, r, c, 0].mean().item()
    post_ret = post_gen[:, r, c, 0].mean().item()
    diff = post_ret - pre_ret
    print(f"    {sym:<10s}: pre={pre_ret:+.4f}, post={post_ret:+.4f}, change={diff:+.4f}")

print("=" * 64)
PYEOF
