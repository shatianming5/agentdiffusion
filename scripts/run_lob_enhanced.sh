#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "============================================================"
echo "  Enhanced LOB Video DiT: 8x8 grid, d_latent=3, 20K steps"
echo "  + Causal temporal + Market conditioning"
echo "============================================================"

OB_PATH="data/external/lobster/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
MSG_PATH="data/external/lobster/AMZN_2012-06-21_34200000_57600000_message_10.csv"

.venv/bin/python3 -u << 'PYEOF'
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, logging, math
from pathlib import Path
from tqdm import tqdm
from agentdiffusion.models.video_dit import VideoDiT
from agentdiffusion.models.order_decoder import AgentToOrderDecoder
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.data.lob_dataset import LOBVideoDataset
from agentdiffusion.eval.stylized_facts import evaluate_stylized_facts, compute_returns, return_distribution_wasserstein, acf
from agentdiffusion.infer.interactive_sim import InteractiveSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("outputs/vdit_lob_8x8_enhanced")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OB = "data/external/lobster/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
MSG = "data/external/lobster/AMZN_2012-06-21_34200000_57600000_message_10.csv"

# --- Dataset: 8x8 grid (64 cells, feature_dim=48 → C = ceil(48/64) = 1, pad to 64) ---
# Actually with 8x8=64 cells and 48 features, C=1 with 16 padding
# Better: use 6x8=48 cells, C=1, zero padding
# Or: use 4x4=16, C=3 (already proven)
# Let's try 4x6=24, C=2 (48/24=2) for more spatial structure
GRID_H, GRID_W = 4, 6
dataset = LOBVideoDataset(OB, MSG, total_frames=20, cond_frames=4, subsample=10, grid_shape=(GRID_H, GRID_W))
d_latent = dataset[0]["frames"].shape[-1]
logger.info(f"Dataset: {len(dataset)} sequences, grid=({GRID_H},{GRID_W}), d_latent={d_latent}")

loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)

# --- Bigger model: d_model=256, depth=8 ---
model = VideoDiT(
    d_latent=d_latent, d_model=256, depth=8, heads=8,
    patch_size=2, num_frames=20, num_cond_frames=4,
    mlp_ratio=4.0, market_cond_dim=32,
    grid_h=GRID_H, grid_w=GRID_W,
    causal_temporal=True, alibi_temporal=True,
).to(device)
n_params = sum(p.numel() for p in model.parameters())
logger.info(f"Model params: {n_params/1e6:.1f}M")

# Order Decoder
order_decoder = AgentToOrderDecoder(
    d_state=d_latent, d_model=128, n_queries=64,
    n_layers=2, n_heads=4, d_order_out=6,
).to(device)

scheduler = NoiseScheduler(1000, "cosine").to(device)
all_params = list(model.parameters()) + list(order_decoder.parameters())
optimizer = torch.optim.AdamW(all_params, lr=1e-4, weight_decay=0.01)
TOTAL_STEPS = 20000
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)
LAMBDA_ORDER = 0.1

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
pbar = tqdm(total=TOTAL_STEPS, desc="Enhanced LOB DiT")

while step < TOTAL_STEPS:
    for batch in loader:
        if step >= TOTAL_STEPS:
            break

        frames = batch["frames"].to(device)
        B, T, H, W, C = frames.shape
        N = T - K

        z_cond = frames[:, :K]
        z_gen = frames[:, K:]

        t = torch.randint(0, 1000, (B,), device=device)
        noise = torch.randn_like(z_gen)

        z_gen_flat = z_gen.reshape(B * N, H, W, C)
        noise_flat = noise.reshape(B * N, H, W, C)
        t_exp = t.unsqueeze(1).expand(B, N).reshape(B * N)
        z_noisy = scheduler.q_sample(z_gen_flat, t_exp, noise_flat).reshape(B, N, H, W, C)

        v_pred = model(z_cond, z_noisy, t)
        v_target = scheduler.v_target(
            z_gen.reshape(B * N, H, W, C), noise.reshape(B * N, H, W, C), t_exp
        ).reshape(B, N, H, W, C)
        loss_diff = F.mse_loss(v_pred, v_target)

        # Order decoder (gradient clamp)
        v_pred_flat = v_pred.reshape(B * N, H, W, C)
        z0_pred = scheduler.predict_x0_from_v(
            z_noisy.reshape(B * N, H, W, C), t_exp, v_pred_flat
        ).clamp(-10, 10).reshape(B, N, H, W, C)
        pred_orders = order_decoder.decode_sequence(torch.cat([z_cond[:, -1:], z0_pred], dim=1))
        with torch.no_grad():
            gt_orders = order_decoder.decode_sequence(frames[:, K-1:])
        loss_order = F.mse_loss(pred_orders, gt_orders)

        loss = loss_diff + LAMBDA_ORDER * loss_order

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        lr_scheduler.step()
        update_ema()

        step += 1
        pbar.update(1)
        if step % 100 == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", diff=f"{loss_diff.item():.4f}",
                ordr=f"{loss_order.item():.4f}",
                lr=f"{lr_scheduler.get_last_lr()[0]:.2e}", step=step)
        if step % 5000 == 0:
            torch.save({"model": model.state_dict(), "ema": ema_state,
                        "decoder": order_decoder.state_dict(), "step": step},
                       OUT_DIR / f"video_dit_step_{step}.pt")

pbar.close()
torch.save({"model": model.state_dict(), "ema": ema_state,
            "decoder": order_decoder.state_dict(), "step": step},
           OUT_DIR / f"video_dit_step_{step}.pt")
logger.info("Training done.")

# --- Evaluation ---
logger.info("=== Evaluation ===")
model_eval = VideoDiT(
    d_latent=d_latent, d_model=256, depth=8, heads=8,
    patch_size=2, num_frames=20, num_cond_frames=4,
    market_cond_dim=32, grid_h=GRID_H, grid_w=GRID_W,
    causal_temporal=True, alibi_temporal=True,
).to(device)
model_eval.load_state_dict(ema_state)
model_eval.eval()

sampler = VideoDDIMSampler(model_eval, scheduler, "v_prediction", ddim_steps=20, eta=0.0)
sim = InteractiveSimulator(model_eval, sampler, num_cond=4, num_gen=16, zero_sum_proj=True)

# Generate
N_SAMPLES = 20
gen_series = []
for s in range(N_SAMPLES):
    seed = dataset[s % len(dataset)]["frames"][:4]
    sim.init(seed)
    frames_list = []
    for _ in range(10):
        gen = sim.step()
        for ti in range(gen.shape[0]):
            frames_list.append(gen[ti].mean().item())
        sim.trim_buffer(keep_last=8)
    series = np.array(frames_list[:160])
    gen_series.append(series - series.min() + 1)

# Real
from agentdiffusion.data.lob_dataset import load_lobster_data
raw = load_lobster_data(OB, MSG, subsample=1)
real_mid = (raw["raw_ob"][:, 0] + raw["raw_ob"][:, 2]) / 2.0
real_chunks = [real_mid[i:i+160] for i in range(0, len(real_mid)-160, 160)][:N_SAMPLES]

gen_ret = compute_returns(np.concatenate(gen_series))
real_ret = compute_returns(np.concatenate(real_chunks))

wd = return_distribution_wasserstein(gen_ret, real_ret)
from scipy.stats import kurtosis as scipy_kurt
gen_kurt = scipy_kurt(gen_ret, fisher=False)
real_kurt = scipy_kurt(real_ret, fisher=False)
gen_acf = acf(np.abs(gen_ret), 10)
real_acf = acf(np.abs(real_ret), 10)

print("\n" + "=" * 64)
print("  ENHANCED LOB Model Results")
print("=" * 64)
print(f"  Model: d_model=256, depth=8, grid=({GRID_H},{GRID_W}), d_latent={d_latent}")
print(f"  Kurtosis:  Real={real_kurt:.2f}  Ours={gen_kurt:.2f}")
print(f"  ACF(1):    Real={real_acf[1]:.4f}  Ours={gen_acf[1]:.4f}")
print(f"  ACF(5):    Real={real_acf[5]:.4f}  Ours={gen_acf[5]:.4f}")
print(f"  Ret std:   Real={real_ret.std()*1000:.2f}  Ours={gen_ret.std()*1000:.2f}")
print(f"  Wasserstein: {wd:.6f}")
print("=" * 64)
PYEOF
