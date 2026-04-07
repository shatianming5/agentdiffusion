#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "============================================================"
echo "  10K Agent Video DiT: Real A-Share L3 → 100×100 Grid"
echo "============================================================"

.venv/bin/python3 -u << 'PYEOF'
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, logging
from pathlib import Path
from tqdm import tqdm
from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.data.ashare_10k_agents import AShare10KAgentDataset, GRID_H, GRID_W, D_STATE
from agentdiffusion.infer.interactive_sim import InteractiveSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("outputs/vdit_10k_agents")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Dataset ---
dataset = AShare10KAgentDataset(
    "data/external/20240619",
    total_frames=20, cond_frames=4,
    window_seconds=5.0,  # 5s windows → ~3960 windows per day
    max_stocks=30,
    n_clusters=10000,
)
logger.info("Dataset: %d sequences", len(dataset))

if len(dataset) == 0:
    logger.error("No data!")
    import sys; sys.exit(1)

loader = torch.utils.data.DataLoader(
    dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)

d_latent = D_STATE  # 6
logger.info("grid=(%d,%d), d_latent=%d, patch_size=10", GRID_H, GRID_W, d_latent)

# --- Model: 100×100 grid, patch_size=10 → 10×10=100 patches ---
model = VideoDiT(
    d_latent=d_latent, d_model=256, depth=8, heads=8,
    patch_size=10, num_frames=20, num_cond_frames=4,
    mlp_ratio=4.0, market_cond_dim=32,
    grid_h=GRID_H, grid_w=GRID_W,
    causal_temporal=True, alibi_temporal=True,
).to(device)
n_params = sum(p.numel() for p in model.parameters())
logger.info("Model params: %.1fM", n_params / 1e6)

scheduler = NoiseScheduler(1000, "cosine").to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
TOTAL_STEPS = 10000
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
pbar = tqdm(total=TOTAL_STEPS, desc="10K Agent DiT")

while step < TOTAL_STEPS:
    for batch in loader:
        if step >= TOTAL_STEPS:
            break
        frames = batch["frames"].to(device)  # [B, T, 100, 100, 6]
        B, T, H, W, C = frames.shape
        N = T - K
        z_cond, z_gen = frames[:, :K], frames[:, K:]
        t = torch.randint(0, 1000, (B,), device=device)
        noise = torch.randn_like(z_gen)
        t_exp = t.unsqueeze(1).expand(B, N).reshape(B * N)
        z_noisy = scheduler.q_sample(
            z_gen.reshape(B*N, H, W, C), t_exp, noise.reshape(B*N, H, W, C)
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
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{lr_scheduler.get_last_lr()[0]:.2e}", step=step)
        if step % 2500 == 0:
            torch.save({"model": model.state_dict(), "ema": ema_state, "step": step},
                       OUT_DIR / f"video_dit_step_{step}.pt")

pbar.close()
torch.save({"model": model.state_dict(), "ema": ema_state, "step": step},
           OUT_DIR / f"video_dit_step_{step}.pt")
logger.info("Training done.")

# --- Quick Eval ---
logger.info("=== 10K Agent Evaluation ===")
model.load_state_dict(ema_state)
model.eval()

sampler = VideoDDIMSampler(model, scheduler, "v_prediction", ddim_steps=20, eta=0.0)
sim = InteractiveSimulator(model, sampler, num_cond=4, num_gen=16, zero_sum_proj=True)

seed = dataset[0]["frames"][:4]
sim.init(seed)

print("\n=== 10K Agent Interactive Simulation ===")
print(f"Grid: {GRID_H}x{GRID_W} = {GRID_H*GRID_W} agent clusters")
for i in range(5):
    gen = sim.step()
    m = gen.mean().item()
    s = gen.std().item()
    active = (gen.abs() > 0.1).float().mean().item()
    print(f"  Round {i+1}: mean={m:.4f}, std={s:.4f}, active_cells={active*100:.1f}%")
    sim.trim_buffer(keep_last=8)

    if i == 2:
        print("  >>> Injecting market-wide shock <<<")
        shock = torch.zeros(d_latent)
        shock[0] = -3.0  # large sell pressure
        sim.intervene(frame_idx=-1, delta=shock)

# Check spatial structure preservation
gen_last = sim.latest_frame  # [100, 100, 6]
print(f"\nFinal frame stats:")
print(f"  Shape: {gen_last.shape}")
print(f"  Non-zero cells: {(gen_last.abs() > 0.01).any(dim=-1).float().mean().item()*100:.1f}%")
print(f"  Spatial std (row): {gen_last[:,:,0].std(dim=1).mean().item():.4f}")
print(f"  Spatial std (col): {gen_last[:,:,0].std(dim=0).mean().item():.4f}")
print("=" * 64)
PYEOF
