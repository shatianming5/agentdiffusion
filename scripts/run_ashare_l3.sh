#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "============================================================"
echo "  A-Share L3 Pipeline: Real Order-by-Order → Video DiT"
echo "============================================================"

DATA_7Z="data/external/20220601.7z"
DATA_DIR="data/external/20220601"

# ============================================================
# Step 0: Extract if needed
# ============================================================
if [ ! -d "$DATA_DIR" ]; then
    echo "=== Step 0: Extracting $DATA_7Z ==="
    7z x "$DATA_7Z" -o"data/external/" || p7zip -d "$DATA_7Z"
    echo "Extracted to $DATA_DIR"
else
    echo "=== Step 0: SKIP (data already extracted) ==="
fi

echo "Stocks: $(ls $DATA_DIR | wc -l)"

# ============================================================
# Step 1: Train Video DiT on A-Share L3 agent grids
# ============================================================
echo "=== Step 1: Train Video DiT on A-Share L3 (top 50 stocks) ==="
.venv/bin/python3 -u << 'PYEOF'
import torch, torch.nn as nn, torch.nn.functional as F
import logging, time
from pathlib import Path
from tqdm import tqdm
from agentdiffusion.models.video_dit import VideoDiT
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.data.ashare_l3_dataset import AShareL3VideoDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("outputs/vdit_ashare_l3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Dataset ---
# Use top 50 liquid SZ stocks (000001-000099 range typically liquid)
dataset = AShareL3VideoDataset(
    "data/external/20220601",
    total_frames=20, cond_frames=4,
    window_seconds=1.0,
    grid_shape=(4, 4),
    max_stocks=50,
)
logger.info(f"Dataset: {len(dataset)} sequences")

if len(dataset) == 0:
    logger.error("No sequences! Check data.")
    import sys; sys.exit(1)

loader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True,
)

# --- Model ---
sample = dataset[0]
d_latent = sample["frames"].shape[-1]  # 6 (agent state dim)
logger.info(f"d_latent={d_latent}, grid=4x4")

model = VideoDiT(
    d_latent=d_latent, d_model=128, depth=6, heads=4,
    patch_size=2, num_frames=20, num_cond_frames=4,
    mlp_ratio=4.0, market_cond_dim=32,
    grid_h=4, grid_w=4,
).to(device)
logger.info(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

scheduler = NoiseScheduler(1000, "cosine").to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
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
pbar = tqdm(total=TOTAL_STEPS, desc="A-Share L3 DiT")

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
            z_gen.reshape(B * N, H, W, C),
            noise.reshape(B * N, H, W, C),
            t_exp,
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
logger.info(f"Done. Saved to {OUT_DIR}")
PYEOF

# ============================================================
# Step 2: Eval — stylized facts + interactive demo
# ============================================================
echo "=== Step 2: Quick Evaluation ==="
.venv/bin/python3 -u << 'PYEOF'
import torch, numpy as np
from pathlib import Path
from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.data.ashare_l3_dataset import AShareL3VideoDataset
from agentdiffusion.infer.interactive_sim import InteractiveSimulator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VDIT_DIR = Path("outputs/vdit_ashare_l3")

dataset = AShareL3VideoDataset(
    "data/external/20220601",
    total_frames=20, cond_frames=4,
    window_seconds=1.0, grid_shape=(4, 4), max_stocks=50,
)
d_latent = dataset[0]["frames"].shape[-1]

ckpt_path = sorted(VDIT_DIR.glob("video_dit_step_*.pt"))[-1]
model = VideoDiT(
    d_latent=d_latent, d_model=128, depth=6, heads=4,
    patch_size=2, num_frames=20, num_cond_frames=4,
    grid_h=4, grid_w=4,
).to(device)
ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
model.load_state_dict(ckpt.get("ema", ckpt["model"]))
model.eval()
print(f"Loaded model from {ckpt_path}")

scheduler = NoiseScheduler(1000, "cosine").to(device)
sampler = VideoDDIMSampler(model, scheduler, "v_prediction", ddim_steps=50, eta=0.0)

# Interactive simulation
sim = InteractiveSimulator(
    model, sampler, num_cond=4, num_gen=16,
    zero_sum_proj=True, scheduler=scheduler,
    recalibrate_every=5, recalibrate_strength=0.3,
)

seed = dataset[0]["frames"][:4]
sim.init(seed)
print(f"Init: grid=4x4, d_state={d_latent}, frames={seed.shape[0]}")

print("\n=== A-Share L3 Interactive Simulation (10 rounds) ===")
for i in range(10):
    gen = sim.step()
    traj = sim.get_trajectory(dim=0)  # net_position
    vol_traj = sim.get_trajectory(dim=1)  # order_rate
    print(f"Round {i+1}: position={traj[-1]:.4f}, order_rate={vol_traj[-1]:.4f}")

    if i == 4:
        print("  >>> Shock: large sell pressure <<<")
        shock = torch.zeros(d_latent)
        shock[0] = -3.0  # negative position = sell
        sim.intervene(frame_idx=-1, delta=shock)

    sim.trim_buffer(keep_last=8)

print("\nA-Share L3 simulation complete.")
PYEOF

echo "=== ALL DONE ==="
