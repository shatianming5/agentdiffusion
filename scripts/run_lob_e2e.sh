#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "============================================================"
echo "  End-to-End: LOBSTER → Encoder → Video DiT → Decoder"
echo "  Stage A: Train 16x16 Video DiT on LOB-derived agent grids"
echo "  Stage B: Train Encoder+Decoder with frozen DiT"
echo "  Stage C: Interactive simulation demo"
echo "============================================================"

OB_PATH="data/external/lobster/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
MSG_PATH="data/external/lobster/AMZN_2012-06-21_34200000_57600000_message_10.csv"
GRID_H=4
GRID_W=4
D_STATE=3
VDIT_OUT="outputs/vdit_lob_4x4"
OA_OUT="outputs/order_agent_lob_v2"

# ============================================================
# Stage A: Train small Video DiT on LOB data (16x16 grid)
# ============================================================
echo "=== Stage A: Train 16x16 Video DiT on LOB ==="
.venv/bin/python3 -u << 'PYEOF'
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, logging, time
from pathlib import Path
from tqdm import tqdm
from agentdiffusion.models.video_dit import VideoDiT
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.data.lob_dataset import LOBVideoDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("outputs/vdit_lob_4x4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Dataset: LOB snapshots as video frames ---
OB = "data/external/lobster/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
MSG = "data/external/lobster/AMZN_2012-06-21_34200000_57600000_message_10.csv"

dataset = LOBVideoDataset(OB, MSG, total_frames=20, cond_frames=4, subsample=10, grid_shape=(4, 4))
logger.info(f"Dataset: {len(dataset)} sequences")
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)

# --- Model: Video DiT for 4x4 grid with d_latent=3 (48 LOB features / 16 cells) ---
sample = dataset[0]
d_latent = sample["frames"].shape[-1]
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
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20000)

# --- EMA ---
ema_state = {k: v.clone() for k, v in model.state_dict().items()}
ema_decay = 0.999

def update_ema():
    with torch.no_grad():
        for k, v in model.state_dict().items():
            ema_state[k].lerp_(v, 1 - ema_decay)

# --- Training loop ---
TOTAL_STEPS = 20000
K = 4
step = 0
pbar = tqdm(total=TOTAL_STEPS, desc="Video DiT 16x16")

while step < TOTAL_STEPS:
    for batch in loader:
        if step >= TOTAL_STEPS:
            break

        frames = batch["frames"].to(device)  # [B, T, H, W, C]
        B, T, H, W, C = frames.shape
        N = T - K

        z_cond = frames[:, :K]
        z_gen = frames[:, K:]

        t = torch.randint(0, 1000, (B,), device=device)
        noise = torch.randn_like(z_gen)

        z_gen_flat = z_gen.reshape(B * N, H, W, C)
        noise_flat = noise.reshape(B * N, H, W, C)
        t_exp = t.unsqueeze(1).expand(B, N).reshape(B * N)
        z_noisy_flat = scheduler.q_sample(z_gen_flat, t_exp, noise_flat)
        z_noisy = z_noisy_flat.reshape(B, N, H, W, C)

        v_pred = model(z_cond, z_noisy, t)
        z_gen_flat2 = z_gen.reshape(B * N, H, W, C)
        noise_flat2 = noise.reshape(B * N, H, W, C)
        v_target_flat = scheduler.v_target(z_gen_flat2, noise_flat2, t_exp)
        v_target = v_target_flat.reshape(B, N, H, W, C)

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
            logger.info(f"Saved step {step}")

pbar.close()
torch.save({"model": model.state_dict(), "ema": ema_state, "step": step},
           OUT_DIR / f"video_dit_step_{step}.pt")
logger.info(f"Stage A done. Saved to {OUT_DIR}")
PYEOF

# ============================================================
# Stage B: Train Encoder+Decoder with frozen DiT
# ============================================================
echo "=== Stage B: Train Encoder+Decoder (frozen DiT) ==="
VDIT_CKPT=$(ls -t ${VDIT_OUT}/video_dit_step_*.pt | head -1)
echo "Using Video DiT: ${VDIT_CKPT}"

# Get d_latent from LOBVideoDataset
D_LATENT=$(.venv/bin/python3 -c "
from agentdiffusion.data.lob_dataset import LOBVideoDataset
d = LOBVideoDataset('${OB_PATH}','${MSG_PATH}',total_frames=20,cond_frames=4,subsample=10,grid_shape=(16,16))
print(d[0]['frames'].shape[-1])
")
echo "d_latent from LOB: ${D_LATENT}"

.venv/bin/python3 -m agentdiffusion.train.train_order_agent \
    --vdit-ckpt "${VDIT_CKPT}" \
    --ob-path "${OB_PATH}" \
    --msg-path "${MSG_PATH}" \
    --grid-h ${GRID_H} --grid-w ${GRID_W} \
    --d-state ${D_LATENT} --d-embed 128 --d-model 128 \
    --total-steps 3000 --batch-size 8 --lr 3e-4 \
    --output-dir "${OA_OUT}"

# ============================================================
# Stage C: Interactive simulation demo
# ============================================================
echo "=== Stage C: Interactive Simulation Demo ==="
.venv/bin/python3 -u << 'PYEOF'
import torch
from pathlib import Path
from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.data.lob_dataset import LOBVideoDataset
from agentdiffusion.infer.interactive_sim import InteractiveSimulator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VDIT_DIR = Path("outputs/vdit_lob_4x4")
OB = "data/external/lobster/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
MSG = "data/external/lobster/AMZN_2012-06-21_34200000_57600000_message_10.csv"

# Load dataset for seed frames
dataset = LOBVideoDataset(OB, MSG, total_frames=20, cond_frames=4, subsample=10, grid_shape=(4, 4))
d_latent = dataset[0]["frames"].shape[-1]

# Load model
ckpt_path = sorted(VDIT_DIR.glob("video_dit_step_*.pt"))[-1]
model = VideoDiT(
    d_latent=d_latent, d_model=128, depth=6, heads=4,
    patch_size=2, num_frames=20, num_cond_frames=4,
    grid_h=4, grid_w=4,
).to(device)
ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
model.load_state_dict(ckpt.get("ema", ckpt["model"]))
model.eval()

scheduler = NoiseScheduler(1000, "cosine").to(device)
sampler = VideoDDIMSampler(model, scheduler, "v_prediction", ddim_steps=50, eta=0.0)

# Create interactive simulator with SDEdit recalibration every 5 rounds
sim = InteractiveSimulator(
    model, sampler, num_cond=4, num_gen=16, zero_sum_proj=True,
    recalibrate_every=5, recalibrate_strength=0.3, scheduler=scheduler,
)

# Init with real LOB frames
seed = dataset[0]["frames"][:4]  # [4, 16, 16, d_latent]
sim.init(seed)
print(f"Initialized with {seed.shape[0]} seed frames, grid={seed.shape[1]}x{seed.shape[2]}, d={seed.shape[3]}")

# Run 5 rounds of generation
print("\n=== Rolling simulation (5 rounds × 16 frames = 80 frames) ===")
for i in range(5):
    gen = sim.step()
    traj = sim.get_trajectory(dim=0)
    print(f"Round {i+1}: buffer={sim.buffer.shape[1]} frames, "
          f"latest mean_pos={traj[-1]:.4f}")

    # Inject shock at round 3: spike position dim 0
    if i == 2:
        print("  >>> Injecting shock: +2.0 to all positions <<<")
        shock = torch.zeros(d_latent)
        shock[0] = 2.0
        sim.intervene(frame_idx=-1, delta=shock)

    sim.trim_buffer(keep_last=8)

# Final trajectory
traj = sim.get_trajectory(dim=0)
print(f"\nFull trajectory ({len(traj)} frames): "
      f"start={traj[0]:.4f}, end={traj[-1]:.4f}")
print("Interactive simulation complete.")
PYEOF

echo "=== ALL DONE ==="
