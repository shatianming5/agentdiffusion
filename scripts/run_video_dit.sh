#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHON="${PYTHON:-.venv/bin/python3}"
DATA_DIR="${DATA_DIR:-data/abides_video}"
AE_CKPT="${AE_CKPT:-outputs/ae_norm/ae_step_10000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/video_dit}"
NUM_SIMS="${NUM_SIMS:-200}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"

echo "============================================================"
echo "  Video DiT Pipeline"
echo "  Phase 1: Generate ABIDES sequence data (40-frame sequences)"
echo "  Phase 2: Train Video DiT (${TRAIN_STEPS} steps)"
echo "  Phase 3: Evaluate (generate 32 frames from 8 real)"
echo "============================================================"

# ---------------------------------------------------------------
# Phase 1: Generate ABIDES sequence data (40-frame sequences)
# ---------------------------------------------------------------
echo ""
echo "=== Phase 1: Generate ABIDES sequence data ==="

if [ -d "${DATA_DIR}" ] && [ "$(ls ${DATA_DIR}/*.pt 2>/dev/null | head -1)" ]; then
    echo "Data directory ${DATA_DIR} already exists with .pt files, skipping generation."
else
    mkdir -p "${DATA_DIR}"
    ${PYTHON} -c "
import logging; logging.disable(logging.INFO)
from agentdiffusion.data.abides_generator import generate_abides_dataset
generate_abides_dataset(
    output_dir='${DATA_DIR}',
    num_simulations=${NUM_SIMS},
    seed_start=0,
    end_time='11:30:00',
    num_snapshots=50,
)
"
    echo "Generated data in ${DATA_DIR}"
fi

echo "Data files: $(ls ${DATA_DIR}/*.pt 2>/dev/null | wc -l | tr -d ' ') .pt files"

# ---------------------------------------------------------------
# Phase 2: Train AE if checkpoint not found
# ---------------------------------------------------------------
echo ""
echo "=== Phase 2: Verify AE checkpoint ==="

if [ ! -f "${AE_CKPT}" ]; then
    echo "AE checkpoint not found at ${AE_CKPT}. Training AE first..."
    ${PYTHON} -m agentdiffusion.train.train_ae \
        --config configs/train/stage1_ae.yaml \
        data.data_dir="${DATA_DIR}" \
        patch.grid_h=36 patch.grid_w=36 \
        train.total_steps=10000 train.batch_size=2048 \
        train.log_every=200 train.save_every=5000 \
        data.num_workers=0 data.pin_memory=false \
        output_dir=outputs/ae_norm
    AE_CKPT="outputs/ae_norm/ae_step_10000.pt"
    echo "AE trained and saved to ${AE_CKPT}"
else
    echo "Using existing AE checkpoint: ${AE_CKPT}"
fi

# ---------------------------------------------------------------
# Phase 3: Train Video DiT
# ---------------------------------------------------------------
echo ""
echo "=== Phase 3: Train Video DiT (${TRAIN_STEPS} steps) ==="

${PYTHON} -m agentdiffusion.train.train_video_dit \
    --config configs/train/stage_video_dit.yaml \
    --ae-ckpt "${AE_CKPT}" \
    data.data_dir="${DATA_DIR}" \
    patch.grid_h=36 patch.grid_w=36 \
    patch.patch_size=4 \
    model.d_model=256 model.depth=6 model.heads=4 \
    agent.latent_dim=16 \
    video.num_frames=40 video.num_cond_frames=8 \
    diffusion.timesteps=1000 \
    train.total_steps="${TRAIN_STEPS}" train.batch_size=8 \
    train.log_every=100 train.save_every=10000 \
    data.num_workers=0 data.pin_memory=false \
    output_dir="${OUTPUT_DIR}"

echo "Video DiT training complete."

# ---------------------------------------------------------------
# Phase 4: Evaluate — generate 32 frames from 8 real, compute fidelity
# ---------------------------------------------------------------
echo ""
echo "=== Phase 4: Evaluation ==="

${PYTHON} << 'PYEOF'
import sys
import torch
import numpy as np
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from agentdiffusion.models.autoencoder import AgentAutoencoder
from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.data.video_dataset import AgentVideoDataset, SyntheticVideoDataset

import os
data_dir = os.environ.get("DATA_DIR", "data/abides_video")
output_dir = os.environ.get("OUTPUT_DIR", "outputs/video_dit")
ae_ckpt = os.environ.get("AE_CKPT", "outputs/ae_norm/ae_step_10000.pt")

# Load AE
ae = AgentAutoencoder(raw_dim=128, latent_dim=16).to(device)
ae.load_state_dict(
    torch.load(ae_ckpt, map_location=device, weights_only=True)["model"]
)
ae.eval()

# Load Video DiT
model = VideoDiT(
    d_latent=16, d_model=256, depth=6, heads=4,
    patch_size=4, grid_h=36, grid_w=36,
    num_frames=40, num_cond_frames=8,
).to(device)

# Find latest checkpoint
ckpt_files = sorted(Path(output_dir).glob("video_dit_step_*.pt"))
if not ckpt_files:
    print("[ERROR] No Video DiT checkpoints found.")
    sys.exit(1)
ckpt_path = ckpt_files[-1]
print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(ckpt["model"])
model.eval()

# Scheduler + sampler
scheduler = NoiseScheduler(1000, "cosine").to(device)
sampler = VideoDDIMSampler(model, scheduler, "v_prediction", ddim_steps=50)

# Load test data (first sequence)
try:
    dataset = AgentVideoDataset(
        data_dir=data_dir, total_frames=40, cond_frames=8, pad_to=(36, 36)
    )
except (FileNotFoundError, ValueError):
    dataset = SyntheticVideoDataset(
        num_samples=10, total_frames=40, cond_frames=8, grid_h=36, grid_w=36
    )

sample = dataset[0]
frames = sample["frames"].unsqueeze(0).to(device)  # [1, 40, 36, 36, 128]

# Encode through AE
with torch.no_grad():
    B, T, H, W, C = frames.shape
    latents = ae.encode(frames.reshape(B * T, H, W, C))
    latents = latents.reshape(B, T, H, W, -1)

z_cond = latents[:, :8]    # [1, 8, 36, 36, 16]
z_gen_gt = latents[:, 8:]  # [1, 32, 36, 36, 16]

# Generate 32 frames from 8 condition frames
print("Generating 32 frames from 8 condition frames (50-step DDIM)...")
with torch.no_grad():
    z_gen_pred = sampler.sample(
        x_cond=z_cond,
        gen_shape=(1, 32, 36, 36, 16),
        device=device,
    )

# Decode generated latents
with torch.no_grad():
    gen_decoded = ae.decode(z_gen_pred.reshape(32, 36, 36, 16))
    gen_decoded = gen_decoded.reshape(1, 32, 36, 36, 128)
    gt_decoded = ae.decode(z_gen_gt.reshape(32, 36, 36, 16))
    gt_decoded = gt_decoded.reshape(1, 32, 36, 36, 128)

# Compute fidelity metrics
latent_mse = (z_gen_pred - z_gen_gt).pow(2).mean().item()
decoded_mse = (gen_decoded - gt_decoded).pow(2).mean().item()

print(f"\n=== Fidelity Metrics ===")
print(f"  Latent MSE:  {latent_mse:.6f}")
print(f"  Decoded MSE: {decoded_mse:.6f}")

# Per-frame MSE
print(f"\n=== Per-frame Decoded MSE ===")
for f_idx in range(0, 32, 4):
    frame_mse = (gen_decoded[0, f_idx] - gt_decoded[0, f_idx]).pow(2).mean().item()
    print(f"  Frame {f_idx + 8:2d} (gen {f_idx:2d}): MSE = {frame_mse:.6f}")

# Temporal consistency: measure frame-to-frame differences
gen_diffs = (gen_decoded[0, 1:] - gen_decoded[0, :-1]).pow(2).mean(dim=(1, 2, 3))
gt_diffs = (gt_decoded[0, 1:] - gt_decoded[0, :-1]).pow(2).mean(dim=(1, 2, 3))
print(f"\n=== Temporal Consistency ===")
print(f"  Generated avg frame-to-frame MSE: {gen_diffs.mean().item():.6f}")
print(f"  Ground truth avg frame-to-frame MSE: {gt_diffs.mean().item():.6f}")
print(f"  Ratio (gen/gt): {gen_diffs.mean().item() / max(gt_diffs.mean().item(), 1e-8):.4f}")

# Timing benchmark
import time
torch.cuda.synchronize() if device.type == "cuda" else None
t0 = time.perf_counter()
N_bench = 5
for _ in range(N_bench):
    with torch.no_grad():
        _ = sampler.sample(z_cond, (1, 32, 36, 36, 16), device=device)
if device.type == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"\n=== Inference Speed ===")
print(f"  {(t1 - t0) / N_bench:.3f}s per 32-frame generation (50-step DDIM)")

print("\n=== Evaluation Complete ===")
PYEOF

echo ""
echo "=== ALL DONE ==="
