#!/bin/bash
set -Eeuo pipefail
cd ~/agentdiffusion
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=3

echo "=== LeWorldModel Training Pipeline ==="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Data: data/abides_norm"
echo "Steps: 50K"
echo ""

# -------------------------------------------------------
# Phase 1: Verify data exists
# -------------------------------------------------------
echo "=== Phase 1: Verify data ==="
.venv/bin/python3 -c "
from pathlib import Path
files = sorted(Path('data/abides_norm').glob('*.pt'))
assert len(files) > 0, 'No data files found in data/abides_norm'
print(f'Found {len(files)} data files')
import torch
s = torch.load(files[0], map_location='cpu', weights_only=True)
print(f'state_t shape: {s[\"state_t\"].shape}')
print(f'market_cond shape: {s[\"market_cond\"].shape}')
"

# -------------------------------------------------------
# Phase 2: Train LeWorldModel (50K steps)
# -------------------------------------------------------
echo "=== Phase 2: Train LeWorldModel (50K steps) ==="
.venv/bin/python3 -m agentdiffusion.train.train_lewm \
    --config configs/train/stage_lewm.yaml \
    data.data_dir=data/abides_norm \
    patch.grid_h=34 patch.grid_w=33 \
    patch.patch_size=4 \
    lewm.d_enc=256 lewm.d_latent=256 \
    lewm.d_pred=384 lewm.d_cond=32 \
    lewm.enc_depth=6 lewm.enc_heads=8 \
    lewm.pred_depth=6 lewm.pred_heads=8 \
    lewm.enc_mlp_ratio=4.0 lewm.pred_mlp_ratio=2.0 \
    lewm.num_projections=512 lewm.lambda_sigreg=5.0 \
    train.total_steps=50000 train.batch_size=16 \
    train.lr=3e-4 train.weight_decay=0.05 \
    train.warmup_steps=2000 \
    train.log_every=100 train.save_every=10000 \
    data.num_workers=4 data.pin_memory=true \
    output_dir=outputs/lewm

# -------------------------------------------------------
# Phase 3: Evaluation
# -------------------------------------------------------
echo "=== Phase 3: Evaluation ==="
.venv/bin/python3 << 'PYEOF'
import sys, torch, numpy as np
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from agentdiffusion.models.lewm import LeWorldModel
from agentdiffusion.data.dataset import _pad_grid

# Load trained model (~15M params)
model = LeWorldModel(
    d_agent=128, d_enc=256, d_latent=256, d_pred=384, d_cond=32,
    patch_size=4, enc_depth=6, enc_heads=8, pred_depth=6, pred_heads=8,
    enc_mlp_ratio=4.0, pred_mlp_ratio=2.0,
    num_projections=512, lambda_sigreg=0.1,
).to(device)

ckpt = torch.load("outputs/lewm/lewm_step_50000.pt", map_location=device, weights_only=True)
model.load_state_dict(ckpt["model"])
model.eval()

# Load test data
files = sorted(Path("data/abides_norm").glob("*.pt"))
sample = torch.load(files[-1], map_location=device, weights_only=True)
state_t = _pad_grid(sample["state_t"], 36, 36).unsqueeze(0).to(device)
state_t1_gt = _pad_grid(sample["state_t1"], 36, 36).unsqueeze(0).to(device)
mc = sample["market_cond"].unsqueeze(0).to(device)

# Single-step evaluation: encode ground truth, predict, compare
with torch.no_grad():
    z_t = model.encode(state_t)
    z_t1_gt = model.encode(state_t1_gt)
    z_t1_pred = model.predict(z_t, mc)

    latent_mse = (z_t1_pred - z_t1_gt).pow(2).mean().item()
    cosine_sim = torch.nn.functional.cosine_similarity(
        z_t1_pred, z_t1_gt, dim=-1
    ).mean().item()

print(f"Single-step latent MSE:    {latent_mse:.6f}")
print(f"Single-step cosine sim:    {cosine_sim:.4f}")
print(f"z_t std (per dim mean):    {z_t.std(dim=0).mean().item():.4f}")
print(f"z_t1_pred std:             {z_t1_pred.std(dim=0).mean().item():.4f}")

# Multi-step rollout in latent space (200 steps)
print("\nRunning 200-step latent rollout...")
z_trajectory = [z_t]
z_current = z_t
for step in range(200):
    with torch.no_grad():
        z_current = model.predict(z_current, mc)
    z_trajectory.append(z_current)

z_traj = torch.stack(z_trajectory, dim=0)  # [201, 1, d_latent]
z_drift = (z_traj[-1] - z_traj[0]).pow(2).mean().item()
z_var_over_time = z_traj.squeeze(1).var(dim=0).mean().item()
print(f"Rollout drift (MSE first->last): {z_drift:.4f}")
print(f"Rollout variance (per dim mean): {z_var_over_time:.4f}")

# Check for collapse: are latent dimensions still active?
z_stds = z_traj.squeeze(1).std(dim=0)  # [d_latent]
active_dims = (z_stds > 0.01).sum().item()
print(f"Active latent dims (std>0.01):   {active_dims}/{z_stds.shape[0]}")

# Speedup benchmark
import time
if device.type == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(200):
    with torch.no_grad():
        _ = model.predict(z_t, mc)
if device.type == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"\nLatent rollout speed: {(t1-t0)/200*1000:.2f}ms per step")

print("\n=== LeWM Evaluation Complete ===")
PYEOF

echo "=== LeWorldModel Pipeline Complete ==="
