#!/bin/bash
set -Eeuo pipefail
cd ~/agentdiffusion
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4

echo "=== Phase 1: Regenerate normalized data (2000 sims) ==="
.venv/bin/python3 -c "
import logging; logging.disable(logging.INFO)
from agentdiffusion.data.abides_generator import generate_abides_dataset
generate_abides_dataset(
    output_dir='data/abides_norm',
    num_simulations=2000,
    seed_start=0,
    end_time='11:00:00',
    num_snapshots=20,
)
"

echo "=== Phase 2: Verify data normalization ==="
.venv/bin/python3 -c "
import torch
from pathlib import Path
files = sorted(Path('data/abides_norm').glob('*.pt'))
print(f'Total files: {len(files)}')
s = torch.load(files[0], map_location='cpu', weights_only=True)
t = s['state_t']
print(f'Shape: {t.shape}')
print(f'Global: min={t.min():.4f}, max={t.max():.4f}, mean={t.mean():.4f}, std={t.std():.4f}')
for name, sl in [('position[0]', 0), ('cash_rel[1]', 1), ('leverage[2]', 2), ('price[98]', 98)]:
    v = t[:, :, sl]
    print(f'  {name}: min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}')
"

echo "=== Phase 3: Train AE (10K steps) ==="
.venv/bin/python3 -m agentdiffusion.train.train_ae \
    --config configs/train/stage1_ae.yaml \
    data.data_dir=data/abides_norm patch.grid_h=34 patch.grid_w=33 \
    train.total_steps=10000 train.batch_size=2048 \
    train.log_every=200 train.save_every=5000 \
    data.num_workers=0 data.pin_memory=false \
    output_dir=outputs/ae_norm

echo "=== Phase 4: Train Diffusion - Large (50K steps) ==="
.venv/bin/python3 -m agentdiffusion.train.train_diffusion \
    --config configs/train/stage2_diffusion.yaml \
    --ae-ckpt outputs/ae_norm/ae_step_10000.pt \
    data.data_dir=data/abides_norm patch.grid_h=34 patch.grid_w=33 \
    patch.patch_size=4 \
    model.d_model=512 model.depth=12 model.heads=8 \
    model.num_market_tokens=64 model.local_window_size=4 \
    agent.latent_dim=16 \
    diffusion.timesteps=1000 \
    train.total_steps=50000 train.batch_size=8 \
    train.log_every=100 train.save_every=10000 \
    data.num_workers=0 data.pin_memory=false \
    output_dir=outputs/dit_large

echo "=== Phase 5: Evaluation ==="
.venv/bin/python3 << 'PYEOF'
import sys, torch, numpy as np
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from agentdiffusion.models.autoencoder import AgentAutoencoder
from agentdiffusion.models.agent_dit import AgentDiT
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.diffusion.ddim import DDIMSampler
from agentdiffusion.constraints.projection import apply_all_projections
from agentdiffusion.data.dataset import _pad_grid

# Load models
ae = AgentAutoencoder(raw_dim=128, latent_dim=16).to(device)
ae.load_state_dict(torch.load("outputs/ae_norm/ae_step_10000.pt", map_location=device, weights_only=True)["model"])
ae.eval()

model = AgentDiT(
    raw_dim=128, latent_dim=16, d_model=512, depth=12, heads=8,
    patch_size=4, num_market_tokens=64, local_window_size=4,
).to(device)
model.load_state_dict(torch.load("outputs/dit_large/dit_step_50000.pt", map_location=device, weights_only=True)["model"])
model.eval()

sched = NoiseScheduler(1000, "cosine").to(device)
sampler = DDIMSampler(model, sched, "v_prediction", ddim_steps=50)

# Test data
files = sorted(Path("data/abides_norm").glob("*.pt"))
sample = torch.load(files[-1], map_location=device, weights_only=True)
state_t = _pad_grid(sample["state_t"], 36, 36).unsqueeze(0).to(device)
state_t1_gt = _pad_grid(sample["state_t1"], 36, 36).unsqueeze(0).to(device)
mc = sample["market_cond"].unsqueeze(0).to(device)

# Single-step inference
with torch.no_grad():
    z_pred = sampler.sample((1, 36, 36, 16), mc, device=device)
    pred = ae.decode(z_pred)
    pred = apply_all_projections(state_t, pred)

mse = (pred - state_t1_gt).pow(2).mean().item()
delta = pred[..., :32] - state_t[..., :32]
clearing = delta.sum(dim=(1,2)).abs().max().item()
print(f"Single-step MSE: {mse:.6f}")
print(f"Clearing violation: {clearing:.8f}")

# 200-step rollout
print("Running 200-step rollout...")
states = [state_t]
for _ in range(200):
    with torch.no_grad():
        z = sampler.sample((1, 36, 36, 16), mc, device=device)
        s = ae.decode(z)
        s = apply_all_projections(states[-1], s)
    states.append(s)

traj = torch.stack(states, dim=0)
# Use masked mean to exclude padding agents (padding has all-zero features)
# traj shape: [T+1, B, H, W, C]
agent_mask = (traj[:, 0, :, :, 0] != 0) | (traj[:, 0, :, :, 1] != 0)  # [T+1, H, W]
raw_prices = traj[:, 0, :, :, 98]  # [T+1, H, W] — dim 98 = mid price
masked_prices = raw_prices * agent_mask.float()
prices = (masked_prices.sum(dim=(1,2)) / agent_mask.float().sum(dim=(1,2)).clamp(min=1)).cpu().numpy()
prices = np.clip(prices, 0.01, None)

from agentdiffusion.eval.stylized_facts import evaluate_stylized_facts
report = evaluate_stylized_facts(prices)
print(f"\nStylized Facts: {report.summary}")
print(f"  Fat tail alpha: {report.fat_tail_alpha:.2f} ({'PASS' if report.fat_tail_pass else 'FAIL'})")
print(f"  Vol clustering: {'PASS' if report.volatility_clustering_pass else 'FAIL'}")
print(f"  Leverage effect: {report.leverage_effect_corr:.4f} ({'PASS' if report.leverage_effect_pass else 'FAIL'})")
print(f"  Return autocorr: {'PASS' if report.return_autocorr_pass else 'FAIL'}")
print(f"  Gain/loss asym: {'PASS' if report.gain_loss_asymmetry_pass else 'FAIL'}")

from agentdiffusion.eval.emergence import run_emergence_analysis
events = run_emergence_analysis(prices)
for etype, evts in events.items():
    print(f"  {etype}: {len(evts)} events")

# Speedup
import time
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    with torch.no_grad():
        _ = sampler.sample((1, 36, 36, 16), mc, device=device)
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"\nSpeedup: {(t1-t0)/50:.3f}s per step (50-step DDIM, 1296 agents)")

# Save visualization
from agentdiffusion.eval.visualize import plot_stylized_facts
plot_stylized_facts(report, prices, "outputs/eval_plots", prefix="large")
print("Plots saved to outputs/eval_plots/")
PYEOF

echo "=== ALL DONE ==="
