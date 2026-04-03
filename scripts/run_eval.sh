#!/bin/bash
set -Eeuo pipefail
cd ~/agentdiffusion
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4

echo '=== Running Evaluation ==='
.venv/bin/python3 << 'PYEOF'
import sys, torch, numpy as np
from pathlib import Path

# --- 1. Load models ---
print("=== Loading models ===")
from agentdiffusion.models.autoencoder import AgentAutoencoder
from agentdiffusion.models.agent_dit import AgentDiT
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.diffusion.ddim import DDIMSampler
from agentdiffusion.constraints.projection import apply_all_projections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# AE
ae = AgentAutoencoder(raw_dim=128, latent_dim=16).to(device)
ae_ckpt = torch.load("outputs/stage1_ae_real/ae_step_5000.pt", map_location=device, weights_only=True)
ae.load_state_dict(ae_ckpt["model"])
ae.eval()
print("AE loaded")

# DiT
model = AgentDiT(
    raw_dim=128, latent_dim=16, d_model=128, depth=4, heads=4,
    patch_size=4, num_market_tokens=16, local_window_size=4,
).to(device)
dit_ckpt = torch.load("outputs/stage2_dit_real/dit_step_5000.pt", map_location=device, weights_only=True)
model.load_state_dict(dit_ckpt["model"])
model.eval()
print("DiT loaded")

# Scheduler + Sampler
sched = NoiseScheduler(200, "cosine").to(device)
sampler = DDIMSampler(model, sched, "v_prediction", ddim_steps=20)

# --- 2. Load test data ---
print("\n=== Loading test data ===")
from agentdiffusion.data.dataset import _pad_grid
files = sorted(Path("data/abides_real").glob("*.pt"))
test_files = files[-20:]  # last 20 samples as test
print(f"Test samples: {len(test_files)}")

# --- 3. Single-step generation test ---
print("\n=== Single-step inference ===")
sample = torch.load(test_files[0], map_location=device, weights_only=True)
state_t = _pad_grid(sample["state_t"], 36, 36).unsqueeze(0).to(device)
state_t1_gt = _pad_grid(sample["state_t1"], 36, 36).unsqueeze(0).to(device)
mc = sample["market_cond"].unsqueeze(0).to(device)

with torch.no_grad():
    z_shape = (1, 36, 36, 16)
    z_pred = sampler.sample(z_shape, mc, device=device)
    state_t1_pred = ae.decode(z_pred)
    state_t1_pred = apply_all_projections(state_t, state_t1_pred)

mse = (state_t1_pred - state_t1_gt).pow(2).mean().item()
print(f"Single-step MSE: {mse:.4f}")

# Clearing violation check
delta = state_t1_pred[..., :32] - state_t[..., :32]
clearing = delta.sum(dim=(1,2)).abs().max().item()
print(f"Clearing violation (after projection): {clearing:.6f}")

# --- 4. Multi-step rollout ---
print("\n=== Multi-step rollout (10 steps) ===")
states = [state_t]
for step in range(10):
    with torch.no_grad():
        z_next = sampler.sample(z_shape, mc, device=device)
        s_next = ae.decode(z_next)
        s_next = apply_all_projections(states[-1], s_next)
    states.append(s_next)

trajectory = torch.stack(states, dim=0)  # [11, 1, 36, 36, 128]
print(f"Trajectory shape: {trajectory.shape}")

# Extract price proxy
prices = trajectory[:, 0, :, :, 0].mean(dim=(1,2)).cpu().numpy()
print(f"Price proxy over 10 steps: {prices}")

# --- 5. Stylized Facts ---
print("\n=== Stylized Facts (on generated trajectory) ===")
from agentdiffusion.eval.stylized_facts import evaluate_stylized_facts
if len(prices) > 5:
    # Extend with more rollout for meaningful stats
    print("Running 100-step rollout for stylized facts...")
    for step in range(90):
        with torch.no_grad():
            z_next = sampler.sample(z_shape, mc, device=device)
            s_next = ae.decode(z_next)
            s_next = apply_all_projections(states[-1], s_next)
        states.append(s_next)

    long_traj = torch.stack(states, dim=0)
    long_prices = long_traj[:, 0, :, :, 0].mean(dim=(1,2)).cpu().numpy()
    long_prices = np.clip(long_prices, 1, None)  # ensure positive

    report = evaluate_stylized_facts(long_prices)
    print(f"Results: {report.summary}")
    print(f"  Fat tail alpha: {report.fat_tail_alpha:.2f} ({'PASS' if report.fat_tail_pass else 'FAIL'})")
    print(f"  Vol clustering: {'PASS' if report.volatility_clustering_pass else 'FAIL'}")
    print(f"  Leverage effect: {report.leverage_effect_corr:.3f} ({'PASS' if report.leverage_effect_pass else 'FAIL'})")
    print(f"  Return autocorr: {'PASS' if report.return_autocorr_pass else 'FAIL'}")

# --- 6. Speedup estimate ---
print("\n=== Speedup estimate ===")
import time
# Diffusion inference time (10 steps)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    with torch.no_grad():
        _ = sampler.sample(z_shape, mc, device=device)
torch.cuda.synchronize()
diff_time = time.perf_counter() - t0
print(f"Diffusion 10-step: {diff_time:.2f}s ({diff_time/10:.3f}s per step)")
print(f"Grid size: 36x36 = 1296 agents")

# --- 7. Emergence detection ---
print("\n=== Emergence detection ===")
from agentdiffusion.eval.emergence import run_emergence_analysis
events = run_emergence_analysis(long_prices)
for etype, evts in events.items():
    if evts:
        print(f"  {etype}: {len(evts)} events")
    else:
        print(f"  {etype}: none")

print("\n=== EVALUATION COMPLETE ===")
PYEOF
