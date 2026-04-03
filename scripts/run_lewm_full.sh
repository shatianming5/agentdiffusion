#!/bin/bash
set -Eeuo pipefail
cd ~/agentdiffusion
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=3

echo "=== LeWorldModel Full Pipeline (with Decoder) ==="
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
# Phase 2: Train LeWorldModel WITH decoder (50K steps)
# -------------------------------------------------------
echo "=== Phase 2: Train LeWorldModel with Decoder (50K steps) ==="
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
    lewm.num_projections=512 lewm.lambda_sigreg=0.5 \
    lewm.lambda_price=10.0 lewm.lambda_returns=5.0 \
    lewm.use_decoder=true \
    lewm.d_dec=256 lewm.dec_depth=4 lewm.dec_heads=4 lewm.dec_mlp_ratio=4.0 \
    lewm.lambda_recon=1.0 \
    train.total_steps=50000 train.batch_size=16 \
    train.lr=3e-4 train.weight_decay=0.05 \
    train.warmup_steps=2000 \
    train.log_every=100 train.save_every=10000 \
    data.num_workers=4 data.pin_memory=true \
    output_dir=outputs/lewm_full

# -------------------------------------------------------
# Phase 3: Full Evaluation
# -------------------------------------------------------
echo "=== Phase 3: Full Evaluation ==="
.venv/bin/python3 << 'PYEOF'
import sys, time, torch, numpy as np
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from agentdiffusion.models.lewm import LeWorldModel, masked_mean
from agentdiffusion.data.dataset import _pad_grid
from agentdiffusion.constraints.projection import apply_all_projections
from agentdiffusion.eval.stylized_facts import evaluate_stylized_facts
from agentdiffusion.eval.emergence import run_emergence_analysis

PAD_H, PAD_W = 36, 36
D_AGENT = 128
MID_PRICE_DIM = 98  # observation slice starts at 96, mid price offset = 2

# -------------------------------------------------------------------
# Load trained model (with decoder)
# -------------------------------------------------------------------
print("\n--- Loading model ---")
model = LeWorldModel(
    d_agent=D_AGENT, d_enc=256, d_latent=256, d_pred=384, d_cond=32,
    patch_size=4, enc_depth=6, enc_heads=8, pred_depth=6, pred_heads=8,
    enc_mlp_ratio=4.0, pred_mlp_ratio=2.0,
    num_projections=512, lambda_sigreg=0.5, lambda_recon=1.0,
    lambda_price=10.0, lambda_returns=5.0,
    use_decoder=True, d_dec=256, dec_depth=4, dec_heads=4, dec_mlp_ratio=4.0,
    dec_grid_h=PAD_H, dec_grid_w=PAD_W,
).to(device)

ckpt_path = "outputs/lewm_full/lewm_step_50000.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Loaded checkpoint: {ckpt_path}")

total_params = sum(p.numel() for p in model.parameters())
dec_params = sum(p.numel() for p in model.decoder.parameters())
print(f"Total params: {total_params/1e6:.2f}M (decoder: {dec_params/1e6:.2f}M)")

# -------------------------------------------------------------------
# Load test data (last few samples for evaluation)
# -------------------------------------------------------------------
files = sorted(Path("data/abides_norm").glob("*.pt"))
assert len(files) > 0, "No data files found"
num_test = min(10, len(files))
test_files = files[-num_test:]
print(f"Using {num_test} test samples from {len(files)} total")

# Load first test sample for single-step eval
sample = torch.load(test_files[0], map_location=device, weights_only=True)
state_t = _pad_grid(sample["state_t"], PAD_H, PAD_W).unsqueeze(0).to(device)
state_t1_gt = _pad_grid(sample["state_t1"], PAD_H, PAD_W).unsqueeze(0).to(device)
mc = sample["market_cond"].unsqueeze(0).to(device)

# ===================================================================
# (a) Single-step evaluation: encode, predict, decode, compare raw
# ===================================================================
print("\n" + "="*60)
print("(a) Single-step Evaluation (latent + raw space)")
print("="*60)

with torch.no_grad():
    z_t = model.encode(state_t)
    z_t1_gt_latent = model.encode(state_t1_gt)
    z_t1_pred = model.predict(z_t, mc)

    # Latent space metrics
    latent_mse = (z_t1_pred - z_t1_gt_latent).pow(2).mean().item()
    cosine_sim = torch.nn.functional.cosine_similarity(
        z_t1_pred, z_t1_gt_latent, dim=-1
    ).mean().item()

    # Decode predicted latent to raw space
    state_t1_pred_raw = model.decode(z_t1_pred)    # [1, H, W, C]

    # Raw space MSE: predicted vs ground truth
    raw_mse = (state_t1_pred_raw - state_t1_gt).pow(2).mean().item()

    # Also check reconstruction quality: encode then decode
    state_t_recon = model.decode(z_t)
    recon_mse = (state_t_recon - state_t).pow(2).mean().item()

print(f"  Latent MSE (pred vs gt):     {latent_mse:.6f}")
print(f"  Latent cosine sim:           {cosine_sim:.4f}")
print(f"  Raw-space MSE (pred vs gt):  {raw_mse:.6f}")
print(f"  Reconstruction MSE (AE):     {recon_mse:.6f}")
print(f"  z_t std (per-dim mean):      {z_t.std(dim=0).mean().item():.4f}")

# ===================================================================
# (b) Apply constraint projections
# ===================================================================
print("\n" + "="*60)
print("(b) Constraint Projections")
print("="*60)

with torch.no_grad():
    state_t1_proj = apply_all_projections(
        state_t, state_t1_pred_raw, target_totals=None, max_leverage=10.0,
    )
    proj_mse = (state_t1_proj - state_t1_gt).pow(2).mean().item()
    delta_from_unprojected = (state_t1_proj - state_t1_pred_raw).pow(2).mean().item()

print(f"  Post-projection MSE vs gt:   {proj_mse:.6f}")
print(f"  Projection adjustment MSE:   {delta_from_unprojected:.6f}")

# ===================================================================
# (c) 200-step rollout: latent predict -> decode -> project
# ===================================================================
print("\n" + "="*60)
print("(c) 200-step Rollout (latent -> decode -> project)")
print("="*60)

ROLLOUT_STEPS = 200

with torch.no_grad():
    z_current = z_t.clone()
    decoded_states = [state_t]  # initial state
    z_trajectory = [z_t]

    for step in range(ROLLOUT_STEPS):
        # Predict next latent
        z_current = model.predict(z_current, mc)
        z_trajectory.append(z_current)

        # Decode to raw space
        state_raw = model.decode(z_current)

        # Apply constraint projections
        state_proj = apply_all_projections(
            decoded_states[-1], state_raw, target_totals=None, max_leverage=10.0,
        )
        decoded_states.append(state_proj)

    # Stack trajectories
    z_traj = torch.stack(z_trajectory, dim=0)       # [201, 1, d_latent]
    state_traj = torch.stack(decoded_states, dim=0)  # [201, 1, H, W, C]

# Latent trajectory stats
z_drift = (z_traj[-1] - z_traj[0]).pow(2).mean().item()
z_var_over_time = z_traj.squeeze(1).var(dim=0).mean().item()
z_stds = z_traj.squeeze(1).std(dim=0)
active_dims = (z_stds > 0.01).sum().item()

print(f"  Rollout drift (MSE first->last):  {z_drift:.4f}")
print(f"  Rollout variance (per-dim mean):  {z_var_over_time:.4f}")
print(f"  Active latent dims (std>0.01):    {active_dims}/{z_stds.shape[0]}")

# Raw trajectory stats
state_traj_np = state_traj.squeeze(1).cpu().numpy()  # [201, H, W, C]
raw_drift = np.mean((state_traj_np[-1] - state_traj_np[0])**2)
print(f"  Raw state drift (MSE):            {raw_drift:.4f}")

# ===================================================================
# (d) Extract price series from rollout (dim 98 = mid price)
# ===================================================================
print("\n" + "="*60)
print("(d) Price Series Extraction")
print("="*60)

# Mid price is at dim 98 (observation slice 96:112, offset 2 = mid price)
# Use masked mean to exclude padding agents (padding has all-zero features)
# Load agent_types from a sample to build the validity mask
_sample_at = sample.get("agent_types", None)
if _sample_at is not None:
    _at_padded = _pad_grid(_sample_at.cpu(), PAD_H, PAD_W)
    valid_mask = (_at_padded != -1).numpy()  # [H, W]
else:
    # Fallback: non-padding = any non-zero feature in dims 0 or 1
    valid_mask = (state_traj_np[0, :, :, 0] != 0) | (state_traj_np[0, :, :, 1] != 0)

price_raw = state_traj_np[:, :, :, MID_PRICE_DIM]  # [201, H, W]
valid_f = valid_mask.astype(np.float32)[None, :, :]  # [1, H, W]
valid_sum = valid_f.sum()
prices = (price_raw * valid_f).sum(axis=(1, 2)) / max(valid_sum, 1.0)  # [201]
print(f"  Price series length:   {len(prices)}")
print(f"  Price range:           [{prices.min():.4f}, {prices.max():.4f}]")
print(f"  Price mean:            {prices.mean():.4f}")
print(f"  Price std:             {prices.std():.4f}")

# ===================================================================
# (e) Stylized facts evaluation
# ===================================================================
print("\n" + "="*60)
print("(e) Stylized Facts Evaluation")
print("="*60)

sf_report = evaluate_stylized_facts(prices, volumes=None)
print(f"  Fat tail alpha:             {sf_report.fat_tail_alpha:.3f}  "
      f"{'PASS' if sf_report.fat_tail_pass else 'FAIL'}")
print(f"  Volatility clustering:      "
      f"{'PASS' if sf_report.volatility_clustering_pass else 'FAIL'}")
print(f"  Leverage effect (corr):     {sf_report.leverage_effect_corr:.4f}  "
      f"{'PASS' if sf_report.leverage_effect_pass else 'FAIL'}")
print(f"  Volume-volatility corr:     {sf_report.volume_volatility_corr:.4f}  "
      f"{'PASS' if sf_report.volume_volatility_pass else 'FAIL'}")
print(f"  Return autocorr:            "
      f"{'PASS' if sf_report.return_autocorr_pass else 'FAIL'}")
print(f"  Gain-loss asymmetry (p):    {sf_report.gain_loss_asymmetry_pvalue:.4f}  "
      f"{'PASS' if sf_report.gain_loss_asymmetry_pass else 'FAIL'}")
print(f"  >>> {sf_report.summary}")

# ===================================================================
# (f) Emergence detection
# ===================================================================
print("\n" + "="*60)
print("(f) Emergence Detection")
print("="*60)

emergence = run_emergence_analysis(prices, volumes=None, spreads=None)
total_events = 0
for etype, events in emergence.items():
    print(f"  {etype}: {len(events)} events detected")
    total_events += len(events)
    for e in events[:3]:  # show at most 3 per type
        print(f"    step {e.start_step}-{e.end_step}: severity={e.severity:.3f} | {e.description}")
print(f"  Total emergence events:     {total_events}")

# ===================================================================
# (g) Speedup benchmark
# ===================================================================
print("\n" + "="*60)
print("(g) Speedup Benchmark")
print("="*60)

# Latent-only rollout speed (predict only)
if device.type == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    z_bench = z_t.clone()
    for _ in range(ROLLOUT_STEPS):
        z_bench = model.predict(z_bench, mc)
if device.type == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()
latent_only_ms = (t1 - t0) / ROLLOUT_STEPS * 1000

# Full pipeline: predict + decode + project
if device.type == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    z_bench = z_t.clone()
    prev_state = state_t.clone()
    for _ in range(ROLLOUT_STEPS):
        z_bench = model.predict(z_bench, mc)
        raw = model.decode(z_bench)
        prev_state = apply_all_projections(prev_state, raw, max_leverage=10.0)
if device.type == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()
full_pipeline_ms = (t1 - t0) / ROLLOUT_STEPS * 1000

# Encode speed
if device.type == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(100):
        _ = model.encode(state_t)
if device.type == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()
encode_ms = (t1 - t0) / 100 * 1000

# Decode speed
if device.type == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(100):
        _ = model.decode(z_t)
if device.type == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()
decode_ms = (t1 - t0) / 100 * 1000

print(f"  Encode speed:                  {encode_ms:.2f} ms/step")
print(f"  Decode speed:                  {decode_ms:.2f} ms/step")
print(f"  Latent-only rollout:           {latent_only_ms:.2f} ms/step")
print(f"  Full pipeline (pred+dec+proj): {full_pipeline_ms:.2f} ms/step")
print(f"  Decode overhead per step:      {full_pipeline_ms - latent_only_ms:.2f} ms")

# ===================================================================
# (h) Full comparison table
# ===================================================================
print("\n" + "="*60)
print("(h) Full Comparison Table")
print("="*60)

results = {
    "Latent MSE (single-step)":       latent_mse,
    "Cosine sim (single-step)":       cosine_sim,
    "Raw MSE (single-step pred)":     raw_mse,
    "Recon MSE (encode-decode)":      recon_mse,
    "Post-proj MSE vs GT":            proj_mse,
    "Projection adjustment MSE":      delta_from_unprojected,
    "Latent drift (200-step)":        z_drift,
    "Latent variance (200-step)":     z_var_over_time,
    "Active latent dims":             active_dims,
    "Raw state drift (200-step)":     raw_drift,
    "Stylized facts passed":          sf_report.total_passed,
    "Fat tail alpha":                 sf_report.fat_tail_alpha,
    "Emergence events":               total_events,
    "Encode ms/step":                 encode_ms,
    "Decode ms/step":                 decode_ms,
    "Latent rollout ms/step":         latent_only_ms,
    "Full pipeline ms/step":          full_pipeline_ms,
}

print(f"{'Metric':<35} {'Value':>12}")
print("-" * 49)
for k, v in results.items():
    if isinstance(v, float):
        print(f"  {k:<33} {v:>12.4f}")
    else:
        print(f"  {k:<33} {v:>12}")

print("\n=== LeWM Full Evaluation Complete ===")
PYEOF

echo "=== LeWorldModel Full Pipeline Complete ==="
