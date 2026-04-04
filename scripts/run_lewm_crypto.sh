#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

echo "=== LeWorldModel Crypto Training Pipeline ==="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Data: data/external/crypto (Binance BTCUSDT 1-min klines)"
echo "Steps: 50K"
echo ""

# -------------------------------------------------------
# Phase 1: Verify crypto data exists
# -------------------------------------------------------
echo "=== Phase 1: Verify data ==="
.venv/bin/python3 -c "
from pathlib import Path
import zipfile

data_dir = Path('data/external/crypto')
zips = sorted(data_dir.glob('BTCUSDT*.zip'))
assert len(zips) > 0, f'No BTCUSDT zip files in {data_dir}'
print(f'Found {len(zips)} BTCUSDT zip files')

# Quick load test
total_rows = 0
for z in zips:
    with zipfile.ZipFile(z) as zf:
        for name in zf.namelist():
            if name.endswith('.csv'):
                with zf.open(name) as f:
                    lines = f.read().decode('utf-8').strip().splitlines()
                    total_rows += len(lines)
                    print(f'  {z.name}: {len(lines)} bars')
print(f'Total: {total_rows} bars')
print()

# Test dataset loading (first 1000 bars)
from agentdiffusion.data.crypto_dataset import CryptoKlineDataset
ds = CryptoKlineDataset(
    data_dir='data/external/crypto',
    symbol='BTCUSDT',
    window_size=64,
    feature_dim=32,
    stride=1,
    grid_h=8,
    grid_w=8,
)
sample = ds[0]
print(f'Dataset size: {len(ds)} samples')
print(f'state_t shape: {sample[\"state_t\"].shape}')
print(f'state_t1 shape: {sample[\"state_t1\"].shape}')
print(f'market_cond shape: {sample[\"market_cond\"].shape}')
print(f'state_t range: [{sample[\"state_t\"].min():.3f}, {sample[\"state_t\"].max():.3f}]')
print(f'market_cond range: [{sample[\"market_cond\"].min():.3f}, {sample[\"market_cond\"].max():.3f}]')
"

# -------------------------------------------------------
# Phase 2: Train LeWorldModel on crypto data (50K steps)
# -------------------------------------------------------
echo ""
echo "=== Phase 2: Train LeWorldModel on crypto (50K steps) ==="
.venv/bin/python3 -m agentdiffusion.train.train_lewm_crypto \
    --config configs/train/stage_lewm_crypto.yaml \
    agent.raw_dim=32 \
    patch.grid_h=8 patch.grid_w=8 \
    patch.patch_size=2 \
    data.data_dir=data/external/crypto \
    data.symbol=BTCUSDT \
    data.stride=1 \
    lewm.d_enc=256 lewm.d_latent=256 \
    lewm.d_pred=384 lewm.d_cond=32 \
    lewm.enc_depth=6 lewm.enc_heads=8 \
    lewm.pred_depth=6 lewm.pred_heads=8 \
    lewm.use_decoder=true \
    lewm.lambda_price=5.0 lewm.lambda_returns=2.0 \
    lewm.lambda_recon=1.0 lewm.lambda_sigreg=0.5 \
    train.total_steps=50000 train.batch_size=64 \
    train.lr=3e-4 train.weight_decay=0.05 \
    train.warmup_steps=2000 \
    train.rollout_steps=4 train.seq_len=8 \
    train.log_every=100 train.save_every=10000 train.eval_every=5000 \
    train.mixed_precision=bf16 \
    output_dir=outputs/lewm_crypto

# -------------------------------------------------------
# Phase 3: Full evaluation (20 trials, Wasserstein distance)
# -------------------------------------------------------
echo ""
echo "=== Phase 3: Full Evaluation (20 trials) ==="
.venv/bin/python3 << 'PYEOF'
import sys
import numpy as np
import torch
from pathlib import Path
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from agentdiffusion.models.lewm import LeWorldModel
from agentdiffusion.data.crypto_dataset import CryptoKlineDataset
from agentdiffusion.eval.stylized_facts import (
    evaluate_stylized_facts,
    compute_returns,
    return_distribution_wasserstein,
)

# Build model with crypto dimensions
model = LeWorldModel(
    d_agent=32, d_enc=256, d_latent=256, d_pred=384, d_cond=32,
    patch_size=2, enc_depth=6, enc_heads=8, pred_depth=6, pred_heads=8,
    enc_mlp_ratio=4.0, pred_mlp_ratio=2.0,
    use_decoder=True, d_dec=256, dec_depth=4, dec_heads=4,
    dec_grid_h=8, dec_grid_w=8,
    lambda_price=5.0, lambda_returns=2.0,
).to(device)

# Load best checkpoint
ckpt_dir = Path("outputs/lewm_crypto")
ckpts = sorted(ckpt_dir.glob("lewm_crypto_step_*.pt"))
if not ckpts:
    print("No checkpoints found, skipping evaluation")
    sys.exit(0)

ckpt_path = ckpts[-1]
print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(ckpt["model"])
model.eval()

# Load dataset for ground truth
dataset = CryptoKlineDataset(
    data_dir="data/external/crypto",
    symbol="BTCUSDT",
    window_size=64,
    feature_dim=32,
    stride=1,
    grid_h=8,
    grid_w=8,
)
norm_params = dataset.get_normalisation_params()

# -------------------------------------------------------
# Evaluation 1: Single-step prediction quality
# -------------------------------------------------------
print("\n=== Single-step Prediction Quality ===")
latent_mses = []
cosine_sims = []
for i in range(0, min(1000, len(dataset)), 10):
    sample = dataset[i]
    st = sample["state_t"].unsqueeze(0).to(device)
    st1 = sample["state_t1"].unsqueeze(0).to(device)
    mc = sample["market_cond"].unsqueeze(0).to(device)

    with torch.no_grad():
        z_t = model.encode(st)
        z_t1_gt = model.encode(st1)
        z_t1_pred = model.predict(z_t, mc)

        mse = (z_t1_pred - z_t1_gt).pow(2).mean().item()
        cos = torch.nn.functional.cosine_similarity(z_t1_pred, z_t1_gt, dim=-1).mean().item()
        latent_mses.append(mse)
        cosine_sims.append(cos)

print(f"  Latent MSE:     {np.mean(latent_mses):.6f} +/- {np.std(latent_mses):.6f}")
print(f"  Cosine sim:     {np.mean(cosine_sims):.4f} +/- {np.std(cosine_sims):.4f}")

# -------------------------------------------------------
# Evaluation 2: Reconstruction quality (if decoder)
# -------------------------------------------------------
if model.decoder is not None:
    print("\n=== Reconstruction Quality ===")
    recon_mses = []
    for i in range(0, min(500, len(dataset)), 10):
        sample = dataset[i]
        st = sample["state_t"].unsqueeze(0).to(device)
        with torch.no_grad():
            z = model.encode(st)
            recon = model.decode(z)
            mse = (recon - st).pow(2).mean().item()
            recon_mses.append(mse)
    print(f"  Reconstruction MSE: {np.mean(recon_mses):.6f}")

# -------------------------------------------------------
# Evaluation 3: 20-trial rollout evaluation
# -------------------------------------------------------
NUM_TRIALS = 20
ROLLOUT_STEPS = 1000
ret_scale = norm_params["iqr"][0]
ret_center = norm_params["medians"][0]

# Spread starting points evenly across the dataset
ds_len = len(dataset)
start_indices = [int(ds_len * (i + 1) / (NUM_TRIALS + 1)) for i in range(NUM_TRIALS)]

print(f"\n=== {NUM_TRIALS}-trial Rollout Evaluation ({ROLLOUT_STEPS} steps each) ===")

trial_wasserstein = []
trial_stylized_passed = []
trial_ret_stds = []

# Collect all GT returns once (from the full close price series)
all_gt_prices = dataset.get_close_prices(0, ds_len + ROLLOUT_STEPS + 100)
all_gt_returns = compute_returns(all_gt_prices)

for trial_idx, start_idx in enumerate(start_indices):
    sample = dataset[start_idx]
    state_t = sample["state_t"].unsqueeze(0).to(device)
    market_cond = sample["market_cond"].unsqueeze(0).to(device)

    generated_returns = []
    generated_volumes = []

    with torch.no_grad():
        z = model.encode(state_t)

        for step in range(ROLLOUT_STEPS):
            z_next = model.predict(z, market_cond, stochastic=False)

            mu, sigma, nu = model.return_dist_head(z_next)
            generated_returns.append(mu.item())

            if model.decoder is not None:
                decoded = model.decode(z_next)
                flat = decoded.reshape(1, -1, 32)
                log_vol_normed = flat[0, :, 1].mean().item()
                log_vol = log_vol_normed * norm_params["iqr"][1] + norm_params["medians"][1]
                vol = max(np.exp(log_vol) - 1.0, 0.0)
                generated_volumes.append(vol)

                mean_ret = flat[0, :, 0].mean().item()
                mc_np = market_cond.cpu().numpy()[0].copy()
                mc_np[0] = mean_ret
                mc_np[7] = mean_ret
                market_cond = torch.from_numpy(mc_np).unsqueeze(0).float().to(device)
            else:
                generated_volumes.append(100.0)

            z = z_next

    gen_returns = np.array(generated_returns)
    raw_returns = gen_returns * ret_scale + ret_center
    gen_prices = np.exp(np.cumsum(raw_returns)) * dataset.close_prices[start_idx]
    gen_volumes = np.array(generated_volumes)

    # Compute generated log-returns for Wasserstein
    gen_log_returns = compute_returns(gen_prices)

    # Ground truth segment for this trial
    gt_seg_prices = dataset.get_close_prices(start_idx, ROLLOUT_STEPS + 100)
    gt_seg_returns = compute_returns(gt_seg_prices[:ROLLOUT_STEPS])

    # Wasserstein distance
    w_dist = return_distribution_wasserstein(gen_log_returns, gt_seg_returns)
    trial_wasserstein.append(w_dist)

    # Stylized facts
    gen_report = evaluate_stylized_facts(gen_prices, gen_volumes[:len(gen_prices)])
    trial_stylized_passed.append(gen_report.total_passed)
    trial_ret_stds.append(gen_returns.std())

    print(f"  Trial {trial_idx+1:2d} (start={start_idx:6d}): "
          f"stylized={gen_report.total_passed}/6  "
          f"W-dist={w_dist:.6f}  "
          f"ret_std={gen_returns.std():.6f}")

# Summary statistics
w_arr = np.array(trial_wasserstein)
sf_arr = np.array(trial_stylized_passed)
rs_arr = np.array(trial_ret_stds)

print(f"\n=== 20-Trial Summary ===")
print(f"  Wasserstein distance: {w_arr.mean():.6f} +/- {w_arr.std():.6f}")
print(f"  Stylized facts passed: {sf_arr.mean():.2f} +/- {sf_arr.std():.2f} out of 6")
print(f"  Generated return std:  {rs_arr.mean():.6f} +/- {rs_arr.std():.6f}")
print(f"  GT return std:         {np.std(all_gt_returns):.6f}")

# Ground truth stylized facts (single reference)
gt_prices_ref = dataset.get_close_prices(ds_len // 4, ROLLOUT_STEPS + 100)
gt_volumes_ref = dataset.get_volumes(ds_len // 4, ROLLOUT_STEPS + 100)
gt_report = evaluate_stylized_facts(gt_prices_ref[:ROLLOUT_STEPS], gt_volumes_ref[:ROLLOUT_STEPS])
print(f"\n--- Stylized Facts (Ground Truth BTC reference) ---")
print(f"  Fat tails:              {'PASS' if gt_report.fat_tail_pass else 'FAIL'} (alpha={gt_report.fat_tail_alpha:.2f})")
print(f"  Volatility clustering:  {'PASS' if gt_report.volatility_clustering_pass else 'FAIL'}")
print(f"  Leverage effect:        {'PASS' if gt_report.leverage_effect_pass else 'FAIL'} (corr={gt_report.leverage_effect_corr:.4f})")
print(f"  Vol-volume corr:        {'PASS' if gt_report.volume_volatility_pass else 'FAIL'} (corr={gt_report.volume_volatility_corr:.4f})")
print(f"  Return autocorr:        {'PASS' if gt_report.return_autocorr_pass else 'FAIL'}")
print(f"  Gain-loss asymmetry:    {'PASS' if gt_report.gain_loss_asymmetry_pass else 'FAIL'} (p={gt_report.gain_loss_asymmetry_pvalue:.4f})")
print(f"  TOTAL: {gt_report.summary}")

# -------------------------------------------------------
# Evaluation 4: Speed benchmark
# -------------------------------------------------------
print("\n=== Speed Benchmark ===")
sample0 = dataset[0]
state_t_bench = sample0["state_t"].unsqueeze(0).to(device)
mc_bench = sample0["market_cond"].unsqueeze(0).to(device)
z_test = model.encode(state_t_bench)
if device.type == "cuda":
    torch.cuda.synchronize()

t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(1000):
        z_test = model.predict(z_test, mc_bench)
if device.type == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"  Latent rollout speed: {(t1-t0)/1000*1000:.2f} ms/step")
print(f"  Throughput: {1000/(t1-t0):.0f} steps/sec")

print("\n=== LeWM-Crypto Evaluation Complete ===")
PYEOF

echo "=== LeWorldModel Crypto Pipeline Complete ==="
