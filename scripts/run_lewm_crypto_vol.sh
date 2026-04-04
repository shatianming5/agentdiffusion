#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

echo "=== LeWorldModel Crypto + Stochastic Volatility Pipeline ==="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Data: data/external/crypto (Binance BTCUSDT 1-min klines)"
echo "Stage A: 20K steps (LeWM latent dynamics)"
echo "Stage B: 30K steps (vol_head on frozen encoder+predictor)"
echo ""

OUTPUT_DIR="outputs/lewm_crypto_vol"
STAGE_A_STEPS=20000
STAGE_B_STEPS=30000

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

from agentdiffusion.data.crypto_dataset import CryptoKlineDataset
ds = CryptoKlineDataset(
    data_dir='data/external/crypto', symbol='BTCUSDT',
    window_size=64, feature_dim=32, stride=1, grid_h=8, grid_w=8,
)
print(f'Dataset size: {len(ds)} samples')
sample = ds[0]
print(f'state_t shape: {sample[\"state_t\"].shape}')
print(f'market_cond shape: {sample[\"market_cond\"].shape}')
"

# -------------------------------------------------------
# Phase 2A: Train LeWorldModel on crypto (Stage A — 20K steps)
# -------------------------------------------------------
echo ""
echo "=== Phase 2A: Train LeWorldModel (${STAGE_A_STEPS} steps) ==="
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
    train.total_steps=${STAGE_A_STEPS} train.batch_size=64 \
    train.lr=3e-4 train.weight_decay=0.05 \
    train.warmup_steps=2000 \
    train.rollout_steps=4 train.seq_len=8 \
    train.log_every=100 train.save_every=10000 train.eval_every=5000 \
    train.mixed_precision=bf16 \
    output_dir=${OUTPUT_DIR}

# Find latest Stage A checkpoint
STAGE_A_CKPT=$(ls -1t ${OUTPUT_DIR}/lewm_crypto_step_*.pt 2>/dev/null | head -1)
if [ -z "$STAGE_A_CKPT" ]; then
    echo "ERROR: No Stage A checkpoint found in ${OUTPUT_DIR}"
    exit 1
fi
echo "Stage A checkpoint: ${STAGE_A_CKPT}"

# -------------------------------------------------------
# Phase 2B: Train vol_head with frozen encoder+predictor (Stage B — 30K steps)
# -------------------------------------------------------
echo ""
echo "=== Phase 2B: Train StochasticVolatilityHead (${STAGE_B_STEPS} steps) ==="
.venv/bin/python3 << PYEOF
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from agentdiffusion.models.lewm import LeWorldModel
from agentdiffusion.models.stochastic_vol import (
    StochasticVolatilityHead, SkewedStudentT, train_step as vol_train_step,
)
from agentdiffusion.data.crypto_dataset import CryptoKlineDataset

# -------------------------------------------------------------------
# 1) Build model and load Stage A checkpoint
# -------------------------------------------------------------------
model = LeWorldModel(
    d_agent=32, d_enc=256, d_latent=256, d_pred=384, d_cond=32,
    patch_size=2, enc_depth=6, enc_heads=8, pred_depth=6, pred_heads=8,
    enc_mlp_ratio=4.0, pred_mlp_ratio=2.0,
    use_decoder=True, d_dec=256, dec_depth=4, dec_heads=4,
    dec_grid_h=8, dec_grid_w=8,
    lambda_price=5.0, lambda_returns=2.0,
).to(device)

ckpt_path = "${STAGE_A_CKPT}"
print(f"Loading Stage A checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
# Load only matching keys (Stage A checkpoint has no vol_head weights)
state_dict = ckpt["model"]
model_state = model.state_dict()
loaded_keys = []
for k, v in state_dict.items():
    if k in model_state and model_state[k].shape == v.shape:
        model_state[k] = v
        loaded_keys.append(k)
model.load_state_dict(model_state)
print(f"Loaded {len(loaded_keys)}/{len(model_state)} keys from Stage A")

# -------------------------------------------------------------------
# 2) Freeze encoder + predictor, train only vol_head
# -------------------------------------------------------------------
for name, param in model.named_parameters():
    if not name.startswith("vol_head"):
        param.requires_grad = False

vol_params = [p for p in model.vol_head.parameters() if p.requires_grad]
n_vol_params = sum(p.numel() for p in vol_params)
print(f"vol_head trainable parameters: {n_vol_params:,}")

optimizer = torch.optim.AdamW(vol_params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.95))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=${STAGE_B_STEPS})
warmup_steps = 1000

# -------------------------------------------------------------------
# 3) Load dataset and compute real log-returns for teacher forcing
# -------------------------------------------------------------------
dataset = CryptoKlineDataset(
    data_dir="data/external/crypto", symbol="BTCUSDT",
    window_size=64, feature_dim=32, stride=1, grid_h=8, grid_w=8,
)
norm_params = dataset.get_normalisation_params()
ret_scale = norm_params["iqr"][0]   # IQR of log_return feature
ret_center = norm_params["medians"][0]  # median of log_return feature

# Raw log-returns from close prices (un-normalised)
close_prices = dataset.close_prices
raw_log_returns = np.zeros(len(close_prices))
raw_log_returns[1:] = np.log(close_prices[1:] / np.maximum(close_prices[:-1], 1e-10))

print(f"Dataset: {len(dataset)} samples, {len(close_prices)} bars")
print(f"Return stats: mean={raw_log_returns.mean():.8f}, std={raw_log_returns.std():.6f}")

# -------------------------------------------------------------------
# 4) Training loop: iterate over sliding windows, encode to z, train vol_head
#
# For each batch we:
#   a) Sample random starting indices
#   b) For each index, extract a sequence of T consecutive windows
#   c) Encode each window through frozen encoder to get z_seq [B, T, d_latent]
#   d) Collect the real log-returns at each step
#   e) Call vol_train_step(vol_head, optimizer, z_seq, returns, ...)
# -------------------------------------------------------------------
BATCH_SIZE = 32
SEQ_LEN = 64       # teacher-forcing horizon (number of steps per sequence)
TOTAL_STEPS = ${STAGE_B_STEPS}
LOG_EVERY = 200
SAVE_EVERY = 10000

ds_len = len(dataset)
# Valid starting sample indices: need SEQ_LEN consecutive samples
max_start_sample = ds_len - SEQ_LEN - 1

model.eval()  # encoder/predictor in eval mode
model.vol_head.train()  # vol_head in train mode

output_dir = Path("${OUTPUT_DIR}")
output_dir.mkdir(parents=True, exist_ok=True)

pbar = tqdm(total=TOTAL_STEPS, desc="Stage B: vol_head")
global_step = 0
running_loss = 0.0
running_nll = 0.0

while global_step < TOTAL_STEPS:
    # Sample a batch of random starting positions
    batch_starts = np.random.randint(0, max_start_sample, size=BATCH_SIZE)

    # Build z_seq and returns tensors
    z_seq_list = []
    returns_list = []
    prev_return_list = []
    prev_sigma_list = []

    with torch.no_grad():
        for b_idx in range(BATCH_SIZE):
            sample_start = batch_starts[b_idx]

            # Encode T consecutive windows to get z sequence
            z_steps = []
            ret_steps = []
            for t in range(SEQ_LEN):
                s_idx = sample_start + t
                sample = dataset[s_idx]
                state_t = sample["state_t"].unsqueeze(0).to(device)
                z = model.encode(state_t)  # [1, d_latent]
                z_steps.append(z)

                # Real log-return at the bar corresponding to the END of this window
                bar_idx = dataset.indices[s_idx] + dataset.window_size - 1
                ret_steps.append(raw_log_returns[bar_idx])

            z_seq_b = torch.cat(z_steps, dim=0)  # [T, d_latent]
            z_seq_list.append(z_seq_b)
            returns_list.append(torch.tensor(ret_steps, dtype=torch.float32))

            # Previous return and sigma (from bar before the first window)
            bar_before = dataset.indices[sample_start] + dataset.window_size - 2
            prev_ret = raw_log_returns[max(bar_before, 0)]
            prev_sigma = abs(prev_ret) + 1e-5  # simple initialisation
            prev_return_list.append(prev_ret)
            prev_sigma_list.append(prev_sigma)

    z_seq = torch.stack(z_seq_list, dim=0).to(device)  # [B, T, d_latent]
    returns = torch.stack(returns_list, dim=0).to(device)  # [B, T]
    prev_return = torch.tensor(prev_return_list, dtype=torch.float32, device=device)
    prev_sigma = torch.tensor(prev_sigma_list, dtype=torch.float32, device=device)

    # Warmup LR
    if global_step < warmup_steps:
        lr_factor = global_step / max(1, warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = 1e-3 * lr_factor

    # Train step
    model.vol_head.train()
    step_out = vol_train_step(
        model=model.vol_head,
        optimizer=optimizer,
        z_seq=z_seq,
        returns=returns,
        prev_return=prev_return,
        prev_sigma=prev_sigma,
        h0=None,
        nll_weight=1.0,
        mu_penalty_weight=1e4,
        acf_weight=1.0,
        leverage_weight=1.0,
        grad_clip=1.0,
    )

    if global_step >= warmup_steps:
        scheduler.step()

    running_loss += step_out.loss.item()
    running_nll += step_out.nll.item()
    global_step += 1
    pbar.update(1)

    if global_step % LOG_EVERY == 0:
        avg_loss = running_loss / LOG_EVERY
        avg_nll = running_nll / LOG_EVERY
        lr_now = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            loss=f"{avg_loss:.4f}",
            nll=f"{avg_nll:.4f}",
            acf=f"{step_out.acf_penalty.item():.4f}",
            lev=f"{step_out.leverage_penalty.item():.4f}",
            lr=f"{lr_now:.2e}",
        )
        running_loss = 0.0
        running_nll = 0.0

    if global_step % SAVE_EVERY == 0:
        save_path = output_dir / f"lewm_crypto_vol_step_{global_step}.pt"
        torch.save({
            "model": model.state_dict(),
            "vol_head": model.vol_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": global_step,
            "stage": "B",
        }, save_path)
        print(f"\nCheckpoint saved: {save_path}")

pbar.close()

# Final save
final_path = output_dir / f"lewm_crypto_vol_final.pt"
torch.save({
    "model": model.state_dict(),
    "vol_head": model.vol_head.state_dict(),
    "step": global_step,
    "stage": "B_final",
}, final_path)
print(f"Final vol_head model saved: {final_path}")
PYEOF

# -------------------------------------------------------
# Phase 3: Full Evaluation (20 trials, Wasserstein distance)
# -------------------------------------------------------
echo ""
echo "=== Phase 3: Full Evaluation (20 trials, vol_head rollout) ==="
.venv/bin/python3 << 'PYEOF'
import sys
import numpy as np
import torch
import time
from pathlib import Path
from scipy import stats as sp_stats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from agentdiffusion.models.lewm import LeWorldModel
from agentdiffusion.models.stochastic_vol import SkewedStudentT
from agentdiffusion.data.crypto_dataset import CryptoKlineDataset
from agentdiffusion.eval.stylized_facts import (
    evaluate_stylized_facts,
    compute_returns,
    return_distribution_wasserstein,
)

# -------------------------------------------------------------------
# 1) Build model and load final checkpoint (with vol_head)
# -------------------------------------------------------------------
model = LeWorldModel(
    d_agent=32, d_enc=256, d_latent=256, d_pred=384, d_cond=32,
    patch_size=2, enc_depth=6, enc_heads=8, pred_depth=6, pred_heads=8,
    enc_mlp_ratio=4.0, pred_mlp_ratio=2.0,
    use_decoder=True, d_dec=256, dec_depth=4, dec_heads=4,
    dec_grid_h=8, dec_grid_w=8,
    lambda_price=5.0, lambda_returns=2.0,
).to(device)

output_dir = Path("outputs/lewm_crypto_vol")
# Try final checkpoint first, then latest numbered
final_ckpt = output_dir / "lewm_crypto_vol_final.pt"
if final_ckpt.exists():
    ckpt_path = final_ckpt
else:
    ckpts = sorted(output_dir.glob("lewm_crypto_vol_step_*.pt"))
    if not ckpts:
        print("No vol_head checkpoints found, skipping evaluation")
        sys.exit(0)
    ckpt_path = ckpts[-1]

print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

print(f"vol_head params: {sum(p.numel() for p in model.vol_head.parameters()):,}")

# -------------------------------------------------------------------
# 2) Load dataset
# -------------------------------------------------------------------
dataset = CryptoKlineDataset(
    data_dir="data/external/crypto", symbol="BTCUSDT",
    window_size=64, feature_dim=32, stride=1, grid_h=8, grid_w=8,
)
norm_params = dataset.get_normalisation_params()
ret_scale = norm_params["iqr"][0]
ret_center = norm_params["medians"][0]

close_prices = dataset.close_prices
raw_log_returns = np.zeros(len(close_prices))
raw_log_returns[1:] = np.log(close_prices[1:] / np.maximum(close_prices[:-1], 1e-10))

# -------------------------------------------------------------------
# 3) Single-step prediction quality
# -------------------------------------------------------------------
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
        cos = F.cosine_similarity(z_t1_pred, z_t1_gt, dim=-1).mean().item()
        latent_mses.append(mse)
        cosine_sims.append(cos)

# Need the import for F
import torch.nn.functional as F
print(f"  Latent MSE:  {np.mean(latent_mses):.6f} +/- {np.std(latent_mses):.6f}")
print(f"  Cosine sim:  {np.mean(cosine_sims):.4f} +/- {np.std(cosine_sims):.4f}")

# -------------------------------------------------------------------
# 4) 20-trial rollout evaluation with vol_head
# -------------------------------------------------------------------
NUM_TRIALS = 20
ROLLOUT_STEPS = 1000

ds_len = len(dataset)
start_indices = [int(ds_len * (i + 1) / (NUM_TRIALS + 1)) for i in range(NUM_TRIALS)]

print(f"\n=== {NUM_TRIALS}-trial Vol-Head Rollout ({ROLLOUT_STEPS} steps each) ===")

all_gt_prices = dataset.get_close_prices(0, ds_len + ROLLOUT_STEPS + 100)
all_gt_returns = compute_returns(all_gt_prices)

trial_wasserstein = []
trial_stylized_passed = []
trial_ret_stds = []
trial_ret_kurtosis = []
trial_ret_skewness = []

for trial_idx, start_idx in enumerate(start_indices):
    sample = dataset[start_idx]
    state_t = sample["state_t"].unsqueeze(0).to(device)
    market_cond = sample["market_cond"].unsqueeze(0).to(device)

    # Initialise vol_head state
    bar_start = dataset.indices[start_idx] + dataset.window_size - 1
    r_prev = torch.tensor([raw_log_returns[bar_start]], device=device, dtype=torch.float32)
    sigma_prev = torch.tensor([abs(raw_log_returns[bar_start]) + 1e-5],
                              device=device, dtype=torch.float32)
    h_vol = model.vol_head.initial_state(1, device=device, dtype=torch.float32)
    p_current = close_prices[bar_start]

    generated_returns = []
    generated_prices = [p_current]
    generated_volumes = []

    with torch.no_grad():
        z = model.encode(state_t)

        for step in range(ROLLOUT_STEPS):
            # Predict next latent + vol distribution
            z_next, vol_out = model.generate_with_vol(
                z_t=z,
                market_cond=market_cond,
                r_prev=r_prev,
                sigma_prev=sigma_prev,
                h_vol=h_vol,
                stochastic_latent=True,
            )

            # Sample return from SkewedStudentT
            dist = SkewedStudentT(
                loc=vol_out.mu,
                scale=vol_out.sigma,
                df=vol_out.nu,
                skew=vol_out.skew,
            )
            r_sample = dist.rsample()  # [1]
            generated_returns.append(r_sample.item())

            # Update price
            p_current = p_current * np.exp(r_sample.item())
            generated_prices.append(p_current)

            # Record volume proxy
            generated_volumes.append(np.exp(vol_out.log_volume.item()))

            # Update state for next step
            h_vol = vol_out.h_next
            r_prev = r_sample
            sigma_prev = vol_out.sigma

            # Update market_cond if decoder available
            if model.decoder is not None:
                decoded = model.decode(z_next)
                flat = decoded.reshape(1, -1, 32)
                mean_ret = flat[0, :, 0].mean().item()
                mc_np = market_cond.cpu().numpy()[0].copy()
                mc_np[0] = mean_ret
                mc_np[7] = mean_ret
                market_cond = torch.from_numpy(mc_np).unsqueeze(0).float().to(device)

            z = z_next

    gen_returns = np.array(generated_returns)
    gen_prices = np.array(generated_prices)
    gen_volumes = np.array(generated_volumes)

    # Wasserstein distance on return distributions
    gen_log_returns = compute_returns(gen_prices)
    gt_seg_prices = dataset.get_close_prices(start_idx, ROLLOUT_STEPS + 100)
    gt_seg_returns = compute_returns(gt_seg_prices[:ROLLOUT_STEPS])

    w_dist = return_distribution_wasserstein(gen_log_returns, gt_seg_returns)
    trial_wasserstein.append(w_dist)

    # Stylized facts
    gen_report = evaluate_stylized_facts(gen_prices[1:], gen_volumes[:len(gen_prices)-1])
    trial_stylized_passed.append(gen_report.total_passed)
    trial_ret_stds.append(gen_returns.std())
    trial_ret_kurtosis.append(float(sp_stats.kurtosis(gen_returns)))
    trial_ret_skewness.append(float(sp_stats.skew(gen_returns)))

    print(f"  Trial {trial_idx+1:2d} (start={start_idx:6d}): "
          f"stylized={gen_report.total_passed}/6  "
          f"W-dist={w_dist:.6f}  "
          f"ret_std={gen_returns.std():.6f}  "
          f"kurt={sp_stats.kurtosis(gen_returns):.2f}  "
          f"skew={sp_stats.skew(gen_returns):.3f}")

# -------------------------------------------------------------------
# 5) Summary
# -------------------------------------------------------------------
w_arr = np.array(trial_wasserstein)
sf_arr = np.array(trial_stylized_passed)
rs_arr = np.array(trial_ret_stds)
ku_arr = np.array(trial_ret_kurtosis)
sk_arr = np.array(trial_ret_skewness)

print(f"\n=== {NUM_TRIALS}-Trial Summary (Vol-Head) ===")
print(f"  Wasserstein distance:  {w_arr.mean():.6f} +/- {w_arr.std():.6f}")
print(f"  Stylized facts passed: {sf_arr.mean():.2f} +/- {sf_arr.std():.2f} out of 6")
print(f"  Generated return std:  {rs_arr.mean():.6f} +/- {rs_arr.std():.6f}")
print(f"  Generated kurtosis:    {ku_arr.mean():.2f} +/- {ku_arr.std():.2f}")
print(f"  Generated skewness:    {sk_arr.mean():.3f} +/- {sk_arr.std():.3f}")
print(f"  GT return std:         {np.std(all_gt_returns):.6f}")
print(f"  GT kurtosis:           {sp_stats.kurtosis(all_gt_returns):.2f}")
print(f"  GT skewness:           {sp_stats.skew(all_gt_returns):.3f}")

# Ground truth stylized facts reference
gt_prices_ref = dataset.get_close_prices(ds_len // 4, ROLLOUT_STEPS + 100)
gt_volumes_ref = dataset.get_volumes(ds_len // 4, ROLLOUT_STEPS + 100)
gt_report = evaluate_stylized_facts(gt_prices_ref[:ROLLOUT_STEPS], gt_volumes_ref[:ROLLOUT_STEPS])
print(f"\n--- Stylized Facts (Ground Truth BTC reference) ---")
print(f"  Fat tails:             {'PASS' if gt_report.fat_tail_pass else 'FAIL'} (alpha={gt_report.fat_tail_alpha:.2f})")
print(f"  Volatility clustering: {'PASS' if gt_report.volatility_clustering_pass else 'FAIL'}")
print(f"  Leverage effect:       {'PASS' if gt_report.leverage_effect_pass else 'FAIL'} (corr={gt_report.leverage_effect_corr:.4f})")
print(f"  Vol-volume corr:       {'PASS' if gt_report.volume_volatility_pass else 'FAIL'} (corr={gt_report.volume_volatility_corr:.4f})")
print(f"  Return autocorr:       {'PASS' if gt_report.return_autocorr_pass else 'FAIL'}")
print(f"  Gain-loss asymmetry:   {'PASS' if gt_report.gain_loss_asymmetry_pass else 'FAIL'} (p={gt_report.gain_loss_asymmetry_pvalue:.4f})")
print(f"  TOTAL: {gt_report.summary}")

# -------------------------------------------------------------------
# 6) Speed benchmark
# -------------------------------------------------------------------
print("\n=== Speed Benchmark (with vol_head) ===")
sample0 = dataset[0]
state_t_bench = sample0["state_t"].unsqueeze(0).to(device)
mc_bench = sample0["market_cond"].unsqueeze(0).to(device)
z_bench = model.encode(state_t_bench)
r_bench = torch.zeros(1, device=device)
s_bench = torch.ones(1, device=device) * 0.001
h_bench = model.vol_head.initial_state(1, device=device)

if device.type == "cuda":
    torch.cuda.synchronize()

t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(1000):
        z_bench, vol_out = model.generate_with_vol(
            z_bench, mc_bench, r_bench, s_bench, h_bench
        )
        h_bench = vol_out.h_next
        r_bench = vol_out.mu
        s_bench = vol_out.sigma
if device.type == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"  Latent + vol_head rollout: {(t1-t0)/1000*1000:.2f} ms/step")
print(f"  Throughput: {1000/(t1-t0):.0f} steps/sec")

print("\n=== LeWM-Crypto + Vol-Head Evaluation Complete ===")
PYEOF

echo "=== LeWorldModel Crypto + Stochastic Volatility Pipeline Complete ==="
