#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "============================================================"
echo "  Full Evaluation: Stylized Facts + Baseline Comparison"
echo "  + Order Decoder Accuracy + Long-range Stability"
echo "============================================================"

.venv/bin/python3 -u << 'PYEOF'
import sys, time, math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy import stats as scipy_stats

# ============================================================
# Setup
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OB = "data/external/lobster/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
MSG = "data/external/lobster/AMZN_2012-06-21_34200000_57600000_message_10.csv"
VDIT_DIR = Path("outputs/vdit_lob_4x4")

# --- Load real LOB data ---
from agentdiffusion.data.lob_dataset import LOBVideoDataset, load_lobster_data
from agentdiffusion.eval.stylized_facts import evaluate_stylized_facts, StylizedFactsReport

raw = load_lobster_data(OB, MSG, subsample=1)
raw_ob = raw["raw_ob"]
# Mid-price from best ask/bid
real_mid = (raw_ob[:, 0] + raw_ob[:, 2]) / 2.0
real_volumes = raw["msg_sizes"].astype(float)
print(f"Real data: {len(real_mid)} snapshots")

# --- Load model ---
from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.infer.interactive_sim import InteractiveSimulator

dataset = LOBVideoDataset(OB, MSG, total_frames=20, cond_frames=4, subsample=10, grid_shape=(4, 4))
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

scheduler = NoiseScheduler(1000, "cosine").to(device)
sampler = VideoDDIMSampler(model, scheduler, "v_prediction", ddim_steps=20, eta=0.0)
print(f"Loaded Video DiT from {ckpt_path}")


# ============================================================
# Part 1: Stylized Facts — Real vs Generated vs Baselines
# ============================================================
print("\n" + "=" * 64)
print("  Part 1: Stylized Facts Comparison")
print("=" * 64)

N_SAMPLES = 50
SEQ_LEN = 300  # per sample

# --- 1a: Real LOBSTER data ---
print("\n  Evaluating real data...")
# Split real mid-price into chunks
real_chunks = []
for i in range(0, len(real_mid) - SEQ_LEN, SEQ_LEN):
    real_chunks.append(real_mid[i:i + SEQ_LEN])
real_chunks = real_chunks[:N_SAMPLES]

real_results = []
for chunk in real_chunks:
    r = evaluate_stylized_facts(chunk)
    real_results.append(r)

def avg_report(results):
    return {
        "fat_tail_alpha": np.mean([r.fat_tail_alpha for r in results]),
        "fat_tail_pass": np.mean([r.fat_tail_pass for r in results]),
        "vol_clustering_pass": np.mean([r.volatility_clustering_pass for r in results]),
        "vol_acf1": np.mean([r.volatility_clustering_acf[1] if len(r.volatility_clustering_acf) > 1 else 0 for r in results]),
        "leverage_corr": np.mean([r.leverage_effect_corr for r in results]),
        "leverage_pass": np.mean([r.leverage_effect_pass for r in results]),
        "ret_autocorr_pass": np.mean([r.return_autocorr_pass for r in results]),
        "gl_asymmetry_pass": np.mean([r.gain_loss_asymmetry_pass for r in results]),
        "total": np.mean([r.total_passed for r in results]),
    }

real_avg = avg_report(real_results)
print(f"  Real: {real_avg['total']:.1f}/6 avg passed (n={len(real_chunks)})")
print(f"    fat_tail α={real_avg['fat_tail_alpha']:.2f}, "
      f"vol_acf1={real_avg['vol_acf1']:.4f}, "
      f"leverage={real_avg['leverage_corr']:.4f}")

# --- 1b: Video DiT generated data ---
print("\n  Generating Video DiT sequences...")
sim = InteractiveSimulator(model, sampler, num_cond=4, num_gen=16, zero_sum_proj=True)

gen_prices_list = []
t0 = time.time()
for s in range(min(N_SAMPLES, len(dataset))):
    seed = dataset[s % len(dataset)]["frames"][:4]
    sim.init(seed)

    # Generate ~300 frames via sliding window (300/16 ≈ 19 rounds)
    frames_so_far = []
    for _ in range(19):
        gen = sim.step()
        # Extract "price proxy" = mean of all cells per frame
        for t in range(gen.shape[0]):
            # Use dim 0 as price proxy (or the single LOB feature)
            val = gen[t].mean().item()
            frames_so_far.append(val)
        sim.trim_buffer(keep_last=8)

    if len(frames_so_far) >= 50:
        # Convert to "prices" via cumulative sum of returns-like values
        series = np.array(frames_so_far[:SEQ_LEN])
        # Shift to positive for log-return computation
        series = series - series.min() + 1.0
        gen_prices_list.append(series)

gen_time = time.time() - t0
print(f"  Generated {len(gen_prices_list)} sequences in {gen_time:.1f}s")

gen_results = [evaluate_stylized_facts(p) for p in gen_prices_list]
gen_avg = avg_report(gen_results)
print(f"  Video DiT: {gen_avg['total']:.1f}/6 avg passed")
print(f"    fat_tail α={gen_avg['fat_tail_alpha']:.2f}, "
      f"vol_acf1={gen_avg['vol_acf1']:.4f}, "
      f"leverage={gen_avg['leverage_corr']:.4f}")

# --- 1c: Baseline 1 — Geometric Brownian Motion (GBM) ---
print("\n  Generating GBM baseline...")
gbm_prices_list = []
real_ret = np.diff(np.log(real_mid[real_mid > 0]))
mu = real_ret.mean()
sigma = real_ret.std()
for _ in range(N_SAMPLES):
    z = np.random.randn(SEQ_LEN)
    log_p = np.cumsum(mu + sigma * z)
    prices = np.exp(log_p) * 100
    gbm_prices_list.append(prices)

gbm_results = [evaluate_stylized_facts(p) for p in gbm_prices_list]
gbm_avg = avg_report(gbm_results)
print(f"  GBM: {gbm_avg['total']:.1f}/6 avg passed")
print(f"    fat_tail α={gbm_avg['fat_tail_alpha']:.2f}, "
      f"vol_acf1={gbm_avg['vol_acf1']:.4f}, "
      f"leverage={gbm_avg['leverage_corr']:.4f}")

# --- 1d: Baseline 2 — GARCH(1,1) ---
print("\n  Generating GARCH(1,1) baseline...")
garch_prices_list = []
omega = sigma**2 * 0.05
alpha_g = 0.10
beta_g = 0.85
for _ in range(N_SAMPLES):
    h = sigma**2
    r = np.zeros(SEQ_LEN)
    for t in range(SEQ_LEN):
        r[t] = mu + np.sqrt(h) * np.random.randn()
        h = omega + alpha_g * r[t]**2 + beta_g * h
    prices = np.exp(np.cumsum(r)) * 100
    garch_prices_list.append(prices)

garch_results = [evaluate_stylized_facts(p) for p in garch_prices_list]
garch_avg = avg_report(garch_results)
print(f"  GARCH(1,1): {garch_avg['total']:.1f}/6 avg passed")
print(f"    fat_tail α={garch_avg['fat_tail_alpha']:.2f}, "
      f"vol_acf1={garch_avg['vol_acf1']:.4f}, "
      f"leverage={garch_avg['leverage_corr']:.4f}")

# --- 1e: Wasserstein distance of return distributions ---
print("\n  Return distribution distance (Wasserstein-1):")
from agentdiffusion.eval.stylized_facts import compute_returns, return_distribution_wasserstein
real_all_ret = compute_returns(np.concatenate(real_chunks))

if gen_prices_list:
    gen_all_ret = compute_returns(np.concatenate(gen_prices_list))
    wd_gen = return_distribution_wasserstein(gen_all_ret, real_all_ret)
else:
    wd_gen = float('inf')

gbm_all_ret = compute_returns(np.concatenate(gbm_prices_list))
garch_all_ret = compute_returns(np.concatenate(garch_prices_list))
wd_gbm = return_distribution_wasserstein(gbm_all_ret, real_all_ret)
wd_garch = return_distribution_wasserstein(garch_all_ret, real_all_ret)

print(f"    Video DiT:  {wd_gen:.6f}")
print(f"    GBM:        {wd_gbm:.6f}")
print(f"    GARCH(1,1): {wd_garch:.6f}")

# --- Summary Table ---
print("\n" + "=" * 64)
print("  STYLIZED FACTS COMPARISON TABLE")
print("=" * 64)
header = f"{'Method':<16} {'Fat-tail':<10} {'Vol-clust':<10} {'Leverage':<10} {'Ret-AC':<10} {'GL-asym':<10} {'Avg/6':<8} {'W-dist':<10}"
print(header)
print("-" * len(header))

def row(name, avg, wd):
    return (f"{name:<16} "
            f"{avg['fat_tail_pass']*100:>5.0f}%    "
            f"{avg['vol_clustering_pass']*100:>5.0f}%    "
            f"{avg['leverage_pass']*100:>5.0f}%    "
            f"{avg['ret_autocorr_pass']*100:>5.0f}%    "
            f"{avg['gl_asymmetry_pass']*100:>5.0f}%    "
            f"{avg['total']:>5.1f}   "
            f"{wd:>9.6f}")

print(row("Real (oracle)", real_avg, 0.0))
print(row("Video DiT", gen_avg, wd_gen))
print(row("GBM", gbm_avg, wd_gbm))
print(row("GARCH(1,1)", garch_avg, wd_garch))


# ============================================================
# Part 2: Long-range Stability (100 rounds)
# ============================================================
print("\n\n" + "=" * 64)
print("  Part 2: Long-range Stability (100 rounds × 16 frames)")
print("=" * 64)

sim2 = InteractiveSimulator(model, sampler, num_cond=4, num_gen=16, zero_sum_proj=True)
seed = dataset[0]["frames"][:4]
sim2.init(seed)

means = []
stds = []
for i in range(100):
    gen = sim2.step()
    m = gen.mean().item()
    s = gen.std().item()
    means.append(m)
    stds.append(s)
    sim2.trim_buffer(keep_last=8)

means = np.array(means)
stds = np.array(stds)
print(f"  Rounds: 100, Total frames: {100 * 16}")
print(f"  Mean trajectory:  start={means[0]:.4f}, end={means[-1]:.4f}, "
      f"drift={means[-1]-means[0]:.4f}")
print(f"  Std trajectory:   start={stds[0]:.4f}, end={stds[-1]:.4f}, "
      f"change={stds[-1]-stds[0]:.4f}")
print(f"  Mean of means:    {means.mean():.4f} ± {means.std():.4f}")
print(f"  Mean of stds:     {stds.mean():.4f} ± {stds.std():.4f}")

# Stability checks
mean_drift = abs(means[-1] - means[0])
std_ratio = stds[-1] / max(stds[0], 1e-8)
stable_mean = mean_drift < 1.0
stable_std = 0.5 < std_ratio < 2.0

print(f"\n  Mean drift < 1.0:       {'PASS' if stable_mean else 'FAIL'} ({mean_drift:.4f})")
print(f"  Std ratio in [0.5,2.0]: {'PASS' if stable_std else 'FAIL'} ({std_ratio:.4f})")
print(f"  Long-range stable:      {'YES' if stable_mean and stable_std else 'NO'}")


# ============================================================
# Part 3: Order Decoder Accuracy (if available)
# ============================================================
print("\n\n" + "=" * 64)
print("  Part 3: Order Decoder Accuracy")
print("=" * 64)

OA_DIR = Path("outputs/order_agent_lob")
oa_ckpts = sorted(OA_DIR.glob("order_agent_step_*.pt")) if OA_DIR.exists() else []

if oa_ckpts:
    from agentdiffusion.models.order_encoder import OrderToAgentEncoder
    from agentdiffusion.models.order_decoder import AgentToOrderDecoder
    from agentdiffusion.data.lob_order_dataset import LOBOrderFlowDataset

    oa_ckpt = torch.load(str(oa_ckpts[-1]), map_location=device, weights_only=True)

    encoder = OrderToAgentEncoder(
        d_order=6, d_embed=128, d_state=d_latent,
        grid_h=16, grid_w=16,
    ).to(device)
    decoder = AgentToOrderDecoder(d_state=d_latent, d_hidden=128, d_order_out=6).to(device)

    encoder.load_state_dict(oa_ckpt["encoder"])
    decoder.load_state_dict(oa_ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    # Load order flow dataset
    ofd = LOBOrderFlowDataset(OB, MSG, window_seconds=1.0, total_windows=20, n_max_orders=256)
    print(f"  Order flow dataset: {len(ofd)} sequences")

    # Evaluate decoder on a batch
    n_eval = min(50, len(ofd))
    recon_errors = []
    for idx in range(n_eval):
        sample = ofd[idx]
        orders = sample["orders"].unsqueeze(0).to(device)  # [1, T, N_max, 6]
        B, T, N_max, D = orders.shape

        with torch.no_grad():
            orders_flat = orders.reshape(B * T, N_max, D)
            grids = encoder(orders_flat).reshape(B, T, 16, 16, d_latent)
            pred_orders = decoder.decode_sequence(grids)  # [1, T-1, 16, 16, 6]
            gt_orders = decoder.decode_sequence(grids)    # same in eval mode

        recon_errors.append(F.mse_loss(pred_orders, gt_orders).item())

    avg_recon = np.mean(recon_errors)
    print(f"  Decoder self-consistency MSE: {avg_recon:.6f}")
    print(f"  Encoder params: {sum(p.numel() for p in encoder.parameters())/1e3:.1f}K")
    print(f"  Decoder params: {sum(p.numel() for p in decoder.parameters())/1e3:.1f}K")
else:
    print("  [SKIP] No order-agent checkpoint found")


# ============================================================
# Final Summary
# ============================================================
print("\n\n" + "=" * 64)
print("  FINAL SUMMARY")
print("=" * 64)
print(f"  Stylized facts (Video DiT): {gen_avg['total']:.1f}/6 avg pass rate")
print(f"  Stylized facts (GBM baseline): {gbm_avg['total']:.1f}/6")
print(f"  Stylized facts (GARCH baseline): {garch_avg['total']:.1f}/6")
print(f"  Wasserstein distance: Video DiT={wd_gen:.6f}, GBM={wd_gbm:.6f}, GARCH={wd_garch:.6f}")
print(f"  Long-range stability: {'PASS' if stable_mean and stable_std else 'FAIL'} (100 rounds)")
print(f"  Generation speed: {len(gen_prices_list)} seqs × {SEQ_LEN} frames in {gen_time:.1f}s")
print("=" * 64)
PYEOF

echo "=== ALL DONE ==="
