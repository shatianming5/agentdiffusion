#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "============================================================"
echo "  Decoder -> Price Space Evaluation"
echo "  Video DiT agent grids -> Order Decoder -> price series"
echo "============================================================"

.venv/bin/python3 -u << 'PYEOF'
import sys, os, logging, re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kurtosis as scipy_kurtosis, wasserstein_distance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device: %s", device)

# ============================================================
# Paths
# ============================================================
OB = "data/external/lobster/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
MSG = "data/external/lobster/AMZN_2012-06-21_34200000_57600000_message_10.csv"
GRID_H, GRID_W = 4, 6

# ============================================================
# Imports
# ============================================================
from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.models.order_decoder import AgentToOrderDecoder
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.data.lob_dataset import LOBVideoDataset
from agentdiffusion.eval.stylized_facts import (
    compute_returns, acf, evaluate_stylized_facts,
    return_distribution_wasserstein,
)
from agentdiffusion.infer.interactive_sim import InteractiveSimulator


def checkpoint_step(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


# ============================================================
# 1) Load dataset
# ============================================================
logger.info("=== Loading LOBSTER AMZN dataset ===")
dataset = LOBVideoDataset(
    OB, MSG, total_frames=20, cond_frames=4,
    subsample=10, grid_shape=(GRID_H, GRID_W),
)
d_latent = dataset[0]["frames"].shape[-1]
logger.info("Dataset: %d sequences, grid=(%d,%d), d_latent=%d",
            len(dataset), GRID_H, GRID_W, d_latent)

# ============================================================
# 2) Load model + decoder
# ============================================================
VDIT_DIR_V2 = Path("outputs/vdit_lob_8x8_enhanced_v2")
VDIT_DIR_V1 = Path("outputs/vdit_lob_8x8_enhanced")

vdit_dir = None
for candidate in [VDIT_DIR_V2, VDIT_DIR_V1]:
    if candidate.exists():
        ckpts = sorted(candidate.glob("video_dit_step_*.pt"), key=checkpoint_step)
        if ckpts:
            vdit_dir = candidate
            break

if vdit_dir is None:
    print("[ERROR] No trained Video DiT model found.")
    print("  Looked in:")
    print("    - outputs/vdit_lob_8x8_enhanced_v2/")
    print("    - outputs/vdit_lob_8x8_enhanced/")
    print("  Run scripts/run_lob_enhanced.sh first to train a model.")
    sys.exit(1)

ckpt_path = ckpts[-1]
logger.info("Loading model from %s", ckpt_path)

model = VideoDiT(
    d_latent=d_latent, d_model=256, depth=8, heads=8,
    patch_size=2, num_frames=20, num_cond_frames=4,
    market_cond_dim=32, grid_h=GRID_H, grid_w=GRID_W,
    causal_temporal=True, alibi_temporal=True,
).to(device)

decoder = AgentToOrderDecoder(
    d_state=d_latent, d_model=128, n_queries=64,
    n_layers=2, n_heads=4, d_order_out=6,
).to(device)

ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
model.load_state_dict(ckpt.get("ema", ckpt["model"]))
model.eval()
has_decoder_weights = "decoder" in ckpt
if has_decoder_weights:
    decoder.load_state_dict(ckpt["decoder"])
    logger.info("Loaded decoder weights from checkpoint")
else:
    logger.warning("No decoder weights in checkpoint -- using random init")
decoder.eval()

scheduler = NoiseScheduler(1000, "cosine").to(device)
sampler = VideoDDIMSampler(model, scheduler, "v_prediction", ddim_steps=20, eta=0.0)
sim = InteractiveSimulator(model, sampler, num_cond=4, num_gen=16, zero_sum_proj=True)

# ============================================================
# 3) Generate agent grids and decode to orders
# ============================================================
print()
print("=" * 70)
print("  Part 1: Generate Agent Grids -> Decode to Orders -> Price Series")
print("=" * 70)

N_SAMPLES = 20
N_ROUNDS = 10  # 10 rounds x 16 frames = 160 frames per sample

# Order feature layout (d_order_out=6):
#   dim 0: price_offset
#   dim 1: log_size
#   dim 2: direction_logit (>0 = buy, <0 = sell)
#   dim 3: (other)
#   dim 4: (other)
#   dim 5: activity_logit (>0 = active order)

# Base price for AMZN (from LOBSTER data, price in cents x 100)
ob_df = pd.read_csv(OB, header=None)
ob_vals = ob_df.values
real_ask1 = ob_vals[:, 0]
real_bid1 = ob_vals[:, 2]
real_mid_full = (real_ask1 + real_bid1) / 2.0
BASE_PRICE = float(np.median(real_mid_full[real_mid_full > 0]))
logger.info("Base price (median mid): %.2f", BASE_PRICE)

all_decoded_mid = []      # mid-price series per sample
all_decoded_spread = []   # spread series per sample
all_decoded_volume = []   # total volume per timestep
all_decoded_n_active = [] # number of active orders per timestep

with torch.no_grad():
    for s in range(N_SAMPLES):
        seed_idx = s % len(dataset)
        seed = dataset[seed_idx]["frames"][:4]
        sim.init(seed)

        # Collect all generated frames
        all_frames_list = [seed.unsqueeze(0).to(device)]
        for _ in range(N_ROUNDS):
            gen = sim.step()  # [16, H, W, D]
            all_frames_list.append(gen.unsqueeze(0).to(device))
            sim.trim_buffer(keep_last=8)

        # Concatenate: [1, T_total, H, W, D]
        gen_frames = torch.cat(all_frames_list, dim=1)
        T_total = gen_frames.shape[1]

        # Decode consecutive frame pairs to orders
        # decode_sequence expects [B, T, H, W, D] and returns [B, T-1, n_queries, 6]
        # Process in chunks to avoid OOM
        chunk_size = 20
        orders_list = []
        for c_start in range(0, T_total - 1, chunk_size - 1):
            c_end = min(c_start + chunk_size, T_total)
            chunk = gen_frames[:, c_start:c_end]
            if chunk.shape[1] < 2:
                break
            orders_chunk = decoder.decode_sequence(chunk)  # [1, chunk_T-1, 64, 6]
            orders_list.append(orders_chunk)

        if not orders_list:
            continue
        all_orders = torch.cat(orders_list, dim=1)  # [1, T_transitions, 64, 6]
        orders_np = all_orders[0].cpu().numpy()      # [T_transitions, 64, 6]

        T_out = orders_np.shape[0]

        # Extract price series from decoded orders
        mid_prices = []
        spreads = []
        volumes = []
        n_active_list = []

        for t_idx in range(T_out):
            price_offsets = orders_np[t_idx, :, 0]  # [64]
            log_sizes = orders_np[t_idx, :, 1]      # [64]
            dir_logits = orders_np[t_idx, :, 2]     # [64]
            activity_logits = orders_np[t_idx, :, 5] # [64]

            # Filter to active orders only
            active_mask = activity_logits > 0
            n_active = active_mask.sum()
            n_active_list.append(int(n_active))

            if n_active == 0:
                # No active orders: carry forward previous mid or use base
                mid_prices.append(mid_prices[-1] if mid_prices else BASE_PRICE)
                spreads.append(0.0)
                volumes.append(0.0)
                continue

            active_prices = BASE_PRICE + price_offsets[active_mask] * 100
            active_sizes = np.exp(np.abs(log_sizes[active_mask])).clip(1, 1e6)
            active_dirs = dir_logits[active_mask]

            # Separate buy and sell orders
            buy_mask = active_dirs > 0
            sell_mask = ~buy_mask

            # Compute weighted mid-price from buy/sell order prices
            if buy_mask.any() and sell_mask.any():
                buy_prices = active_prices[buy_mask]
                buy_sizes = active_sizes[buy_mask]
                sell_prices = active_prices[sell_mask]
                sell_sizes = active_sizes[sell_mask]

                # Best bid = highest buy price, best ask = lowest sell price
                best_bid = np.max(buy_prices)
                best_ask = np.min(sell_prices)

                # If crossed, use volume-weighted average as mid
                if best_bid >= best_ask:
                    w_buy = buy_sizes.sum()
                    w_sell = sell_sizes.sum()
                    w_total = w_buy + w_sell
                    mid = (np.sum(buy_prices * buy_sizes) / w_buy * w_buy / w_total
                           + np.sum(sell_prices * sell_sizes) / w_sell * w_sell / w_total)
                else:
                    mid = (best_bid + best_ask) / 2.0

                spread = max(best_ask - best_bid, 0.0)
            elif buy_mask.any():
                mid = np.average(active_prices[buy_mask], weights=active_sizes[buy_mask])
                spread = 0.0
            else:
                mid = np.average(active_prices[sell_mask], weights=active_sizes[sell_mask])
                spread = 0.0

            mid_prices.append(float(mid))
            spreads.append(float(spread))
            volumes.append(float(active_sizes.sum()))

        if len(mid_prices) > 10:
            all_decoded_mid.append(np.array(mid_prices))
            all_decoded_spread.append(np.array(spreads))
            all_decoded_volume.append(np.array(volumes))
            all_decoded_n_active.append(np.array(n_active_list))

        if (s + 1) % 5 == 0:
            logger.info("  Generated %d/%d samples, current T=%d", s + 1, N_SAMPLES, len(mid_prices))

print("  Generated {} price series from decoded orders".format(len(all_decoded_mid)))
if not all_decoded_mid:
    print("[ERROR] No valid decoded price series generated.")
    sys.exit(1)

# ============================================================
# 4) Compute stylized facts on RECONSTRUCTED price series
# ============================================================
print()
print("=" * 70)
print("  Part 2: Stylized Facts on Decoder-Reconstructed Prices")
print("=" * 70)

# Per-sample stylized facts
sf_results = []
for i, prices in enumerate(all_decoded_mid):
    if len(prices) < 30:
        continue
    vols = all_decoded_volume[i][:len(prices)]
    sf = evaluate_stylized_facts(prices, volumes=vols[1:] if len(vols) > 1 else None)
    sf_results.append(sf)

if sf_results:
    n_sf = len(sf_results)
    avg_passed = np.mean([sf.total_passed for sf in sf_results])
    ft_rate = np.mean([sf.fat_tail_pass for sf in sf_results]) * 100
    vc_rate = np.mean([sf.volatility_clustering_pass for sf in sf_results]) * 100
    le_rate = np.mean([sf.leverage_effect_pass for sf in sf_results]) * 100
    vv_rate = np.mean([sf.volume_volatility_pass for sf in sf_results]) * 100
    ra_rate = np.mean([sf.return_autocorr_pass for sf in sf_results]) * 100
    gl_rate = np.mean([sf.gain_loss_asymmetry_pass for sf in sf_results]) * 100

    print("  Evaluated {} samples".format(n_sf))
    print("  Average stylized facts passed: {:.1f}/6".format(avg_passed))
    print()
    print("  {:<30s} {:<10s}".format("Fact", "Pass rate"))
    print("  " + "-" * 40)
    print("  {:<30s} {:.0f}%".format("Fat tails", ft_rate))
    print("  {:<30s} {:.0f}%".format("Volatility clustering", vc_rate))
    print("  {:<30s} {:.0f}%".format("Leverage effect", le_rate))
    print("  {:<30s} {:.0f}%".format("Volume-volatility corr", vv_rate))
    print("  {:<30s} {:.0f}%".format("Return autocorrelation", ra_rate))
    print("  {:<30s} {:.0f}%".format("Gain-loss asymmetry", gl_rate))
else:
    print("  [WARN] No valid stylized fact results.")

# ============================================================
# 5) Compare with real LOBSTER mid-price
# ============================================================
print()
print("=" * 70)
print("  Part 3: Comparison with Real LOBSTER (AMZN)")
print("=" * 70)

# Real metrics (subsample to ~match generated density)
real_mid_sub = real_mid_full[::10]
real_mid_sub = real_mid_sub[real_mid_sub > 0]
real_ret = compute_returns(real_mid_sub)
real_abs_acf = acf(np.abs(real_ret), 10)
real_kurt = float(scipy_kurtosis(real_ret, fisher=False))

# Our decoded metrics (concatenate all samples)
decoded_all_prices = np.concatenate(all_decoded_mid)
decoded_ret = compute_returns(decoded_all_prices)
decoded_abs_acf = acf(np.abs(decoded_ret), 10)
decoded_kurt = float(scipy_kurtosis(decoded_ret, fisher=False))

# Wasserstein distance
min_len = min(len(decoded_ret), len(real_ret))
wd = float(wasserstein_distance(decoded_ret[:min_len], real_ret[:min_len]))

# Real stylized facts (one long series)
real_sf = evaluate_stylized_facts(real_mid_sub[:5000])

hdr = "{:<25s} {:<20s} {:<20s}"
row_f = "{:<25s} {:<20.4f} {:<20.4f}"
row_2f = "{:<25s} {:<20.2f} {:<20.2f}"

print()
print(hdr.format("Metric", "Real (AMZN)", "Decoded (Ours)"))
print("-" * 65)
print(row_2f.format("Kurtosis", real_kurt, decoded_kurt))
print(row_f.format("|Ret| ACF(1)",
    real_abs_acf[1] if len(real_abs_acf) > 1 else 0.0,
    decoded_abs_acf[1] if len(decoded_abs_acf) > 1 else 0.0))
print(row_f.format("|Ret| ACF(5)",
    real_abs_acf[5] if len(real_abs_acf) > 5 else 0.0,
    decoded_abs_acf[5] if len(decoded_abs_acf) > 5 else 0.0))
print(row_2f.format("Ret std (x1000)",
    float(real_ret.std() * 1000),
    float(decoded_ret.std() * 1000)))
print(row_f.format("W-dist to real", 0.0, wd))
if sf_results:
    print("{:<25s} {:<20s} {:<20s}".format("Stylized Facts",
        "{}/6".format(real_sf.total_passed),
        "{:.1f}/6".format(avg_passed)))

# ============================================================
# 6) Spread and volume analysis from decoded orders
# ============================================================
print()
print("=" * 70)
print("  Part 4: Spread & Volume from Decoded Orders")
print("=" * 70)

# Real spread and volume
real_spread = real_ask1 - real_bid1
real_spread_sub = real_spread[::10]
real_spread_bps = real_spread_sub / (real_mid_sub[:len(real_spread_sub)] + 1e-8) * 10000

# Decoded spread
all_spreads_flat = np.concatenate(all_decoded_spread)
decoded_spread_bps = all_spreads_flat / (np.concatenate(all_decoded_mid)[:len(all_spreads_flat)] + 1e-8) * 10000

# Decoded volume
all_volumes_flat = np.concatenate(all_decoded_volume)

# Active order stats
all_n_active_flat = np.concatenate(all_decoded_n_active)

print()
print("  {:<35s} {:<15s} {:<15s}".format("Metric", "Real", "Decoded"))
print("  " + "-" * 65)
print("  {:<35s} {:<15.2f} {:<15.2f}".format(
    "Spread mean (bps)",
    float(real_spread_bps[real_spread_bps > 0].mean()) if (real_spread_bps > 0).any() else 0.0,
    float(decoded_spread_bps[decoded_spread_bps > 0].mean()) if (decoded_spread_bps > 0).any() else 0.0))
print("  {:<35s} {:<15.2f} {:<15.2f}".format(
    "Spread std (bps)",
    float(real_spread_bps[real_spread_bps > 0].std()) if (real_spread_bps > 0).any() else 0.0,
    float(decoded_spread_bps[decoded_spread_bps > 0].std()) if (decoded_spread_bps > 0).any() else 0.0))
print("  {:<35s} {:<15.1f} {:<15.1f}".format(
    "Volume mean (per timestep)",
    0.0,  # N/A for real (would need message-level aggregation)
    float(all_volumes_flat.mean())))
print("  {:<35s} {:<15s} {:<15.1f}".format(
    "Active orders mean (per timestep)",
    "N/A",
    float(all_n_active_flat.mean())))
print("  {:<35s} {:<15s} {:<15.1f}".format(
    "Active orders std (per timestep)",
    "N/A",
    float(all_n_active_flat.std())))

# Zero-spread ratio
n_zero_spread_decoded = (all_spreads_flat == 0).sum()
print("  {:<35s} {:<15s} {:<15.1f}%".format(
    "Zero-spread ratio",
    "N/A",
    float(n_zero_spread_decoded / len(all_spreads_flat) * 100)))

# ============================================================
# 7) Summary
# ============================================================
print()
print("=" * 70)
print("  SUMMARY")
print("=" * 70)
print()
print("  Pipeline: Video DiT -> agent grids -> Order Decoder -> orders -> prices")
print("  The decoder translates diffusion output (agent-state space) into")
print("  interpretable order-level quantities (price, size, direction).")
print()
if sf_results:
    print("  Decoded price stylized facts: {:.1f}/6 (avg over {} samples)".format(
        avg_passed, len(sf_results)))
print("  Wasserstein distance (decoded vs real): {:.6f}".format(wd))
print("  Decoder checkpoint: {} ({})".format(
    "trained" if has_decoder_weights else "random-init", ckpt_path.name))
print()
print("  Key insight: The Order Decoder bridges agent-state space and")
print("  price space, enabling fair comparison with LOB-Bench baselines")
print("  that operate directly on order/price data.")
print()
print("=" * 70)
PYEOF
