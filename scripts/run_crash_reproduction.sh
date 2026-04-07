#!/bin/bash
# ============================================================================
#  Historical Event Reproduction: 2020-03 COVID Crash
#  ---------------------------------------------------
#  1. Load 20200306 (normal Friday) -> build agent grid (K=1000, 32x32)
#  2. Load 20200309 (crash Monday)  -> build agent grid as ground truth
#  3. Train Video DiT on 20200306 data (5000 steps)
#  4. Seed model with LAST 4 frames of 20200306
#  5. Generate "next day" prediction
#  6. Compare generated vs real 20200309 metrics
#  7. Also compare 20200310 (recovery day)
#  8. Output: Normal -> Crash -> Recovery comparison table
# ============================================================================
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "============================================================"
echo "  COVID Crash Reproduction: 20200306 -> 20200309 -> 20200310"
echo "  K=1000, grid=32x32, patch_size=4"
echo "============================================================"

.venv/bin/python3 -u << 'PYEOF'
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, logging, sys, os
from pathlib import Path
from tqdm import tqdm
from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.data.ashare_10k_agents import (
    load_orders_multi_stock, extract_order_features, cluster_agents,
    arrange_grid_2d, build_agent_grid, D_STATE,
)
from agentdiffusion.infer.interactive_sim import InteractiveSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Configuration
# ============================================================
K_CLUSTERS   = 1000
GRID_H, GRID_W = 32, 32
PATCH_SIZE   = 4
D_LATENT     = D_STATE  # 6
TOTAL_FRAMES = 20
COND_FRAMES  = 4
NUM_GEN      = TOTAL_FRAMES - COND_FRAMES
TOTAL_STEPS  = 5000
BATCH_SIZE   = 8
WINDOW_SEC   = 5.0
MAX_STOCKS   = 30

DATA_NORMAL   = "data/external/20200306"
DATA_CRASH    = "data/external/20200309"
DATA_RECOVERY = "data/external/20200310"

OUT_DIR = Path("outputs/crash_reproduction")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Helper: build raw (un-normalized) agent grid from a data dir
# ============================================================
def build_raw_grid(data_dir, labels, grid_pos):
    """Build raw agent grid from data dir using pre-fitted cluster assignments."""
    orders = load_orders_multi_stock(data_dir, MAX_STOCKS)
    if orders.empty:
        logger.error("No orders from %s", data_dir)
        return None, None
    features = extract_order_features(orders)
    # Re-predict labels with the same cluster model
    from sklearn.cluster import MiniBatchKMeans
    local_labels = labels  # passed in from the training clustering
    grid, time_edges = build_agent_grid(orders, local_labels, grid_pos, WINDOW_SEC, GRID_H, GRID_W)
    return grid, time_edges

# ============================================================
# Step 1: Load 20200306 (normal day) and build agent grid + clusters
# ============================================================
print("\n" + "=" * 64)
print("Step 1: Loading 20200306 (normal day) and clustering agents")
print("=" * 64)

for ddir in [DATA_NORMAL, DATA_CRASH, DATA_RECOVERY]:
    if not Path(ddir).exists():
        logger.error("Data directory not found: %s", ddir)
        logger.error("Please ensure A-share L3 data is available at data/external/")
        sys.exit(1)

orders_normal = load_orders_multi_stock(DATA_NORMAL, MAX_STOCKS)
if orders_normal.empty:
    logger.error("No orders loaded from %s!", DATA_NORMAL)
    sys.exit(1)

features_normal = extract_order_features(orders_normal)
labels_normal, centers = cluster_agents(features_normal, K_CLUSTERS)
grid_pos = arrange_grid_2d(centers, GRID_H, GRID_W)

grid_normal, te_normal = build_agent_grid(
    orders_normal, labels_normal, grid_pos, WINDOW_SEC, GRID_H, GRID_W)
logger.info("Normal day grid: %s, T=%d frames", grid_normal.shape, grid_normal.shape[0])

# ============================================================
# Step 2: Load 20200309 (crash day) and 20200310 (recovery)
# ============================================================
print("\n" + "=" * 64)
print("Step 2: Loading crash day (20200309) and recovery day (20200310)")
print("=" * 64)

# For crash/recovery days, we re-cluster with the same features approach
# but re-assign to same grid positions using the centers from normal day
orders_crash = load_orders_multi_stock(DATA_CRASH, MAX_STOCKS)
features_crash = extract_order_features(orders_crash)
# Predict cluster labels using normal-day centers (nearest center)
from sklearn.metrics import pairwise_distances
dists_crash = pairwise_distances(features_crash, centers)
labels_crash = dists_crash.argmin(axis=1)
grid_crash, te_crash = build_agent_grid(
    orders_crash, labels_crash, grid_pos, WINDOW_SEC, GRID_H, GRID_W)
logger.info("Crash day grid: %s, T=%d frames", grid_crash.shape, grid_crash.shape[0])

orders_recovery = load_orders_multi_stock(DATA_RECOVERY, MAX_STOCKS)
features_recovery = extract_order_features(orders_recovery)
dists_recovery = pairwise_distances(features_recovery, centers)
labels_recovery = dists_recovery.argmin(axis=1)
grid_recovery, te_recovery = build_agent_grid(
    orders_recovery, labels_recovery, grid_pos, WINDOW_SEC, GRID_H, GRID_W)
logger.info("Recovery day grid: %s, T=%d frames", grid_recovery.shape, grid_recovery.shape[0])

# ============================================================
# Compute normalization stats from normal day (shared across all)
# ============================================================
flat_normal = grid_normal.reshape(-1, D_STATE)
g_mean = flat_normal.mean(axis=0, keepdims=True)
g_std  = flat_normal.std(axis=0, keepdims=True).clip(min=1e-8)

def normalize(grid):
    return ((grid - g_mean) / g_std).clip(-5, 5)

grid_normal_n = normalize(grid_normal)
grid_crash_n  = normalize(grid_crash)
grid_recov_n  = normalize(grid_recovery)

# ============================================================
# Step 3: Build training dataset from normal day and train
# ============================================================
print("\n" + "=" * 64)
print("Step 3: Training Video DiT on 20200306 data (%d steps)" % TOTAL_STEPS)
print("=" * 64)

# Build sequences from normal day
sequences = []
stride = TOTAL_FRAMES // 2
T_total = grid_normal_n.shape[0]
for i in range(0, T_total - TOTAL_FRAMES + 1, stride):
    seq = grid_normal_n[i:i + TOTAL_FRAMES]
    sequences.append(torch.from_numpy(seq).float())
logger.info("Training sequences: %d (from %d frames)", len(sequences), T_total)

if len(sequences) < 2:
    logger.error("Not enough sequences for training!")
    sys.exit(1)

# Simple dataset
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, seqs):
        self.seqs = seqs
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        return {"frames": self.seqs[idx], "market_conds": torch.zeros(TOTAL_FRAMES, 32)}

loader = torch.utils.data.DataLoader(
    SeqDataset(sequences), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

# Model: 32x32 grid, patch_size=4 -> 8x8=64 patches
model = VideoDiT(
    d_latent=D_LATENT, d_model=256, depth=8, heads=8,
    patch_size=PATCH_SIZE, num_frames=TOTAL_FRAMES, num_cond_frames=COND_FRAMES,
    mlp_ratio=4.0, market_cond_dim=32,
    grid_h=GRID_H, grid_w=GRID_W,
    causal_temporal=True, alibi_temporal=True,
).to(device)
n_params = sum(p.numel() for p in model.parameters())
logger.info("Model params: %.1fM (grid=%dx%d, patch=%d)", n_params / 1e6, GRID_H, GRID_W, PATCH_SIZE)

scheduler = NoiseScheduler(1000, "cosine").to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
lr_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)

ema_state = {k: v.clone() for k, v in model.state_dict().items()}
ema_decay = 0.999
def update_ema():
    with torch.no_grad():
        for k, v in model.state_dict().items():
            ema_state[k].lerp_(v, 1 - ema_decay)

K = COND_FRAMES
step = 0
pbar = tqdm(total=TOTAL_STEPS, desc="Training (normal day)")
while step < TOTAL_STEPS:
    for batch in loader:
        if step >= TOTAL_STEPS:
            break
        frames = batch["frames"].to(device)
        B, T, H, W, C = frames.shape
        N = T - K
        z_cond, z_gen = frames[:, :K], frames[:, K:]
        t = torch.randint(0, 1000, (B,), device=device)
        noise = torch.randn_like(z_gen)
        t_exp = t.unsqueeze(1).expand(B, N).reshape(B * N)
        z_noisy = scheduler.q_sample(
            z_gen.reshape(B*N, H, W, C), t_exp, noise.reshape(B*N, H, W, C)
        ).reshape(B, N, H, W, C)
        v_pred = model(z_cond, z_noisy, t)
        v_target = scheduler.v_target(
            z_gen.reshape(B*N, H, W, C), noise.reshape(B*N, H, W, C), t_exp
        ).reshape(B, N, H, W, C)
        loss = F.mse_loss(v_pred, v_target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_sched.step()
        update_ema()
        step += 1
        pbar.update(1)
        if step % 500 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", step=step)
pbar.close()

torch.save({"model": model.state_dict(), "ema": ema_state, "step": step,
            "g_mean": g_mean, "g_std": g_std},
           OUT_DIR / "crash_model.pt")
logger.info("Training done. Model saved.")

# ============================================================
# Step 4: Generate prediction -- seed with last 4 frames of normal day
# ============================================================
print("\n" + "=" * 64)
print("Step 4: Generating 'next day' prediction from model")
print("=" * 64)

model.load_state_dict(ema_state)
model.eval()

sampler = VideoDDIMSampler(model, scheduler, "v_prediction", ddim_steps=30, eta=0.0)
sim = InteractiveSimulator(model, sampler, num_cond=COND_FRAMES, num_gen=NUM_GEN, zero_sum_proj=True)

# Seed: last 4 frames of normal day
seed_frames = torch.from_numpy(grid_normal_n[-COND_FRAMES:]).float()
sim.init(seed_frames)

# Generate multiple rounds to cover ~1 day equivalent
# Normal day has T_total frames; we generate enough to cover crash day length
T_crash = grid_crash_n.shape[0]
n_rounds_needed = max(1, (T_crash + NUM_GEN - 1) // NUM_GEN)
n_rounds_needed = min(n_rounds_needed, 10)  # cap at 10 rounds

all_generated = []
for r in range(n_rounds_needed):
    gen = sim.step()
    all_generated.append(gen.cpu())
    sim.trim_buffer(keep_last=COND_FRAMES * 2)

generated = torch.cat(all_generated, dim=0).numpy()  # [T_gen, H, W, D]
logger.info("Generated %d frames (covering predicted 'next day')", generated.shape[0])

# ============================================================
# Step 5 & 6: Compare generated vs real crash day
# ============================================================
print("\n" + "=" * 64)
print("Step 5: Comparing Generated vs Real (Normal / Crash / Recovery)")
print("=" * 64)

def compute_day_metrics(grid_normalized, label=""):
    """Compute aggregate metrics from normalized grid [T, H, W, D]."""
    # dim 0: net_signed_volume, dim 1: order_count, dim 2: avg_price
    # dim 3: cancel_rate, dim 4: avg_size, dim 5: buy_ratio
    sv  = grid_normalized[:, :, :, 0]   # signed volume
    cnt = grid_normalized[:, :, :, 1]   # order count / activity
    px  = grid_normalized[:, :, :, 2]   # avg price
    cr  = grid_normalized[:, :, :, 3]   # cancel rate
    br  = grid_normalized[:, :, :, 5]   # buy ratio

    mean_sv       = float(np.mean(sv))
    std_sv        = float(np.std(sv))
    mean_activity = float(np.mean(cnt))
    std_activity  = float(np.std(cnt))
    volatility    = float(np.std(px))
    mean_cancel   = float(np.mean(cr))
    mean_buy_ratio= float(np.mean(br))
    # Activity spike: fraction of cells with activity > 2 std above global mean
    activity_spike = float(np.mean(np.abs(cnt) > 2.0))

    return {
        "label": label,
        "mean_signed_vol": mean_sv,
        "vol_std": std_sv,
        "mean_activity": mean_activity,
        "activity_std": std_activity,
        "price_volatility": volatility,
        "mean_cancel_rate": mean_cancel,
        "mean_buy_ratio": mean_buy_ratio,
        "activity_spike_frac": activity_spike,
    }

# Use comparable number of frames from each day
T_compare = min(generated.shape[0], grid_crash_n.shape[0], grid_recov_n.shape[0], grid_normal_n.shape[0])
# Take first T_compare frames from crash and recovery; last T_compare from normal
metrics_normal    = compute_day_metrics(grid_normal_n[-T_compare:], "Normal (0306)")
metrics_crash     = compute_day_metrics(grid_crash_n[:T_compare], "Crash (0309)")
metrics_recovery  = compute_day_metrics(grid_recov_n[:T_compare], "Recovery (0310)")
metrics_predicted = compute_day_metrics(generated[:T_compare], "Predicted (Model)")

# Spatial pattern similarity: correlation between generated and real crash grids
# Average each grid over time to get [H, W, D] spatial pattern
def spatial_pattern(grid):
    return np.mean(grid, axis=0)  # [H, W, D]

sp_generated = spatial_pattern(generated[:T_compare])
sp_crash     = spatial_pattern(grid_crash_n[:T_compare])
sp_normal    = spatial_pattern(grid_normal_n[-T_compare:])

# Flatten and compute Pearson correlation
corr_gen_crash  = float(np.corrcoef(sp_generated.flatten(), sp_crash.flatten())[0, 1])
corr_gen_normal = float(np.corrcoef(sp_generated.flatten(), sp_normal.flatten())[0, 1])
corr_crash_norm = float(np.corrcoef(sp_crash.flatten(), sp_normal.flatten())[0, 1])

# ============================================================
# Step 7: Print comparison table
# ============================================================
print("\n" + "=" * 80)
print("  COVID Crash Reproduction -- Comparison Table")
print("=" * 80)
header = f"{'Metric':<28} {'Normal(0306)':>14} {'Crash(0309)':>14} {'Recovery(0310)':>14} {'Predicted':>14}"
print(header)
print("-" * 80)

all_m = [metrics_normal, metrics_crash, metrics_recovery, metrics_predicted]
for key in ["mean_signed_vol", "vol_std", "mean_activity", "activity_std",
            "price_volatility", "mean_cancel_rate", "mean_buy_ratio", "activity_spike_frac"]:
    row = f"{key:<28}"
    for m in all_m:
        row += f" {m[key]:>14.4f}"
    print(row)

print("-" * 80)
print(f"\nSpatial Pattern Correlations:")
print(f"  Generated vs Crash (0309):   {corr_gen_crash:+.4f}")
print(f"  Generated vs Normal (0306):  {corr_gen_normal:+.4f}")
print(f"  Crash vs Normal:             {corr_crash_norm:+.4f}")

# Check key crash signatures
print(f"\nCrash Signature Analysis:")
sv_normal = metrics_normal["mean_signed_vol"]
sv_crash  = metrics_crash["mean_signed_vol"]
sv_pred   = metrics_predicted["mean_signed_vol"]
sv_recov  = metrics_recovery["mean_signed_vol"]

print(f"  Signed volume: Normal={sv_normal:+.4f} -> Crash={sv_crash:+.4f} -> Recovery={sv_recov:+.4f}")
print(f"  Model predicted: {sv_pred:+.4f}")
if sv_pred < sv_normal:
    print(f"  [OK] Model predicts selling pressure (more negative than normal)")
else:
    print(f"  [--] Model did NOT predict increased selling")

vol_normal = metrics_normal["price_volatility"]
vol_crash  = metrics_crash["price_volatility"]
vol_pred   = metrics_predicted["price_volatility"]
print(f"  Volatility: Normal={vol_normal:.4f} -> Crash={vol_crash:.4f}")
print(f"  Model predicted: {vol_pred:.4f}")
if vol_pred > vol_normal:
    print(f"  [OK] Model predicts increased volatility")
else:
    print(f"  [--] Model did NOT predict volatility increase")

act_normal = metrics_normal["activity_spike_frac"]
act_crash  = metrics_crash["activity_spike_frac"]
act_pred   = metrics_predicted["activity_spike_frac"]
print(f"  Activity spike: Normal={act_normal:.4f} -> Crash={act_crash:.4f}")
print(f"  Model predicted: {act_pred:.4f}")

print("\n" + "=" * 80)
print("  Crash Reproduction Experiment Complete")
print("=" * 80)
PYEOF
