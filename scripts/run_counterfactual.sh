#!/bin/bash
# ============================================================================
#  Real Counterfactual Causal Analysis: Market Maker Withdrawal
#  -------------------------------------------------------------
#  1. Load A-share L3 data for 20240619 (full trading day)
#  2. Build agent grid with AShareL3VideoDataset
#  3. Identify "market maker behavior" windows (high cancel, balanced buy/sell)
#  4. Identify "MM withdrawal" windows (those agents go inactive)
#  5. Measure REAL spread change from snapshots during withdrawal
#  6. Model predictions: normal generation vs counterfactual (MM zeroed out)
#  7. Compare real vs model vs counterfactual spread changes
#  8. Report correlation
# ============================================================================
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "============================================================"
echo "  Counterfactual Causal Analysis: Market Maker Withdrawal"
echo "  Data: A-Share L3 20240619"
echo "============================================================"

.venv/bin/python3 -u << 'PYEOF'
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, logging, sys, os
from pathlib import Path
from tqdm import tqdm
from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.data.ashare_l3_dataset import (
    AShareL3VideoDataset, load_stock_l3, build_agent_states_from_orders,
    build_market_conditions, MARKET_COND_DIM,
)
from agentdiffusion.infer.interactive_sim import InteractiveSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Configuration
# ============================================================
DATA_DIR      = "data/external/20240619"
GRID_SHAPE    = (4, 4)  # 16 agent types (AShareL3VideoDataset default)
N_AGENTS      = GRID_SHAPE[0] * GRID_SHAPE[1]  # 16
D_STATE       = 6
D_LATENT      = D_STATE
TOTAL_FRAMES  = 20
COND_FRAMES   = 4
NUM_GEN       = TOTAL_FRAMES - COND_FRAMES
PATCH_SIZE    = 2  # 4x4 grid / patch=2 -> 2x2=4 patches
WINDOW_SEC    = 1.0
TRAIN_STEPS   = 5000
BATCH_SIZE    = 8
MAX_STOCKS    = 50

# MM detection thresholds
MM_CANCEL_THRESHOLD  = 0.3   # cancel_rate (dim 3) > 0.3
MM_BALANCE_LOW       = 0.35  # buy_ratio (dim 5) in [0.35, 0.65]
MM_BALANCE_HIGH      = 0.65
MM_INACTIVE_THRESHOLD = 0.1  # order_count (dim 1) drops below this (after normalization)

OUT_DIR = Path("outputs/counterfactual_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Step 1: Load raw L3 data for analysis (per-stock)
# ============================================================
print("\n" + "=" * 64)
print("Step 1: Loading A-Share L3 data from %s" % DATA_DIR)
print("=" * 64)

if not Path(DATA_DIR).exists():
    # Try alternative
    ext_dir = Path("data/external")
    if ext_dir.exists():
        candidates = sorted([d for d in ext_dir.iterdir() if d.is_dir()])
        if candidates:
            DATA_DIR = str(candidates[0])
            logger.info("Using alternative data dir: %s", DATA_DIR)
        else:
            logger.error("No data directories found!"); sys.exit(1)

data_dir = Path(DATA_DIR)
stock_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])[:MAX_STOCKS]
logger.info("Found %d stock directories", len(stock_dirs))

# ============================================================
# Step 2: Build agent grids and analyze per stock
# ============================================================
print("\n" + "=" * 64)
print("Step 2: Building agent grids and identifying MM behavior")
print("=" * 64)

all_stock_data = []  # (stock_name, agent_states, time_edges, snapshots)

for sd in tqdm(stock_dirs, desc="Loading stocks"):
    try:
        data = load_stock_l3(sd)
        states, time_edges = build_agent_states_from_orders(
            data["orders"], window_seconds=WINDOW_SEC, n_agent_types=N_AGENTS)
        if states.shape[0] < TOTAL_FRAMES:
            continue
        market_conds = build_market_conditions(data["snapshots"], time_edges)
        all_stock_data.append((sd.name, states, time_edges, data["snapshots"], market_conds))
    except Exception as e:
        logger.warning("Failed %s: %s", sd.name, e)

logger.info("Successfully loaded %d stocks", len(all_stock_data))

if not all_stock_data:
    logger.error("No stocks loaded!"); sys.exit(1)

# ============================================================
# Step 3: Identify MM-like agents and withdrawal windows
# ============================================================
print("\n" + "=" * 64)
print("Step 3: Identifying market maker behavior and withdrawal events")
print("=" * 64)

mm_events = []  # (stock, agent_type, mm_window_start, withdrawal_start, withdrawal_end)

for stock_name, states, time_edges, snapshots, mconds in all_stock_data:
    T_s = states.shape[0]
    # states: [T, N_agents, D_STATE]
    # D_STATE: 0=net_position, 1=order_rate, 2=avg_price, 3=cancel_rate, 4=avg_size, 5=aggressiveness

    for agent_idx in range(N_AGENTS):
        agent_ts = states[:, agent_idx, :]  # [T, D]
        cancel_rate = agent_ts[:, 3]
        buy_ratio   = agent_ts[:, 5]  # aggressiveness, used as proxy for buy_ratio
        order_count = agent_ts[:, 1]

        # Identify MM windows: high cancel + balanced buy/sell
        # Use rolling windows of 20 frames
        window_size = 20
        for t_start in range(0, T_s - 2 * window_size, window_size):
            t_end = t_start + window_size
            w_cancel = cancel_rate[t_start:t_end]
            w_balance = buy_ratio[t_start:t_end]
            w_count = order_count[t_start:t_end]

            mean_cancel  = float(np.mean(w_cancel))
            mean_balance = float(np.mean(w_balance))
            mean_count   = float(np.mean(w_count))

            is_mm = (mean_cancel > MM_CANCEL_THRESHOLD and
                     MM_BALANCE_LOW < mean_balance < MM_BALANCE_HIGH and
                     mean_count > 0.5)

            if not is_mm:
                continue

            # Check next window for withdrawal
            next_start = t_end
            next_end = min(next_start + window_size, T_s)
            if next_end - next_start < window_size // 2:
                continue

            next_count = order_count[next_start:next_end]
            mean_next_count = float(np.mean(next_count))

            if mean_next_count < MM_INACTIVE_THRESHOLD * mean_count:
                mm_events.append({
                    "stock": stock_name,
                    "agent_type": agent_idx,
                    "mm_window": (t_start, t_end),
                    "withdrawal_window": (next_start, next_end),
                    "mm_cancel_rate": mean_cancel,
                    "mm_balance": mean_balance,
                    "mm_activity": mean_count,
                    "withdrawal_activity": mean_next_count,
                })

logger.info("Found %d MM withdrawal events across %d stocks", len(mm_events), len(all_stock_data))

if not mm_events:
    logger.warning("No clear MM withdrawal events detected. Relaxing thresholds...")
    # Relax and try again
    MM_CANCEL_THRESHOLD = 0.15
    MM_BALANCE_LOW = 0.25
    MM_BALANCE_HIGH = 0.75
    for stock_name, states, time_edges, snapshots, mconds in all_stock_data:
        T_s = states.shape[0]
        for agent_idx in range(N_AGENTS):
            agent_ts = states[:, agent_idx, :]
            cancel_rate = agent_ts[:, 3]
            buy_ratio   = agent_ts[:, 5]
            order_count = agent_ts[:, 1]
            window_size = 20
            for t_start in range(0, T_s - 2 * window_size, window_size):
                t_end = t_start + window_size
                mean_cancel  = float(np.mean(cancel_rate[t_start:t_end]))
                mean_balance = float(np.mean(buy_ratio[t_start:t_end]))
                mean_count   = float(np.mean(order_count[t_start:t_end]))
                is_mm = (mean_cancel > MM_CANCEL_THRESHOLD and
                         MM_BALANCE_LOW < mean_balance < MM_BALANCE_HIGH and
                         mean_count > 0.3)
                if not is_mm:
                    continue
                next_start = t_end
                next_end = min(next_start + window_size, T_s)
                if next_end - next_start < window_size // 2:
                    continue
                mean_next_count = float(np.mean(order_count[next_start:next_end]))
                if mean_next_count < 0.2 * mean_count:
                    mm_events.append({
                        "stock": stock_name, "agent_type": agent_idx,
                        "mm_window": (t_start, t_end),
                        "withdrawal_window": (next_start, next_end),
                        "mm_cancel_rate": mean_cancel, "mm_balance": mean_balance,
                        "mm_activity": mean_count, "withdrawal_activity": mean_next_count,
                    })
    logger.info("After relaxation: %d MM withdrawal events", len(mm_events))

print(f"  Total MM withdrawal events: {len(mm_events)}")
if mm_events:
    print(f"  Example event: stock={mm_events[0]['stock']}, "
          f"agent={mm_events[0]['agent_type']}, "
          f"cancel_rate={mm_events[0]['mm_cancel_rate']:.3f}, "
          f"activity drop: {mm_events[0]['mm_activity']:.2f} -> {mm_events[0]['withdrawal_activity']:.2f}")

# ============================================================
# Step 4: Measure REAL spread changes from snapshots
# ============================================================
print("\n" + "=" * 64)
print("Step 4: Measuring real spread changes during MM withdrawal")
print("=" * 64)

# Build lookup: stock_name -> (snapshots, time_edges)
stock_lookup = {name: (snaps, te, mc) for name, _, te, snaps, mc in all_stock_data}
states_lookup = {name: states for name, states, _, _, _ in all_stock_data}

real_spread_changes = []  # (event_idx, spread_before, spread_during)

for evt_idx, evt in enumerate(mm_events):
    stock = evt["stock"]
    if stock not in stock_lookup:
        continue
    snaps, te, mc = stock_lookup[stock]

    # Get time range for MM window and withdrawal window
    t_mm_start = te[evt["mm_window"][0]] if evt["mm_window"][0] < len(te) else te[-1]
    t_mm_end   = te[min(evt["mm_window"][1], len(te)-1)]
    t_wd_start = te[min(evt["withdrawal_window"][0], len(te)-1)]
    t_wd_end   = te[min(evt["withdrawal_window"][1], len(te)-1)]

    # Compute spread from snapshots in each window
    snap_ts = snaps["timestamp"].values
    ask_p1  = snaps.get("ask_p1")
    bid_p1  = snaps.get("bid_p1")
    if ask_p1 is None or bid_p1 is None:
        continue

    ask_p1 = ask_p1.values.astype(float)
    bid_p1 = bid_p1.values.astype(float)

    # MM window spread
    mask_mm = (snap_ts >= t_mm_start) & (snap_ts < t_mm_end)
    if mask_mm.sum() < 5:
        continue
    mid_mm = (ask_p1[mask_mm] + bid_p1[mask_mm]) / 2.0
    mid_mm = mid_mm[mid_mm > 0]
    if len(mid_mm) == 0:
        continue
    spread_mm = float(np.mean((ask_p1[mask_mm] - bid_p1[mask_mm])[ask_p1[mask_mm] > 0]))

    # Withdrawal window spread
    mask_wd = (snap_ts >= t_wd_start) & (snap_ts < t_wd_end)
    if mask_wd.sum() < 5:
        continue
    spread_wd = float(np.mean((ask_p1[mask_wd] - bid_p1[mask_wd])[ask_p1[mask_wd] > 0]))

    spread_change = spread_wd - spread_mm
    real_spread_changes.append({
        "event_idx": evt_idx,
        "stock": stock,
        "spread_before": spread_mm,
        "spread_during_withdrawal": spread_wd,
        "spread_change": spread_change,
    })

logger.info("Computed real spread changes for %d events", len(real_spread_changes))

# ============================================================
# Step 5: Train / load model on L3 data
# ============================================================
print("\n" + "=" * 64)
print("Step 5: Preparing Video DiT model for counterfactual generation")
print("=" * 64)

# Build dataset
dataset = AShareL3VideoDataset(
    DATA_DIR, total_frames=TOTAL_FRAMES, cond_frames=COND_FRAMES,
    window_seconds=WINDOW_SEC, grid_shape=GRID_SHAPE, max_stocks=MAX_STOCKS,
)
logger.info("L3 Dataset: %d sequences", len(dataset))

if len(dataset) == 0:
    logger.error("No data sequences!"); sys.exit(1)

# Model: 4x4 grid, patch_size=2 -> 2x2=4 patches
model = VideoDiT(
    d_latent=D_LATENT, d_model=256, depth=8, heads=8,
    patch_size=PATCH_SIZE, num_frames=TOTAL_FRAMES, num_cond_frames=COND_FRAMES,
    mlp_ratio=4.0, market_cond_dim=32,
    grid_h=GRID_SHAPE[0], grid_w=GRID_SHAPE[1],
    causal_temporal=True, alibi_temporal=True,
).to(device)

scheduler = NoiseScheduler(1000, "cosine").to(device)

# Check for checkpoint
ckpt_paths = [
    Path("outputs/vdit_ashare_l3/video_dit_latest.pt"),
    Path("outputs/counterfactual_analysis/cf_model.pt"),
]
loaded = False
for cp in ckpt_paths:
    if cp.exists():
        logger.info("Loading checkpoint: %s", cp)
        ckpt = torch.load(cp, map_location=device, weights_only=False)
        if "ema" in ckpt:
            model.load_state_dict(ckpt["ema"])
        else:
            model.load_state_dict(ckpt["model"])
        loaded = True
        break

if not loaded:
    logger.info("Training new model (%d steps)...", TRAIN_STEPS)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    lr_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_STEPS)
    ema_state = {k: v.clone() for k, v in model.state_dict().items()}
    ema_decay = 0.999
    def update_ema():
        with torch.no_grad():
            for k, v in model.state_dict().items():
                ema_state[k].lerp_(v, 1 - ema_decay)

    K = COND_FRAMES
    step = 0
    pbar = tqdm(total=TRAIN_STEPS, desc="Training L3 model")
    while step < TRAIN_STEPS:
        for batch in loader:
            if step >= TRAIN_STEPS:
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
    model.load_state_dict(ema_state)
    torch.save({"model": ema_state, "ema": ema_state, "step": step},
               OUT_DIR / "cf_model.pt")
    logger.info("Training done.")

model.eval()

# ============================================================
# Step 6: For each MM withdrawal event, run normal + counterfactual generation
# ============================================================
print("\n" + "=" * 64)
print("Step 6: Running normal and counterfactual generation for each event")
print("=" * 64)

sampler = VideoDDIMSampler(model, scheduler, "v_prediction", ddim_steps=20, eta=0.0)

# Process events that have real spread data
events_with_spread = {e["event_idx"]: e for e in real_spread_changes}
max_events = min(20, len(events_with_spread))  # cap for runtime

model_predictions = []  # (event_idx, normal_spread_proxy, cf_spread_proxy)

H, W = GRID_SHAPE
processed = 0

for evt_idx in sorted(events_with_spread.keys())[:max_events]:
    evt = mm_events[evt_idx]
    stock = evt["stock"]
    if stock not in states_lookup:
        continue
    states = states_lookup[stock]
    T_s = states.shape[0]

    mm_start, mm_end = evt["mm_window"]
    wd_start, wd_end = evt["withdrawal_window"]

    # We need at least COND_FRAMES before the withdrawal
    if mm_end - COND_FRAMES < 0 or wd_end > T_s:
        continue

    # Normalize states using per-stock stats
    flat = states.reshape(-1, D_STATE)
    s_mean = flat.mean(axis=0, keepdims=True)
    s_std  = flat.std(axis=0, keepdims=True).clip(min=1e-8)
    states_norm = ((states - s_mean) / s_std).clip(-5, 5)

    # Reshape for model: [T, N_agents, D] -> [T, H, W, D]
    grid_states = states_norm.reshape(T_s, H, W, D_STATE)

    # Seed: COND_FRAMES frames just before withdrawal
    seed_start = max(0, wd_start - COND_FRAMES)
    seed_end = wd_start
    if seed_end - seed_start < COND_FRAMES:
        continue

    seed_frames = torch.from_numpy(grid_states[seed_start:seed_end]).float()

    # --- Normal generation ---
    sim_normal = InteractiveSimulator(model, sampler, num_cond=COND_FRAMES, num_gen=NUM_GEN)
    sim_normal.init(seed_frames)
    gen_normal = sim_normal.step()  # [NUM_GEN, H, W, D]

    # --- Counterfactual: zero out MM agent cells ---
    sim_cf = InteractiveSimulator(model, sampler, num_cond=COND_FRAMES, num_gen=NUM_GEN)
    # Create modified seed where MM agent is zeroed
    seed_cf = seed_frames.clone()
    mm_agent = evt["agent_type"]
    mm_row = mm_agent // W
    mm_col = mm_agent % W
    seed_cf[:, mm_row, mm_col, :] = 0.0  # zero out MM agent in all seed frames
    sim_cf.init(seed_cf)
    gen_cf = sim_cf.step()  # [NUM_GEN, H, W, D]

    # --- Compute spread proxy ---
    # Use price volatility across agents as spread proxy
    # dim 2 = avg_price; higher price variance across agents ~ wider spread
    normal_spread_proxy = float(gen_normal[:, :, :, 2].std().item())
    cf_spread_proxy     = float(gen_cf[:, :, :, 2].std().item())

    # Also look at activity change
    normal_activity = float(gen_normal[:, :, :, 1].mean().item())
    cf_activity     = float(gen_cf[:, :, :, 1].mean().item())

    model_predictions.append({
        "event_idx": evt_idx,
        "stock": stock,
        "normal_spread_proxy": normal_spread_proxy,
        "cf_spread_proxy": cf_spread_proxy,
        "spread_change_pred": cf_spread_proxy - normal_spread_proxy,
        "normal_activity": normal_activity,
        "cf_activity": cf_activity,
    })

    processed += 1
    if processed % 5 == 0:
        logger.info("Processed %d/%d events", processed, max_events)

logger.info("Processed %d events with model predictions", len(model_predictions))

# ============================================================
# Step 7: Compare real vs model vs counterfactual
# ============================================================
print("\n" + "=" * 64)
print("Step 7: Comparing real spread changes vs model predictions")
print("=" * 64)

# Merge real and model data
comparison = []
for mp in model_predictions:
    evt_idx = mp["event_idx"]
    if evt_idx in events_with_spread:
        real = events_with_spread[evt_idx]
        comparison.append({
            "stock": mp["stock"],
            "real_spread_change": real["spread_change"],
            "real_spread_before": real["spread_before"],
            "real_spread_during": real["spread_during_withdrawal"],
            "model_normal_proxy": mp["normal_spread_proxy"],
            "model_cf_proxy": mp["cf_spread_proxy"],
            "model_cf_change": mp["spread_change_pred"],
        })

if comparison:
    print(f"\n{'Stock':<16} {'Real dSpread':>12} {'Model Normal':>13} {'Model CF':>13} {'Model dSpread':>13}")
    print("-" * 72)
    for c in comparison:
        print(f"{c['stock']:<16} {c['real_spread_change']:>12.6f} "
              f"{c['model_normal_proxy']:>13.4f} {c['model_cf_proxy']:>13.4f} "
              f"{c['model_cf_change']:>13.4f}")

    # Compute correlation
    real_changes = np.array([c["real_spread_change"] for c in comparison])
    model_cf_changes = np.array([c["model_cf_change"] for c in comparison])

    if len(comparison) >= 3 and np.std(real_changes) > 1e-10 and np.std(model_cf_changes) > 1e-10:
        corr = float(np.corrcoef(real_changes, model_cf_changes)[0, 1])
    else:
        corr = float("nan")

    print("-" * 72)
    print(f"\nCorrelation (real spread change vs model counterfactual): {corr:+.4f}")
    print(f"Number of events compared: {len(comparison)}")

    # Summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Real spread changes:  mean={np.mean(real_changes):.6f}, std={np.std(real_changes):.6f}")
    print(f"  Model CF changes:     mean={np.mean(model_cf_changes):.4f}, std={np.std(model_cf_changes):.4f}")

    # Direction agreement
    if len(comparison) > 0:
        direction_agree = sum(1 for r, m in zip(real_changes, model_cf_changes)
                              if (r > 0 and m > 0) or (r < 0 and m < 0) or (r == 0 and m == 0))
        print(f"  Direction agreement:  {direction_agree}/{len(comparison)} "
              f"({direction_agree/len(comparison)*100:.1f}%)")

    # Key insight
    real_mean = float(np.mean(real_changes))
    cf_mean = float(np.mean(model_cf_changes))
    print(f"\nKey Insight:")
    if real_mean > 0 and cf_mean > 0:
        print(f"  Both real data and model counterfactual show spread WIDENING when MM withdraws")
        print(f"  This supports the causal interpretation: MM presence narrows spreads")
    elif real_mean > 0:
        print(f"  Real data shows spread widening; model counterfactual direction: {'widening' if cf_mean > 0 else 'narrowing'}")
    else:
        print(f"  Real spread change direction: {'widening' if real_mean > 0 else 'narrowing'}")
        print(f"  Model counterfactual direction: {'widening' if cf_mean > 0 else 'narrowing'}")
else:
    print("No events could be compared (no matching real + model data)")
    print("This may happen if the data format does not contain sufficient snapshot data")

# ============================================================
# Step 8: Save results
# ============================================================
import json

results = {
    "n_stocks": len(all_stock_data),
    "n_mm_events": len(mm_events),
    "n_events_with_spread": len(real_spread_changes),
    "n_compared": len(comparison),
    "comparison": comparison,
    "mm_events_sample": mm_events[:10],
}
with open(OUT_DIR / "counterfactual_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "=" * 80)
print("  Counterfactual Causal Analysis Complete")
print(f"  Results: {OUT_DIR}/counterfactual_results.json")
print("=" * 80)
PYEOF
