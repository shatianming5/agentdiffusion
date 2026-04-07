#!/bin/bash
# ============================================================================
#  Institutional Order Impact Simulation
#  --------------------------------------
#  Compare execution strategies using InteractiveSimulator.intervene():
#
#  Setup: Train/load model on A-share data
#  1. Generate baseline: 20 frames of normal market
#  2. Strategy A (block trade):  single large sell at frame 10
#  3. Strategy B (split 10):    10 smaller sells across frames 10-19
#  4. Strategy C (TWAP):        10 decreasing sells across frames 10-19
#  5. For each: measure price impact, execution cost, recovery time, disruption
#  6. Print comparison table
# ============================================================================
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "============================================================"
echo "  Institutional Order Impact Simulation"
echo "  Block Trade vs Split vs TWAP"
echo "============================================================"

.venv/bin/python3 -u << 'PYEOF'
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, logging, sys, os, copy
from pathlib import Path
from tqdm import tqdm
from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.data.ashare_10k_agents import (
    AShare10KAgentDataset, D_STATE,
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
TRAIN_STEPS  = 5000
BATCH_SIZE   = 8

# Impact simulation params
N_TOTAL_ROUNDS   = 5    # generate 5 rounds of NUM_GEN frames each = 80 frames total
SHOCK_ROUND      = 1    # inject at round 1 (after 1 baseline round)
# Within each round, we get NUM_GEN=16 frames; shock starts at beginning of round 1

# Shock definitions -- target a single agent cluster region (5x5 block in center)
SHOCK_ROW_START, SHOCK_ROW_END = GRID_H // 2 - 2, GRID_H // 2 + 3  # 5 rows
SHOCK_COL_START, SHOCK_COL_END = GRID_W // 2 - 2, GRID_W // 2 + 3  # 5 cols

OUT_DIR = Path("outputs/impact_simulation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Step 1: Load data + train/load model
# ============================================================
print("\n" + "=" * 64)
print("Step 1: Preparing model and data")
print("=" * 64)

DATA_DIR = "data/external/20240619"
if not Path(DATA_DIR).exists():
    ext_dir = Path("data/external")
    if ext_dir.exists():
        candidates = sorted([d for d in ext_dir.iterdir() if d.is_dir()])
        if candidates:
            DATA_DIR = str(candidates[0])
            logger.info("Using alternative data dir: %s", DATA_DIR)
        else:
            logger.error("No data directories found!")
            sys.exit(1)

dataset = AShare10KAgentDataset(
    DATA_DIR, total_frames=TOTAL_FRAMES, cond_frames=COND_FRAMES,
    window_seconds=5.0, max_stocks=30,
    n_clusters=K_CLUSTERS, grid_h=GRID_H, grid_w=GRID_W,
)
if len(dataset) == 0:
    logger.error("No data!"); sys.exit(1)

model = VideoDiT(
    d_latent=D_LATENT, d_model=256, depth=8, heads=8,
    patch_size=PATCH_SIZE, num_frames=TOTAL_FRAMES, num_cond_frames=COND_FRAMES,
    mlp_ratio=4.0, market_cond_dim=32,
    grid_h=GRID_H, grid_w=GRID_W,
    causal_temporal=True, alibi_temporal=True,
).to(device)

scheduler = NoiseScheduler(1000, "cosine").to(device)

# Check for existing checkpoint
ckpt_paths = [
    Path("outputs/vdit_10k_agents/video_dit_step_10000.pt"),
    Path("outputs/vdit_10k_agents/video_dit_step_5000.pt"),
    Path("outputs/impact_simulation/impact_model.pt"),
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
    logger.info("No checkpoint found. Training new model (%d steps)...", TRAIN_STEPS)
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
    pbar = tqdm(total=TRAIN_STEPS, desc="Training")
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
               OUT_DIR / "impact_model.pt")
    logger.info("Training done.")

model.eval()

# ============================================================
# Helper: run a full simulation with optional intervention schedule
# ============================================================
def run_simulation(seed_frames, interventions=None, label="baseline"):
    """Run N_TOTAL_ROUNDS of generation, optionally injecting interventions.

    Args:
        seed_frames: [K, H, W, D] initial condition frames
        interventions: dict mapping (round_idx, frame_within_round) -> delta [D]
                       delta is applied to the shock region via intervene()
        label: name for logging

    Returns:
        all_frames: np array [total_frames, H, W, D]
        price_trajectory: list of mean price per frame
        vol_trajectory: list of signed volume per frame
    """
    sampler = VideoDDIMSampler(model, scheduler, "v_prediction", ddim_steps=20, eta=0.0)
    sim = InteractiveSimulator(model, sampler, num_cond=COND_FRAMES, num_gen=NUM_GEN, zero_sum_proj=True)
    sim.init(seed_frames.clone())

    all_gen = [seed_frames.cpu().numpy()]
    price_traj = []
    vol_traj = []

    # Record seed frame stats
    for fi in range(seed_frames.shape[0]):
        f = seed_frames[fi].cpu().numpy()
        price_traj.append(float(np.mean(f[:, :, 2])))
        vol_traj.append(float(np.mean(f[:, :, 0])))

    for r in range(N_TOTAL_ROUNDS):
        gen = sim.step()  # [NUM_GEN, H, W, D]

        # Apply interventions if any
        if interventions is not None:
            for (ri, fi), delta_val in interventions.items():
                if ri == r:
                    # Build mask for shock region
                    mask = torch.zeros(GRID_H, GRID_W, dtype=torch.bool)
                    mask[SHOCK_ROW_START:SHOCK_ROW_END, SHOCK_COL_START:SHOCK_COL_END] = True
                    delta = torch.zeros(D_LATENT)
                    delta[0] = delta_val  # signed volume shock
                    # Intervene on the last generated frame (most recent)
                    sim.intervene(frame_idx=-1, mask=mask, delta=delta)

        gen_np = sim.buffer[0, -NUM_GEN:].cpu().numpy()
        all_gen.append(gen_np)

        for fi in range(gen_np.shape[0]):
            f = gen_np[fi]
            price_traj.append(float(np.mean(f[:, :, 2])))
            vol_traj.append(float(np.mean(f[:, :, 0])))

        sim.trim_buffer(keep_last=COND_FRAMES * 2)

    all_frames = np.concatenate(all_gen, axis=0)
    return all_frames, price_traj, vol_traj

# ============================================================
# Step 2: Run baseline (no intervention)
# ============================================================
print("\n" + "=" * 64)
print("Step 2: Running baseline simulation (no intervention)")
print("=" * 64)

seed = dataset[0]["frames"][:COND_FRAMES]
baseline_frames, baseline_prices, baseline_vols = run_simulation(seed, label="baseline")
logger.info("Baseline: %d frames generated", len(baseline_prices))

# ============================================================
# Step 3: Strategy A -- Block Trade (single large sell at round 1)
# ============================================================
print("\n" + "=" * 64)
print("Step 3: Strategy A -- Block Trade (single large sell)")
print("=" * 64)

# Single massive shock at round 1, applied once
interventions_A = {(SHOCK_ROUND, 0): -5.0}
frames_A, prices_A, vols_A = run_simulation(seed, interventions=interventions_A, label="block_trade")
logger.info("Strategy A done: %d frames", len(prices_A))

# ============================================================
# Step 4: Strategy B -- Split 10 (10 smaller sells)
# ============================================================
print("\n" + "=" * 64)
print("Step 4: Strategy B -- Split 10 (10 x -0.5)")
print("=" * 64)

# Distribute across rounds 1-3 (each round has NUM_GEN=16 frames)
# We inject at 10 consecutive rounds (or spread across available rounds)
interventions_B = {}
for i in range(min(10, N_TOTAL_ROUNDS - SHOCK_ROUND)):
    interventions_B[(SHOCK_ROUND + i, 0)] = -0.5
frames_B, prices_B, vols_B = run_simulation(seed, interventions=interventions_B, label="split_10")
logger.info("Strategy B done: %d frames", len(prices_B))

# ============================================================
# Step 5: Strategy C -- TWAP (decreasing shocks)
# ============================================================
print("\n" + "=" * 64)
print("Step 5: Strategy C -- TWAP (decreasing shocks)")
print("=" * 64)

twap_schedule = [-0.7, -0.6, -0.5, -0.4, -0.4, -0.3, -0.3, -0.2, -0.2, -0.1]
interventions_C = {}
for i, shock in enumerate(twap_schedule):
    if SHOCK_ROUND + i < N_TOTAL_ROUNDS:
        interventions_C[(SHOCK_ROUND + i, 0)] = shock
frames_C, prices_C, vols_C = run_simulation(seed, interventions=interventions_C, label="twap")
logger.info("Strategy C done: %d frames", len(prices_C))

# ============================================================
# Step 6: Compute metrics for each strategy
# ============================================================
print("\n" + "=" * 64)
print("Step 6: Computing impact metrics")
print("=" * 64)

def compute_impact_metrics(baseline_px, strategy_px, baseline_vol, strategy_vol, name):
    """Compute impact metrics comparing strategy to baseline.

    Returns dict with:
      - price_impact: max drop in mean price relative to baseline
      - execution_cost: approximated sum(shock_size * price_at_execution)
      - recovery_time: frames until price returns within 10% of pre-shock
      - market_disruption: std increase during/after intervention
    """
    bp = np.array(baseline_px, dtype=float)
    sp = np.array(strategy_px, dtype=float)
    bv = np.array(baseline_vol, dtype=float)
    sv = np.array(strategy_vol, dtype=float)

    T = min(len(bp), len(sp))
    bp, sp = bp[:T], sp[:T]
    bv, sv = bv[:T], sv[:T]

    # Pre-shock level (average of first COND_FRAMES frames)
    pre_shock_px = float(np.mean(bp[:COND_FRAMES]))
    pre_shock_std = float(np.std(bp[:COND_FRAMES])) if COND_FRAMES > 1 else 0.01

    # Shock onset index
    shock_onset = COND_FRAMES + NUM_GEN * SHOCK_ROUND

    # Price impact: max deviation of strategy price from baseline price after shock
    if shock_onset < T:
        post_shock_diff = sp[shock_onset:] - bp[shock_onset:]
        price_impact = float(np.min(post_shock_diff))  # most negative = biggest drop
    else:
        price_impact = 0.0

    # Execution cost: sum of absolute volume shocks * mean price at those frames
    # Approximate from the volume trajectory difference
    if shock_onset < T:
        vol_diff = sv[shock_onset:] - bv[shock_onset:]
        px_during = sp[shock_onset:]
        exec_cost = float(np.sum(np.abs(vol_diff) * np.abs(px_during)))
    else:
        exec_cost = 0.0

    # Recovery time: frames after shock until |strategy_px - baseline_px| < 10% of pre_shock_std
    recovery_threshold = abs(pre_shock_std) * 0.1 + 0.01
    recovery_time = T - shock_onset  # default: never recovered
    if shock_onset < T:
        for fi in range(shock_onset, T):
            if abs(sp[fi] - bp[fi]) < recovery_threshold:
                recovery_time = fi - shock_onset
                break

    # Market disruption: std increase in strategy vs baseline after shock
    if shock_onset < T:
        base_std = float(np.std(bp[shock_onset:]))
        strat_std = float(np.std(sp[shock_onset:]))
        disruption = strat_std - base_std
    else:
        disruption = 0.0

    return {
        "name": name,
        "price_impact": price_impact,
        "execution_cost": exec_cost,
        "recovery_time": recovery_time,
        "market_disruption": disruption,
        "total_shock": sum(abs(v) for (_, _), v in (
            interventions_A if name == "Block Trade" else
            interventions_B if name == "Split 10" else
            interventions_C).items()),
    }

metrics_A = compute_impact_metrics(baseline_prices, prices_A, baseline_vols, vols_A, "Block Trade")
metrics_B = compute_impact_metrics(baseline_prices, prices_B, baseline_vols, vols_B, "Split 10")
metrics_C = compute_impact_metrics(baseline_prices, prices_C, baseline_vols, vols_C, "TWAP")

# ============================================================
# Step 7: Print comparison table
# ============================================================
print("\n" + "=" * 80)
print("  Institutional Order Impact -- Strategy Comparison")
print("=" * 80)

header = f"{'Metric':<24} {'Baseline':>12} {'Block Trade':>12} {'Split 10':>12} {'TWAP':>12}"
print(header)
print("-" * 80)

# Total shock injected
print(f"{'Total Shock Size':<24} {'0.0':>12} {metrics_A['total_shock']:>12.2f} "
      f"{metrics_B['total_shock']:>12.2f} {metrics_C['total_shock']:>12.2f}")

# Price impact
print(f"{'Price Impact (max)':<24} {'0.0':>12} {metrics_A['price_impact']:>12.4f} "
      f"{metrics_B['price_impact']:>12.4f} {metrics_C['price_impact']:>12.4f}")

# Execution cost
print(f"{'Execution Cost':<24} {'0.0':>12} {metrics_A['execution_cost']:>12.4f} "
      f"{metrics_B['execution_cost']:>12.4f} {metrics_C['execution_cost']:>12.4f}")

# Recovery time
print(f"{'Recovery Time (frames)':<24} {'N/A':>12} {metrics_A['recovery_time']:>12d} "
      f"{metrics_B['recovery_time']:>12d} {metrics_C['recovery_time']:>12d}")

# Market disruption
print(f"{'Market Disruption (std)':<24} {'0.0':>12} {metrics_A['market_disruption']:>12.4f} "
      f"{metrics_B['market_disruption']:>12.4f} {metrics_C['market_disruption']:>12.4f}")

print("-" * 80)

# Summary
print("\nKey Findings:")
best_impact = min(metrics_A["price_impact"], metrics_B["price_impact"], metrics_C["price_impact"])
for m in [metrics_A, metrics_B, metrics_C]:
    marker = " <-- LOWEST IMPACT" if m["price_impact"] == max(
        metrics_A["price_impact"], metrics_B["price_impact"], metrics_C["price_impact"]) else ""
    print(f"  {m['name']:<12}: impact={m['price_impact']:+.4f}, "
          f"cost={m['execution_cost']:.4f}, "
          f"recovery={m['recovery_time']} frames, "
          f"disruption={m['market_disruption']:+.4f}{marker}")

best_recovery = min(metrics_A["recovery_time"], metrics_B["recovery_time"], metrics_C["recovery_time"])
for m in [metrics_A, metrics_B, metrics_C]:
    if m["recovery_time"] == best_recovery:
        print(f"\n  Fastest recovery: {m['name']} ({best_recovery} frames)")

print("\n" + "=" * 80)
print("  Impact Simulation Complete")
print("=" * 80)

# Save results
import json
results = {"baseline": len(baseline_prices),
           "strategies": [metrics_A, metrics_B, metrics_C]}
with open(OUT_DIR / "impact_results.json", "w") as f:
    json.dump(results, f, indent=2)
logger.info("Results saved to %s", OUT_DIR / "impact_results.json")
PYEOF
