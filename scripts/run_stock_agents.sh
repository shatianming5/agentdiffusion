#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "============================================================"
echo "  Stock-Agent Video DiT: Each A-Share Stock = One Agent"
echo "  Grid: 32x32 (up to 1024 stocks), 1-min windows"
echo "============================================================"

# ============================================================
# Step 1: Train Video DiT on stock-agent grid
# ============================================================
echo "=== Step 1: Training ==="
.venv/bin/python3 -u << 'PYEOF'
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, logging, json
from pathlib import Path
from tqdm import tqdm
from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.data.ashare_stock_agents import (
    AShareStockAgentDataset, D_STATE, SECTOR_ORDER, classify_sector,
)
from agentdiffusion.infer.interactive_sim import InteractiveSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("outputs/vdit_stock_agents")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRID_H, GRID_W = 32, 32
MAX_STOCKS = 1000

# --- Dataset ---
dataset = AShareStockAgentDataset(
    "data/external/20240619",
    total_frames=20, cond_frames=4,
    window_seconds=60.0,    # 1-minute windows
    grid_h=GRID_H, grid_w=GRID_W,
    max_stocks=MAX_STOCKS,
)
logger.info("Dataset: %d sequences", len(dataset))

if len(dataset) == 0:
    logger.error("No data! Check data/external/20240619/")
    import sys; sys.exit(1)

loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True)

d_latent = D_STATE  # 6
logger.info("grid=(%d,%d), d_latent=%d, patch_size=4", GRID_H, GRID_W, d_latent)

# --- Model: 32x32 grid, patch_size=4 -> 8x8=64 spatial patches ---
model = VideoDiT(
    d_latent=d_latent, d_model=256, depth=8, heads=8,
    patch_size=4, num_frames=20, num_cond_frames=4,
    mlp_ratio=4.0, market_cond_dim=32,
    grid_h=GRID_H, grid_w=GRID_W,
    causal_temporal=True, alibi_temporal=True,
).to(device)
n_params = sum(p.numel() for p in model.parameters())
logger.info("Model params: %.1fM", n_params / 1e6)

scheduler = NoiseScheduler(1000, "cosine").to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
TOTAL_STEPS = 10000
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)

# EMA
ema_state = {k: v.clone() for k, v in model.state_dict().items()}
ema_decay = 0.999
def update_ema():
    with torch.no_grad():
        for k, v in model.state_dict().items():
            ema_state[k].lerp_(v, 1 - ema_decay)

# --- Training loop ---
K = 4
step = 0
pbar = tqdm(total=TOTAL_STEPS, desc="Stock-Agent DiT")

while step < TOTAL_STEPS:
    for batch in loader:
        if step >= TOTAL_STEPS:
            break
        frames = batch["frames"].to(device)  # [B, T, 32, 32, 6]
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
        lr_scheduler.step()
        update_ema()

        step += 1
        pbar.update(1)
        if step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{lr_scheduler.get_last_lr()[0]:.2e}", step=step)
        if step % 2500 == 0:
            torch.save({"model": model.state_dict(), "ema": ema_state, "step": step},
                       OUT_DIR / f"video_dit_step_{step}.pt")

pbar.close()
torch.save({"model": model.state_dict(), "ema": ema_state, "step": step},
           OUT_DIR / f"video_dit_step_{step}.pt")
logger.info("Training done. Saved to %s", OUT_DIR)
PYEOF

# ============================================================
# Step 2: Evaluation — sector analysis + shock propagation
# ============================================================
echo "=== Step 2: Sector Evaluation ==="
.venv/bin/python3 -u << 'PYEOF'
import torch, numpy as np, json, logging
from pathlib import Path
from collections import defaultdict
from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.data.ashare_stock_agents import (
    AShareStockAgentDataset, D_STATE, SECTOR_ORDER, classify_sector,
)
from agentdiffusion.infer.interactive_sim import InteractiveSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("outputs/vdit_stock_agents")

GRID_H, GRID_W = 32, 32

# --- Reload dataset (for grid positions and sector info) ---
dataset = AShareStockAgentDataset(
    "data/external/20240619",
    total_frames=20, cond_frames=4,
    window_seconds=60.0,
    grid_h=GRID_H, grid_w=GRID_W,
    max_stocks=1000,
)

if len(dataset) == 0:
    logger.error("No data!")
    import sys; sys.exit(1)

# --- Load trained model ---
ckpt_path = sorted(OUT_DIR.glob("video_dit_step_*.pt"))[-1]
model = VideoDiT(
    d_latent=D_STATE, d_model=256, depth=8, heads=8,
    patch_size=4, num_frames=20, num_cond_frames=4,
    mlp_ratio=4.0, market_cond_dim=32,
    grid_h=GRID_H, grid_w=GRID_W,
    causal_temporal=True, alibi_temporal=True,
).to(device)
ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
model.load_state_dict(ckpt.get("ema", ckpt["model"]))
model.eval()
logger.info("Loaded model from %s", ckpt_path)

scheduler = NoiseScheduler(1000, "cosine").to(device)
sampler = VideoDDIMSampler(model, scheduler, "v_prediction", ddim_steps=20, eta=0.0)

# --- Build sector masks ---
sector_masks = {}
for sec in SECTOR_ORDER:
    mask = dataset.get_sector_mask(sec)
    if mask.any():
        sector_masks[sec] = mask

print("\n" + "=" * 72)
print("  Stock-Agent Evaluation: Sector Analysis")
print("=" * 72)
print(f"  Grid: {GRID_H}x{GRID_W}, Stocks: {len(dataset.stock_codes)}")
print(f"  Sectors: {list(sector_masks.keys())}")
print(f"  Sequences: {len(dataset)}")

# --- Generate 5 rounds of 16 frames ---
sim = InteractiveSimulator(model, sampler, num_cond=4, num_gen=16, zero_sum_proj=False)
seed = dataset[0]["frames"][:4]
sim.init(seed)

all_gen = []
for rd in range(5):
    gen = sim.step()
    all_gen.append(gen.cpu())
    sim.trim_buffer(keep_last=8)
    m = gen.mean().item()
    s = gen.std().item()
    print(f"  Round {rd+1}: mean={m:.4f}, std={s:.4f}, "
          f"active_cells={(gen.abs() > 0.1).float().mean().item()*100:.1f}%")

all_gen = torch.cat(all_gen, dim=0)  # [80, 32, 32, 6]
T_gen = all_gen.shape[0]

# --- Per-sector statistics ---
print("\n--- Per-Sector Statistics (generated data) ---")
print(f"{'Sector':<15} {'Count':>6} {'Mean Ret':>10} {'Vol':>10} {'Spread':>10} {'Imbalance':>10}")
print("-" * 65)

sector_returns = {}
for sec, mask in sector_masks.items():
    n_stocks = mask.sum()
    # Extract return series for this sector: all_gen[:, mask, 0]
    rows, cols = np.where(mask)
    returns = []
    for r, c in zip(rows, cols):
        returns.append(all_gen[:, r, c, 0].numpy())
    if not returns:
        continue
    returns = np.stack(returns, axis=1)  # [T_gen, n_stocks_in_sector]
    sector_returns[sec] = returns.mean(axis=1)  # sector-average return

    mean_ret = returns.mean()
    vol = returns.std()
    spread = all_gen[:, rows, cols, 2].numpy().mean() if len(rows) > 0 else 0
    imbal = all_gen[:, rows, cols, 3].numpy().mean() if len(rows) > 0 else 0
    print(f"  {sec:<13} {n_stocks:>6d} {mean_ret:>+10.4f} {vol:>10.4f} "
          f"{spread:>10.4f} {imbal:>10.4f}")

# --- Cross-sector correlation matrix ---
print("\n--- Cross-Sector Correlation Matrix ---")
active_sectors = [s for s in SECTOR_ORDER if s in sector_returns]
n_sec = len(active_sectors)

if n_sec > 1:
    corr_matrix = np.zeros((n_sec, n_sec))
    for i, si in enumerate(active_sectors):
        for j, sj in enumerate(active_sectors):
            ri = sector_returns[si]
            rj = sector_returns[sj]
            if len(ri) > 1 and len(rj) > 1:
                corr_matrix[i, j] = np.corrcoef(ri, rj)[0, 1]
            else:
                corr_matrix[i, j] = float("nan")

    # Print header
    header = f"{'':>13}"
    for sec in active_sectors:
        header += f" {sec[:8]:>8}"
    print(header)

    for i, si in enumerate(active_sectors):
        row = f"  {si[:11]:<11}"
        for j in range(n_sec):
            v = corr_matrix[i, j]
            row += f" {v:>+8.3f}"
        print(row)

    # Save correlation data
    corr_data = {
        "sectors": active_sectors,
        "correlation_matrix": corr_matrix.tolist(),
    }
    with open(OUT_DIR / "sector_correlation.json", "w") as f:
        json.dump(corr_data, f, indent=2)
    print(f"\n  Saved correlation data to {OUT_DIR / 'sector_correlation.json'}")

# --- Shock propagation: crash banking/SH_MAIN sector ---
print("\n" + "=" * 72)
print("  Shock Propagation Test: SH_MAIN (banking) Crash")
print("=" * 72)

# Pick the sector with the most stocks (likely SH_MAIN or SZ_MAIN)
shock_sector = "SH_MAIN" if "SH_MAIN" in sector_masks else active_sectors[0]
shock_mask = sector_masks[shock_sector]
print(f"  Shocking sector: {shock_sector} ({shock_mask.sum()} stocks)")
print(f"  Setting all {shock_sector} cells to zero (crash)")

# Baseline: normal generation from same seed
sim_base = InteractiveSimulator(model, sampler, num_cond=4, num_gen=16, zero_sum_proj=False)
sim_base.init(dataset[0]["frames"][:4])
base_gen = sim_base.step().cpu()

# Shocked: zero out the banking sector cells in the last cond frame, then generate
sim_shock = InteractiveSimulator(model, sampler, num_cond=4, num_gen=16, zero_sum_proj=False)
sim_shock.init(dataset[0]["frames"][:4])

# Inject shock: zero out all features for the target sector
shock_rows, shock_cols = np.where(shock_mask)
for r, c in zip(shock_rows.tolist(), shock_cols.tolist()):
    sim_shock.buffer[0, -1, r, c, :] = 0.0
    sim_shock.buffer[0, -2, r, c, :] = 0.0

shock_gen = sim_shock.step().cpu()

# Compare baseline vs shocked per-sector
print(f"\n{'Sector':<15} {'Base MeanRet':>12} {'Shock MeanRet':>13} {'Delta':>10} {'% Change':>10}")
print("-" * 62)

shock_results = {}
for sec, mask in sector_masks.items():
    rows, cols = np.where(mask)
    if len(rows) == 0:
        continue
    base_ret = base_gen[:, rows, cols, 0].numpy().mean()
    shock_ret = shock_gen[:, rows, cols, 0].numpy().mean()
    delta = shock_ret - base_ret
    pct = (delta / (abs(base_ret) + 1e-8)) * 100
    is_shocked = " (SHOCKED)" if sec == shock_sector else ""
    print(f"  {sec:<13} {base_ret:>+12.4f} {shock_ret:>+13.4f} "
          f"{delta:>+10.4f} {pct:>+9.1f}%{is_shocked}")
    shock_results[sec] = {
        "baseline_mean_return": float(base_ret),
        "shocked_mean_return": float(shock_ret),
        "delta": float(delta),
        "pct_change": float(pct),
    }

# Analyse propagation: which non-shocked sectors are most affected?
print("\n--- Shock Propagation Ranking (non-shocked sectors) ---")
propagation = []
for sec, data in shock_results.items():
    if sec != shock_sector:
        propagation.append((abs(data["delta"]), sec, data["delta"]))
propagation.sort(key=lambda x: -x[0])

for rank, (abs_d, sec, delta) in enumerate(propagation, 1):
    direction = "contagion" if delta < 0 else "flight-to-quality"
    print(f"  {rank}. {sec:<13}: delta={delta:+.4f} ({direction})")

# Save shock results
shock_output = {
    "shocked_sector": shock_sector,
    "n_shocked_stocks": int(shock_mask.sum()),
    "per_sector": shock_results,
    "propagation_ranking": [
        {"sector": sec, "abs_delta": float(ad), "delta": float(d)}
        for ad, sec, d in propagation
    ],
}
with open(OUT_DIR / "shock_propagation.json", "w") as f:
    json.dump(shock_output, f, indent=2)
print(f"\n  Saved shock results to {OUT_DIR / 'shock_propagation.json'}")

print("\n" + "=" * 72)
print("  Stock-Agent Experiment Complete")
print("=" * 72)
PYEOF

echo "=== ALL DONE ==="
