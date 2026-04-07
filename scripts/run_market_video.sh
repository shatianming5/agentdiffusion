#!/bin/bash
# ============================================================================
#  Market Video: Render agent grid as visual heatmap frames + animated GIF
#  -------------------------------------------------------------------------
#  1. Load trained 10K model or train a new K=1000 model
#  2. Generate 50 frames via sliding window
#  3. For each frame: matplotlib heatmap (signed volume), side panel, time series
#  4. Save as outputs/market_video/frame_NNNN.png
#  5. Create animated GIF with PIL/imageio
#  6. Print summary stats per frame
# ============================================================================
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "============================================================"
echo "  Market Video: Agent Grid Heatmap Visualization"
echo "  100x100 grid -> heatmap PNGs + animated GIF"
echo "============================================================"

.venv/bin/python3 -u << 'PYEOF'
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, logging, sys, os
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = Path("outputs/market_video")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Configuration
# ============================================================
K_CLUSTERS   = 1000
GRID_H, GRID_W = 32, 32
PATCH_SIZE   = 4
D_LATENT     = 6
TOTAL_FRAMES = 20
COND_FRAMES  = 4
NUM_GEN      = TOTAL_FRAMES - COND_FRAMES
NUM_VIS_FRAMES = 50  # total frames to generate for visualization
TRAIN_STEPS  = 5000
BATCH_SIZE   = 8

# ============================================================
# Step 1: Load or train model
# ============================================================
from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.data.ashare_10k_agents import (
    AShare10KAgentDataset, D_STATE,
    load_orders_multi_stock, extract_order_features, cluster_agents,
    arrange_grid_2d, build_agent_grid,
)
from agentdiffusion.infer.interactive_sim import InteractiveSimulator

# Check for existing trained model
ckpt_paths = [
    Path("outputs/vdit_10k_agents/video_dit_step_10000.pt"),
    Path("outputs/vdit_10k_agents/video_dit_step_5000.pt"),
]
ckpt_path = None
for cp in ckpt_paths:
    if cp.exists():
        ckpt_path = cp
        break

# Build model architecture
model = VideoDiT(
    d_latent=D_LATENT, d_model=256, depth=8, heads=8,
    patch_size=PATCH_SIZE, num_frames=TOTAL_FRAMES, num_cond_frames=COND_FRAMES,
    mlp_ratio=4.0, market_cond_dim=32,
    grid_h=GRID_H, grid_w=GRID_W,
    causal_temporal=True, alibi_temporal=True,
).to(device)

scheduler = NoiseScheduler(1000, "cosine").to(device)

# Load training data for seed frames regardless
DATA_DIR = "data/external/20240619"
if not Path(DATA_DIR).exists():
    # Try other available data dirs
    ext_dir = Path("data/external")
    if ext_dir.exists():
        candidates = sorted([d for d in ext_dir.iterdir() if d.is_dir()])
        if candidates:
            DATA_DIR = str(candidates[0])
            logger.info("Using alternative data dir: %s", DATA_DIR)
        else:
            logger.error("No data directories found in data/external/")
            sys.exit(1)

dataset = AShare10KAgentDataset(
    DATA_DIR, total_frames=TOTAL_FRAMES, cond_frames=COND_FRAMES,
    window_seconds=5.0, max_stocks=30,
    n_clusters=K_CLUSTERS, grid_h=GRID_H, grid_w=GRID_W,
)

if len(dataset) == 0:
    logger.error("No data loaded!")
    sys.exit(1)

if ckpt_path is not None and ckpt_path.exists():
    logger.info("Loading existing checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Try EMA first, fall back to model
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])
    logger.info("Loaded checkpoint at step %d", ckpt.get("step", -1))
else:
    logger.info("No existing checkpoint found. Training new model (%d steps)...", TRAIN_STEPS)
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
    pbar = tqdm(total=TRAIN_STEPS, desc="Training for video")
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
               OUT_DIR / "video_model.pt")
    logger.info("Training done.")

model.eval()

# ============================================================
# Step 2: Generate frames via sliding window
# ============================================================
print("\n" + "=" * 64)
print("Step 2: Generating %d frames via sliding window" % NUM_VIS_FRAMES)
print("=" * 64)

sampler = VideoDDIMSampler(model, scheduler, "v_prediction", ddim_steps=20, eta=0.0)
sim = InteractiveSimulator(model, sampler, num_cond=COND_FRAMES, num_gen=NUM_GEN, zero_sum_proj=True)

# Seed with first 4 frames from dataset
seed = dataset[0]["frames"][:COND_FRAMES]
sim.init(seed)

all_frames = [seed.numpy()]  # start with seed frames
n_rounds = max(1, (NUM_VIS_FRAMES + NUM_GEN - 1) // NUM_GEN)

for r in tqdm(range(n_rounds), desc="Generating"):
    gen = sim.step()
    all_frames.append(gen.cpu().numpy())
    sim.trim_buffer(keep_last=COND_FRAMES * 2)

all_frames = np.concatenate(all_frames, axis=0)[:NUM_VIS_FRAMES]
logger.info("Generated %d total frames, shape per frame: %s", all_frames.shape[0], all_frames.shape[1:])

# ============================================================
# Step 3: Render each frame as matplotlib heatmap
# ============================================================
print("\n" + "=" * 64)
print("Step 3: Rendering %d frames as heatmap PNGs" % NUM_VIS_FRAMES)
print("=" * 64)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Accumulate time series for bottom panel
ts_mean_sv     = []  # mean signed volume
ts_volatility  = []  # spread/volatility proxy
ts_activity    = []  # mean activity

for f_idx in tqdm(range(len(all_frames)), desc="Rendering"):
    frame = all_frames[f_idx]  # [H, W, D]
    sv  = frame[:, :, 0]  # signed volume: red=sell(negative), green=buy(positive)
    cnt = frame[:, :, 1]  # order count / activity
    px  = frame[:, :, 2]  # avg price
    br  = frame[:, :, 5]  # buy ratio

    ts_mean_sv.append(float(np.mean(sv)))
    ts_volatility.append(float(np.std(px)))
    ts_activity.append(float(np.mean(np.abs(cnt))))

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[3, 1],
                          hspace=0.25, wspace=0.15)

    # --- Main heatmap: signed volume ---
    ax_main = fig.add_subplot(gs[0, 0])
    vmax = max(abs(np.percentile(sv, 5)), abs(np.percentile(sv, 95)), 0.5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax_main.imshow(sv, cmap="RdYlGn", norm=norm, aspect="auto", interpolation="nearest")
    ax_main.set_title(f"Frame {f_idx:04d} -- Signed Volume (Red=Sell, Green=Buy)", fontsize=12)
    ax_main.set_xlabel("Agent Column")
    ax_main.set_ylabel("Agent Row")
    plt.colorbar(im, ax=ax_main, label="Signed Volume", shrink=0.8)

    # --- Side panel: mean activity per row (sector view) ---
    ax_side = fig.add_subplot(gs[0, 1])
    row_activity = np.mean(np.abs(cnt), axis=1)  # [H]
    ax_side.barh(range(len(row_activity)), row_activity, color="steelblue", height=1.0)
    ax_side.set_ylim(len(row_activity) - 0.5, -0.5)
    ax_side.set_title("Activity\n(per row)", fontsize=10)
    ax_side.set_xlabel("Mean |count|")
    ax_side.tick_params(axis="y", labelleft=False)

    # --- Bottom panel: time series so far ---
    ax_ts = fig.add_subplot(gs[1, :])
    x_ts = np.arange(len(ts_mean_sv))
    ax_ts.plot(x_ts, ts_mean_sv, "g-", label="Mean Signed Vol", linewidth=1.5)
    ax_ts2 = ax_ts.twinx()
    ax_ts2.plot(x_ts, ts_volatility, "r--", label="Volatility (px std)", linewidth=1.0)
    ax_ts2.plot(x_ts, ts_activity, "b:", label="Activity", linewidth=1.0)
    ax_ts.axvline(x=f_idx, color="black", linestyle="-", alpha=0.5, linewidth=0.8)
    ax_ts.set_xlabel("Frame")
    ax_ts.set_ylabel("Signed Volume", color="green")
    ax_ts2.set_ylabel("Volatility / Activity", color="red")
    ax_ts.set_title("Time Series", fontsize=10)
    # Combine legends
    lines1, labels1 = ax_ts.get_legend_handles_labels()
    lines2, labels2 = ax_ts2.get_legend_handles_labels()
    ax_ts.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    fig.savefig(OUT_DIR / f"frame_{f_idx:04d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

logger.info("Saved %d frame PNGs to %s", len(all_frames), OUT_DIR)

# ============================================================
# Step 4: Create animated GIF
# ============================================================
print("\n" + "=" * 64)
print("Step 4: Creating animated GIF")
print("=" * 64)

try:
    from PIL import Image
    png_files = sorted(OUT_DIR.glob("frame_*.png"))
    if png_files:
        imgs = [Image.open(str(p)) for p in png_files]
        gif_path = OUT_DIR / "market_simulation.gif"
        imgs[0].save(
            gif_path, save_all=True, append_images=imgs[1:],
            duration=200, loop=0, optimize=True,
        )
        logger.info("GIF saved: %s (%.1f MB)", gif_path, gif_path.stat().st_size / 1e6)
    else:
        logger.warning("No PNG files found for GIF creation")
except ImportError:
    logger.warning("PIL not available; trying imageio for GIF")
    try:
        import imageio
        png_files = sorted(OUT_DIR.glob("frame_*.png"))
        frames_for_gif = [imageio.imread(str(p)) for p in png_files]
        gif_path = OUT_DIR / "market_simulation.gif"
        imageio.mimsave(str(gif_path), frames_for_gif, duration=0.2, loop=0)
        logger.info("GIF saved: %s", gif_path)
    except ImportError:
        logger.warning("Neither PIL nor imageio available; skipping GIF creation")

# ============================================================
# Step 5: Print summary stats per frame
# ============================================================
print("\n" + "=" * 64)
print("Step 5: Frame-by-Frame Summary Stats")
print("=" * 64)

print(f"{'Frame':>6} {'MeanSV':>10} {'StdSV':>10} {'Activity':>10} {'Volatility':>10} {'Active%':>10}")
print("-" * 62)
for f_idx in range(len(all_frames)):
    frame = all_frames[f_idx]
    sv  = frame[:, :, 0]
    cnt = frame[:, :, 1]
    px  = frame[:, :, 2]
    active_pct = float((np.abs(frame).sum(axis=-1) > 0.1).mean()) * 100
    print(f"{f_idx:>6d} {np.mean(sv):>10.4f} {np.std(sv):>10.4f} "
          f"{np.mean(np.abs(cnt)):>10.4f} {np.std(px):>10.4f} {active_pct:>9.1f}%")

print("\n" + "=" * 64)
print(f"  Output: {OUT_DIR}/frame_NNNN.png + market_simulation.gif")
print(f"  Total frames rendered: {len(all_frames)}")
print("=" * 64)
PYEOF
