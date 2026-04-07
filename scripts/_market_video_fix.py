"""Market video visualization — uses vdit_10k_agents (100x100) or falls back to ashare_l3 (4x4)."""
import torch
import numpy as np
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.infer.interactive_sim import InteractiveSimulator

OUT_DIR = Path("outputs/market_video")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Try 10K model first, fall back to ashare_l3
CKPT_10K = Path("outputs/vdit_10k_agents/video_dit_step_10000.pt")
CKPT_L3 = Path("outputs/vdit_ashare_l3/video_dit_step_20000.pt")

if CKPT_10K.exists():
    print("Using 10K agent model (100x100 grid)")
    from agentdiffusion.data.ashare_10k_agents import AShare10KAgentDataset
    dataset = AShare10KAgentDataset(
        "data/external/20240619", total_frames=20, cond_frames=4,
        window_seconds=5.0, max_stocks=30, n_clusters=10000, grid_h=100, grid_w=100)
    d_latent = 6
    grid_h, grid_w = 100, 100
    model = VideoDiT(
        d_latent=d_latent, d_model=256, depth=8, heads=8,
        patch_size=10, num_frames=20, num_cond_frames=4,
        market_cond_dim=32, grid_h=100, grid_w=100,
        causal_temporal=True, alibi_temporal=True).to(device)
    ckpt = torch.load(str(CKPT_10K), map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))
else:
    print("Using A-Share L3 model (4x4 grid)")
    from agentdiffusion.data.ashare_l3_dataset import AShareL3VideoDataset
    dataset = AShareL3VideoDataset(
        "data/external/20240619", total_frames=20, cond_frames=4,
        window_seconds=1.0, grid_shape=(4, 4), max_stocks=50)
    d_latent = dataset[0]["frames"].shape[-1]
    mc_dim = dataset[0]["market_conds"].shape[-1]
    grid_h, grid_w = 4, 4
    model = VideoDiT(
        d_latent=d_latent, d_model=128, depth=6, heads=4,
        patch_size=2, num_frames=20, num_cond_frames=4,
        market_cond_dim=mc_dim, grid_h=4, grid_w=4,
        causal_temporal=True).to(device)
    ckpt = torch.load(str(CKPT_L3), map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))

model.eval()
print(f"Grid: {grid_h}x{grid_w}, d_latent={d_latent}")

scheduler = NoiseScheduler(1000, "cosine").to(device)
sampler = VideoDDIMSampler(model, scheduler, "v_prediction", ddim_steps=20, eta=0.0)
sim = InteractiveSimulator(model, sampler, num_cond=4, num_gen=16, zero_sum_proj=True)

seed = dataset[0]["frames"][:4]
sim.init(seed)

# Generate 50 frames
N_ROUNDS = 5
all_frames = []
mean_series = []
vol_series = []
activity_series = []

for r in range(N_ROUNDS):
    gen = sim.step()  # [16, H, W, d_latent]
    for t in range(gen.shape[0]):
        frame = gen[t].cpu().numpy()  # [H, W, d_latent]
        all_frames.append(frame)
        mean_series.append(frame[:, :, 0].mean())
        vol_series.append(frame[:, :, 0].std())
        activity_series.append((np.abs(frame) > 0.1).any(axis=-1).mean())
    sim.trim_buffer(keep_last=8)
    print(f"  Round {r+1}/{N_ROUNDS}: {len(all_frames)} frames")

# Render frames
print(f"\nRendering {len(all_frames)} frames...")
for i, frame in enumerate(all_frames):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 1]})

    # Main heatmap: signed volume (dim 0)
    signed_vol = frame[:, :, 0]
    vmax = max(abs(signed_vol.min()), abs(signed_vol.max()), 0.01)
    im = axes[0].imshow(signed_vol, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                        interpolation="nearest", aspect="equal")
    axes[0].set_title(f"Agent Grid (Signed Volume) — Frame {i+1}/{len(all_frames)}", fontsize=12)
    axes[0].set_xlabel("Agent Column")
    axes[0].set_ylabel("Agent Row")
    plt.colorbar(im, ax=axes[0], label="Signed Volume (red=sell, green=buy)")

    # Side panel: time series
    ax_ts = axes[1]
    t_range = range(max(0, i - 30), i + 1)
    ax_ts.plot(list(t_range), mean_series[max(0, i-30):i+1], "b-", label="Mean", linewidth=1)
    ax_ts.plot(list(t_range), vol_series[max(0, i-30):i+1], "r--", label="Vol", linewidth=1)
    ax_ts.set_title("Time Series", fontsize=10)
    ax_ts.set_xlabel("Frame")
    ax_ts.legend(fontsize=8)
    ax_ts.set_ylim(-2, 2)

    plt.tight_layout()
    fig.savefig(OUT_DIR / f"frame_{i:04d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

print(f"Saved {len(all_frames)} frames to {OUT_DIR}/")

# Create GIF
print("Creating animated GIF...")
try:
    from PIL import Image
    images = []
    for i in range(len(all_frames)):
        img = Image.open(OUT_DIR / f"frame_{i:04d}.png")
        images.append(img)
    images[0].save(OUT_DIR / "market_evolution.gif", save_all=True,
                   append_images=images[1:], duration=200, loop=0)
    print(f"GIF saved: {OUT_DIR}/market_evolution.gif")
except Exception as e:
    print(f"GIF creation failed: {e}")

print("\n" + "=" * 60)
print(f"  Market Video: {len(all_frames)} frames, grid={grid_h}x{grid_w}")
print(f"  Activity: {np.mean(activity_series)*100:.1f}% cells active on avg")
print("=" * 60)
