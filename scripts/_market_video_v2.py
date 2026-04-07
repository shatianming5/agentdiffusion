"""Market video v2: Multiple visualization styles.

1. Multi-asset 4x4 grid (13 crypto) — each cell is a named asset
2. 10K agent grid with improved rendering (clustered heatmap + annotations)
3. A-Share L3 4x4 with agent type labels
"""
import torch
import numpy as np
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.infer.interactive_sim import InteractiveSimulator


def render_multi_asset(all_frames, out_dir, symbols):
    """Render 4x4 multi-asset grid with asset names and rich annotations."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_series = []
    vol_series = []

    for idx, frame in enumerate(all_frames):
        # frame: [4, 4, d_latent]
        H, W, C = frame.shape
        signed_vol = frame[:, :, 0]  # dim 0 = return/signed volume

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1],
                               hspace=0.3, wspace=0.25)

        # === Main heatmap ===
        ax_main = fig.add_subplot(gs[0, 0])
        vmax = max(np.abs(signed_vol).max(), 0.3)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax_main.imshow(signed_vol, cmap="RdYlGn", norm=norm,
                            interpolation="nearest", aspect="equal")

        # Asset labels on each cell
        for i in range(H):
            for j in range(W):
                sym_idx = i * W + j
                if sym_idx < len(symbols):
                    name = symbols[sym_idx].replace("USDT", "")
                    val = signed_vol[i, j]
                    color = "white" if abs(val) > vmax * 0.5 else "black"
                    ax_main.text(j, i, f"{name}\n{val:+.2f}",
                                ha="center", va="center", fontsize=9,
                                fontweight="bold", color=color)
                else:
                    ax_main.text(j, i, "—", ha="center", va="center",
                                fontsize=9, color="gray")

        ax_main.set_xticks([])
        ax_main.set_yticks([])
        ax_main.set_title(f"Crypto Market State — Frame {idx+1}/{len(all_frames)}",
                         fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax_main, label="Return Signal (green=↑ red=↓)", shrink=0.8)

        # === Volume bars (right panel) ===
        ax_vol = fig.add_subplot(gs[0, 1])
        volumes = frame[:, :, 1].flatten()[:len(symbols)]  # dim 1 = volume
        names = [s.replace("USDT", "") for s in symbols]
        colors_bar = ["#2ecc71" if v > 0 else "#e74c3c" for v in signed_vol.flatten()[:len(symbols)]]
        ax_vol.barh(range(len(names)), volumes[:len(names)], color=colors_bar)
        ax_vol.set_yticks(range(len(names)))
        ax_vol.set_yticklabels(names, fontsize=8)
        ax_vol.set_xlabel("Volume", fontsize=9)
        ax_vol.set_title("Activity", fontsize=11)
        ax_vol.invert_yaxis()

        # === Time series (bottom panel) ===
        ax_ts = fig.add_subplot(gs[1, :])
        mean_series.append(signed_vol.mean())
        vol_series.append(signed_vol.std())
        t_range = range(len(mean_series))
        ax_ts.fill_between(t_range,
                          [m - v for m, v in zip(mean_series, vol_series)],
                          [m + v for m, v in zip(mean_series, vol_series)],
                          alpha=0.2, color="blue")
        ax_ts.plot(t_range, mean_series, "b-", linewidth=2, label="Market Mean")
        ax_ts.plot(t_range, vol_series, "r--", linewidth=1, label="Volatility")
        ax_ts.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax_ts.set_xlabel("Frame", fontsize=10)
        ax_ts.set_ylabel("Signal", fontsize=10)
        ax_ts.legend(fontsize=9, loc="upper left")
        ax_ts.set_title("Market Trajectory", fontsize=11)
        ax_ts.set_xlim(0, max(len(all_frames), len(mean_series)))

        fig.savefig(out_dir / f"frame_{idx:04d}.png", dpi=120, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)

    return len(all_frames)


def render_10k_improved(all_frames, out_dir):
    """Render 100x100 grid with sector grouping and zoom panels."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_series = []
    hotspot_series = []

    for idx, frame in enumerate(all_frames):
        H, W, C = frame.shape
        signed_vol = frame[:, :, 0]

        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1], hspace=0.3, wspace=0.3)

        # === Main overview heatmap ===
        ax_main = fig.add_subplot(gs[0, 0:2])
        vmax = max(np.abs(signed_vol).max(), 0.1)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax_main.imshow(signed_vol, cmap="RdYlGn", norm=norm,
                            interpolation="bilinear", aspect="equal")
        ax_main.set_title(f"10,000 Agent Market — Frame {idx+1}/{len(all_frames)}",
                         fontsize=14, fontweight="bold")
        ax_main.set_xlabel("Agent Cluster (behavioral similarity →)")
        ax_main.set_ylabel("Agent Cluster (behavioral similarity ↓)")
        plt.colorbar(im, ax=ax_main, label="Net Position (green=buy, red=sell)", shrink=0.7)

        # === Zoom panel: hottest 20x20 region ===
        ax_zoom = fig.add_subplot(gs[0, 2])
        # Find most active 20x20 region
        activity = np.abs(signed_vol)
        best_r, best_c, best_val = 0, 0, 0
        step = 10
        for r in range(0, max(H - 20, 1), step):
            for c in range(0, max(W - 20, 1), step):
                val = activity[r:r+20, c:c+20].sum()
                if val > best_val:
                    best_r, best_c, best_val = r, c, val
        zoom = signed_vol[best_r:best_r+20, best_c:best_c+20]
        ax_zoom.imshow(zoom, cmap="RdYlGn", norm=norm, interpolation="nearest")
        ax_zoom.set_title(f"Hotspot [{best_r}:{best_r+20}, {best_c}:{best_c+20}]",
                         fontsize=10)
        # Mark hotspot on main
        rect = plt.Rectangle((best_c-0.5, best_r-0.5), 20, 20,
                             linewidth=2, edgecolor="blue", facecolor="none")
        ax_main.add_patch(rect)

        # === Bottom: stats ===
        ax_stats = fig.add_subplot(gs[1, :])
        mean_series.append(signed_vol.mean())
        n_buy = (signed_vol > 0.1).sum()
        n_sell = (signed_vol < -0.1).sum()
        n_neutral = H * W - n_buy - n_sell
        hotspot_series.append(best_val)

        ax_stats.bar(range(len(mean_series)),
                    [abs(m) for m in mean_series],
                    color=["#2ecc71" if m >= 0 else "#e74c3c" for m in mean_series],
                    alpha=0.7)
        ax_stats.set_xlabel("Frame")
        ax_stats.set_ylabel("|Mean Signal|")
        ax_stats.set_title(f"Buy: {n_buy} | Sell: {n_sell} | Neutral: {n_neutral}",
                          fontsize=11)

        fig.savefig(out_dir / f"frame_{idx:04d}.png", dpi=100, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)

    return len(all_frames)


def make_gif(frame_dir, output_path, duration=300):
    """Create GIF from PNG frames."""
    from PIL import Image
    frame_dir = Path(frame_dir)
    pngs = sorted(frame_dir.glob("frame_*.png"))
    if not pngs:
        print(f"No frames in {frame_dir}")
        return
    images = [Image.open(p) for p in pngs]
    images[0].save(output_path, save_all=True, append_images=images[1:],
                   duration=duration, loop=0)
    print(f"GIF saved: {output_path} ({len(images)} frames)")


# ============================================================
# Style 1: Multi-asset 4x4 (13 crypto)
# ============================================================
print("=" * 60)
print("  Style 1: Multi-Asset Crypto Grid (4x4)")
print("=" * 60)

CKPT_MA = Path("outputs/vdit_multi_asset/video_dit_step_20000.pt")
if CKPT_MA.exists():
    from agentdiffusion.data.binance_multi_asset import BinanceMultiAssetDataset, SYMBOLS

    dataset_ma = BinanceMultiAssetDataset(
        "data/external/binance/aggTrades", total_frames=20, cond_frames=4,
        window_seconds=60.0, max_months=1)

    d_latent = 8
    model_ma = VideoDiT(
        d_latent=d_latent, d_model=256, depth=8, heads=8,
        patch_size=2, num_frames=20, num_cond_frames=4,
        market_cond_dim=32, grid_h=4, grid_w=4,
        causal_temporal=True, alibi_temporal=True).to(device)
    ckpt = torch.load(str(CKPT_MA), map_location=device, weights_only=True)
    model_ma.load_state_dict(ckpt.get("ema", ckpt["model"]))
    model_ma.eval()

    scheduler = NoiseScheduler(1000, "cosine").to(device)
    sampler = VideoDDIMSampler(model_ma, scheduler, "v_prediction", ddim_steps=20, eta=0.0)
    sim = InteractiveSimulator(model_ma, sampler, num_cond=4, num_gen=16, zero_sum_proj=False)

    seed = dataset_ma[0]["frames"][:4]
    sim.init(seed)

    frames_ma = []
    for r in range(5):
        gen = sim.step()
        for t in range(gen.shape[0]):
            frames_ma.append(gen[t].cpu().numpy())
        sim.trim_buffer(keep_last=8)
        print(f"  Round {r+1}/5: {len(frames_ma)} frames")

    n = render_multi_asset(frames_ma, "outputs/market_video_crypto", SYMBOLS)
    make_gif("outputs/market_video_crypto", "outputs/market_video_crypto/crypto_market.gif", 300)
    print(f"  Rendered {n} frames\n")
else:
    print("  [SKIP] No multi-asset model found\n")

# ============================================================
# Style 2: 10K agent grid (improved rendering)
# ============================================================
print("=" * 60)
print("  Style 2: 10K Agent Grid (improved)")
print("=" * 60)

CKPT_10K = Path("outputs/vdit_10k_agents/video_dit_step_10000.pt")
if CKPT_10K.exists():
    from agentdiffusion.data.ashare_10k_agents import AShare10KAgentDataset

    dataset_10k = AShare10KAgentDataset(
        "data/external/20240619", total_frames=20, cond_frames=4,
        window_seconds=5.0, max_stocks=30, n_clusters=10000, grid_h=100, grid_w=100)

    model_10k = VideoDiT(
        d_latent=6, d_model=256, depth=8, heads=8,
        patch_size=10, num_frames=20, num_cond_frames=4,
        market_cond_dim=32, grid_h=100, grid_w=100,
        causal_temporal=True, alibi_temporal=True).to(device)
    ckpt = torch.load(str(CKPT_10K), map_location=device, weights_only=True)
    model_10k.load_state_dict(ckpt.get("ema", ckpt["model"]))
    model_10k.eval()

    scheduler = NoiseScheduler(1000, "cosine").to(device)
    sampler = VideoDDIMSampler(model_10k, scheduler, "v_prediction", ddim_steps=20, eta=0.0)
    sim = InteractiveSimulator(model_10k, sampler, num_cond=4, num_gen=16, zero_sum_proj=True)

    seed = dataset_10k[0]["frames"][:4]
    sim.init(seed)

    frames_10k = []
    for r in range(3):
        gen = sim.step()
        for t in range(gen.shape[0]):
            frames_10k.append(gen[t].cpu().numpy())
        sim.trim_buffer(keep_last=8)
        print(f"  Round {r+1}/3: {len(frames_10k)} frames")

    # Also inject a shock at frame 24
    if len(frames_10k) >= 24:
        shock = torch.zeros(6)
        shock[0] = -3.0
        sim.intervene(frame_idx=-1, delta=shock)
        gen = sim.step()
        for t in range(gen.shape[0]):
            frames_10k.append(gen[t].cpu().numpy())
        print(f"  Post-shock: {len(frames_10k)} frames")

    n = render_10k_improved(frames_10k, "outputs/market_video_10k")
    make_gif("outputs/market_video_10k", "outputs/market_video_10k/agents_10k.gif", 250)
    print(f"  Rendered {n} frames\n")
else:
    print("  [SKIP] No 10K agent model found\n")

# ============================================================
# Style 3: A-Share L3 4x4 with agent type labels
# ============================================================
print("=" * 60)
print("  Style 3: A-Share Agent Types (4x4)")
print("=" * 60)

CKPT_L3 = Path("outputs/vdit_ashare_l3/video_dit_step_20000.pt")
if CKPT_L3.exists():
    from agentdiffusion.data.ashare_l3_dataset import AShareL3VideoDataset

    dataset_l3 = AShareL3VideoDataset(
        "data/external/20240619", total_frames=20, cond_frames=4,
        window_seconds=1.0, grid_shape=(4, 4), max_stocks=50)

    d_latent = dataset_l3[0]["frames"].shape[-1]
    mc_dim = dataset_l3[0]["market_conds"].shape[-1]

    model_l3 = VideoDiT(
        d_latent=d_latent, d_model=128, depth=6, heads=4,
        patch_size=2, num_frames=20, num_cond_frames=4,
        market_cond_dim=mc_dim, grid_h=4, grid_w=4,
        causal_temporal=True).to(device)
    ckpt = torch.load(str(CKPT_L3), map_location=device, weights_only=True)
    model_l3.load_state_dict(ckpt.get("ema", ckpt["model"]))
    model_l3.eval()

    AGENT_LABELS = [
        "Micro\nBuy\nPassive", "Micro\nBuy\nAggr", "Micro\nSell\nPassive", "Micro\nSell\nAggr",
        "Small\nBuy\nPassive", "Small\nBuy\nAggr", "Small\nSell\nPassive", "Small\nSell\nAggr",
        "Med\nBuy\nPassive", "Med\nBuy\nAggr", "Med\nSell\nPassive", "Med\nSell\nAggr",
        "Large\nBuy\nPassive", "Large\nBuy\nAggr", "Large\nSell\nPassive", "Large\nSell\nAggr",
    ]

    scheduler = NoiseScheduler(1000, "cosine").to(device)
    sampler = VideoDDIMSampler(model_l3, scheduler, "v_prediction", ddim_steps=20, eta=0.0)
    sim = InteractiveSimulator(model_l3, sampler, num_cond=4, num_gen=16, zero_sum_proj=True)

    seed = dataset_l3[0]["frames"][:4]
    sim.init(seed)

    out_dir = Path("outputs/market_video_ashare")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_l3 = []
    mean_s, vol_s = [], []

    for r in range(5):
        gen = sim.step()
        for t in range(gen.shape[0]):
            frame = gen[t].cpu().numpy()
            frames_l3.append(frame)
            mean_s.append(frame[:,:,0].mean())
            vol_s.append(frame[:,:,0].std())
        sim.trim_buffer(keep_last=8)

    for idx, frame in enumerate(frames_l3):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [2, 1]})

        sv = frame[:, :, 0]
        vmax = max(np.abs(sv).max(), 0.3)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = axes[0].imshow(sv, cmap="RdYlGn", norm=norm, interpolation="nearest")

        for i in range(4):
            for j in range(4):
                label = AGENT_LABELS[i * 4 + j]
                val = sv[i, j]
                color = "white" if abs(val) > vmax * 0.4 else "black"
                axes[0].text(j, i, f"{label}\n{val:+.2f}",
                            ha="center", va="center", fontsize=7, color=color)

        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_title(f"A-Share Agent Types — Frame {idx+1}", fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=axes[0], label="Net Position", shrink=0.8)

        # Time series
        axes[1].fill_between(range(idx+1),
                            [m-v for m, v in zip(mean_s[:idx+1], vol_s[:idx+1])],
                            [m+v for m, v in zip(mean_s[:idx+1], vol_s[:idx+1])],
                            alpha=0.2, color="steelblue")
        axes[1].plot(range(idx+1), mean_s[:idx+1], "b-", linewidth=2, label="Mean")
        axes[1].plot(range(idx+1), vol_s[:idx+1], "r--", linewidth=1, label="Vol")
        axes[1].axhline(0, color="gray", linewidth=0.5)
        axes[1].legend(fontsize=9)
        axes[1].set_title("Trajectory", fontsize=11)
        axes[1].set_xlim(0, len(frames_l3))

        fig.savefig(out_dir / f"frame_{idx:04d}.png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    make_gif(out_dir, out_dir / "ashare_agents.gif", 250)
    print(f"  Rendered {len(frames_l3)} frames\n")
else:
    print("  [SKIP] No A-Share L3 model found\n")

print("=" * 60)
print("  All styles complete!")
print("=" * 60)
