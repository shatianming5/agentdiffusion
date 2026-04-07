#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "============================================================"
echo "  Ablation Study: Video DiT on A-Share L3 (5 configs)"
echo "============================================================"

DATA_DIR="data/external/20240619"

if [ ! -d "$DATA_DIR" ]; then
    # Try extracting from 7z if available
    if [ -f "${DATA_DIR}.7z" ]; then
        echo "=== Extracting ${DATA_DIR}.7z ==="
        7z x "${DATA_DIR}.7z" -o"data/external/" || p7zip -d "${DATA_DIR}.7z"
    else
        echo "[ERROR] Data directory $DATA_DIR not found and no .7z archive available."
        echo "Please place A-Share L3 data at $DATA_DIR before running."
        exit 1
    fi
fi

echo "Data dir: $DATA_DIR"
echo "Stocks:   $(ls "$DATA_DIR" | wc -l)"
echo ""

# ============================================================
# Run all 5 ablation configs sequentially
# ============================================================
.venv/bin/python3 -u << 'PYEOF'
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import logging, time, json
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict

from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.models.order_decoder import AgentToOrderDecoder
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.data.ashare_l3_dataset import AShareL3VideoDataset
from agentdiffusion.infer.interactive_sim import InteractiveSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

# ----------------------------------------------------------------
# Shared hyper-parameters
# ----------------------------------------------------------------
DATA_DIR       = "data/external/20240619"
TOTAL_FRAMES   = 20
COND_FRAMES    = 4
GRID_H, GRID_W = 4, 4
D_LATENT       = 6
D_MODEL        = 128
DEPTH          = 6
HEADS          = 4
BATCH_SIZE     = 32
TOTAL_STEPS    = 5000
MARKET_COND_DIM = 8
LAMBDA_ORDER   = 0.1
NUM_GEN_SEQS   = 20    # sequences to generate for stylized facts
STAB_ROUNDS    = 50    # stability test rounds
NUM_GEN_FRAMES = TOTAL_FRAMES - COND_FRAMES  # 16

# ----------------------------------------------------------------
# 5 ablation configurations
# ----------------------------------------------------------------
CONFIGS = OrderedDict({
    "full_model": {
        "causal_temporal": True,
        "alibi_temporal": True,
        "market_cond_dim": MARKET_COND_DIM,
        "zero_market_cond": False,
        "recalibrate_every": 5,
        "recalibrate_strength": 0.3,
        "desc": "Full model (causal+ALiBi+market_cond+SDEdit)",
    },
    "no_causal": {
        "causal_temporal": False,
        "alibi_temporal": False,
        "market_cond_dim": MARKET_COND_DIM,
        "zero_market_cond": False,
        "recalibrate_every": 5,
        "recalibrate_strength": 0.3,
        "desc": "No causal (causal=False, alibi=False)",
    },
    "no_alibi": {
        "causal_temporal": True,
        "alibi_temporal": False,
        "market_cond_dim": MARKET_COND_DIM,
        "zero_market_cond": False,
        "recalibrate_every": 5,
        "recalibrate_strength": 0.3,
        "desc": "No ALiBi (causal=True, alibi=False)",
    },
    "no_market_cond": {
        "causal_temporal": True,
        "alibi_temporal": True,
        "market_cond_dim": MARKET_COND_DIM,
        "zero_market_cond": True,
        "recalibrate_every": 5,
        "recalibrate_strength": 0.3,
        "desc": "No market conditioning (zeros)",
    },
    "no_sdedit": {
        "causal_temporal": True,
        "alibi_temporal": True,
        "market_cond_dim": MARKET_COND_DIM,
        "zero_market_cond": False,
        "recalibrate_every": 0,
        "recalibrate_strength": 0.0,
        "desc": "No SDEdit (recalibrate_every=0)",
    },
})


# ----------------------------------------------------------------
# Load dataset once (shared across all configs)
# ----------------------------------------------------------------
logger.info("Loading A-Share L3 dataset ...")
dataset = AShareL3VideoDataset(
    DATA_DIR,
    total_frames=TOTAL_FRAMES,
    cond_frames=COND_FRAMES,
    window_seconds=1.0,
    grid_shape=(GRID_H, GRID_W),
    max_stocks=50,
)
logger.info(f"Dataset: {len(dataset)} sequences")
if len(dataset) == 0:
    logger.error("No sequences found! Check data directory.")
    import sys; sys.exit(1)

loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, drop_last=True,
)


# ================================================================
# Helper: compute stylized facts (kurtosis, ACF)
# ================================================================
def compute_acf(x: np.ndarray, lag: int) -> float:
    """Compute autocorrelation at given lag."""
    n = len(x)
    if n <= lag:
        return 0.0
    mean = x.mean()
    var = np.var(x)
    if var < 1e-12:
        return 0.0
    c = np.sum((x[:n - lag] - mean) * (x[lag:] - mean)) / (n * var)
    return float(c)


def compute_stylized_facts(
    model: VideoDiT,
    sampler: VideoDDIMSampler,
    scheduler: NoiseScheduler,
    dataset: AShareL3VideoDataset,
    num_seqs: int,
    cfg: dict,
) -> dict:
    """Generate sequences and compute kurtosis, ACF(1), ACF(5)."""
    model.eval()
    all_returns = []

    with torch.no_grad():
        for i in range(min(num_seqs, len(dataset))):
            sample = dataset[i]
            frames = sample["frames"].unsqueeze(0).to(device)  # [1, T, H, W, C]
            mconds = sample["market_conds"].unsqueeze(0).to(device)  # [1, T, 8]

            K = COND_FRAMES
            x_cond = frames[:, :K]  # [1, K, H, W, C]

            # Market conditioning
            if cfg["zero_market_cond"]:
                market_cond = torch.zeros(1, MARKET_COND_DIM, device=device)
            else:
                market_cond = mconds.mean(dim=1)  # [1, 8]

            gen_shape = (1, NUM_GEN_FRAMES, GRID_H, GRID_W, D_LATENT)
            generated = sampler.sample(
                x_cond, gen_shape, market_cond=market_cond,
                device=device, zero_sum_proj=True,
            )  # [1, N, H, W, C]

            # Extract net_position (dim 0) trajectory: mean over grid
            pos = generated[0, :, :, :, 0].mean(dim=(-2, -1)).cpu().numpy()  # [N]
            # Compute returns (first differences)
            rets = np.diff(pos)
            if len(rets) > 0:
                all_returns.append(rets)

    if len(all_returns) == 0:
        return {"kurtosis": float("nan"), "acf1": float("nan"), "acf5": float("nan")}

    all_rets = np.concatenate(all_returns)
    kurt = float(np.mean((all_rets - all_rets.mean()) ** 4) /
                 (np.var(all_rets) ** 2 + 1e-12)) - 3.0  # excess kurtosis
    acf1 = compute_acf(all_rets, 1)
    acf5 = compute_acf(all_rets, 5)

    return {"kurtosis": kurt, "acf1": acf1, "acf5": acf5}


# ================================================================
# Helper: stability test (mean drift, std ratio over N rounds)
# ================================================================
def run_stability_test(
    model: VideoDiT,
    sampler: VideoDDIMSampler,
    scheduler: NoiseScheduler,
    dataset: AShareL3VideoDataset,
    n_rounds: int,
    cfg: dict,
) -> dict:
    """Run interactive simulation for n_rounds, report drift & std ratio."""
    model.eval()

    sim = InteractiveSimulator(
        model, sampler,
        num_cond=COND_FRAMES,
        num_gen=NUM_GEN_FRAMES,
        zero_sum_proj=True,
        scheduler=scheduler,
        recalibrate_every=cfg["recalibrate_every"],
        recalibrate_strength=cfg["recalibrate_strength"],
    )

    # Use first dataset sample as seed
    seed = dataset[0]["frames"][:COND_FRAMES]  # [K, H, W, C]
    sim.init(seed)

    # Record trajectory statistics
    round_means = []
    round_stds = []

    for r in range(n_rounds):
        gen = sim.step()  # [N, H, W, C]
        pos = gen[:, :, :, 0]  # [N, H, W] net_position
        round_means.append(float(pos.mean().cpu()))
        round_stds.append(float(pos.std().cpu()))
        sim.trim_buffer(keep_last=COND_FRAMES * 2)

    round_means = np.array(round_means)
    round_stds = np.array(round_stds)

    # Mean drift: absolute change in mean from first 5 to last 5 rounds
    early_mean = round_means[:5].mean()
    late_mean = round_means[-5:].mean()
    mean_drift = float(abs(late_mean - early_mean))

    # Std ratio: std of last 5 rounds / std of first 5 rounds
    early_std = round_stds[:5].mean()
    late_std = round_stds[-5:].mean()
    std_ratio = float(late_std / (early_std + 1e-8))

    return {
        "mean_drift": mean_drift,
        "std_ratio": std_ratio,
        "final_mean": float(round_means[-1]),
        "final_std": float(round_stds[-1]),
    }


# ================================================================
# Main ablation loop
# ================================================================
all_results = {}

for config_name, cfg in CONFIGS.items():
    logger.info("=" * 60)
    logger.info(f"  CONFIG: {config_name}")
    logger.info(f"  {cfg['desc']}")
    logger.info("=" * 60)

    out_dir = Path(f"outputs/ablation/{config_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build model ----
    model = VideoDiT(
        d_latent=D_LATENT,
        d_model=D_MODEL,
        depth=DEPTH,
        heads=HEADS,
        patch_size=2,
        num_frames=TOTAL_FRAMES,
        num_cond_frames=COND_FRAMES,
        mlp_ratio=4.0,
        market_cond_dim=cfg["market_cond_dim"],
        grid_h=GRID_H,
        grid_w=GRID_W,
        causal_temporal=cfg["causal_temporal"],
        alibi_temporal=cfg["alibi_temporal"],
    ).to(device)
    logger.info(f"DiT params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ---- Order decoder (shared architecture) ----
    order_decoder = AgentToOrderDecoder(
        d_state=D_LATENT, d_model=D_MODEL, n_queries=64,
        n_layers=2, n_heads=HEADS, d_order_out=6,
    ).to(device)

    # ---- Scheduler & optimizer ----
    scheduler = NoiseScheduler(1000, "cosine").to(device)
    all_params = list(model.parameters()) + list(order_decoder.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=1e-4, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)

    # ---- EMA ----
    ema_state = {k: v.clone() for k, v in model.state_dict().items()}
    ema_decay = 0.999
    def update_ema(model_ref=model, ema_ref=ema_state):
        with torch.no_grad():
            for k, v in model_ref.state_dict().items():
                ema_ref[k].lerp_(v, 1 - ema_decay)

    # ---- Training loop ----
    K = COND_FRAMES
    step = 0
    t_start = time.time()
    pbar = tqdm(total=TOTAL_STEPS, desc=f"Train [{config_name}]")

    while step < TOTAL_STEPS:
        for batch in loader:
            if step >= TOTAL_STEPS:
                break

            frames = batch["frames"].to(device)     # [B, T, H, W, C]
            mconds = batch["market_conds"].to(device)  # [B, T, 8]
            B, T, H, W, C = frames.shape
            N = T - K

            z_cond = frames[:, :K]
            z_gen = frames[:, K:]

            # Market conditioning
            if cfg["zero_market_cond"]:
                market_cond = torch.zeros(B, MARKET_COND_DIM, device=device)
            else:
                market_cond = mconds.mean(dim=1)  # [B, 8]

            t_diff = torch.randint(0, 1000, (B,), device=device)
            noise = torch.randn_like(z_gen)

            z_gen_flat = z_gen.reshape(B * N, H, W, C)
            noise_flat = noise.reshape(B * N, H, W, C)
            t_exp = t_diff.unsqueeze(1).expand(B, N).reshape(B * N)
            z_noisy = scheduler.q_sample(z_gen_flat, t_exp, noise_flat).reshape(B, N, H, W, C)

            v_pred = model(z_cond, z_noisy, t_diff, market_cond=market_cond)

            v_target = scheduler.v_target(
                z_gen.reshape(B * N, H, W, C),
                noise.reshape(B * N, H, W, C),
                t_exp,
            ).reshape(B, N, H, W, C)

            loss_diff = F.mse_loss(v_pred, v_target)

            # ---- Order decoder loss ----
            v_pred_flat = v_pred.reshape(B * N, H, W, C)
            z0_pred_flat = scheduler.predict_x0_from_v(
                z_noisy.reshape(B * N, H, W, C), t_exp, v_pred_flat)
            z0_pred = z0_pred_flat.clamp(-10, 10).reshape(B, N, H, W, C)

            pred_orders = order_decoder.decode_sequence(
                torch.cat([z_cond[:, -1:], z0_pred], dim=1))
            with torch.no_grad():
                gt_orders = order_decoder.decode_sequence(frames[:, K - 1:])
            gt_scale = gt_orders.detach().abs().mean().clamp(min=1e-4)
            loss_order = F.mse_loss(pred_orders / gt_scale, gt_orders / gt_scale)

            loss = loss_diff + LAMBDA_ORDER * loss_order

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            update_ema()

            step += 1
            pbar.update(1)
            if step % 200 == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    diff=f"{loss_diff.item():.4f}",
                    ordr=f"{loss_order.item():.4f}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                )

    pbar.close()
    train_time = time.time() - t_start
    logger.info(f"[{config_name}] Training done in {train_time:.0f}s")

    # ---- Save checkpoint (use EMA weights) ----
    ckpt_path = out_dir / "checkpoint.pt"
    torch.save({
        "model": model.state_dict(),
        "ema": ema_state,
        "decoder": order_decoder.state_dict(),
        "step": step,
        "config": cfg,
    }, ckpt_path)
    logger.info(f"[{config_name}] Saved checkpoint to {ckpt_path}")

    # ---- Load EMA for evaluation ----
    model.load_state_dict(ema_state)
    model.eval()

    sampler = VideoDDIMSampler(model, scheduler, "v_prediction", ddim_steps=50, eta=0.0)

    # ---- Evaluation 1: Stylized facts ----
    logger.info(f"[{config_name}] Computing stylized facts ({NUM_GEN_SEQS} sequences) ...")
    sfacts = compute_stylized_facts(model, sampler, scheduler, dataset, NUM_GEN_SEQS, cfg)
    logger.info(f"[{config_name}] Kurtosis={sfacts['kurtosis']:.3f}, "
                f"ACF(1)={sfacts['acf1']:.4f}, ACF(5)={sfacts['acf5']:.4f}")

    # ---- Evaluation 2: Stability test ----
    logger.info(f"[{config_name}] Running stability test ({STAB_ROUNDS} rounds) ...")
    stab = run_stability_test(model, sampler, scheduler, dataset, STAB_ROUNDS, cfg)
    logger.info(f"[{config_name}] MeanDrift={stab['mean_drift']:.4f}, "
                f"StdRatio={stab['std_ratio']:.3f}")

    # ---- Store results ----
    all_results[config_name] = {
        "desc": cfg["desc"],
        "train_time_s": round(train_time, 1),
        "final_loss_diff": round(loss_diff.item(), 5),
        "final_loss_order": round(loss_order.item(), 5),
        **{k: round(v, 4) for k, v in sfacts.items()},
        **{k: round(v, 4) for k, v in stab.items()},
    }

    # Save per-config results
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results[config_name], f, indent=2)


# ================================================================
# Final comparison table
# ================================================================
print("\n" + "=" * 100)
print("  ABLATION STUDY RESULTS: Video DiT on A-Share L3")
print("=" * 100)

header = (
    f"{'Config':<20s} | {'Kurtosis':>9s} | {'ACF(1)':>8s} | {'ACF(5)':>8s} | "
    f"{'MeanDrift':>10s} | {'StdRatio':>9s} | {'DiffLoss':>9s} | {'OrdLoss':>9s} | "
    f"{'Time(s)':>8s}"
)
print(header)
print("-" * 100)

for cname, res in all_results.items():
    row = (
        f"{cname:<20s} | "
        f"{res['kurtosis']:>9.3f} | "
        f"{res['acf1']:>8.4f} | "
        f"{res['acf5']:>8.4f} | "
        f"{res['mean_drift']:>10.4f} | "
        f"{res['std_ratio']:>9.3f} | "
        f"{res['final_loss_diff']:>9.5f} | "
        f"{res['final_loss_order']:>9.5f} | "
        f"{res['train_time_s']:>8.1f}"
    )
    print(row)

print("-" * 100)
print()

# Identify best config per metric
for metric, ascending in [
    ("kurtosis", False),  # higher excess kurtosis = heavier tails (more realistic)
    ("acf1", False),      # higher ACF(1) = more temporal structure
    ("mean_drift", True), # lower drift = more stable
    ("std_ratio", True),  # closer to 1.0 = more stable
]:
    vals = {k: abs(v[metric]) if metric == "std_ratio" else v[metric]
            for k, v in all_results.items()}
    if ascending:
        if metric == "std_ratio":
            # Closest to 1.0 is best
            best = min(vals, key=lambda k: abs(all_results[k]["std_ratio"] - 1.0))
        else:
            best = min(vals, key=vals.get)
    else:
        best = max(vals, key=vals.get)
    print(f"  Best {metric:<12s}: {best} ({all_results[best][metric]:.4f})")

print()

# Save full results table
results_path = Path("outputs/ablation/ablation_results.json")
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Full results saved to {results_path}")
print("Ablation study complete.")
PYEOF

echo "=== ABLATION STUDY COMPLETE ==="
