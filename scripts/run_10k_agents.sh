#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "============================================================"
echo "  10K Agent Video DiT: Real A-Share L3 -> 100x100 Grid"
echo "============================================================"

.venv/bin/python3 -u <<'PYEOF'
import json
import logging
import os
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from agentdiffusion.data.ashare_10k_agents import (
    AShare10KAgentDataset,
    D_STATE,
    GRID_H,
    GRID_W,
)
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.infer.interactive_sim import InteractiveSimulator
from agentdiffusion.models.video_dit import VideoDDIMSampler, VideoDiT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def parse_step(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def latest_checkpoint(out_dir: Path) -> Path | None:
    ckpts = sorted(out_dir.glob("video_dit_step_*.pt"), key=parse_step)
    return ckpts[-1] if ckpts else None


def make_market_cond(batch_or_item: torch.Tensor, device: torch.device) -> torch.Tensor | None:
    if batch_or_item.numel() == 0:
        return None
    if batch_or_item.dim() == 3:
        return batch_or_item[:, 0].to(device)
    if batch_or_item.dim() == 2:
        return batch_or_item[0].to(device)
    if batch_or_item.dim() == 1:
        return batch_or_item.to(device)
    raise ValueError(f"Unsupported market_conds shape: {tuple(batch_or_item.shape)}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path(os.getenv("OUT_DIR", "outputs/vdit_10k_agents"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_STEPS = env_int("TOTAL_STEPS", 10000)
BATCH_SIZE = env_int("BATCH_SIZE", 8)
SAVE_EVERY = env_int("SAVE_EVERY", 2500)
LOG_EVERY = env_int("LOG_EVERY", 100)
NUM_WORKERS = env_int("NUM_WORKERS", 0)
DATA_DIR = os.getenv("DATA_DIR", "data/external/20240619")
WINDOW_SECONDS = env_float("WINDOW_SECONDS", 5.0)
MAX_STOCKS = env_int("MAX_STOCKS", 30)
N_CLUSTERS = env_int("N_CLUSTERS", 10000)
USE_DATASET_CACHE = env_flag("USE_DATASET_CACHE", True)
CACHE_DIR = os.getenv("CACHE_DIR", "outputs/cache/ashare_10k_agents")
EVAL_ONLY = env_flag("EVAL_ONLY", False)
RESUME = env_flag("RESUME", True)
EVAL_ROUNDS = env_int("EVAL_ROUNDS", 5)
EVAL_SEED_INDEX = env_int("EVAL_SEED_INDEX", 0)
EVAL_NUM_COND = env_int("EVAL_NUM_COND", 4)
EVAL_NUM_GEN = env_int("EVAL_NUM_GEN", 16)
EVAL_TRIM_KEEP = env_int("EVAL_TRIM_KEEP", 8)
EVAL_SHOCK_ROUND = env_int("EVAL_SHOCK_ROUND", 3)
EVAL_SHOCK_DIM = env_int("EVAL_SHOCK_DIM", 0)
EVAL_SHOCK_VALUE = env_float("EVAL_SHOCK_VALUE", -3.0)
ACTIVE_THRESHOLD = env_float("ACTIVE_THRESHOLD", 0.1)
STRICT_MIN_STD = env_float("STRICT_MIN_STD", 0.10)
STRICT_MAX_STD = env_float("STRICT_MAX_STD", 2.50)
STRICT_MAX_MEAN_ABS = env_float("STRICT_MAX_MEAN_ABS", 0.10)
EVAL_ANCHOR_STATE_STATS = env_flag("EVAL_ANCHOR_STATE_STATS", False)
DDIM_STEPS = env_int("DDIM_STEPS", 20)
LR = env_float("LR", 1e-4)
WEIGHT_DECAY = env_float("WEIGHT_DECAY", 0.01)
CKPT_PATH_ENV = os.getenv("CKPT_PATH", "").strip()

logger.info(
    "Config: eval_only=%s resume=%s total_steps=%d batch_size=%d anchor=%s cache=%s",
    EVAL_ONLY, RESUME, TOTAL_STEPS, BATCH_SIZE, EVAL_ANCHOR_STATE_STATS, USE_DATASET_CACHE,
)

# --- Dataset ---
dataset = AShare10KAgentDataset(
    DATA_DIR,
    total_frames=20,
    cond_frames=EVAL_NUM_COND,
    window_seconds=WINDOW_SECONDS,
    max_stocks=MAX_STOCKS,
    n_clusters=N_CLUSTERS,
    cache_dir=CACHE_DIR,
    use_cache=USE_DATASET_CACHE,
)
logger.info("Dataset: %d sequences", len(dataset))

if len(dataset) == 0:
    raise SystemExit("No data!")

loader = None
if not EVAL_ONLY:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )

d_latent = D_STATE
logger.info("grid=(%d,%d), d_latent=%d, patch_size=10", GRID_H, GRID_W, d_latent)

# --- Model ---
model = VideoDiT(
    d_latent=d_latent,
    d_model=256,
    depth=8,
    heads=8,
    patch_size=10,
    num_frames=20,
    num_cond_frames=EVAL_NUM_COND,
    mlp_ratio=4.0,
    market_cond_dim=32,
    grid_h=GRID_H,
    grid_w=GRID_W,
    causal_temporal=True,
    alibi_temporal=True,
).to(device)
n_params = sum(p.numel() for p in model.parameters())
logger.info("Model params: %.1fM", n_params / 1e6)

scheduler = NoiseScheduler(1000, "cosine").to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
for group in optimizer.param_groups:
    group.setdefault("initial_lr", group["lr"])
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)

ema_state = {k: v.clone() for k, v in model.state_dict().items()}
ema_decay = 0.999


def update_ema() -> None:
    with torch.no_grad():
        for k, v in model.state_dict().items():
            ema_state[k].lerp_(v, 1 - ema_decay)


def save_checkpoint(step: int) -> Path:
    ckpt_path = OUT_DIR / f"video_dit_step_{step}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema_state,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "step": step,
            "config": {
                "data_dir": DATA_DIR,
                "total_steps": TOTAL_STEPS,
                "batch_size": BATCH_SIZE,
                "window_seconds": WINDOW_SECONDS,
                "max_stocks": MAX_STOCKS,
                "n_clusters": N_CLUSTERS,
            },
        },
        ckpt_path,
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return ckpt_path


def load_checkpoint(path: Path, for_training: bool) -> int:
    global ema_state
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if for_training:
        model.load_state_dict(ckpt["model"])
        logger.info("Loaded training weights from %s", path)
    else:
        eval_state = ckpt.get("ema", ckpt["model"])
        model.load_state_dict(eval_state)
        logger.info("Loaded eval weights from %s", path)

    loaded_ema = ckpt.get("ema")
    if loaded_ema is not None:
        ema_state = {k: v.clone() for k, v in loaded_ema.items()}
    else:
        ema_state = {k: v.clone() for k, v in model.state_dict().items()}

    if for_training and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    elif for_training:
        logger.warning("Checkpoint missing optimizer state; continuing with fresh optimizer")

    if for_training and "lr_scheduler" in ckpt:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    elif for_training:
        logger.warning("Checkpoint missing LR scheduler state; scheduler restarts from current config")

    return int(ckpt.get("step", 0))


start_step = 0
ckpt_path: Path | None = Path(CKPT_PATH_ENV) if CKPT_PATH_ENV else None
if ckpt_path is None and (EVAL_ONLY or RESUME):
    ckpt_path = latest_checkpoint(OUT_DIR)

if ckpt_path is not None and not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

if EVAL_ONLY:
    if ckpt_path is None:
        raise FileNotFoundError("EVAL_ONLY=1 but no checkpoint was found")
    start_step = load_checkpoint(ckpt_path, for_training=False)
elif ckpt_path is not None:
    start_step = load_checkpoint(ckpt_path, for_training=True)
    if start_step >= TOTAL_STEPS:
        logger.info(
            "Checkpoint step=%d already reaches total_steps=%d; skipping training",
            start_step,
            TOTAL_STEPS,
        )


def run_eval(eval_ckpt: Path | None, anchor_state_stats: bool) -> Path:
    eval_source = eval_ckpt if eval_ckpt is not None else OUT_DIR / f"video_dit_step_{start_step}.pt"
    if eval_ckpt is not None:
        load_checkpoint(eval_ckpt, for_training=False)
    else:
        model.load_state_dict(ema_state)
    model.eval()

    sampler = VideoDDIMSampler(
        model,
        scheduler,
        "v_prediction",
        ddim_steps=DDIM_STEPS,
        eta=0.0,
    )
    seed_item = dataset[EVAL_SEED_INDEX]
    seed = seed_item["frames"][:EVAL_NUM_COND]
    seed_market_cond = make_market_cond(seed_item["market_conds"], device)
    sim = InteractiveSimulator(
        model,
        sampler,
        num_cond=EVAL_NUM_COND,
        num_gen=EVAL_NUM_GEN,
        zero_sum_proj=True,
        market_cond=seed_market_cond,
        anchor_state_stats=anchor_state_stats,
    )
    sim.init(seed)

    seed_std = seed.std().item()
    seed_mean = seed.mean().item()
    logger.info(
        "=== 10K Agent Evaluation (%s) ===",
        "anchor" if anchor_state_stats else "raw",
    )
    print(f"\n=== 10K Agent Interactive Simulation ({'anchor' if anchor_state_stats else 'raw'}) ===")
    print(f"Grid: {GRID_H}x{GRID_W} = {GRID_H * GRID_W} agent clusters")
    print(f"Seed: mean={seed_mean:.4f}, std={seed_std:.4f}")

    rounds = []
    for round_idx in range(EVAL_ROUNDS):
        gen = sim.step()
        pos = gen[..., 0]
        round_metrics = {
            "round": round_idx + 1,
            "mean": float(gen.mean().item()),
            "std": float(gen.std().item()),
            "state0_mean": float(pos.mean().item()),
            "state0_std": float(pos.std().item()),
            "feature_std_mean": float(gen.std(dim=-1).mean().item()),
            "active_ratio": float((gen.abs() > ACTIVE_THRESHOLD).float().mean().item()),
        }
        rounds.append(round_metrics)
        print(
            "  Round {round}: mean={mean:.4f}, std={std:.4f}, "
            "state0_mean={state0_mean:.4f}, state0_std={state0_std:.4f}, "
            "active_cells={active_ratio:.1%}".format(**round_metrics)
        )
        sim.trim_buffer(keep_last=EVAL_TRIM_KEEP)

        if round_idx + 1 == EVAL_SHOCK_ROUND:
            print("  >>> Injecting market-wide shock <<<")
            shock = torch.zeros(d_latent)
            shock[EVAL_SHOCK_DIM] = EVAL_SHOCK_VALUE
            sim.intervene(frame_idx=-1, delta=shock)

    final_frame = sim.latest_frame
    std_values = [r["std"] for r in rounds]
    mean_values = [abs(r["mean"]) for r in rounds]
    stable = (
        min(std_values) >= STRICT_MIN_STD
        and max(std_values) <= STRICT_MAX_STD
        and max(mean_values) <= STRICT_MAX_MEAN_ABS
    )

    summary = {
        "checkpoint": str(eval_source),
        "anchor_state_stats": anchor_state_stats,
        "seed_index": EVAL_SEED_INDEX,
        "seed_mean": seed_mean,
        "seed_std": seed_std,
        "strict_thresholds": {
            "min_std": STRICT_MIN_STD,
            "max_std": STRICT_MAX_STD,
            "max_abs_mean": STRICT_MAX_MEAN_ABS,
        },
        "rounds": rounds,
        "stable": stable,
        "final_frame": {
            "non_zero_ratio": float((final_frame.abs() > 0.01).any(dim=-1).float().mean().item()),
            "spatial_std_row": float(final_frame[:, :, 0].std(dim=1).mean().item()),
            "spatial_std_col": float(final_frame[:, :, 0].std(dim=0).mean().item()),
        },
    }
    summary_path = OUT_DIR / (
        f"eval_{Path(eval_source).stem}_{'anchor' if anchor_state_stats else 'raw'}.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nStable: {stable}")
    print(f"Eval summary: {summary_path}")
    print("=" * 64)
    logger.info("Strict stability verdict: %s", stable)
    logger.info("Wrote eval summary to %s", summary_path)
    return summary_path


# --- Training ---
last_saved_ckpt: Path | None = ckpt_path if ckpt_path is not None and ckpt_path.exists() else None
if not EVAL_ONLY and start_step < TOTAL_STEPS:
    assert loader is not None
    train_start = time.time()
    step = start_step
    pbar = tqdm(total=TOTAL_STEPS, initial=step, desc="10K Agent DiT")
    K = EVAL_NUM_COND

    while step < TOTAL_STEPS:
        for batch in loader:
            if step >= TOTAL_STEPS:
                break
            frames = batch["frames"].to(device)
            market_cond = make_market_cond(batch["market_conds"], device)
            B, T, H, W, C = frames.shape
            N = T - K
            z_cond, z_gen = frames[:, :K], frames[:, K:]
            t = torch.randint(0, 1000, (B,), device=device)
            noise = torch.randn_like(z_gen)
            t_exp = t.unsqueeze(1).expand(B, N).reshape(B * N)
            z_noisy = scheduler.q_sample(
                z_gen.reshape(B * N, H, W, C),
                t_exp,
                noise.reshape(B * N, H, W, C),
            ).reshape(B, N, H, W, C)

            v_pred = model(z_cond, z_noisy, t, market_cond=market_cond)
            v_target = scheduler.v_target(
                z_gen.reshape(B * N, H, W, C),
                noise.reshape(B * N, H, W, C),
                t_exp,
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
            if step % LOG_EVERY == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    step=step,
                )
            if step % SAVE_EVERY == 0:
                last_saved_ckpt = save_checkpoint(step)

    pbar.close()
    last_saved_ckpt = save_checkpoint(step)
    logger.info(
        "Training done: %d steps in %.0fs (%.2f steps/s)",
        step,
        time.time() - train_start,
        max(step - start_step, 1) / max(time.time() - train_start, 1e-6),
    )

if last_saved_ckpt is None:
    last_saved_ckpt = ckpt_path

if last_saved_ckpt is None:
    raise FileNotFoundError("No checkpoint available for evaluation")

run_eval(last_saved_ckpt, anchor_state_stats=EVAL_ANCHOR_STATE_STATS)
PYEOF
