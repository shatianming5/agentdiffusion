#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

echo "============================================================"
echo "  100x100 Rich Latent Stage 2: cached Rich encoder -> Video DiT"
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

from agentdiffusion.data.news_conditioning import NewsConditioner
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


def checkpoint_step(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def latest_checkpoint(out_dir: Path) -> Path | None:
    ckpts = sorted(out_dir.glob("stage2_10k_rich_step_*.pt"), key=checkpoint_step)
    return ckpts[-1] if ckpts else None


def build_news_mean(stock_codes: list[str], device: torch.device, d_news: int) -> torch.Tensor:
    news_path = Path("data/external/news/2024_News_Security.xlsx")
    content_csv = Path("data/external/news/news_20240619_with_content.csv")
    if not news_path.exists():
        logger.warning("News file missing: %s", news_path)
        return torch.zeros(d_news, device=device)

    try:
        news_cond = NewsConditioner(
            news_excel_path=news_path,
            target_date="2024-06-19",
            d_news=d_news,
            content_csv_path=content_csv if content_csv.exists() else None,
            use_content=content_csv.exists(),
        )
        embeddings = news_cond.get_stock_embeddings(stock_codes)
        if embeddings.size == 0:
            return torch.zeros(d_news, device=device)
        return torch.from_numpy(embeddings.mean(axis=0)).float().to(device)
    except Exception as exc:
        logger.warning("News conditioning unavailable: %s", exc)
        return torch.zeros(d_news, device=device)


class EncodedGridDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_grid: torch.Tensor, sequence_starts: list[int], total_frames: int):
        self.encoded_grid = encoded_grid
        self.sequence_starts = sequence_starts
        self.total_frames = total_frames

    def __len__(self) -> int:
        return len(self.sequence_starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.sequence_starts[idx]
        frames = self.encoded_grid[start:start + self.total_frames].float()
        return {"frames": frames}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()

D_LATENT = env_int("D_LATENT", 64)
GRID_H = env_int("GRID_H", 100)
GRID_W = env_int("GRID_W", 100)
TOTAL_FRAMES = env_int("TOTAL_FRAMES", 20)
COND_FRAMES = env_int("COND_FRAMES", 4)
D_MODEL = env_int("D_MODEL", 384)
DEPTH = env_int("DEPTH", 8)
HEADS = env_int("HEADS", 8)
PATCH_SIZE = env_int("PATCH_SIZE", 10)
BATCH_SIZE = env_int("BATCH_SIZE", max(2 * max(n_gpus, 1), 2))
TOTAL_STEPS = env_int("TOTAL_STEPS", 20000)
LR = env_float("LR", 1e-4)
WEIGHT_DECAY = env_float("WEIGHT_DECAY", 0.01)
SAVE_EVERY = env_int("SAVE_EVERY", 2500)
LOG_EVERY = env_int("LOG_EVERY", 100)
NUM_WORKERS = env_int("NUM_WORKERS", 0)
MAX_STOCKS = env_int("MAX_STOCKS", 30)
N_CLUSTERS = env_int("N_CLUSTERS", 10000)
N_MAX_ORDERS = env_int("N_MAX_ORDERS", 16)
WINDOW_SECONDS = env_float("WINDOW_SECONDS", 60.0)
NEWS_DIM = env_int("NEWS_DIM", 32)
TOTAL_MARKET_DIM = 8 + NEWS_DIM
EVAL_ONLY = env_flag("EVAL_ONLY", False)
RESUME = env_flag("RESUME", True)
EVAL_BOTH = env_flag("EVAL_BOTH", True)
EVAL_ROUNDS = env_int("EVAL_ROUNDS", 5)
EVAL_SEED_INDEX = env_int("EVAL_SEED_INDEX", 0)
EVAL_TRIM_KEEP = env_int("EVAL_TRIM_KEEP", 8)
ACTIVE_THRESHOLD = env_float("ACTIVE_THRESHOLD", 0.1)
STRICT_MIN_STD = env_float("STRICT_MIN_STD", 0.05)
STRICT_MAX_STD = env_float("STRICT_MAX_STD", 2.50)
STRICT_MAX_MEAN_ABS = env_float("STRICT_MAX_MEAN_ABS", 0.10)
DDIM_STEPS = env_int("DDIM_STEPS", 20)
ALLOW_RANDOM_ENCODER = env_flag("ALLOW_RANDOM_ENCODER", False)

LATENT_CACHE_DIR = Path(os.getenv("LATENT_CACHE_DIR", "outputs/cache/ashare_10k_rich_latents"))
OUT_DIR = Path(os.getenv("OUT_DIR", "outputs/vdit_10k_rich_stage2"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

e2e_dir = Path("outputs/vdit_rich_e2e")
e2e_ckpts = sorted(e2e_dir.glob("rich_e2e_step_*.pt"), key=checkpoint_step)
encoder_cache_tag = "random_encoder"
if e2e_ckpts:
    encoder_cache_tag = e2e_ckpts[-1].stem
elif not ALLOW_RANDOM_ENCODER:
    raise RuntimeError(
        "No E2E encoder checkpoint found. Run scripts/cache_10k_rich_latents.sh after "
        "scripts/run_rich_e2e.sh, or set ALLOW_RANDOM_ENCODER=1 for ablations."
    )

latent_cache_path_env = os.getenv("LATENT_CACHE_PATH", "").strip()
latent_cache_path = (
    Path(latent_cache_path_env)
    if latent_cache_path_env
    else LATENT_CACHE_DIR / (
        f"rich10k_latents_{MAX_STOCKS}stocks_k{N_CLUSTERS}_{GRID_H}x{GRID_W}_"
        f"nmax{N_MAX_ORDERS}_tf{TOTAL_FRAMES}_ws{WINDOW_SECONDS:g}_"
        f"{D_LATENT}d_{encoder_cache_tag}.pt"
    )
)
if not latent_cache_path.exists():
    raise FileNotFoundError(
        f"Latent cache not found: {latent_cache_path}. "
        "Run scripts/cache_10k_rich_latents.sh first."
    )

logger.info(
    "Config: latent=%s d_model=%d depth=%d heads=%d patch=%d batch=%d eval_only=%s resume=%s",
    latent_cache_path,
    D_MODEL,
    DEPTH,
    HEADS,
    PATCH_SIZE,
    BATCH_SIZE,
    EVAL_ONLY,
    RESUME,
)

latent_cache = torch.load(latent_cache_path, map_location="cpu", weights_only=False)
encoded_grid = latent_cache["encoded_grid"]
sequence_starts = list(latent_cache["sequence_starts"])
stock_codes = list(latent_cache.get("stock_codes", []))
logger.info(
    "Loaded latent cache: grid=%s sequences=%d cell_semantics=%s",
    tuple(encoded_grid.shape),
    len(sequence_starts),
    latent_cache.get("cell_semantics", "unknown"),
)

dataset = EncodedGridDataset(encoded_grid, sequence_starts, total_frames=TOTAL_FRAMES)
if len(dataset) == 0:
    raise RuntimeError("EncodedGridDataset is empty")

loader = None
if not EVAL_ONLY:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )

news_mean = build_news_mean(stock_codes, device, NEWS_DIM)
logger.info("News mean norm: %.4f", news_mean.norm().item())


def build_market_cond(batch_size: int) -> torch.Tensor:
    market = torch.zeros(batch_size, 8, device=device)
    news = news_mean.unsqueeze(0).expand(batch_size, -1)
    return torch.cat([market, news], dim=-1)


model = VideoDiT(
    d_latent=D_LATENT,
    d_model=D_MODEL,
    depth=DEPTH,
    heads=HEADS,
    patch_size=PATCH_SIZE,
    num_frames=TOTAL_FRAMES,
    num_cond_frames=COND_FRAMES,
    mlp_ratio=4.0,
    market_cond_dim=TOTAL_MARKET_DIM,
    grid_h=GRID_H,
    grid_w=GRID_W,
    causal_temporal=True,
    alibi_temporal=True,
).to(device)
logger.info("Model params: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)

if n_gpus > 1:
    model = nn.DataParallel(model)
    logger.info("VideoDiT DataParallel across %d GPUs", n_gpus)


def unwrap(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


noise_sched = NoiseScheduler(1000, "cosine").to(device)
optimizer = torch.optim.AdamW(unwrap(model).parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
for group in optimizer.param_groups:
    group.setdefault("initial_lr", group["lr"])
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)
ema_state = {k: v.clone() for k, v in unwrap(model).state_dict().items()}


def update_ema(decay: float = 0.999) -> None:
    with torch.no_grad():
        for key, value in unwrap(model).state_dict().items():
            ema_state[key].lerp_(value, 1 - decay)


def save_checkpoint(step: int) -> Path:
    ckpt_path = OUT_DIR / f"stage2_10k_rich_step_{step}.pt"
    torch.save(
        {
            "dit": unwrap(model).state_dict(),
            "ema_dit": ema_state,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "step": step,
            "latent_cache_path": str(latent_cache_path),
            "config": {
                "d_latent": D_LATENT,
                "d_model": D_MODEL,
                "depth": DEPTH,
                "heads": HEADS,
                "patch_size": PATCH_SIZE,
                "total_frames": TOTAL_FRAMES,
                "cond_frames": COND_FRAMES,
                "grid_h": GRID_H,
                "grid_w": GRID_W,
                "max_stocks": MAX_STOCKS,
                "n_clusters": N_CLUSTERS,
                "n_max_orders": N_MAX_ORDERS,
                "window_seconds": WINDOW_SECONDS,
                "cell_semantics": latent_cache.get("cell_semantics", "behavioral_archetype_cluster"),
            },
        },
        ckpt_path,
    )
    logger.info("Saved checkpoint: %s", ckpt_path)
    return ckpt_path


def load_checkpoint(path: Path, for_training: bool) -> int:
    global ema_state
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if for_training:
        unwrap(model).load_state_dict(ckpt["dit"])
        logger.info("Loaded training weights from %s", path)
    else:
        eval_state = ckpt.get("ema_dit", ckpt["dit"])
        unwrap(model).load_state_dict(eval_state)
        logger.info("Loaded eval weights from %s", path)

    loaded_ema = ckpt.get("ema_dit")
    if loaded_ema is not None:
        ema_state = {k: v.clone() for k, v in loaded_ema.items()}
    else:
        ema_state = {k: v.clone() for k, v in unwrap(model).state_dict().items()}

    if for_training and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if for_training and "lr_scheduler" in ckpt:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

    return int(ckpt.get("step", 0))


start_step = 0
ckpt_path_env = os.getenv("CKPT_PATH", "").strip()
train_ckpt_path = Path(ckpt_path_env) if ckpt_path_env else None
if train_ckpt_path is None and (EVAL_ONLY or RESUME):
    train_ckpt_path = latest_checkpoint(OUT_DIR)
if train_ckpt_path is not None and not train_ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {train_ckpt_path}")

if EVAL_ONLY:
    if train_ckpt_path is None:
        raise FileNotFoundError("EVAL_ONLY=1 but no stage2 checkpoint was found")
    start_step = load_checkpoint(train_ckpt_path, for_training=False)
elif train_ckpt_path is not None:
    start_step = load_checkpoint(train_ckpt_path, for_training=True)
    if start_step >= TOTAL_STEPS:
        logger.info("Checkpoint already reached step %d >= total_steps %d", start_step, TOTAL_STEPS)


def run_eval(eval_ckpt: Path | None, anchor_state_stats: bool) -> Path:
    eval_source = eval_ckpt if eval_ckpt is not None else OUT_DIR / f"stage2_10k_rich_step_{start_step}.pt"
    if eval_ckpt is not None:
        load_checkpoint(eval_ckpt, for_training=False)
    else:
        unwrap(model).load_state_dict(ema_state)
    unwrap(model).eval()

    sampler = VideoDDIMSampler(unwrap(model), noise_sched, "v_prediction", ddim_steps=DDIM_STEPS, eta=0.0)
    seed_item = dataset[EVAL_SEED_INDEX]
    seed = seed_item["frames"][:COND_FRAMES]
    sim = InteractiveSimulator(
        unwrap(model),
        sampler,
        num_cond=COND_FRAMES,
        num_gen=TOTAL_FRAMES - COND_FRAMES,
        zero_sum_proj=True,
        market_cond=build_market_cond(1),
        anchor_state_stats=anchor_state_stats,
    )
    sim.init(seed)

    seed_mean = seed.mean().item()
    seed_std = seed.std().item()
    logger.info("=== 10K Rich Stage2 Evaluation (%s) ===", "anchor" if anchor_state_stats else "raw")
    print(f"\n=== 10K Rich Stage2 Evaluation ({'anchor' if anchor_state_stats else 'raw'}) ===")
    print(f"Seed: mean={seed_mean:.4f}, std={seed_std:.4f}")

    rounds = []
    for round_idx in range(EVAL_ROUNDS):
        gen = sim.step()
        metrics = {
            "round": round_idx + 1,
            "mean": float(gen.mean().item()),
            "std": float(gen.std().item()),
            "dim0_std": float(gen[..., 0].std().item()),
            "feature_std_mean": float(gen.std(dim=-1).mean().item()),
            "active_ratio": float((gen.abs() > ACTIVE_THRESHOLD).float().mean().item()),
        }
        rounds.append(metrics)
        print(
            "  Round {round}: mean={mean:.4f}, std={std:.4f}, dim0_std={dim0_std:.4f}, "
            "feature_std_mean={feature_std_mean:.4f}, active={active_ratio:.1%}".format(**metrics)
        )
        sim.trim_buffer(keep_last=EVAL_TRIM_KEEP)

    final_frame = sim.latest_frame
    std_values = [item["std"] for item in rounds]
    mean_values = [abs(item["mean"]) for item in rounds]
    stable = (
        min(std_values) >= STRICT_MIN_STD
        and max(std_values) <= STRICT_MAX_STD
        and max(mean_values) <= STRICT_MAX_MEAN_ABS
    )

    summary = {
        "checkpoint": str(eval_source),
        "latent_cache_path": str(latent_cache_path),
        "anchor_state_stats": anchor_state_stats,
        "cell_semantics": latent_cache.get("cell_semantics", "behavioral_archetype_cluster"),
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
    logger.info("Strict stability verdict (%s): %s", "anchor" if anchor_state_stats else "raw", stable)
    logger.info("Wrote eval summary to %s", summary_path)
    return summary_path


last_saved_ckpt: Path | None = train_ckpt_path if train_ckpt_path is not None and train_ckpt_path.exists() else None
if not EVAL_ONLY and start_step < TOTAL_STEPS:
    assert loader is not None
    train_start = time.time()
    step = start_step
    K = COND_FRAMES
    pbar = tqdm(total=TOTAL_STEPS, initial=step, desc="10K Rich Stage2")

    while step < TOTAL_STEPS:
        for batch in loader:
            if step >= TOTAL_STEPS:
                break
            frames = batch["frames"].to(device)
            B, T, H, W, C = frames.shape
            N = T - K
            z_cond, z_gen = frames[:, :K], frames[:, K:]
            t_diff = torch.randint(0, 1000, (B,), device=device)
            noise = torch.randn_like(z_gen)
            t_exp = t_diff.unsqueeze(1).expand(B, N).reshape(B * N)
            z_noisy = noise_sched.q_sample(
                z_gen.reshape(B * N, H, W, C),
                t_exp,
                noise.reshape(B * N, H, W, C),
            ).reshape(B, N, H, W, C)

            v_pred = model(z_cond, z_noisy, t_diff, market_cond=build_market_cond(B))
            v_target = noise_sched.v_target(
                z_gen.reshape(B * N, H, W, C),
                noise.reshape(B * N, H, W, C),
                t_exp,
            ).reshape(B, N, H, W, C)
            loss = F.mse_loss(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(unwrap(model).parameters(), 1.0)
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
    elapsed = time.time() - train_start
    logger.info("Training done: %d steps in %.0fs (%.2f steps/s)", step, elapsed, max(step - start_step, 1) / max(elapsed, 1e-6))

if last_saved_ckpt is None:
    last_saved_ckpt = train_ckpt_path
if last_saved_ckpt is None:
    raise FileNotFoundError("No checkpoint available for evaluation")

run_eval(last_saved_ckpt, anchor_state_stats=False)
if EVAL_BOTH:
    run_eval(last_saved_ckpt, anchor_state_stats=True)
PYEOF
