#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

echo "============================================================"
echo "  Cache 10K Rich Latents: behavioral clusters -> Rich encoder"
echo "============================================================"

.venv/bin/python3 -u <<'PYEOF'
import logging
import os
import re
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from agentdiffusion.data.ashare_10k_rich_orders import AShare10KRichOrderDataset
from agentdiffusion.models.rich_agent_encoder import RichAgentEncoder

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


def adapt_pos_embed(enc_state: dict[str, torch.Tensor], target_n_max: int) -> dict[str, torch.Tensor]:
    pos_embed = enc_state.get("pos_embed")
    if pos_embed is None:
        return enc_state
    src_n = pos_embed.shape[1]
    if src_n == target_n_max:
        return enc_state

    logger.warning("Adapting encoder pos_embed from n_max=%d to n_max=%d", src_n, target_n_max)
    if src_n > target_n_max:
        enc_state["pos_embed"] = pos_embed[:, :target_n_max, :].clone()
    else:
        pad = pos_embed[:, -1:, :].expand(-1, target_n_max - src_n, -1).clone()
        enc_state["pos_embed"] = torch.cat([pos_embed, pad], dim=1)
    return enc_state


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()

D_LATENT = env_int("D_LATENT", 64)
GRID_H = env_int("GRID_H", 100)
GRID_W = env_int("GRID_W", 100)
MAX_STOCKS = env_int("MAX_STOCKS", 30)
N_CLUSTERS = env_int("N_CLUSTERS", 10000)
N_MAX_ORDERS = env_int("N_MAX_ORDERS", 16)
TOTAL_FRAMES = env_int("TOTAL_FRAMES", 20)
COND_FRAMES = env_int("COND_FRAMES", 4)
WINDOW_SECONDS = env_float("WINDOW_SECONDS", 60.0)
ENCODE_CHUNK_FRAMES = env_int("ENCODE_CHUNK_FRAMES", 8)
FORCE_REBUILD = env_flag("FORCE_REBUILD", False)
ALLOW_RANDOM_ENCODER = env_flag("ALLOW_RANDOM_ENCODER", False)

RAW_CACHE_DIR = Path(os.getenv("RAW_CACHE_DIR", "outputs/cache/ashare_10k_rich_orders"))
LATENT_CACHE_DIR = Path(os.getenv("LATENT_CACHE_DIR", "outputs/cache/ashare_10k_rich_latents"))
LATENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = os.getenv("DATA_DIR", "data/external/20240619")

logger.info(
    "Config: stocks=%d clusters=%d grid=%dx%d n_max=%d latent=%d chunk_frames=%d",
    MAX_STOCKS,
    N_CLUSTERS,
    GRID_H,
    GRID_W,
    N_MAX_ORDERS,
    D_LATENT,
    ENCODE_CHUNK_FRAMES,
)

e2e_dir = Path("outputs/vdit_rich_e2e")
e2e_ckpts = sorted(e2e_dir.glob("rich_e2e_step_*.pt"), key=checkpoint_step)
encoder_source = "random"
encoder_cache_tag = "random_encoder"

encoder = RichAgentEncoder(
    d_raw_order=10,
    d_embed=64,
    d_state=D_LATENT,
    n_heads=4,
    n_layers=2,
    n_max_orders=N_MAX_ORDERS,
    dropout=0.0,
)
if e2e_ckpts:
    ckpt_path = e2e_ckpts[-1]
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    enc_state = ckpt.get("ema_enc", ckpt.get("encoder"))
    enc_state = adapt_pos_embed(dict(enc_state), N_MAX_ORDERS)
    encoder.load_state_dict(enc_state, strict=False)
    encoder_source = "trained"
    encoder_cache_tag = ckpt_path.stem
    logger.info("Loaded trained encoder from %s", ckpt_path)
elif not ALLOW_RANDOM_ENCODER:
    raise RuntimeError(
        "No E2E encoder checkpoint found. Run scripts/run_rich_e2e.sh first, "
        "or set ALLOW_RANDOM_ENCODER=1 for explicit ablations."
    )
else:
    logger.warning("Using random encoder because ALLOW_RANDOM_ENCODER=1")

latent_cache_file = LATENT_CACHE_DIR / (
    f"rich10k_latents_{MAX_STOCKS}stocks_k{N_CLUSTERS}_{GRID_H}x{GRID_W}_"
    f"nmax{N_MAX_ORDERS}_tf{TOTAL_FRAMES}_ws{WINDOW_SECONDS:g}_"
    f"{D_LATENT}d_{encoder_cache_tag}.pt"
)
logger.info("Latent cache target: %s", latent_cache_file)
if latent_cache_file.exists() and not FORCE_REBUILD:
    logger.info("Latent cache already exists. Nothing to do.")
    raise SystemExit(0)

dataset = AShare10KRichOrderDataset(
    DATA_DIR,
    total_frames=TOTAL_FRAMES,
    cond_frames=COND_FRAMES,
    window_seconds=WINDOW_SECONDS,
    max_stocks=MAX_STOCKS,
    n_clusters=N_CLUSTERS,
    grid_h=GRID_H,
    grid_w=GRID_W,
    n_max_orders=N_MAX_ORDERS,
    cache_dir=RAW_CACHE_DIR,
    use_cache=True,
)
if len(dataset) == 0:
    raise RuntimeError("AShare10KRichOrderDataset is empty")

encoder = encoder.to(device).eval()
if n_gpus > 1:
    encoder = nn.DataParallel(encoder)
    logger.info("Encoder DataParallel across %d GPUs", n_gpus)

raw_grid = dataset.raw_order_grid
mask_grid = dataset.mask_grid
T_total, H, W, N, D = raw_grid.shape
logger.info("Raw rich-order grid: shape=%s", tuple(raw_grid.shape))

chunks = []
with torch.no_grad():
    for start in tqdm(range(0, T_total, ENCODE_CHUNK_FRAMES), desc="Encoding rich latent grid"):
        end = min(start + ENCODE_CHUNK_FRAMES, T_total)
        chunk_orders = raw_grid[start:end].float().to(device)
        chunk_masks = mask_grid[start:end].to(device)
        t_chunk = end - start
        flat_orders = chunk_orders.reshape(t_chunk * H * W, N, D)
        flat_masks = chunk_masks.reshape(t_chunk * H * W, N)
        chunk_states = encoder(flat_orders, flat_masks)
        chunk_states = chunk_states.reshape(t_chunk, H, W, D_LATENT).cpu().to(torch.float16)
        chunks.append(chunk_states)

encoded_grid = torch.cat(chunks, dim=0)
torch.save(
    {
        "encoded_grid": encoded_grid,
        "sequence_starts": dataset.sequence_starts,
        "stock_codes": dataset.stock_codes,
        "encoder_source": encoder_source,
        "encoder_cache_tag": encoder_cache_tag,
        "raw_cache_path": str(dataset.cache_path),
        "cell_semantics": "behavioral_archetype_cluster",
        "config": {
            "data_dir": DATA_DIR,
            "total_frames": TOTAL_FRAMES,
            "cond_frames": COND_FRAMES,
            "window_seconds": WINDOW_SECONDS,
            "max_stocks": MAX_STOCKS,
            "n_clusters": N_CLUSTERS,
            "grid_h": GRID_H,
            "grid_w": GRID_W,
            "n_max_orders": N_MAX_ORDERS,
            "d_latent": D_LATENT,
        },
    },
    latent_cache_file,
)
logger.info("Saved encoded rich latent grid: shape=%s", tuple(encoded_grid.shape))
logger.info("Done: %s", latent_cache_file)
PYEOF
