#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

echo "============================================================"
echo "  Rich Agent Stage 2: Pre-compute encoder → fast DiT training"
echo "  Runs AFTER run_rich_e2e.sh completes"
echo "  Set ALLOW_RANDOM_ENCODER=1 only for ablations/debug runs"
echo "============================================================"

.venv/bin/python3 -u << 'PYEOF'
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, logging, os, time
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()

# ---- Config: LARGER model now that encoder is frozen ----
D_LATENT = 64
GRID_H, GRID_W = 10, 10     # Scale up to 100 stocks
MAX_STOCKS = 100
N_MAX_ORDERS = 16
BATCH_SIZE = 8 * max(n_gpus, 1)
TOTAL_FRAMES = 20
COND_FRAMES = 4
TOTAL_STEPS = 20000
LR = 1e-4
NEWS_DIM = 32

OUT_DIR = Path("outputs/vdit_rich_stage2")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path("outputs/rich_encoder_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
ALLOW_RANDOM_ENCODER = os.environ.get("ALLOW_RANDOM_ENCODER", "").lower() in {"1", "true", "yes"}

# ============================================================
# Phase 1: Load E2E checkpoint encoder, pre-compute all states
# ============================================================
E2E_DIR = Path("outputs/vdit_rich_e2e")
def checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_step_", 1)[1])
    except (IndexError, ValueError):
        return -1

e2e_ckpts = sorted(E2E_DIR.glob("rich_e2e_step_*.pt"), key=checkpoint_step)
encoder_source = "random"
encoder_cache_tag = "random_encoder"

from agentdiffusion.models.rich_agent_encoder import RichAgentEncoder

# Load E2E encoder unless this is an explicit random-encoder ablation.
if e2e_ckpts:
    logger.info("Loading trained encoder from %s", e2e_ckpts[-1])
    ckpt = torch.load(str(e2e_ckpts[-1]), map_location="cpu", weights_only=True)
    encoder = RichAgentEncoder(
        d_raw_order=10, d_embed=64, d_state=D_LATENT,
        n_heads=4, n_layers=2, n_max_orders=N_MAX_ORDERS, dropout=0.0,
    )
    enc_state = ckpt.get("ema_enc", ckpt.get("encoder"))
    encoder.load_state_dict(enc_state)
    logger.info("Encoder loaded (trained)")
    encoder_source = "trained"
    encoder_cache_tag = e2e_ckpts[-1].stem
else:
    if not ALLOW_RANDOM_ENCODER:
        raise RuntimeError(
            "No E2E checkpoint found in outputs/vdit_rich_e2e. "
            "Run scripts/run_rich_e2e.sh first, or set ALLOW_RANDOM_ENCODER=1 "
            "for an explicit random-encoder ablation."
        )
    logger.warning("No E2E checkpoint found, using random encoder (ALLOW_RANDOM_ENCODER=1)")
    encoder = RichAgentEncoder(
        d_raw_order=10, d_embed=64, d_state=D_LATENT,
        n_heads=4, n_layers=2, n_max_orders=N_MAX_ORDERS, dropout=0.0,
    )

encoder = encoder.to(device).eval()

# Load dataset (100 stocks now)
from agentdiffusion.data.ashare_rich_agent_dataset import AShareRichAgentDataset

cache_file = CACHE_DIR / (
    f"encoded_states_{MAX_STOCKS}stocks_{D_LATENT}d_{encoder_cache_tag}.pt"
)
logger.info("Encoder source: %s", encoder_source)
logger.info("Encoded-state cache: %s", cache_file)

if cache_file.exists():
    logger.info("Loading cached encoded states from %s", cache_file)
    cached = torch.load(str(cache_file), weights_only=True)
    all_encoded = cached["states"]  # list of [T, H, W, d_latent] tensors
    logger.info("Loaded %d cached sequences", len(all_encoded))
else:
    logger.info("Pre-computing encoder states for %d stocks...", MAX_STOCKS)
    dataset_raw = AShareRichAgentDataset(
        "data/external/20240619",
        total_frames=TOTAL_FRAMES, cond_frames=COND_FRAMES,
        window_seconds=60.0, max_stocks=MAX_STOCKS,
        grid_h=GRID_H, grid_w=GRID_W, n_max_orders=N_MAX_ORDERS,
    )

    all_encoded = []
    with torch.no_grad():
        for idx in tqdm(range(len(dataset_raw)), desc="Encoding"):
            sample = dataset_raw[idx]
            ro = sample["raw_orders"].to(device)    # [T, H*W, N, 10]
            om = sample["order_masks"].to(device)   # [T, H*W, N]
            T, C, N, D = ro.shape
            flat_ro = ro.reshape(T * C, N, D)
            flat_om = om.reshape(T * C, N)
            states = encoder(flat_ro, flat_om)       # [T*C, d_latent]
            states = states.reshape(T, GRID_H, GRID_W, D_LATENT).cpu()
            all_encoded.append(states)

    torch.save({
        "states": all_encoded,
        "encoder_source": encoder_source,
        "encoder_cache_tag": encoder_cache_tag,
    }, str(cache_file))
    logger.info("Saved %d encoded sequences to %s", len(all_encoded), cache_file)

# Build simple dataset from pre-computed states
class PrecomputedDataset(torch.utils.data.Dataset):
    def __init__(self, states_list):
        self.states = states_list
    def __len__(self):
        return len(self.states)
    def __getitem__(self, idx):
        return {"frames": self.states[idx]}

dataset = PrecomputedDataset(all_encoded)
logger.info("Pre-computed dataset: %d sequences, shape=%s", len(dataset), dataset[0]["frames"].shape)

loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

# ============================================================
# Phase 2: Train DiT on pre-computed states (FAST)
# ============================================================
logger.info("=== Phase 2: Fast DiT training on pre-computed states ===")

from agentdiffusion.models.video_dit import VideoDiT
from agentdiffusion.diffusion.scheduler import NoiseScheduler

TOTAL_MARKET_DIM = 8 + NEWS_DIM

dit = VideoDiT(
    d_latent=D_LATENT, d_model=512, depth=8, heads=8,
    patch_size=2, num_frames=TOTAL_FRAMES, num_cond_frames=COND_FRAMES,
    mlp_ratio=4.0, market_cond_dim=TOTAL_MARKET_DIM,
    grid_h=GRID_H, grid_w=GRID_W,
    causal_temporal=True, alibi_temporal=True,
).to(device)
logger.info("DiT params: %.2fM", sum(p.numel() for p in dit.parameters()) / 1e6)

# Load E2E DiT weights if available (warm start)
if e2e_ckpts:
    try:
        e2e_dit_state = ckpt.get("ema_dit", ckpt.get("dit"))
        dit.load_state_dict(e2e_dit_state, strict=False)
        logger.info("Warm-started DiT from E2E checkpoint")
    except Exception as e:
        logger.warning("Could not warm-start DiT: %s", e)

if n_gpus > 1:
    dit = nn.DataParallel(dit)
    logger.info("DiT on %d GPUs", n_gpus)

# News conditioning
from agentdiffusion.data.news_conditioning import NewsConditioner
news_cond = NewsConditioner(
    "data/external/news/2024_News_Security.xlsx",
    target_date="2024-06-19", d_news=NEWS_DIM,
    content_csv_path="data/external/news/news_20240619_with_content.csv",
    use_content=True,
)
news_mean = torch.zeros(NEWS_DIM, device=device)
try:
    stock_codes = [f"stock_{i}" for i in range(MAX_STOCKS)]
    emb = news_cond.get_stock_embeddings(stock_codes)
    news_mean = torch.from_numpy(emb.mean(axis=0)).float().to(device)
except:
    pass
logger.info("News norm: %.4f", news_mean.norm().item())

noise_sched = NoiseScheduler(1000, "cosine").to(device)
optimizer = torch.optim.AdamW(dit.parameters() if not isinstance(dit, nn.DataParallel) else dit.module.parameters(), lr=LR, weight_decay=0.01)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)

def _unwrap(m):
    return m.module if hasattr(m, 'module') else m

ema_dit = {k: v.clone() for k, v in _unwrap(dit).state_dict().items()}
def update_ema():
    with torch.no_grad():
        for k, v in _unwrap(dit).state_dict().items():
            ema_dit[k].lerp_(v, 1 - 0.999)

def build_mc(bs):
    return torch.cat([torch.zeros(bs, 8, device=device),
                      news_mean.unsqueeze(0).expand(bs, -1)], dim=-1)

K = COND_FRAMES
step = 0
pbar = tqdm(total=TOTAL_STEPS, desc="Stage2 DiT")
t0 = time.time()

while step < TOTAL_STEPS:
    for batch in loader:
        if step >= TOTAL_STEPS:
            break
        frames = batch["frames"].to(device)  # [B, T, H, W, d_latent]
        B, T, H, W, C = frames.shape
        N = T - K
        z_cond, z_gen = frames[:, :K], frames[:, K:]

        t_diff = torch.randint(0, 1000, (B,), device=device)
        noise = torch.randn_like(z_gen)
        t_exp = t_diff.unsqueeze(1).expand(B, N).reshape(B * N)
        z_noisy = noise_sched.q_sample(
            z_gen.reshape(B*N, H, W, C), t_exp, noise.reshape(B*N, H, W, C)
        ).reshape(B, N, H, W, C)

        mc = build_mc(B)
        v_pred = dit(z_cond, z_noisy, t_diff, market_cond=mc)
        v_target = noise_sched.v_target(
            z_gen.reshape(B*N, H, W, C), noise.reshape(B*N, H, W, C), t_exp
        ).reshape(B, N, H, W, C)
        loss = F.mse_loss(v_pred, v_target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(_unwrap(dit).parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        update_ema()

        step += 1
        pbar.update(1)
        if step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")
        if step % 5000 == 0:
            torch.save({"dit": _unwrap(dit).state_dict(), "ema_dit": ema_dit,
                        "step": step}, OUT_DIR / f"stage2_step_{step}.pt")

pbar.close()
elapsed = time.time() - t0
logger.info("Stage2 done: %d steps in %.0fs (%.1f steps/s)", step, elapsed, step/elapsed)
torch.save({"dit": _unwrap(dit).state_dict(), "ema_dit": ema_dit,
            "step": step}, OUT_DIR / f"stage2_step_{step}.pt")

# Quick eval
logger.info("=== Eval ===")
_unwrap(dit).load_state_dict(ema_dit)
dit.eval()
from agentdiffusion.models.video_dit import VideoDDIMSampler
from agentdiffusion.infer.interactive_sim import InteractiveSimulator
sampler = VideoDDIMSampler(_unwrap(dit), noise_sched, "v_prediction", ddim_steps=20, eta=0.0)
sim = InteractiveSimulator(
    _unwrap(dit),
    sampler,
    num_cond=K,
    num_gen=T-K,
    zero_sum_proj=True,
    market_cond=build_mc(1),
    anchor_state_stats=True,
)
seed = all_encoded[0][:K].to(device)
sim.init(seed)
for r in range(5):
    gen = sim.step()
    print(f"  Round {r+1}: mean={gen.mean():.4f}, std={gen.std():.4f}")
    sim.trim_buffer(keep_last=8)
print("=== Stage2 Complete ===")
PYEOF
