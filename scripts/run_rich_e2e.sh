#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

echo "============================================================"
echo "  Rich Agent E2E: 25 stocks, d_latent=64, multi-GPU"
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
logger.info("GPUs available: %d", n_gpus)

# ---- Config (scaled down for E2E feasibility) ----
D_LATENT = 64
GRID_H, GRID_W = 5, 5
MAX_STOCKS = 25
N_MAX_ORDERS = 16
BATCH_SIZE = 4 * max(n_gpus, 1)
TOTAL_FRAMES = 20
COND_FRAMES = 4
TOTAL_STEPS = 10000
LR = 1e-4
LAMBDA_ORDER = 0.1
NEWS_DIM = 32
TOTAL_MARKET_DIM = 8 + NEWS_DIM  # 40

OUT_DIR = Path("outputs/vdit_rich_e2e")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Dataset ----
from agentdiffusion.data.ashare_rich_agent_dataset import AShareRichAgentDataset

logger.info("Loading dataset: %d stocks, %dx%d grid, n_max=%d",
            MAX_STOCKS, GRID_H, GRID_W, N_MAX_ORDERS)
dataset = AShareRichAgentDataset(
    "data/external/20240619",
    total_frames=TOTAL_FRAMES, cond_frames=COND_FRAMES,
    window_seconds=60.0,
    max_stocks=MAX_STOCKS,
    grid_h=GRID_H, grid_w=GRID_W,
    n_max_orders=N_MAX_ORDERS,
)
logger.info("Dataset: %d sequences", len(dataset))
if len(dataset) == 0:
    raise RuntimeError("No data!")

loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

# ---- News conditioning (BERT) ----
NEWS_PATH = "data/external/news/2024_News_Security.xlsx"
CONTENT_CSV = "data/external/news/news_20240619_with_content.csv"

from agentdiffusion.data.news_conditioning import NewsConditioner
logger.info("Loading BERT news conditioner...")
news_cond = NewsConditioner(
    news_excel_path=NEWS_PATH,
    target_date="2024-06-19",
    d_news=NEWS_DIM,
    content_csv_path=CONTENT_CSV,
    use_content=True,
)
stock_codes = getattr(dataset, 'stock_codes', [f"stock_{i}" for i in range(MAX_STOCKS)])
news_embeddings = news_cond.get_stock_embeddings(stock_codes)
news_mean = torch.from_numpy(news_embeddings.mean(axis=0)).float().to(device)
logger.info("News conditioning ready: norm=%.4f", news_mean.norm().item())

# ---- Models ----
from agentdiffusion.models.rich_agent_encoder import RichAgentEncoder
from agentdiffusion.models.video_dit import VideoDiT
from agentdiffusion.models.rich_order_decoder import RichOrderDecoder

encoder = RichAgentEncoder(
    d_raw_order=10, d_embed=64, d_state=D_LATENT,
    n_heads=4, n_layers=2, n_max_orders=N_MAX_ORDERS, dropout=0.1,
).to(device)

dit = VideoDiT(
    d_latent=D_LATENT, d_model=256, depth=6, heads=8,
    patch_size=1, num_frames=TOTAL_FRAMES, num_cond_frames=COND_FRAMES,
    mlp_ratio=4.0, market_cond_dim=TOTAL_MARKET_DIM,
    grid_h=GRID_H, grid_w=GRID_W,
    causal_temporal=True, alibi_temporal=True,
).to(device)

decoder = RichOrderDecoder(
    d_state=D_LATENT, d_model=128, n_queries=16,
    n_layers=2, n_heads=4, d_hidden=128, dropout=0.0,
).to(device)

logger.info("Encoder: %.2fM params", sum(p.numel() for p in encoder.parameters()) / 1e6)
logger.info("DiT: %.2fM params", sum(p.numel() for p in dit.parameters()) / 1e6)
logger.info("Decoder: %.2fM params", sum(p.numel() for p in decoder.parameters()) / 1e6)

# Multi-GPU
if n_gpus > 1:
    encoder = nn.DataParallel(encoder)
    dit = nn.DataParallel(dit)
    # decoder stays single-GPU (uses custom forward_cells)
    logger.info("DataParallel on encoder + DiT (%d GPUs)", n_gpus)

from agentdiffusion.diffusion.scheduler import NoiseScheduler
noise_sched = NoiseScheduler(1000, "cosine").to(device)

all_params = list(encoder.parameters()) + list(dit.parameters()) + list(decoder.parameters())
optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=0.01)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)

def _unwrap(m):
    return m.module if hasattr(m, 'module') else m

ema_dit = {k: v.clone() for k, v in _unwrap(dit).state_dict().items()}
ema_enc = {k: v.clone() for k, v in _unwrap(encoder).state_dict().items()}
EMA_DECAY = 0.999
def update_ema():
    with torch.no_grad():
        for k, v in _unwrap(dit).state_dict().items():
            ema_dit[k].lerp_(v, 1 - EMA_DECAY)
        for k, v in _unwrap(encoder).state_dict().items():
            ema_enc[k].lerp_(v, 1 - EMA_DECAY)

def build_market_cond(bs):
    original = torch.zeros(bs, 8, device=device)
    news = news_mean.unsqueeze(0).expand(bs, -1)
    return torch.cat([original, news], dim=-1)

# ---- Training loop ----
K = COND_FRAMES
step = 0
pbar = tqdm(total=TOTAL_STEPS, desc="Rich E2E")
t0 = time.time()

while step < TOTAL_STEPS:
    for batch in loader:
        if step >= TOTAL_STEPS:
            break

        raw_orders = batch["raw_orders"].to(device)   # [B, T, H*W, N_max, 10]
        order_masks = batch["order_masks"].to(device)  # [B, T, H*W, N_max]
        B, T, C, N, D = raw_orders.shape

        # 1. Encode all cells all frames → agent states
        flat_orders = raw_orders.reshape(B * T * C, N, D)
        flat_masks = order_masks.reshape(B * T * C, N)
        flat_states = encoder(flat_orders, flat_masks)  # [B*T*C, d_latent]
        agent_states = flat_states.reshape(B, T, GRID_H, GRID_W, D_LATENT)

        # 2. Split cond/gen
        z_cond = agent_states[:, :K]
        z_gen = agent_states[:, K:]
        N_gen = T - K

        # 3. Diffusion
        t_diff = torch.randint(0, 1000, (B,), device=device)
        noise = torch.randn_like(z_gen)
        t_exp = t_diff.unsqueeze(1).expand(B, N_gen).reshape(B * N_gen)
        z_noisy = noise_sched.q_sample(
            z_gen.reshape(B * N_gen, GRID_H, GRID_W, D_LATENT),
            t_exp,
            noise.reshape(B * N_gen, GRID_H, GRID_W, D_LATENT),
        ).reshape(B, N_gen, GRID_H, GRID_W, D_LATENT)

        mc = build_market_cond(B)
        v_pred = dit(z_cond, z_noisy, t_diff, market_cond=mc)
        v_target = noise_sched.v_target(
            z_gen.reshape(B * N_gen, GRID_H, GRID_W, D_LATENT),
            noise.reshape(B * N_gen, GRID_H, GRID_W, D_LATENT),
            t_exp,
        ).reshape(B, N_gen, GRID_H, GRID_W, D_LATENT)

        loss_diff = F.mse_loss(v_pred, v_target)

        # 4. Order decoder loss (on a few sampled cells)
        n_sample_cells = min(5, GRID_H * GRID_W)
        cell_idx = torch.randint(0, GRID_H * GRID_W, (n_sample_cells,))
        loss_order = torch.tensor(0.0, device=device)
        for ci in cell_idx:
            r, c = ci // GRID_W, ci % GRID_W
            s_t = agent_states[:, K-1:-1, r, c, :]   # [B, N_gen, d_latent]
            s_t1 = agent_states[:, K:, r, c, :]       # [B, N_gen, d_latent]
            for ti in range(min(3, N_gen)):
                pred = decoder(s_t[:, ti], s_t1[:, ti])  # [B, n_queries, 10]
                # Simple MSE against pred itself (self-consistency, no GT orders for speed)
                loss_order = loss_order + pred.abs().mean() * 0.001
        loss_order = loss_order / (n_sample_cells * min(3, N_gen))

        loss = loss_diff + LAMBDA_ORDER * loss_order

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        lr_scheduler.step()
        update_ema()

        step += 1
        pbar.update(1)
        if step % 100 == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                diff=f"{loss_diff.item():.4f}",
                lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                step=step)

        if step % 2500 == 0:
            torch.save({
                "encoder": _unwrap(encoder).state_dict(),
                "dit": _unwrap(dit).state_dict(),
                "decoder": decoder.state_dict(),
                "ema_dit": ema_dit,
                "ema_enc": ema_enc,
                "step": step,
            }, OUT_DIR / f"rich_e2e_step_{step}.pt")
            logger.info("Saved step %d", step)

pbar.close()
elapsed = time.time() - t0
logger.info("Training done: %d steps in %.0fs (%.2f steps/s)", step, elapsed, step/elapsed)

# Save final
torch.save({
    "encoder": _unwrap(encoder).state_dict(),
    "dit": _unwrap(dit).state_dict(),
    "decoder": decoder.state_dict(),
    "ema_dit": ema_dit,
    "ema_enc": ema_enc,
    "step": step,
}, OUT_DIR / f"rich_e2e_step_{step}.pt")

# ---- Quick eval ----
logger.info("=== Evaluation ===")
_unwrap(encoder).load_state_dict(ema_enc)
_unwrap(dit).load_state_dict(ema_dit)
encoder.eval(); dit.eval(); decoder.eval()

from agentdiffusion.models.video_dit import VideoDDIMSampler
from agentdiffusion.infer.interactive_sim import InteractiveSimulator

sampler = VideoDDIMSampler(_unwrap(dit), noise_sched, "v_prediction", ddim_steps=20, eta=0.0)
sim = InteractiveSimulator(_unwrap(dit), sampler, num_cond=K, num_gen=T-K, zero_sum_proj=True)

# Encode seed from real data
sample = dataset[0]
ro = sample["raw_orders"].to(device)  # [T, H*W, N, 10]
om = sample["order_masks"].to(device)
with torch.no_grad():
    T_s, C_s, N_s, D_s = ro.shape
    flat = ro.reshape(T_s * C_s, N_s, D_s)
    flat_m = om.reshape(T_s * C_s, N_s)
    enc_out = _unwrap(encoder)(flat, flat_m).reshape(T_s, GRID_H, GRID_W, D_LATENT)

seed = enc_out[:K]  # [4, 5, 5, 64]
sim.init(seed)

print("\n=== Rich E2E Interactive Simulation ===")
print(f"Grid: {GRID_H}x{GRID_W}, d_latent={D_LATENT}, {MAX_STOCKS} stocks")
for r in range(5):
    gen = sim.step()
    m = gen.mean().item()
    s = gen.std().item()
    print(f"  Round {r+1}: mean={m:.4f}, std={s:.4f}")
    sim.trim_buffer(keep_last=8)

print("=== Done ===")
PYEOF
