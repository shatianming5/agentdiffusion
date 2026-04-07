#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "============================================================"
echo "  News-Conditioned Rich Agent Video DiT"
echo "  market_cond_dim: 8 (original) + 32 (news) = 40"
echo "============================================================"

# Install jieba if missing
.venv/bin/python3 -c "import jieba" 2>/dev/null || .venv/bin/pip install jieba

# ============================================================
# Phase 1-2: Data Loading + Joint Training with News Conditioning
# ============================================================
echo "=== Phase 1-2: News-Conditioned Training ==="
.venv/bin/python3 -u << 'PYEOF'
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, logging, json, copy, math
from pathlib import Path
from tqdm import tqdm

from agentdiffusion.data.ashare_rich_agent_dataset import AShareRichAgentDataset
from agentdiffusion.data.news_conditioning import NewsConditioner, D_NEWS
from agentdiffusion.models.rich_agent_encoder import RichAgentEncoder, D_STATE as D_LATENT
from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.models.rich_order_decoder import RichOrderDecoder, RichOrderLoss
from agentdiffusion.diffusion.scheduler import NoiseScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("outputs/vdit_news_conditioned")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Hyperparameters ----
GRID_H, GRID_W = 10, 10
MAX_STOCKS = 100
N_MAX_ORDERS = 64
TOTAL_FRAMES = 20
COND_FRAMES = 4
BATCH_SIZE = 2
LR = 5e-5
TOTAL_STEPS = 10000
ORDER_LOSS_WEIGHT = 0.1
EMA_DECAY = 0.999
GRAD_CLIP = 1.0

# Conditioning dimensions
ORIGINAL_MARKET_DIM = 8
NEWS_DIM = D_NEWS           # 32
TOTAL_MARKET_DIM = ORIGINAL_MARKET_DIM + NEWS_DIM  # 40

# ---- Data paths ----
DATA_DIR = "data/external/20240619"
NEWS_PATH = "data/external/news/2024_News_Security.xlsx"
TARGET_DATE = "2024-06-19"

# ---- Dataset ----
logger.info("Loading AShareRichAgentDataset...")
dataset = AShareRichAgentDataset(
    DATA_DIR,
    total_frames=TOTAL_FRAMES,
    cond_frames=COND_FRAMES,
    window_seconds=60.0,
    grid_h=GRID_H, grid_w=GRID_W,
    max_stocks=MAX_STOCKS,
    n_max_orders=N_MAX_ORDERS,
)
logger.info("Dataset: %d sequences", len(dataset))

if len(dataset) == 0:
    logger.error("No data! Check %s", DATA_DIR)
    import sys; sys.exit(1)

# ---- News Conditioner ----
logger.info("Loading NewsConditioner from %s ...", NEWS_PATH)
CONTENT_CSV = "data/external/news/news_20240619_with_content.csv"
news_cond = NewsConditioner(
    news_excel_path=NEWS_PATH,
    target_date=TARGET_DATE,
    d_news=NEWS_DIM,
    content_csv_path=CONTENT_CSV,
    use_content=True,
)

# Print news summary
news_summary = news_cond.summary(dataset.stock_codes)
logger.info("News summary: %s", json.dumps(news_summary, indent=2, ensure_ascii=False))

# Pre-compute the static news conditioning vector for all batches.
# Since news is per-day (static), we compute it once: mean across the grid.
# Shape: [d_news] -> broadcast to [B, d_news] at training time.
news_embeddings = news_cond.get_stock_embeddings(dataset.stock_codes)  # [100, 32]
news_mean = torch.from_numpy(news_embeddings.mean(axis=0)).float().to(device)  # [32]
logger.info("News mean conditioning norm: %.4f", news_mean.norm().item())

# Per-stock sentiments for later analysis
stock_sentiments = news_cond.get_stock_sentiments(dataset.stock_codes)  # [100]
logger.info(
    "Sentiment stats: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
    stock_sentiments.mean(), stock_sentiments.std(),
    stock_sentiments.min(), stock_sentiments.max(),
)

loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True,
)

# ---- Models ----
# RichAgentEncoder: [B, N_max, 10] -> [B, 200]
encoder = RichAgentEncoder(
    d_raw_order=10,
    d_embed=64,
    d_state=D_LATENT,  # 200
    n_heads=4,
    n_layers=2,
    n_max_orders=N_MAX_ORDERS,
    dropout=0.1,
).to(device)

# Video DiT: d_latent=200, 10x10 grid, market_cond_dim=40 (8 + 32 news)
dit = VideoDiT(
    d_latent=D_LATENT,
    d_model=512,
    depth=8,
    heads=8,
    patch_size=2,
    num_frames=TOTAL_FRAMES,
    num_cond_frames=COND_FRAMES,
    mlp_ratio=4.0,
    market_cond_dim=TOTAL_MARKET_DIM,  # 40 = 8 + 32
    grid_h=GRID_H, grid_w=GRID_W,
    causal_temporal=True,
    alibi_temporal=True,
).to(device)

# RichOrderDecoder
decoder = RichOrderDecoder(
    d_state=D_LATENT,
    d_model=256,
    n_queries=32,
    n_layers=3,
    n_heads=8,
    d_hidden=256,
    dropout=0.0,
).to(device)

n_enc = sum(p.numel() for p in encoder.parameters())
n_dit = sum(p.numel() for p in dit.parameters())
n_dec = sum(p.numel() for p in decoder.parameters())
logger.info("Encoder params: %.2fM", n_enc / 1e6)
logger.info("DiT params: %.2fM", n_dit / 1e6)
logger.info("Decoder params: %.2fM", n_dec / 1e6)
logger.info("Total params: %.2fM", (n_enc + n_dit + n_dec) / 1e6)

# ---- Optimizer + scheduler ----
all_params = (
    list(encoder.parameters())
    + list(dit.parameters())
    + list(decoder.parameters())
)
optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=0.01)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=TOTAL_STEPS,
)

# ---- Noise scheduler ----
noise_sched = NoiseScheduler(1000, "cosine").to(device)

# ---- Order loss ----
order_loss_fn = RichOrderLoss(
    w_direction=1.0, w_price=1.0, w_size=1.0,
    w_type=1.0, w_urgency=0.5, w_active=2.0,
)

# ---- EMA ----
ema_dit = {k: v.clone() for k, v in dit.state_dict().items()}
ema_enc = {k: v.clone() for k, v in encoder.state_dict().items()}
ema_dec = {k: v.clone() for k, v in decoder.state_dict().items()}

def update_ema():
    with torch.no_grad():
        for k, v in dit.state_dict().items():
            ema_dit[k].lerp_(v, 1 - EMA_DECAY)
        for k, v in encoder.state_dict().items():
            ema_enc[k].lerp_(v, 1 - EMA_DECAY)
        for k, v in decoder.state_dict().items():
            ema_dec[k].lerp_(v, 1 - EMA_DECAY)


def encode_raw_orders(raw_orders, order_masks):
    """Encode raw orders -> agent states.

    Args:
        raw_orders: [B, T, H*W, N_max, 10]
        order_masks: [B, T, H*W, N_max]

    Returns:
        agent_states: [B, T, H, W, D_LATENT]
    """
    B, T, C, N, D = raw_orders.shape
    flat_orders = raw_orders.reshape(B * T * C, N, D)
    flat_masks = order_masks.reshape(B * T * C, N)
    flat_states = encoder(flat_orders, flat_masks)
    return flat_states.reshape(B, T, GRID_H, GRID_W, D_LATENT)


def build_market_cond(batch_size: int) -> torch.Tensor:
    """Build [B, 40] market conditioning = concat(original_8, news_32).

    The original 8-dim market_cond is zero (as in the base dataset).
    The news 32-dim is the static mean news embedding for this day.
    """
    original = torch.zeros(batch_size, ORIGINAL_MARKET_DIM, device=device)
    news = news_mean.unsqueeze(0).expand(batch_size, -1)  # [B, 32]
    return torch.cat([original, news], dim=-1)  # [B, 40]


# ---- Training loop ----
K = COND_FRAMES
step = 0
pbar = tqdm(total=TOTAL_STEPS, desc="News-Conditioned DiT")
log_losses = {"diffusion": [], "order": [], "total": []}

while step < TOTAL_STEPS:
    for batch in loader:
        if step >= TOTAL_STEPS:
            break

        raw_orders = batch["raw_orders"].to(device)
        order_masks = batch["order_masks"].to(device)
        B = raw_orders.shape[0]
        T = raw_orders.shape[1]
        N_gen = T - K
        n_cells = GRID_H * GRID_W

        # 1. Encode: raw_orders -> agent_states [B, T, H, W, D_LATENT]
        agent_states = encode_raw_orders(raw_orders, order_masks)

        # 2. Split cond/gen
        z_cond = agent_states[:, :K]
        z_gen = agent_states[:, K:]

        # 3. Noise gen states
        t_diff = torch.randint(0, 1000, (B,), device=device)
        noise = torch.randn_like(z_gen)
        t_exp = t_diff.unsqueeze(1).expand(B, N_gen).reshape(B * N_gen)
        z_noisy = noise_sched.q_sample(
            z_gen.reshape(B * N_gen, GRID_H, GRID_W, D_LATENT),
            t_exp,
            noise.reshape(B * N_gen, GRID_H, GRID_W, D_LATENT),
        ).reshape(B, N_gen, GRID_H, GRID_W, D_LATENT)

        # 4. Build news-augmented market conditioning [B, 40]
        market_cond = build_market_cond(B)

        # 5. Predict v with DiT (now with news conditioning)
        v_pred = dit(z_cond, z_noisy, t_diff, market_cond=market_cond)

        # 6. Diffusion loss
        v_target = noise_sched.v_target(
            z_gen.reshape(B * N_gen, GRID_H, GRID_W, D_LATENT),
            noise.reshape(B * N_gen, GRID_H, GRID_W, D_LATENT),
            t_exp,
        ).reshape(B, N_gen, GRID_H, GRID_W, D_LATENT)

        diffusion_loss = F.mse_loss(v_pred, v_target)

        # 7. Recover x0 from v, decode with RichOrderDecoder
        with torch.no_grad():
            x0_pred = noise_sched.predict_x0_from_v(
                z_noisy.reshape(B * N_gen, GRID_H, GRID_W, D_LATENT),
                t_exp,
                v_pred.detach().reshape(B * N_gen, GRID_H, GRID_W, D_LATENT),
            ).reshape(B, N_gen, GRID_H, GRID_W, D_LATENT)

        if N_gen > 1:
            dec_st = z_gen[:, :-1].reshape(B * (N_gen - 1), GRID_H, GRID_W, D_LATENT)
            dec_st1 = z_gen[:, 1:].reshape(B * (N_gen - 1), GRID_H, GRID_W, D_LATENT)

            pred_orders = decoder.forward_cells(dec_st, dec_st1)
            pred_orders = pred_orders.reshape(
                B, N_gen - 1, n_cells, decoder.n_queries, 10
            )

            gt_frames = raw_orders[:, K + 1:K + N_gen]
            gt_masks = order_masks[:, K + 1:K + N_gen]

            order_loss_accum = torch.tensor(0.0, device=device)
            n_valid_cells = 0

            n_sample_cells = min(n_cells, 20)
            cell_indices = torch.randperm(n_cells, device=device)[:n_sample_cells]

            for ci in cell_indices:
                c_idx = ci.item()
                for bt in range(B):
                    for tf in range(N_gen - 1):
                        pred_cell = pred_orders[bt, tf, c_idx]
                        gt_cell = gt_frames[bt, tf, c_idx]
                        gt_mask = gt_masks[bt, tf, c_idx]
                        n_gt = gt_mask.sum().item()

                        if n_gt == 0:
                            active_logits = pred_cell[:, 9]
                            inactive_target = torch.zeros_like(active_logits)
                            cell_loss = F.binary_cross_entropy_with_logits(
                                active_logits, inactive_target
                            )
                        else:
                            gt_valid = gt_cell[:int(n_gt)]
                            loss_dict = order_loss_fn._matched_loss(pred_cell, gt_valid)
                            cell_loss = loss_dict["active"]
                            if loss_dict["direction"].item() > 0:
                                cell_loss = cell_loss + loss_dict["direction"]
                                cell_loss = cell_loss + loss_dict["price"]
                                cell_loss = cell_loss + loss_dict["size"]

                        order_loss_accum = order_loss_accum + cell_loss
                        n_valid_cells += 1

            order_loss = order_loss_accum / max(n_valid_cells, 1)
        else:
            order_loss = torch.tensor(0.0, device=device)

        # 8. Total loss
        total_loss = diffusion_loss + ORDER_LOSS_WEIGHT * order_loss

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)
        optimizer.step()
        lr_scheduler.step()
        update_ema()

        step += 1
        pbar.update(1)

        if step % 50 == 0:
            dl = diffusion_loss.item()
            ol = order_loss.item()
            tl = total_loss.item()
            log_losses["diffusion"].append(dl)
            log_losses["order"].append(ol)
            log_losses["total"].append(tl)
            pbar.set_postfix(
                diff=f"{dl:.4f}",
                order=f"{ol:.4f}",
                total=f"{tl:.4f}",
                lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                step=step,
            )

        if step % 2500 == 0:
            ckpt = {
                "encoder": encoder.state_dict(),
                "dit": dit.state_dict(),
                "decoder": decoder.state_dict(),
                "ema_encoder": ema_enc,
                "ema_dit": ema_dit,
                "ema_decoder": ema_dec,
                "step": step,
                "config": {
                    "grid_h": GRID_H, "grid_w": GRID_W,
                    "d_latent": D_LATENT, "n_max_orders": N_MAX_ORDERS,
                    "total_frames": TOTAL_FRAMES, "cond_frames": COND_FRAMES,
                    "market_cond_dim": TOTAL_MARKET_DIM,
                    "news_dim": NEWS_DIM,
                },
                "news_summary": news_summary,
            }
            torch.save(ckpt, OUT_DIR / f"news_cond_step_{step}.pt")
            logger.info("Saved checkpoint at step %d", step)

pbar.close()

# Save final checkpoint
ckpt = {
    "encoder": encoder.state_dict(),
    "dit": dit.state_dict(),
    "decoder": decoder.state_dict(),
    "ema_encoder": ema_enc,
    "ema_dit": ema_dit,
    "ema_decoder": ema_dec,
    "step": step,
    "config": {
        "grid_h": GRID_H, "grid_w": GRID_W,
        "d_latent": D_LATENT, "n_max_orders": N_MAX_ORDERS,
        "total_frames": TOTAL_FRAMES, "cond_frames": COND_FRAMES,
        "market_cond_dim": TOTAL_MARKET_DIM,
        "news_dim": NEWS_DIM,
    },
    "news_summary": news_summary,
}
torch.save(ckpt, OUT_DIR / f"news_cond_step_{step}.pt")

with open(OUT_DIR / "loss_history.json", "w") as f:
    json.dump(log_losses, f)

logger.info("Training done. Saved to %s", OUT_DIR)

PYEOF

# ============================================================
# Phase 3: Evaluation + News Counterfactual Experiment
# ============================================================
echo "=== Phase 3: News-Conditioned Evaluation ==="
.venv/bin/python3 -u << 'PYEOF'
import torch, torch.nn.functional as F
import numpy as np, logging, json
from pathlib import Path
from collections import defaultdict

from agentdiffusion.data.ashare_rich_agent_dataset import AShareRichAgentDataset
from agentdiffusion.data.news_conditioning import NewsConditioner, D_NEWS
from agentdiffusion.models.rich_agent_encoder import RichAgentEncoder, D_STATE as D_LATENT
from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.models.rich_order_decoder import RichOrderDecoder
from agentdiffusion.diffusion.scheduler import NoiseScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("outputs/vdit_news_conditioned")

GRID_H, GRID_W = 10, 10
N_MAX_ORDERS = 64
TOTAL_FRAMES = 20
COND_FRAMES = 4
ORIGINAL_MARKET_DIM = 8
NEWS_DIM = D_NEWS
TOTAL_MARKET_DIM = ORIGINAL_MARKET_DIM + NEWS_DIM

DATA_DIR = "data/external/20240619"
NEWS_PATH = "data/external/news/2024_News_Security.xlsx"
TARGET_DATE = "2024-06-19"

# ---- Reload dataset ----
dataset = AShareRichAgentDataset(
    DATA_DIR,
    total_frames=TOTAL_FRAMES, cond_frames=COND_FRAMES,
    window_seconds=60.0,
    grid_h=GRID_H, grid_w=GRID_W,
    max_stocks=100, n_max_orders=N_MAX_ORDERS,
)
if len(dataset) == 0:
    logger.error("No data!"); import sys; sys.exit(1)

# ---- News conditioner (original + flipped copy) ----
CONTENT_CSV = "data/external/news/news_20240619_with_content.csv"
news_cond = NewsConditioner(
    news_excel_path=NEWS_PATH,
    target_date=TARGET_DATE,
    d_news=NEWS_DIM,
    content_csv_path=CONTENT_CSV,
    use_content=True,
)

news_embeddings = news_cond.get_stock_embeddings(dataset.stock_codes)  # [100, 32]
news_mean = torch.from_numpy(news_embeddings.mean(axis=0)).float().to(device)
stock_sentiments = news_cond.get_stock_sentiments(dataset.stock_codes)  # [100]

# ---- Load checkpoint ----
ckpt_path = sorted(OUT_DIR.glob("news_cond_step_*.pt"))[-1]
ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
logger.info("Loaded checkpoint from %s (step %d)", ckpt_path, ckpt["step"])

# ---- Rebuild models ----
encoder = RichAgentEncoder(
    d_raw_order=10, d_embed=64, d_state=D_LATENT,
    n_heads=4, n_layers=2, n_max_orders=N_MAX_ORDERS, dropout=0.1,
).to(device)

dit = VideoDiT(
    d_latent=D_LATENT, d_model=512, depth=8, heads=8,
    patch_size=2, num_frames=TOTAL_FRAMES, num_cond_frames=COND_FRAMES,
    mlp_ratio=4.0, market_cond_dim=TOTAL_MARKET_DIM,
    grid_h=GRID_H, grid_w=GRID_W,
    causal_temporal=True, alibi_temporal=True,
).to(device)

decoder = RichOrderDecoder(
    d_state=D_LATENT, d_model=256, n_queries=32,
    n_layers=3, n_heads=8, d_hidden=256, dropout=0.0,
).to(device)

encoder.load_state_dict(ckpt.get("ema_encoder", ckpt["encoder"]))
dit.load_state_dict(ckpt.get("ema_dit", ckpt["dit"]))
decoder.load_state_dict(ckpt.get("ema_decoder", ckpt["decoder"]))
encoder.eval(); dit.eval(); decoder.eval()

noise_sched = NoiseScheduler(1000, "cosine").to(device)
sampler = VideoDDIMSampler(dit, noise_sched, "v_prediction", ddim_steps=20, eta=0.0)

print("\n" + "=" * 72)
print("  News-Conditioned Evaluation + Counterfactual Experiment")
print("=" * 72)

# ---- Encode seed ----
sample = dataset[0]
raw_orders = sample["raw_orders"].unsqueeze(0).to(device)
order_masks = sample["order_masks"].unsqueeze(0).to(device)

B, T, C, N, D = raw_orders.shape
flat_orders = raw_orders.reshape(B * T * C, N, D)
flat_masks = order_masks.reshape(B * T * C, N)

with torch.no_grad():
    flat_states = encoder(flat_orders, flat_masks)
    seed_states = flat_states.reshape(1, T, GRID_H, GRID_W, D_LATENT)

x_cond = seed_states[:, :COND_FRAMES]
N_gen = TOTAL_FRAMES - COND_FRAMES
gen_shape = (1, N_gen, GRID_H, GRID_W, D_LATENT)


def build_market_cond_eval(batch_size, news_vec):
    """Build [B, 40] market cond with a specific news vector."""
    original = torch.zeros(batch_size, ORIGINAL_MARKET_DIM, device=device)
    news = news_vec.unsqueeze(0).expand(batch_size, -1)
    return torch.cat([original, news], dim=-1)


def generate_and_decode(market_cond_tensor, label=""):
    """Generate states and decode orders given a market_cond."""
    with torch.no_grad():
        gen_states = sampler.sample(
            x_cond, gen_shape, market_cond=market_cond_tensor, device=device,
        )
        all_states = torch.cat([x_cond, gen_states], dim=1)
        gen_orders = decoder.decode_sequence(all_states)
    return gen_states, gen_orders


# ---- Experiment 1: Generate with REAL news conditioning ----
print("\n--- Experiment 1: Generation with REAL news ---")
mc_real = build_market_cond_eval(1, news_mean)
gen_states_real, gen_orders_real = generate_and_decode(mc_real, "real_news")

gen_np_real = gen_orders_real[0].cpu().numpy()  # [T-1, H*W, 32, 10]
active_logits_real = gen_np_real[:, :, :, 9]
active_probs_real = 1.0 / (1.0 + np.exp(-np.clip(active_logits_real, -20, 20)))
pred_active_real = active_probs_real > 0.5
dir_logits_real = gen_np_real[:, :, :, :3]
dir_preds_real = dir_logits_real.argmax(axis=-1)

print(f"  Active ratio: {pred_active_real.mean():.4f}")
buy_r = (dir_preds_real == 0).mean()
sell_r = (dir_preds_real == 1).mean()
print(f"  Direction: buy={buy_r:.3f}, sell={sell_r:.3f}")

# ---- Experiment 2: Generate with FLIPPED news (counterfactual) ----
print("\n--- Experiment 2: Generation with FLIPPED news (counterfactual) ---")
flipped_news_mean = -news_mean  # invert the news embedding
mc_flipped = build_market_cond_eval(1, flipped_news_mean)
gen_states_flip, gen_orders_flip = generate_and_decode(mc_flipped, "flipped_news")

gen_np_flip = gen_orders_flip[0].cpu().numpy()
active_logits_flip = gen_np_flip[:, :, :, 9]
active_probs_flip = 1.0 / (1.0 + np.exp(-np.clip(active_logits_flip, -20, 20)))
pred_active_flip = active_probs_flip > 0.5
dir_logits_flip = gen_np_flip[:, :, :, :3]
dir_preds_flip = dir_logits_flip.argmax(axis=-1)

print(f"  Active ratio: {pred_active_flip.mean():.4f}")
buy_f = (dir_preds_flip == 0).mean()
sell_f = (dir_preds_flip == 1).mean()
print(f"  Direction: buy={buy_f:.3f}, sell={sell_f:.3f}")

# ---- Experiment 3: Generate with ZERO news (no news baseline) ----
print("\n--- Experiment 3: Generation with ZERO news (baseline) ---")
zero_news = torch.zeros_like(news_mean)
mc_zero = build_market_cond_eval(1, zero_news)
gen_states_zero, gen_orders_zero = generate_and_decode(mc_zero, "zero_news")

gen_np_zero = gen_orders_zero[0].cpu().numpy()
active_logits_zero = gen_np_zero[:, :, :, 9]
active_probs_zero = 1.0 / (1.0 + np.exp(-np.clip(active_logits_zero, -20, 20)))
pred_active_zero = active_probs_zero > 0.5
dir_logits_zero = gen_np_zero[:, :, :, :3]
dir_preds_zero = dir_logits_zero.argmax(axis=-1)

print(f"  Active ratio: {pred_active_zero.mean():.4f}")
buy_z = (dir_preds_zero == 0).mean()
sell_z = (dir_preds_zero == 1).mean()
print(f"  Direction: buy={buy_z:.3f}, sell={sell_z:.3f}")

# ---- Per-stock sentiment vs generated behavior correlation ----
print("\n" + "=" * 72)
print("  Per-Stock: News Sentiment vs Generated Behavior")
print("=" * 72)

# For each stock (grid cell), compute sell ratio in the real-news generation
n_cells = GRID_H * GRID_W
per_stock_sell_real = np.zeros(n_cells)
per_stock_sell_flip = np.zeros(n_cells)

for c in range(n_cells):
    # Real news generation
    cell_dirs = dir_preds_real[:, c, :]  # [T-1, 32]
    cell_active = pred_active_real[:, c, :]
    if cell_active.any():
        per_stock_sell_real[c] = (cell_dirs[cell_active] == 1).mean()
    # Flipped news generation
    cell_dirs_f = dir_preds_flip[:, c, :]
    cell_active_f = pred_active_flip[:, c, :]
    if cell_active_f.any():
        per_stock_sell_flip[c] = (cell_dirs_f[cell_active_f] == 1).mean()

# Correlation: sentiment vs sell ratio
# Negative sentiment should correlate with higher sell ratio
sentiments = stock_sentiments[:n_cells]
has_news = sentiments != 0.0

if has_news.any():
    from numpy import corrcoef
    # Pearson correlation: sentiment vs sell ratio (real news)
    corr_real = corrcoef(sentiments[has_news], per_stock_sell_real[has_news])[0, 1]
    # Pearson correlation: sentiment vs sell ratio (flipped news)
    corr_flip = corrcoef(sentiments[has_news], per_stock_sell_flip[has_news])[0, 1]

    print(f"\n  Stocks with news: {has_news.sum()}")
    print(f"  Correlation (sentiment vs sell_ratio, real news):    {corr_real:+.4f}")
    print(f"  Correlation (sentiment vs sell_ratio, flipped news): {corr_flip:+.4f}")
    print(f"  (Negative correlation = negative news -> more selling = model learned)")
else:
    corr_real = 0.0
    corr_flip = 0.0
    print("\n  No stocks with news found for correlation analysis.")

# Show top-5 most negative and positive sentiment stocks
sorted_idx = np.argsort(sentiments)
print("\n  Top-5 NEGATIVE sentiment stocks:")
for i in range(min(5, n_cells)):
    idx = sorted_idx[i]
    code = dataset.stock_codes[idx] if idx < len(dataset.stock_codes) else "?"
    print(f"    {code}: sentiment={sentiments[idx]:+.3f}, "
          f"sell_real={per_stock_sell_real[idx]:.3f}, "
          f"sell_flip={per_stock_sell_flip[idx]:.3f}")

print("\n  Top-5 POSITIVE sentiment stocks:")
for i in range(min(5, n_cells)):
    idx = sorted_idx[-(i + 1)]
    code = dataset.stock_codes[idx] if idx < len(dataset.stock_codes) else "?"
    print(f"    {code}: sentiment={sentiments[idx]:+.3f}, "
          f"sell_real={per_stock_sell_real[idx]:.3f}, "
          f"sell_flip={per_stock_sell_flip[idx]:.3f}")

# ---- Counterfactual delta ----
print("\n" + "=" * 72)
print("  Counterfactual Summary")
print("=" * 72)
print(f"  Real news  -> buy={buy_r:.3f}, sell={sell_r:.3f}")
print(f"  Flipped    -> buy={buy_f:.3f}, sell={sell_f:.3f}")
print(f"  Zero (base)-> buy={buy_z:.3f}, sell={sell_z:.3f}")
delta_sell = sell_f - sell_r
delta_buy = buy_f - buy_r
print(f"  Delta (flipped - real): buy={delta_buy:+.4f}, sell={delta_sell:+.4f}")
print(f"  (Positive delta_sell = flipping news increases selling = model responds to news)")

# ---- State-level divergence ----
with torch.no_grad():
    state_diff_rf = (gen_states_real - gen_states_flip).pow(2).mean().item()
    state_diff_rz = (gen_states_real - gen_states_zero).pow(2).mean().item()
print(f"\n  State MSE (real vs flipped): {state_diff_rf:.6f}")
print(f"  State MSE (real vs zero):    {state_diff_rz:.6f}")

# ---- Save results ----
eval_results = {
    "step": int(ckpt["step"]),
    "real_news": {
        "active_ratio": float(pred_active_real.mean()),
        "buy": float(buy_r), "sell": float(sell_r),
    },
    "flipped_news": {
        "active_ratio": float(pred_active_flip.mean()),
        "buy": float(buy_f), "sell": float(sell_f),
    },
    "zero_news": {
        "active_ratio": float(pred_active_zero.mean()),
        "buy": float(buy_z), "sell": float(sell_z),
    },
    "counterfactual_delta": {
        "delta_buy": float(delta_buy),
        "delta_sell": float(delta_sell),
    },
    "correlation": {
        "sentiment_vs_sell_real": float(corr_real) if has_news.any() else None,
        "sentiment_vs_sell_flipped": float(corr_flip) if has_news.any() else None,
    },
    "state_divergence": {
        "real_vs_flipped": float(state_diff_rf),
        "real_vs_zero": float(state_diff_rz),
    },
    "news_summary": news_cond.summary(dataset.stock_codes),
}

with open(OUT_DIR / "news_eval_results.json", "w") as f:
    json.dump(eval_results, f, indent=2, ensure_ascii=False)
print(f"\n  Saved evaluation to {OUT_DIR / 'news_eval_results.json'}")

print("\n" + "=" * 72)
print("  News-Conditioned Experiment Complete")
print("=" * 72)

PYEOF

echo "=== ALL DONE ==="
