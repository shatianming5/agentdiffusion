#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "============================================================"
echo "  Rich Agent Video DiT: Raw Order Sequences -> Agent States"
echo "  Grid: 10x10 (100 stocks), d_latent=200, n_max_orders=64"
echo "============================================================"

# ============================================================
# Phase 1 & 2: Load data + Joint Training (Encoder + DiT + Decoder)
# ============================================================
echo "=== Phase 1-2: Data Loading + Joint Training ==="
.venv/bin/python3 -u << 'PYEOF'
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, logging, json, copy, math
from pathlib import Path
from tqdm import tqdm

from agentdiffusion.data.ashare_rich_agent_dataset import AShareRichAgentDataset
from agentdiffusion.models.rich_agent_encoder import RichAgentEncoder, D_STATE as D_LATENT
from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.models.rich_order_decoder import RichOrderDecoder, RichOrderLoss
from agentdiffusion.diffusion.scheduler import NoiseScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("outputs/vdit_rich_agent")
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

# ---- Dataset ----
logger.info("Loading AShareRichAgentDataset...")
dataset = AShareRichAgentDataset(
    "data/external/20240619",
    total_frames=TOTAL_FRAMES,
    cond_frames=COND_FRAMES,
    window_seconds=60.0,
    grid_h=GRID_H, grid_w=GRID_W,
    max_stocks=MAX_STOCKS,
    n_max_orders=N_MAX_ORDERS,
)
logger.info("Dataset: %d sequences", len(dataset))

if len(dataset) == 0:
    logger.error("No data! Check data/external/20240619/")
    import sys; sys.exit(1)

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

# Video DiT: d_latent=200, 10x10 grid, patch_size=2 -> 5x5=25 patches
dit = VideoDiT(
    d_latent=D_LATENT,   # 200
    d_model=512,
    depth=8,
    heads=8,
    patch_size=2,
    num_frames=TOTAL_FRAMES,
    num_cond_frames=COND_FRAMES,
    mlp_ratio=4.0,
    market_cond_dim=8,
    grid_h=GRID_H, grid_w=GRID_W,
    causal_temporal=True,
    alibi_temporal=True,
).to(device)

# RichOrderDecoder: [B, 200] x [B, 200] -> [B, 32, 10]
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
    optimizer, T_max=TOTAL_STEPS
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
    # Flatten batch, time, cells
    flat_orders = raw_orders.reshape(B * T * C, N, D)
    flat_masks = order_masks.reshape(B * T * C, N)
    # Encode
    flat_states = encoder(flat_orders, flat_masks)  # [B*T*C, D_LATENT]
    # Reshape to [B, T, H, W, D_LATENT]
    return flat_states.reshape(B, T, GRID_H, GRID_W, D_LATENT)


# ---- Training loop ----
K = COND_FRAMES
step = 0
pbar = tqdm(total=TOTAL_STEPS, desc="Rich Agent DiT")
log_losses = {"diffusion": [], "order": [], "total": []}

while step < TOTAL_STEPS:
    for batch in loader:
        if step >= TOTAL_STEPS:
            break

        raw_orders = batch["raw_orders"].to(device)   # [B, T, H*W, N_max, 10]
        order_masks = batch["order_masks"].to(device)  # [B, T, H*W, N_max]
        B = raw_orders.shape[0]
        T = raw_orders.shape[1]
        N_gen = T - K
        n_cells = GRID_H * GRID_W

        # 1. Encode: raw_orders -> agent_states [B, T, H, W, D_LATENT]
        agent_states = encode_raw_orders(raw_orders, order_masks)

        # 2. Split cond/gen
        z_cond = agent_states[:, :K]    # [B, K, H, W, D_LATENT]
        z_gen = agent_states[:, K:]     # [B, N_gen, H, W, D_LATENT]

        # 3. Noise gen states
        t_diff = torch.randint(0, 1000, (B,), device=device)
        noise = torch.randn_like(z_gen)
        t_exp = t_diff.unsqueeze(1).expand(B, N_gen).reshape(B * N_gen)
        z_noisy = noise_sched.q_sample(
            z_gen.reshape(B * N_gen, GRID_H, GRID_W, D_LATENT),
            t_exp,
            noise.reshape(B * N_gen, GRID_H, GRID_W, D_LATENT),
        ).reshape(B, N_gen, GRID_H, GRID_W, D_LATENT)

        # 4. Predict v with DiT
        v_pred = dit(z_cond, z_noisy, t_diff)

        # 5. Diffusion loss
        v_target = noise_sched.v_target(
            z_gen.reshape(B * N_gen, GRID_H, GRID_W, D_LATENT),
            noise.reshape(B * N_gen, GRID_H, GRID_W, D_LATENT),
            t_exp,
        ).reshape(B, N_gen, GRID_H, GRID_W, D_LATENT)

        diffusion_loss = F.mse_loss(v_pred, v_target)

        # 6. Recover x0 from v, decode with RichOrderDecoder
        with torch.no_grad():
            x0_pred = noise_sched.predict_x0_from_v(
                z_noisy.reshape(B * N_gen, GRID_H, GRID_W, D_LATENT),
                t_exp,
                v_pred.detach().reshape(B * N_gen, GRID_H, GRID_W, D_LATENT),
            ).reshape(B, N_gen, GRID_H, GRID_W, D_LATENT)

        # Use teacher-forced z_gen as state_t, x0_pred as state_t1
        # Decode: predict orders from state transitions
        # We use the first N_gen-1 transitions for decoder supervision
        if N_gen > 1:
            # state_t = z_gen[:, :-1]   [B, N_gen-1, H, W, D]
            # state_t1 = z_gen[:, 1:]   [B, N_gen-1, H, W, D]
            dec_st = z_gen[:, :-1].reshape(B * (N_gen - 1), GRID_H, GRID_W, D_LATENT)
            dec_st1 = z_gen[:, 1:].reshape(B * (N_gen - 1), GRID_H, GRID_W, D_LATENT)

            pred_orders = decoder.forward_cells(
                dec_st, dec_st1
            )  # [B*(N_gen-1), H*W, n_queries, 10]
            pred_orders = pred_orders.reshape(
                B, N_gen - 1, n_cells, decoder.n_queries, 10
            )

            # 7. Order reconstruction loss against next-window real orders
            # Ground truth: raw_orders at frames K+1 ... K+N_gen-1
            gt_frames = raw_orders[:, K + 1:K + N_gen]  # [B, N_gen-1, H*W, N_max, 10]
            gt_masks = order_masks[:, K + 1:K + N_gen]   # [B, N_gen-1, H*W, N_max]

            # Compute order loss per cell, averaged
            order_loss_accum = torch.tensor(0.0, device=device)
            n_valid_cells = 0

            # Sample a subset of cells for efficiency
            n_sample_cells = min(n_cells, 20)
            cell_indices = torch.randperm(n_cells, device=device)[:n_sample_cells]

            for ci in cell_indices:
                c_idx = ci.item()
                for bt in range(B):
                    for tf in range(N_gen - 1):
                        pred_cell = pred_orders[bt, tf, c_idx]  # [n_queries, 10]
                        gt_cell = gt_frames[bt, tf, c_idx]       # [N_max, 10]
                        gt_mask = gt_masks[bt, tf, c_idx]        # [N_max]
                        n_gt = gt_mask.sum().item()

                        if n_gt == 0:
                            # No GT orders: just penalize active predictions
                            active_logits = pred_cell[:, 9]
                            inactive_target = torch.zeros_like(active_logits)
                            cell_loss = F.binary_cross_entropy_with_logits(
                                active_logits, inactive_target
                            )
                        else:
                            gt_valid = gt_cell[:int(n_gt)]
                            loss_dict = order_loss_fn._matched_loss(
                                pred_cell, gt_valid
                            )
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
                },
            }
            torch.save(ckpt, OUT_DIR / f"rich_agent_step_{step}.pt")
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
    },
}
torch.save(ckpt, OUT_DIR / f"rich_agent_step_{step}.pt")

# Save loss history
with open(OUT_DIR / "loss_history.json", "w") as f:
    json.dump(log_losses, f)

logger.info("Training done. Saved to %s", OUT_DIR)

PYEOF

# ============================================================
# Phase 3: Evaluation
# ============================================================
echo "=== Phase 3: Evaluation ==="
.venv/bin/python3 -u << 'PYEOF'
import torch, torch.nn.functional as F
import numpy as np, logging, json
from pathlib import Path
from collections import defaultdict

from agentdiffusion.data.ashare_rich_agent_dataset import AShareRichAgentDataset
from agentdiffusion.models.rich_agent_encoder import RichAgentEncoder, D_STATE as D_LATENT
from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.models.rich_order_decoder import RichOrderDecoder
from agentdiffusion.diffusion.scheduler import NoiseScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("outputs/vdit_rich_agent")

GRID_H, GRID_W = 10, 10
N_MAX_ORDERS = 64
TOTAL_FRAMES = 20
COND_FRAMES = 4

# ---- Reload dataset ----
dataset = AShareRichAgentDataset(
    "data/external/20240619",
    total_frames=TOTAL_FRAMES, cond_frames=COND_FRAMES,
    window_seconds=60.0,
    grid_h=GRID_H, grid_w=GRID_W,
    max_stocks=100, n_max_orders=N_MAX_ORDERS,
)
if len(dataset) == 0:
    logger.error("No data!"); import sys; sys.exit(1)

# ---- Load checkpoint ----
ckpt_path = sorted(OUT_DIR.glob("rich_agent_step_*.pt"))[-1]
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
    mlp_ratio=4.0, market_cond_dim=8,
    grid_h=GRID_H, grid_w=GRID_W,
    causal_temporal=True, alibi_temporal=True,
).to(device)

decoder = RichOrderDecoder(
    d_state=D_LATENT, d_model=256, n_queries=32,
    n_layers=3, n_heads=8, d_hidden=256, dropout=0.0,
).to(device)

# Load EMA weights
encoder.load_state_dict(ckpt.get("ema_encoder", ckpt["encoder"]))
dit.load_state_dict(ckpt.get("ema_dit", ckpt["dit"]))
decoder.load_state_dict(ckpt.get("ema_decoder", ckpt["decoder"]))
encoder.eval(); dit.eval(); decoder.eval()

noise_sched = NoiseScheduler(1000, "cosine").to(device)
sampler = VideoDDIMSampler(dit, noise_sched, "v_prediction", ddim_steps=20, eta=0.0)

print("\n" + "=" * 72)
print("  Rich Agent Evaluation: Order-Level Generation Quality")
print("=" * 72)
print(f"  Grid: {GRID_H}x{GRID_W}, d_latent={D_LATENT}, n_max_orders={N_MAX_ORDERS}")
print(f"  Sequences: {len(dataset)}")

# ---- Encode first sample as seed ----
sample = dataset[0]
raw_orders = sample["raw_orders"].unsqueeze(0).to(device)   # [1, T, H*W, N_max, 10]
order_masks = sample["order_masks"].unsqueeze(0).to(device)  # [1, T, H*W, N_max]

B, T, C, N, D = raw_orders.shape
flat_orders = raw_orders.reshape(B * T * C, N, D)
flat_masks = order_masks.reshape(B * T * C, N)

with torch.no_grad():
    flat_states = encoder(flat_orders, flat_masks)
    seed_states = flat_states.reshape(1, T, GRID_H, GRID_W, D_LATENT)

# Use first K frames as conditioning
x_cond = seed_states[:, :COND_FRAMES]

# ---- Generate via DDIM ----
N_gen = TOTAL_FRAMES - COND_FRAMES
gen_shape = (1, N_gen, GRID_H, GRID_W, D_LATENT)

with torch.no_grad():
    generated_states = sampler.sample(x_cond, gen_shape, device=device)
    # generated_states: [1, N_gen, H, W, D_LATENT]

    # Decode orders from generated states
    all_states = torch.cat([x_cond, generated_states], dim=1)  # [1, T, H, W, D]
    gen_orders = decoder.decode_sequence(all_states)  # [1, T-1, H*W, 32, 10]

print(f"\n  Generated states shape: {generated_states.shape}")
print(f"  Generated orders shape: {gen_orders.shape}")

# ---- Evaluation metrics ----
gen_orders_np = gen_orders[0].cpu().numpy()  # [T-1, H*W, 32, 10]
T_eval = gen_orders_np.shape[0]
n_cells = GRID_H * GRID_W
n_queries = 32

# Direction accuracy against real data
real_states = seed_states[:, COND_FRAMES:]  # [1, N_gen, H, W, D_LATENT]
real_orders_raw = raw_orders[0, COND_FRAMES:].cpu().numpy()  # [N_gen, H*W, N_max, 10]
real_masks_raw = order_masks[0, COND_FRAMES:].cpu().numpy()   # [N_gen, H*W, N_max]

# Active ratio: fraction of predicted slots with active > 0
active_logits = gen_orders_np[:, :, :, 9]  # [T-1, H*W, 32]
active_probs = 1.0 / (1.0 + np.exp(-np.clip(active_logits, -20, 20)))
pred_active = active_probs > 0.5
active_ratio = pred_active.mean()

# Direction distribution
dir_logits = gen_orders_np[:, :, :, :3]  # [T-1, H*W, 32, 3]
dir_preds = dir_logits.argmax(axis=-1)   # 0=buy, 1=sell, 2=hold
buy_ratio = (dir_preds == 0).mean()
sell_ratio = (dir_preds == 1).mean()
hold_ratio = (dir_preds == 2).mean()

# Price distribution (predicted relative price offset)
price_preds = gen_orders_np[:, :, :, 3]
price_mean = price_preds[pred_active].mean() if pred_active.any() else 0.0
price_std = price_preds[pred_active].std() if pred_active.any() else 0.0

# Size distribution (predicted log size)
size_preds = gen_orders_np[:, :, :, 4]
size_mean = size_preds[pred_active].mean() if pred_active.any() else 0.0
size_std = size_preds[pred_active].std() if pred_active.any() else 0.0

# Direction accuracy: compare with real order directions in the same frames
correct = 0
total = 0
for t in range(min(T_eval, N_gen)):
    for c in range(n_cells):
        real_mask = real_masks_raw[t, c]
        n_real = real_mask.sum()
        if n_real == 0:
            continue
        # Real direction: feature index 2, sign -> class
        real_dirs = real_orders_raw[t, c, :int(n_real), 2]
        # Map: +1 -> 0 (buy), -1 -> 1 (sell), 0 -> 2 (hold)
        real_classes = np.where(real_dirs > 0, 0, np.where(real_dirs < 0, 1, 2))
        # Pick the most common predicted direction for this cell
        if pred_active[t, c].any():
            active_dirs = dir_preds[t, c][pred_active[t, c]]
            if len(active_dirs) > 0:
                pred_majority = np.bincount(active_dirs, minlength=3).argmax()
                # Compare with most common real direction
                real_majority = np.bincount(real_classes.astype(int), minlength=3).argmax()
                if pred_majority == real_majority:
                    correct += 1
                total += 1

dir_accuracy = correct / max(total, 1)

# Real order statistics for comparison
real_active_count = real_masks_raw.sum()
real_total_slots = real_masks_raw.size
real_active_ratio = real_active_count / max(real_total_slots, 1)
real_dirs_all = real_orders_raw[:, :, :, 2][real_masks_raw.astype(bool)]
real_buy = (real_dirs_all > 0).mean() if len(real_dirs_all) > 0 else 0
real_sell = (real_dirs_all < 0).mean() if len(real_dirs_all) > 0 else 0

print("\n--- Generated Order Statistics ---")
print(f"  Active ratio:      {active_ratio:.4f}")
print(f"  Direction dist:    buy={buy_ratio:.3f}, sell={sell_ratio:.3f}, hold={hold_ratio:.3f}")
print(f"  Price offset:      mean={price_mean:.4f}, std={price_std:.4f}")
print(f"  Log size:          mean={size_mean:.4f}, std={size_std:.4f}")

print("\n--- Real Order Statistics (comparison) ---")
print(f"  Active ratio:      {real_active_ratio:.4f}")
print(f"  Direction dist:    buy={real_buy:.3f}, sell={real_sell:.3f}")

print("\n--- Direction Accuracy ---")
print(f"  Majority-vote accuracy: {dir_accuracy:.4f} ({correct}/{total} cells)")

# ---- Per-cell activity heatmap ----
cell_activity = pred_active.mean(axis=(0, 2))  # [H*W]
cell_activity_grid = cell_activity.reshape(GRID_H, GRID_W)
print("\n--- Per-Cell Activity Heatmap (10x10) ---")
for r in range(GRID_H):
    row_str = "  "
    for c in range(GRID_W):
        v = cell_activity_grid[r, c]
        if v > 0.3:
            row_str += "# "
        elif v > 0.1:
            row_str += "o "
        elif v > 0.01:
            row_str += ". "
        else:
            row_str += "  "
    print(row_str)

# ---- Save evaluation results ----
eval_results = {
    "step": int(ckpt["step"]),
    "active_ratio": float(active_ratio),
    "direction_accuracy": float(dir_accuracy),
    "direction_dist": {"buy": float(buy_ratio), "sell": float(sell_ratio), "hold": float(hold_ratio)},
    "price_offset": {"mean": float(price_mean), "std": float(price_std)},
    "log_size": {"mean": float(size_mean), "std": float(size_std)},
    "real_stats": {
        "active_ratio": float(real_active_ratio),
        "buy_ratio": float(real_buy),
        "sell_ratio": float(real_sell),
    },
}
with open(OUT_DIR / "eval_results.json", "w") as f:
    json.dump(eval_results, f, indent=2)
print(f"\n  Saved evaluation to {OUT_DIR / 'eval_results.json'}")

print("\n" + "=" * 72)
print("  Rich Agent Experiment Complete")
print("=" * 72)

PYEOF

echo "=== ALL DONE ==="
