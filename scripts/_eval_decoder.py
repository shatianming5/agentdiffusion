"""Evaluate Order Decoder accuracy and generate LOBSTER-format output.

Uses the enhanced v2 model's decoder to:
1. Generate agent grids from Video DiT
2. Decode to order features
3. Measure decoder accuracy (direction, size, activity)
4. Convert to LOBSTER message format for LOB-Bench
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path

from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.models.order_decoder import AgentToOrderDecoder
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.data.lob_dataset import LOBVideoDataset
from agentdiffusion.infer.interactive_sim import InteractiveSimulator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OB = "data/external/lobster/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
MSG = "data/external/lobster/AMZN_2012-06-21_34200000_57600000_message_10.csv"
VDIT_DIR = Path("outputs/vdit_lob_8x8_enhanced_v2")

# Check if v2 model exists
ckpts = sorted(VDIT_DIR.glob("video_dit_step_*.pt")) if VDIT_DIR.exists() else []
if not ckpts:
    # Fall back to v1
    VDIT_DIR = Path("outputs/vdit_lob_8x8_enhanced")
    ckpts = sorted(VDIT_DIR.glob("video_dit_step_*.pt"))
    if not ckpts:
        print("[ERROR] No enhanced model found")
        import sys; sys.exit(1)

print("=" * 64)
print("  Order Decoder Evaluation + LOBSTER Output")
print("=" * 64)

# Load dataset
dataset = LOBVideoDataset(OB, MSG, total_frames=20, cond_frames=4, subsample=10, grid_shape=(4, 6))
d_latent = dataset[0]["frames"].shape[-1]
print("d_latent={}, grid=(4,6)".format(d_latent))

# Load model + decoder
ckpt_path = ckpts[-1]
model = VideoDiT(
    d_latent=d_latent, d_model=256, depth=8, heads=8,
    patch_size=2, num_frames=20, num_cond_frames=4,
    market_cond_dim=32, grid_h=4, grid_w=6,
    causal_temporal=True, alibi_temporal=True,
).to(device)

decoder = AgentToOrderDecoder(
    d_state=d_latent, d_model=128, n_queries=64,
    n_layers=2, n_heads=4, d_order_out=6,
).to(device)

ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
model.load_state_dict(ckpt.get("ema", ckpt["model"]))
model.eval()
if "decoder" in ckpt:
    decoder.load_state_dict(ckpt["decoder"])
    print("Loaded decoder from checkpoint")
else:
    print("No decoder in checkpoint, using random init")
decoder.eval()

print("Loaded {}".format(ckpt_path))

scheduler = NoiseScheduler(1000, "cosine").to(device)
sampler = VideoDDIMSampler(model, scheduler, "v_prediction", ddim_steps=20, eta=0.0)

# ============================================================
# Part 1: Decoder Accuracy
# ============================================================
print("\n--- Part 1: Decoder Accuracy ---")

N_EVAL = 50
all_pred_orders = []
all_gt_orders = []

with torch.no_grad():
    for i in range(min(N_EVAL, len(dataset))):
        frames = dataset[i]["frames"].unsqueeze(0).to(device)  # [1, T, H, W, C]
        # Decoder on ground truth frames
        gt_orders = decoder.decode_sequence(frames)  # [1, T-1, 64, 6]
        # Generate then decode
        x_cond = frames[:, :4]
        gen_shape = (1, 16, 4, 6, d_latent)
        gen = sampler.sample(x_cond, gen_shape, device=device, zero_sum_proj=True)
        gen_seq = torch.cat([x_cond[:, -1:], gen], dim=1)
        pred_orders = decoder.decode_sequence(gen_seq)

        # Align lengths
        min_t = min(gt_orders.shape[1], pred_orders.shape[1])
        all_gt_orders.append(gt_orders[:, :min_t].cpu())
        all_pred_orders.append(pred_orders[:, :min_t].cpu())

gt_cat = torch.cat(all_gt_orders, dim=0)    # [N, T, 64, 6]
pred_cat = torch.cat(all_pred_orders, dim=0)

# Metrics
# dim 0: price_offset
# dim 1: log_size
# dim 2: direction_logit
# dim 5: activity_logit

# Direction accuracy (dim 2)
gt_dir = (gt_cat[..., 2] > 0).float()
pred_dir = (pred_cat[..., 2] > 0).float()
dir_acc = (gt_dir == pred_dir).float().mean().item()

# Activity accuracy (dim 5)
gt_act = (gt_cat[..., 5] > 0).float()
pred_act = (pred_cat[..., 5] > 0).float()
act_acc = (gt_act == pred_act).float().mean().item()

# Size MAE (dim 1)
size_mae = (gt_cat[..., 1] - pred_cat[..., 1]).abs().mean().item()

# Price MAE (dim 0)
price_mae = (gt_cat[..., 0] - pred_cat[..., 0]).abs().mean().item()

# Overall MSE
overall_mse = F.mse_loss(pred_cat, gt_cat).item()

print("  Direction accuracy:  {:.1f}%".format(dir_acc * 100))
print("  Activity accuracy:   {:.1f}%".format(act_acc * 100))
print("  Size MAE:            {:.4f}".format(size_mae))
print("  Price MAE:           {:.4f}".format(price_mae))
print("  Overall MSE:         {:.6f}".format(overall_mse))

# ============================================================
# Part 2: Generate LOBSTER-format messages
# ============================================================
print("\n--- Part 2: LOBSTER Message Generation ---")

sim = InteractiveSimulator(model, sampler, num_cond=4, num_gen=16, zero_sum_proj=True)
seed = dataset[0]["frames"][:4]
sim.init(seed)

# Generate 5 rounds
all_frames = [seed]
for _ in range(5):
    gen = sim.step()
    all_frames.append(gen.unsqueeze(0))
    sim.trim_buffer(keep_last=8)

# Decode all frames to orders
gen_frames = torch.cat(all_frames, dim=0 if all_frames[0].dim() == 4 else 1)
if gen_frames.dim() == 4:
    gen_frames = gen_frames.unsqueeze(0)

with torch.no_grad():
    orders = decoder.decode_sequence(gen_frames[:, :20].to(device))  # [1, 19, 64, 6]

orders_np = orders[0].cpu().numpy()  # [19, 64, 6]
T_out, N_queries, D_order = orders_np.shape

# Convert to LOBSTER message format
messages = []
base_time = 34200.0  # 9:30 AM
base_price = 2238200  # AMZN-like price in cents*100

for t in range(T_out):
    timestamp = base_time + t * 1.0
    for q in range(N_queries):
        price_offset = orders_np[t, q, 0]
        log_size = orders_np[t, q, 1]
        dir_logit = orders_np[t, q, 2]
        activity = orders_np[t, q, 5]

        # Only emit active orders
        if activity < 0:
            continue

        direction = 1 if dir_logit > 0 else -1
        size = max(int(np.exp(abs(log_size))), 1)
        price = int(base_price + price_offset * 100)
        order_id = t * N_queries + q + 1

        messages.append([timestamp, 1, order_id, size, price, direction])

msg_df = pd.DataFrame(messages, columns=["Time", "Type", "OrderID", "Size", "Price", "Direction"])

out_dir = Path("outputs/lobster_generated")
out_dir.mkdir(parents=True, exist_ok=True)
msg_path = out_dir / "message.csv"
msg_df.to_csv(msg_path, index=False, header=False)

print("  Generated {} LOBSTER messages".format(len(msg_df)))
print("  Saved to {}".format(msg_path))
print("  Price range: {} - {}".format(msg_df["Price"].min(), msg_df["Price"].max()))
print("  Size range:  {} - {}".format(msg_df["Size"].min(), msg_df["Size"].max()))
print("  Buy/Sell:    {:.1f}% / {:.1f}%".format(
    (msg_df["Direction"] == 1).mean() * 100,
    (msg_df["Direction"] == -1).mean() * 100))

print("\n" + "=" * 64)
print("  DONE")
print("=" * 64)
