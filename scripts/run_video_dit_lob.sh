#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "============================================================"
echo "  Video DiT on Real NASDAQ LOB (LOBSTER AMZN)"
echo "============================================================"

.venv/bin/python3 -u << 'PYEOF'
import sys, os, time, logging, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

# ============================================================
# Phase 1: Load LOB data
# ============================================================
logger.info("=== Phase 1: Load LOB Data ===")
from agentdiffusion.data.lob_dataset import LOBVideoDataset

OB_PATH = "data/external/lobster/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
MSG_PATH = "data/external/lobster/AMZN_2012-06-21_34200000_57600000_message_10.csv"

dataset = LOBVideoDataset(OB_PATH, MSG_PATH, total_frames=20, cond_frames=4, subsample=10, grid_shape=(6, 8))
logger.info(f"Dataset: {len(dataset)} sequences")

loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)

# ============================================================
# Phase 2: Build Video DiT (tiny, for LOB)
# ============================================================
logger.info("=== Phase 2: Build Model ===")
from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.diffusion.scheduler import NoiseScheduler

# LOB: d_latent=1 (raw features, no AE), grid=6x8, patch_size=2
model = VideoDiT(
    d_latent=1, d_model=128, depth=6, heads=4,
    patch_size=2, num_frames=20, num_cond_frames=4,
    mlp_ratio=4.0, market_cond_dim=32,
    grid_h=6, grid_w=8,
).to(device)
logger.info(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

scheduler = NoiseScheduler(1000, "cosine").to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20000)

# ============================================================
# Phase 3: Train (20K steps, directly on LOB frames — no AE needed)
# ============================================================
logger.info("=== Phase 3: Train Video DiT (20K steps) ===")
model.train()
global_step = 0
TOTAL_STEPS = 20000
os.makedirs("outputs/video_dit_lob", exist_ok=True)

pbar = tqdm(total=TOTAL_STEPS, desc="LOB Video DiT")
while global_step < TOTAL_STEPS:
    for batch in loader:
        if global_step >= TOTAL_STEPS:
            break

        frames = batch["frames"].to(device)  # [B, 20, 6, 8, 1]
        B = frames.shape[0]

        # Split condition and generation frames
        z_cond = frames[:, :4]     # [B, 4, 6, 8, 1] clean
        z_gen = frames[:, 4:]      # [B, 16, 6, 8, 1] to noise

        # Sample timestep, add noise
        t = torch.randint(0, 1000, (B,), device=device)

        # Expand t for multi-frame v_target computation
        noise = torch.randn_like(z_gen)

        # q_sample needs [B, ...] shaped t — broadcast across frames
        z_gen_flat = z_gen.reshape(B, -1)
        noise_flat = noise.reshape(B, -1)
        z_noisy_flat = scheduler.q_sample(z_gen_flat, t, noise_flat)
        z_noisy = z_noisy_flat.reshape_as(z_gen)

        # Forward
        v_pred = model(z_cond, z_noisy, t)

        # v-prediction target
        v_target_flat = scheduler.v_target(z_gen_flat, noise_flat, t)
        v_target = v_target_flat.reshape_as(z_gen)

        loss = F.mse_loss(v_pred, v_target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

        global_step += 1
        pbar.update(1)

        if global_step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")

        if global_step % 10000 == 0:
            torch.save({"model": model.state_dict(), "step": global_step},
                       f"outputs/video_dit_lob/step_{global_step}.pt")

pbar.close()
torch.save({"model": model.state_dict(), "step": global_step},
           f"outputs/video_dit_lob/step_{global_step}.pt")

# ============================================================
# Phase 4: Evaluate — generate LOB sequences, extract prices
# ============================================================
logger.info("=== Phase 4: Evaluate ===")
model.eval()
sampler = VideoDDIMSampler(model, scheduler, ddim_steps=50)

from agentdiffusion.eval.stylized_facts import evaluate_stylized_facts

TRIALS = 10
scores = []

for trial in range(TRIALS):
    torch.manual_seed(trial * 42)
    # Take a real 4-frame condition from the dataset
    start_idx = (len(dataset) // (TRIALS + 1)) * (trial + 1)
    sample = dataset[start_idx]
    cond = sample["frames"][:4].unsqueeze(0).to(device)  # [1, 4, 6, 8, 1]

    # Generate sequence: sliding window approach
    # Generate 16 frames, take last 4 as new condition, repeat
    all_frames = [cond.squeeze(0)]  # list of [4, 6, 8, 1]

    with torch.no_grad():
        for chunk in range(30):  # 30 chunks × 16 frames = 480 generated frames
            gen_shape = (1, 16, 6, 8, 1)
            generated = sampler.sample(cond, gen_shape)  # [1, 16, 6, 8, 1]
            all_frames.append(generated.squeeze(0))
            # Use last 4 generated frames as next condition
            cond = generated[:, -4:]

    # Concatenate all frames
    full_seq = torch.cat(all_frames, dim=0)  # [4 + 30*16, 6, 8, 1] = [484, 6, 8, 1]

    # Extract "price" proxy: feature dim 2 in original (mid_price_normalized)
    # After grid reshape and normalization, we need to find the right position
    # The global features start at index 40 (4*10 levels), mid_norm is at index 42
    # In the 6x8=48 grid: position 42 = row 5, col 2
    # But features are normalized — use the mean of all features as a proxy
    price_proxy = full_seq.cpu().numpy().mean(axis=(1, 2, 3))  # [T]
    # Convert to price-like series (cumulative sum of "returns")
    price_series = np.exp(np.cumsum(price_proxy * 0.001)) * 100  # arbitrary scale

    sf = evaluate_stylized_facts(price_series)
    scores.append(sf.total_passed)

    ft = 'Y' if sf.fat_tail_pass else 'N'
    vc = 'Y' if sf.volatility_clustering_pass else 'N'
    le = 'Y' if sf.leverage_effect_pass else 'N'
    ra = 'Y' if sf.return_autocorr_pass else 'N'
    gl = 'Y' if sf.gain_loss_asymmetry_pass else 'N'
    print(f"  Trial {trial}: {sf.total_passed}/6  FT={ft} VC={vc} LE={le} RA={ra} GL={gl}  alpha={sf.fat_tail_alpha:.2f}")

scores = np.array(scores)
print(f"\n=== SUMMARY ===")
print(f"Stylized Facts: {scores.mean():.2f} +/- {scores.std():.2f} / 6")
print(f"Best: {scores.max()}/6")

# Speed benchmark
logger.info("=== Speed Benchmark ===")
cond = dataset[0]["frames"][:4].unsqueeze(0).to(device)
if device.type == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(10):
        _ = sampler.sample(cond, (1, 16, 6, 8, 1))
if device.type == "cuda":
    torch.cuda.synchronize()
elapsed = (time.perf_counter() - t0) / 10
print(f"Generate 16 LOB frames: {elapsed*1000:.1f}ms ({elapsed/16*1000:.1f}ms per frame)")

print("\n=== DONE ===")
PYEOF
