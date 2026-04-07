#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "============================================================"
echo "  K-Ablation: Agent Clustering Scale Study"
echo "  K in {100, 500, 1000, 5000, 10000, 50000}"
echo "============================================================"

.venv/bin/python3 -u << 'PYEOF'
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, logging, time, json, os, gc, sys, traceback
from pathlib import Path
from tqdm import tqdm
from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.data.ashare_10k_agents import AShare10KAgentDataset, D_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Ablation configurations: K -> (grid_h, grid_w, patch_size, batch_size, d_model, depth)
# ============================================================
CONFIGS = [
    # (K, grid_h, grid_w, patch_size, batch_size, d_model, depth)
    (100,    10,  10,   2,  16, 256,  8),
    (500,    25,  20,   5,  16, 256,  8),
    (1000,   32,  32,   4,   8, 256,  8),
    (5000,   72,  72,   8,   8, 256,  8),   # 72x72=5184 >= 5000
    (10000, 100, 100,  10,   4, 256,  8),
    (50000, 224, 224,  16,   4, 256,  8),   # 224x224=50176 >= 50000
]

DATA_DIR     = "data/external/20240619"
MAX_STOCKS   = 30
TOTAL_STEPS  = 5000
TOTAL_FRAMES = 20
COND_FRAMES  = 4
NUM_GEN      = TOTAL_FRAMES - COND_FRAMES
D_LATENT     = D_STATE  # 6

BASE_OUT = Path("outputs/k_ablation")
BASE_OUT.mkdir(parents=True, exist_ok=True)

all_results = []

# ============================================================
# Helper: compute stylized facts from generated sequences
# ============================================================
def compute_stylized_facts(sequences: list[torch.Tensor]) -> dict:
    """Compute kurtosis, ACF-1, ACF-5 from net signed volume (dim 0).

    sequences: list of [N, H, W, d_latent] tensors (generation frames).
    We aggregate across all spatial positions to get a 1D time-series.
    """
    # Concatenate all generation sequences along time
    all_vols = []
    for seq in sequences:
        # seq: [N, H, W, D] -> net volume = dim 0, sum over spatial
        vol = seq[:, :, :, 0].sum(dim=(1, 2))  # [N]
        all_vols.append(vol)
    ts = torch.cat(all_vols, dim=0).numpy()  # [total_T]

    if len(ts) < 10:
        return {"kurtosis": float("nan"), "acf1": float("nan"), "acf5": float("nan")}

    # Returns (log-difference to mimic returns)
    returns = np.diff(ts)
    if returns.std() < 1e-12:
        return {"kurtosis": 0.0, "acf1": 0.0, "acf5": 0.0}

    # Kurtosis (excess)
    r_centered = returns - returns.mean()
    r_std = returns.std()
    kurtosis = float(np.mean((r_centered / r_std) ** 4) - 3.0)

    # ACF at lag k
    def acf(x, lag):
        n = len(x)
        if n <= lag:
            return 0.0
        xm = x - x.mean()
        c0 = np.sum(xm ** 2)
        if c0 < 1e-12:
            return 0.0
        ck = np.sum(xm[:n - lag] * xm[lag:])
        return float(ck / c0)

    # ACF of absolute returns (volatility clustering)
    abs_ret = np.abs(returns)
    acf1 = acf(abs_ret, 1)
    acf5 = acf(abs_ret, 5)

    return {"kurtosis": kurtosis, "acf1": acf1, "acf5": acf5}


# ============================================================
# Main ablation loop
# ============================================================
for K, grid_h, grid_w, patch_size, batch_size, d_model, depth in CONFIGS:
    run_name = f"K{K}_{grid_h}x{grid_w}_p{patch_size}"
    out_dir = BASE_OUT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"K": K, "grid": f"{grid_h}x{grid_w}", "patch_size": patch_size,
              "batch_size": batch_size, "status": "FAILED"}

    logger.info("=" * 64)
    logger.info("START K=%d, grid=%dx%d, patch=%d, bs=%d",
                K, grid_h, grid_w, patch_size, batch_size)
    logger.info("=" * 64)

    try:
        # --- Dataset ---
        t0_data = time.time()
        dataset = AShare10KAgentDataset(
            DATA_DIR,
            total_frames=TOTAL_FRAMES,
            cond_frames=COND_FRAMES,
            window_seconds=5.0,
            max_stocks=MAX_STOCKS,
            n_clusters=K,
            grid_h=grid_h,
            grid_w=grid_w,
        )
        data_time = time.time() - t0_data
        logger.info("Dataset built in %.1fs: %d sequences", data_time, len(dataset))

        if len(dataset) == 0:
            logger.error("No data for K=%d, skipping", K)
            result["status"] = "NO_DATA"
            all_results.append(result)
            continue

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, drop_last=True,
        )

        # --- Model ---
        # Verify grid is divisible by patch_size
        assert grid_h % patch_size == 0, f"grid_h={grid_h} not divisible by patch_size={patch_size}"
        assert grid_w % patch_size == 0, f"grid_w={grid_w} not divisible by patch_size={patch_size}"

        model = VideoDiT(
            d_latent=D_LATENT,
            d_model=d_model,
            depth=depth,
            heads=8,
            patch_size=patch_size,
            num_frames=TOTAL_FRAMES,
            num_cond_frames=COND_FRAMES,
            mlp_ratio=4.0,
            market_cond_dim=32,
            grid_h=grid_h,
            grid_w=grid_w,
            causal_temporal=True,
            alibi_temporal=True,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("Model params: %.2fM", n_params / 1e6)
        result["params_M"] = round(n_params / 1e6, 2)

        # Memory snapshot after model creation
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        scheduler = NoiseScheduler(1000, "cosine").to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)

        # EMA
        ema_state = {k: v.clone() for k, v in model.state_dict().items()}
        ema_decay = 0.999

        def update_ema():
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema_state[k].lerp_(v, 1 - ema_decay)

        # --- Training ---
        model.train()
        step = 0
        loss_history = []
        t0_train = time.time()
        pbar = tqdm(total=TOTAL_STEPS, desc=f"K={K}")

        while step < TOTAL_STEPS:
            for batch in loader:
                if step >= TOTAL_STEPS:
                    break

                frames = batch["frames"].to(device)  # [B, T, H, W, D]
                B, T, H_f, W_f, C = frames.shape
                N = T - COND_FRAMES
                z_cond = frames[:, :COND_FRAMES]
                z_gen  = frames[:, COND_FRAMES:]

                t_diff = torch.randint(0, 1000, (B,), device=device)
                noise = torch.randn_like(z_gen)
                t_exp = t_diff.unsqueeze(1).expand(B, N).reshape(B * N)
                z_noisy = scheduler.q_sample(
                    z_gen.reshape(B * N, H_f, W_f, C),
                    t_exp,
                    noise.reshape(B * N, H_f, W_f, C),
                ).reshape(B, N, H_f, W_f, C)

                v_pred = model(z_cond, z_noisy, t_diff)
                v_target = scheduler.v_target(
                    z_gen.reshape(B * N, H_f, W_f, C),
                    noise.reshape(B * N, H_f, W_f, C),
                    t_exp,
                ).reshape(B, N, H_f, W_f, C)

                loss = F.mse_loss(v_pred, v_target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_sched.step()
                update_ema()

                step += 1
                loss_val = loss.item()
                loss_history.append(loss_val)
                pbar.update(1)

                if step % 200 == 0:
                    pbar.set_postfix(loss=f"{loss_val:.4f}",
                                     lr=f"{lr_sched.get_last_lr()[0]:.2e}")
                if step % 1000 == 0:
                    torch.save(
                        {"model": model.state_dict(), "ema": ema_state, "step": step},
                        out_dir / f"ckpt_step{step}.pt",
                    )

        pbar.close()
        train_time = time.time() - t0_train
        logger.info("K=%d training done in %.1fs (%.2f steps/s)",
                     K, train_time, TOTAL_STEPS / train_time)

        # Save final
        torch.save(
            {"model": model.state_dict(), "ema": ema_state, "step": step},
            out_dir / "ckpt_final.pt",
        )

        # Record training metrics
        result["train_time_s"]   = round(train_time, 1)
        result["steps_per_sec"]  = round(TOTAL_STEPS / train_time, 3)
        result["final_loss"]     = round(float(np.mean(loss_history[-100:])), 5)
        result["loss_history"]   = [round(float(x), 5) for x in loss_history[::100]]

        # Peak GPU memory during training
        if torch.cuda.is_available():
            peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
            result["peak_gpu_mb"] = round(peak_mem_mb, 1)
        else:
            result["peak_gpu_mb"] = 0.0

        # --- Evaluation: generation speed + stylized facts ---
        logger.info("Evaluating K=%d ...", K)
        model.load_state_dict(ema_state)
        model.eval()

        sampler = VideoDDIMSampler(model, scheduler, "v_prediction",
                                    ddim_steps=20, eta=0.0)

        # Measure generation speed: generate 4 batches
        gen_sequences = []
        t0_gen = time.time()
        n_gen_batches = min(4, len(dataset) // batch_size)
        if n_gen_batches < 1:
            n_gen_batches = 1

        for gi in range(n_gen_batches):
            idx = gi % len(dataset)
            seed_frames = dataset[idx]["frames"][:COND_FRAMES].unsqueeze(0).to(device)
            gen_shape = (1, NUM_GEN, grid_h, grid_w, D_LATENT)
            with torch.no_grad():
                gen_out = sampler.sample(seed_frames, gen_shape)  # [1, N, H, W, D]
            gen_sequences.append(gen_out.squeeze(0).cpu())

        gen_time = time.time() - t0_gen
        gen_per_sec = n_gen_batches / gen_time
        result["gen_time_s"]     = round(gen_time, 2)
        result["gen_per_sec"]    = round(gen_per_sec, 3)
        result["n_gen_seqs"]     = n_gen_batches

        # Stylized facts
        facts = compute_stylized_facts(gen_sequences)
        result["kurtosis"] = round(facts["kurtosis"], 4)
        result["acf1"]     = round(facts["acf1"], 4)
        result["acf5"]     = round(facts["acf5"], 4)

        # Spatial structure check on last generated frame
        last_frame = gen_sequences[-1][-1]  # [H, W, D]
        active_pct = (last_frame.abs() > 0.01).any(dim=-1).float().mean().item() * 100
        spatial_std = last_frame[:, :, 0].std().item()
        result["active_pct"]   = round(active_pct, 1)
        result["spatial_std"]  = round(spatial_std, 4)

        result["status"] = "OK"
        logger.info("K=%d eval done: kurtosis=%.3f, acf1=%.3f, acf5=%.3f",
                     K, facts["kurtosis"], facts["acf1"], facts["acf5"])

    except Exception as e:
        logger.error("K=%d FAILED: %s", K, e)
        traceback.print_exc()
        result["error"] = str(e)

    finally:
        # Save per-K result
        with open(out_dir / "result.json", "w") as f:
            # Filter out loss_history for per-file (too large)
            r_save = {k: v for k, v in result.items() if k != "loss_history"}
            json.dump(r_save, f, indent=2)
        all_results.append(result)

        # Aggressive cleanup
        for name in ["model", "optimizer", "lr_sched", "scheduler",
                      "dataset", "loader", "ema_state", "sampler"]:
            if name in dir():
                try:
                    del locals()[name]
                except:
                    pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ============================================================
# Summary comparison table
# ============================================================
print("\n" + "=" * 100)
print("  K-ABLATION COMPARISON TABLE")
print("=" * 100)

header = f"{'K':>7} | {'Grid':>9} | {'Patch':>5} | {'Params':>7} | {'Loss':>8} | " \
         f"{'Kurt':>7} | {'ACF1':>7} | {'ACF5':>7} | " \
         f"{'GPU MB':>8} | {'Train(s)':>9} | {'Gen/s':>6} | {'Active%':>8} | {'Status':>6}"
print(header)
print("-" * 100)

for r in all_results:
    line = (
        f"{r.get('K','?'):>7} | "
        f"{r.get('grid','?'):>9} | "
        f"{r.get('patch_size','?'):>5} | "
        f"{r.get('params_M','?'):>6}M | "
        f"{r.get('final_loss','n/a'):>8} | "
        f"{r.get('kurtosis','n/a'):>7} | "
        f"{r.get('acf1','n/a'):>7} | "
        f"{r.get('acf5','n/a'):>7} | "
        f"{r.get('peak_gpu_mb','n/a'):>8} | "
        f"{r.get('train_time_s','n/a'):>9} | "
        f"{r.get('gen_per_sec','n/a'):>6} | "
        f"{r.get('active_pct','n/a'):>7}% | "
        f"{r.get('status','?'):>6}"
    )
    print(line)

print("=" * 100)

# Save full results
summary_path = BASE_OUT / "ablation_summary.json"
with open(summary_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
logger.info("Full results saved to %s", summary_path)

# Print quick interpretation
print("\nInterpretation guide:")
print("  Kurtosis > 3   : heavy tails (realistic for financial returns)")
print("  ACF1 > 0.1     : volatility clustering present (good)")
print("  ACF5 > 0.05    : longer-range vol persistence (good)")
print("  Active% > 50   : non-trivial spatial structure preserved")
print("  Lower loss      : better reconstruction (but check for overfitting)")
print("=" * 100)
PYEOF
