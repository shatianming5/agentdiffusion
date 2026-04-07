"""COVID Crash Reproduction: compare normal vs crash vs recovery day statistics,
then train on normal day and see if generated dynamics differ from crash reality."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, logging
from pathlib import Path
from tqdm import tqdm
from agentdiffusion.data.ashare_10k_agents import AShare10KAgentDataset
from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.infer.interactive_sim import InteractiveSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("  COVID Crash Reproduction")
print("  Normal(20240619) vs Crash(20200309) vs Recovery(20200310)")
print("=" * 60)

# Load 3 days with K=1000, grid=32x32
days = {
    "Normal (20240619)": "data/external/20240619",
    "Crash (20200309)": "data/external/20200309",
    "Recovery (20200310)": "data/external/20200310",
}

datasets = {}
for name, path in days.items():
    if not Path(path).exists():
        print(f"  [SKIP] {name}: {path} not found")
        continue
    print(f"\nLoading {name}...")
    ds = AShare10KAgentDataset(
        path, total_frames=20, cond_frames=4,
        window_seconds=5.0, max_stocks=30,
        n_clusters=1000, grid_h=32, grid_w=32)
    datasets[name] = ds
    print(f"  {name}: {len(ds)} sequences")

if len(datasets) < 2:
    print("Need at least 2 days. Exiting.")
    import sys; sys.exit(1)

# Compare real statistics
def day_stats(ds):
    n = min(50, len(ds))
    frames = torch.stack([ds[i]["frames"] for i in range(n)])
    sv = frames[:, :, :, :, 0]
    return {
        "mean_sv": sv.mean().item(),
        "std_sv": sv.std().item(),
        "volatility": sv.std(dim=(2, 3)).mean().item(),
        "activity": (frames.abs() > 0.1).float().mean().item(),
        "cancel_rate": frames[:, :, :, :, 3].mean().item() if frames.shape[-1] > 3 else 0,
        "buy_ratio": frames[:, :, :, :, 5].mean().item() if frames.shape[-1] > 5 else 0.5,
    }

print("\n" + "=" * 70)
print("  Cross-Day Real Data Comparison")
print("=" * 70)
fmt = "{:<25s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}"
print(fmt.format("Day", "Mean SV", "Volatility", "Activity", "Cancel", "Buy%"))
print("-" * 75)

all_stats = {}
for name, ds in datasets.items():
    s = day_stats(ds)
    all_stats[name] = s
    print(fmt.format(name,
                     f"{s['mean_sv']:+.4f}",
                     f"{s['volatility']:.4f}",
                     f"{s['activity']:.3f}",
                     f"{s['cancel_rate']:.4f}",
                     f"{s['buy_ratio']:.3f}"))

# Train on normal day
normal_name = [k for k in datasets if "Normal" in k][0]
ds_normal = datasets[normal_name]

print(f"\nTraining Video DiT on {normal_name} (3000 steps)...")
model = VideoDiT(
    d_latent=6, d_model=128, depth=6, heads=4,
    patch_size=4, num_frames=20, num_cond_frames=4,
    market_cond_dim=32, grid_h=32, grid_w=32,
    causal_temporal=True).to(device)

scheduler = NoiseScheduler(1000, "cosine").to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loader = torch.utils.data.DataLoader(ds_normal, batch_size=8, shuffle=True, drop_last=True)

K = 4
step = 0
pbar = tqdm(total=3000, desc="Training")
while step < 3000:
    for batch in loader:
        if step >= 3000:
            break
        frames = batch["frames"].to(device)
        B, T, H, W, C = frames.shape
        N = T - K
        z_cond, z_gen = frames[:, :K], frames[:, K:]
        t = torch.randint(0, 1000, (B,), device=device)
        noise = torch.randn_like(z_gen)
        t_exp = t.unsqueeze(1).expand(B, N).reshape(B * N)
        z_noisy = scheduler.q_sample(
            z_gen.reshape(B*N, H, W, C), t_exp, noise.reshape(B*N, H, W, C)
        ).reshape(B, N, H, W, C)
        v_pred = model(z_cond, z_noisy, t)
        v_target = scheduler.v_target(
            z_gen.reshape(B*N, H, W, C), noise.reshape(B*N, H, W, C), t_exp
        ).reshape(B, N, H, W, C)
        loss = F.mse_loss(v_pred, v_target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        pbar.update(1)
        if step % 500 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

pbar.close()
model.eval()
print(f"Training done, loss={loss.item():.4f}")

# Generate from normal seed
sampler = VideoDDIMSampler(model, scheduler, "v_prediction", ddim_steps=20, eta=0.0)
sim = InteractiveSimulator(model, sampler, num_cond=4, num_gen=16, zero_sum_proj=True)

seed = ds_normal[len(ds_normal) // 2]["frames"][:4]
sim.init(seed)

gen_frames = []
for _ in range(5):
    gen = sim.step()
    gen_frames.append(gen.cpu())
    sim.trim_buffer(keep_last=8)
gen_all = torch.cat(gen_frames, dim=0)

gen_stats = {
    "volatility": gen_all[:, :, :, 0].std(dim=(1, 2)).mean().item(),
    "activity": (gen_all.abs() > 0.1).float().mean().item(),
    "cancel_rate": gen_all[:, :, :, 3].mean().item() if gen_all.shape[-1] > 3 else 0,
    "buy_ratio": gen_all[:, :, :, 5].mean().item() if gen_all.shape[-1] > 5 else 0.5,
}

print("\n" + "=" * 70)
print("  Model Generation vs Real Days")
print("=" * 70)
print(fmt.format("Source", "Mean SV", "Volatility", "Activity", "Cancel", "Buy%"))
print("-" * 75)
for name, s in all_stats.items():
    print(fmt.format(name, f"{s['mean_sv']:+.4f}", f"{s['volatility']:.4f}",
                     f"{s['activity']:.3f}", f"{s['cancel_rate']:.4f}", f"{s['buy_ratio']:.3f}"))
print(fmt.format("Generated (model)",
                 "N/A",
                 f"{gen_stats['volatility']:.4f}",
                 f"{gen_stats['activity']:.3f}",
                 f"{gen_stats['cancel_rate']:.4f}",
                 f"{gen_stats['buy_ratio']:.3f}"))
print("=" * 70)
print("\nDone.")
