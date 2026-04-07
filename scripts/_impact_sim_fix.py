"""Institutional order impact simulation — uses vdit_ashare_l3 model."""
import torch
import numpy as np
import logging
from pathlib import Path
from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.data.ashare_l3_dataset import AShareL3VideoDataset
from agentdiffusion.infer.interactive_sim import InteractiveSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset + model (A-Share L3, 4x4 grid)
dataset = AShareL3VideoDataset(
    "data/external/20240619", total_frames=20, cond_frames=4,
    window_seconds=1.0, grid_shape=(4, 4), max_stocks=50)
d_latent = dataset[0]["frames"].shape[-1]
mc_dim = dataset[0]["market_conds"].shape[-1]

model = VideoDiT(
    d_latent=d_latent, d_model=128, depth=6, heads=4,
    patch_size=2, num_frames=20, num_cond_frames=4,
    market_cond_dim=mc_dim, grid_h=4, grid_w=4,
    causal_temporal=True).to(device)
ckpt = torch.load("outputs/vdit_ashare_l3/video_dit_step_20000.pt",
                   map_location=device, weights_only=True)
model.load_state_dict(ckpt.get("ema", ckpt["model"]))
model.eval()
print(f"Loaded model: d_latent={d_latent}, grid=4x4")

scheduler = NoiseScheduler(1000, "cosine").to(device)
sampler = VideoDDIMSampler(model, scheduler, "v_prediction", ddim_steps=20, eta=0.0)

seed = dataset[0]["frames"][:4]

def run_strategy(name, shocks):
    """Run simulation with given shock schedule.
    shocks: list of (round_idx, delta_value) pairs.
    """
    sim = InteractiveSimulator(model, sampler, num_cond=4, num_gen=16, zero_sum_proj=True)
    sim.init(seed)

    price_series = []  # dim 2 = avg_price proxy
    vol_series = []

    for r in range(20):
        gen = sim.step()  # [16, 4, 4, d_latent]
        price = gen[:, :, :, 2].mean().item() if d_latent > 2 else gen[:, :, :, 0].mean().item()
        vol = gen.std().item()
        price_series.append(price)
        vol_series.append(vol)

        # Apply shocks
        for shock_round, delta_val in shocks:
            if r == shock_round:
                shock = torch.zeros(d_latent)
                shock[0] = delta_val  # signed volume shock
                sim.intervene(frame_idx=-1, delta=shock)

        sim.trim_buffer(keep_last=8)

    return np.array(price_series), np.array(vol_series)

# Define strategies
strategies = {
    "Baseline (no trade)": [],
    "Block trade": [(5, -5.0)],
    "Split-10": [(5+i, -0.5) for i in range(10)],
    "TWAP": [(5+i, v) for i, v in enumerate([-0.7,-0.6,-0.5,-0.5,-0.4,-0.4,-0.3,-0.3,-0.2,-0.1])],
}

print("\n" + "=" * 70)
print("  Institutional Order Impact Simulation")
print("=" * 70)

results = {}
for name, shocks in strategies.items():
    prices, vols = run_strategy(name, shocks)
    pre = prices[:5].mean()
    post = prices[5:10].mean()
    recovery_idx = 20  # default: no recovery
    if pre != 0:
        for i in range(6, 20):
            if abs(prices[i] - pre) / (abs(pre) + 1e-8) < 0.1:
                recovery_idx = i - 5
                break
    impact = (post - pre) / (abs(pre) + 1e-8) * 100
    disruption = vols[5:15].mean() / (vols[:5].mean() + 1e-8)

    results[name] = {
        "impact_pct": impact,
        "recovery_rounds": recovery_idx,
        "disruption": disruption,
        "post_vol": vols[5:15].mean(),
    }
    print(f"\n  {name}:")
    print(f"    Price impact:    {impact:+.2f}%")
    print(f"    Recovery:        {recovery_idx} rounds")
    print(f"    Disruption:      {disruption:.2f}x baseline vol")

# Summary table
print("\n" + "=" * 70)
fmt = "{:<25s} {:>12s} {:>12s} {:>12s}"
print(fmt.format("Strategy", "Impact(%)", "Recovery", "Disruption"))
print("-" * 70)
for name, r in results.items():
    print(fmt.format(name, f"{r['impact_pct']:+.2f}", f"{r['recovery_rounds']}", f"{r['disruption']:.2f}x"))
print("=" * 70)
