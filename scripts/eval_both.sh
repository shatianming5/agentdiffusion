#!/bin/bash
set -Eeuo pipefail
cd ~/agentdiffusion
export PYTHONPATH=$PWD:${PYTHONPATH:-}

echo "============================================"
echo "  Unified Evaluation: LeWM v3.1 vs DiT v3"
echo "  1000-step rollout, masked price extraction"
echo "============================================"

.venv/bin/python3 << 'PYEOF'
import sys, torch, numpy as np, time
from pathlib import Path

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

from agentdiffusion.data.dataset import _pad_grid
from agentdiffusion.utils.masked import masked_mean
from agentdiffusion.eval.stylized_facts import evaluate_stylized_facts
from agentdiffusion.eval.emergence import run_emergence_analysis
from agentdiffusion.constraints.projection import apply_all_projections

# Load test data — get a sequence from same sim_id
files = sorted(Path("data/abides_norm").glob("*.pt"))
# Group by sim_id to get time-varying market_cond
test_samples = []
for f in files[-100:]:
    s = torch.load(f, map_location="cpu", weights_only=True)
    test_samples.append(s)

# Use first sample as initial state
sample0 = test_samples[0]
state_init = _pad_grid(sample0["state_t"], 36, 36).unsqueeze(0).to(device)
agent_types = _pad_grid(sample0["agent_types"], 36, 36)
valid = (agent_types != -1).to(device)  # [36, 36]

# Collect market_conds from sequence
mc_list = [s["market_cond"].to(device) for s in test_samples]

ROLLOUT_STEPS = 1000

def extract_price(state, valid_mask):
    """Extract masked price from state dim 98."""
    p = state[0, :, :, 98]
    return (p * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1)

# ============================================================
# Model 1: LeWM v3.1
# ============================================================
print("\n" + "="*60)
print("  MODEL 1: LeWorldModel v3.1")
print("="*60)

from agentdiffusion.models.lewm import LeWorldModel

lewm = LeWorldModel(
    d_agent=128, d_enc=256, d_latent=256, d_pred=384, d_cond=32,
    patch_size=4, enc_depth=6, enc_heads=8, pred_depth=6, pred_heads=8,
    use_decoder=True, d_dec=256, dec_depth=4, dec_heads=4,
    lambda_price=1.0, lambda_returns=1.0, beta_leverage=0.5,
    lambda_sigreg=0.5, lambda_recon=1.0, dec_grid_h=36, dec_grid_w=36,
).to(device)

ckpt = torch.load("outputs/lewm_full/lewm_step_50000.pt", map_location=device, weights_only=True)
lewm.load_state_dict(ckpt["model"])
lewm.eval()
print(f"Loaded LeWM checkpoint (50K steps)")

# Rollout
lewm_prices = []
state_cur = state_init
with torch.no_grad():
    z = lewm.encode(state_cur)
    lewm_prices.append(extract_price(state_cur, valid).item())

    for step in range(ROLLOUT_STEPS):
        mc_idx = step % len(mc_list)
        mc = mc_list[mc_idx].unsqueeze(0)
        z = lewm.predict(z, mc, stochastic=True)
        decoded = lewm.decode(z)
        decoded = apply_all_projections(state_cur, decoded)
        lewm_prices.append(extract_price(decoded, valid).item())
        state_cur = decoded

lewm_prices = np.array(lewm_prices)
lewm_prices = np.clip(lewm_prices, 0.01, None)
print(f"Rollout: {len(lewm_prices)} steps, range=[{lewm_prices.min():.4f}, {lewm_prices.max():.4f}], std={lewm_prices.std():.6f}")

# Speedup
torch.cuda.synchronize()
t0 = time.perf_counter()
z = lewm.encode(state_init)
for _ in range(100):
    z = lewm.predict(z, mc_list[0].unsqueeze(0), stochastic=True)
    _ = lewm.decode(z)
torch.cuda.synchronize()
lewm_speed = (time.perf_counter() - t0) / 100 * 1000
print(f"Speed: {lewm_speed:.2f} ms/step (encode+predict+decode)")

lewm_sf = evaluate_stylized_facts(lewm_prices)
print(f"\nStylized Facts: {lewm_sf.summary}")
print(f"  Fat tail alpha:    {lewm_sf.fat_tail_alpha:.4f}  {'PASS' if lewm_sf.fat_tail_pass else 'FAIL'}")
print(f"  Vol clustering:    {'PASS' if lewm_sf.volatility_clustering_pass else 'FAIL'}")
print(f"  Leverage effect:   {lewm_sf.leverage_effect_corr:.4f}  {'PASS' if lewm_sf.leverage_effect_pass else 'FAIL'}")
print(f"  Return autocorr:   {'PASS' if lewm_sf.return_autocorr_pass else 'FAIL'}")
print(f"  Gain/loss asym:    {'PASS' if lewm_sf.gain_loss_asymmetry_pass else 'FAIL'}")

lewm_events = run_emergence_analysis(lewm_prices)
for k, v in lewm_events.items():
    if v: print(f"  {k}: {len(v)} events")

# ============================================================
# Model 2: Diffusion DiT v3
# ============================================================
print("\n" + "="*60)
print("  MODEL 2: Diffusion DiT v3")
print("="*60)

device2 = torch.device("cuda:4" if torch.cuda.is_available() else device)

from agentdiffusion.models.autoencoder import AgentAutoencoder
from agentdiffusion.models.agent_dit import AgentDiT
from agentdiffusion.diffusion.scheduler import NoiseScheduler
from agentdiffusion.diffusion.ddim import DDIMSampler

ae = AgentAutoencoder(raw_dim=128, latent_dim=16).to(device2)
ae_ckpt = torch.load("outputs/ae_norm/ae_step_10000.pt", map_location=device2, weights_only=True)
ae.load_state_dict(ae_ckpt["model"])
ae.eval()

dit_path = "outputs/dit_v3/dit_step_50000.pt"
if not Path(dit_path).exists():
    dit_path = "outputs/dit_v3/dit_step_25000.pt"
    print(f"Using intermediate checkpoint: {dit_path}")

dit = AgentDiT(
    raw_dim=128, latent_dim=16, d_model=512, depth=12, heads=8,
    patch_size=4, num_market_tokens=64, local_window_size=4,
).to(device2)
dit_ckpt = torch.load(dit_path, map_location=device2, weights_only=True)
dit.load_state_dict(dit_ckpt["model"])
dit.eval()

sched = NoiseScheduler(1000, "cosine").to(device2)
sampler = DDIMSampler(dit, sched, "v_prediction", ddim_steps=50)

state_init_d = state_init.to(device2)
valid_d = valid.to(device2)
print(f"Loaded DiT checkpoint")

# Rollout
dit_prices = []
state_cur = state_init_d
with torch.no_grad():
    dit_prices.append(extract_price(state_cur, valid_d).item())

    for step in range(ROLLOUT_STEPS):
        mc_idx = step % len(mc_list)
        mc = mc_list[mc_idx].unsqueeze(0).to(device2)
        z_shape = (1, 36, 36, 16)
        z_pred = sampler.sample(z_shape, mc, device=device2)
        decoded = ae.decode(z_pred)
        decoded = apply_all_projections(state_cur, decoded)
        dit_prices.append(extract_price(decoded, valid_d).item())
        state_cur = decoded

dit_prices = np.array(dit_prices)
dit_prices = np.clip(dit_prices, 0.01, None)
print(f"Rollout: {len(dit_prices)} steps, range=[{dit_prices.min():.4f}, {dit_prices.max():.4f}], std={dit_prices.std():.6f}")

# Speedup
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    _ = sampler.sample(z_shape, mc, device=device2)
torch.cuda.synchronize()
dit_speed = (time.perf_counter() - t0) / 10 * 1000
print(f"Speed: {dit_speed:.2f} ms/step (DDIM 50 steps)")

dit_sf = evaluate_stylized_facts(dit_prices)
print(f"\nStylized Facts: {dit_sf.summary}")
print(f"  Fat tail alpha:    {dit_sf.fat_tail_alpha:.4f}  {'PASS' if dit_sf.fat_tail_pass else 'FAIL'}")
print(f"  Vol clustering:    {'PASS' if dit_sf.volatility_clustering_pass else 'FAIL'}")
print(f"  Leverage effect:   {dit_sf.leverage_effect_corr:.4f}  {'PASS' if dit_sf.leverage_effect_pass else 'FAIL'}")
print(f"  Return autocorr:   {'PASS' if dit_sf.return_autocorr_pass else 'FAIL'}")
print(f"  Gain/loss asym:    {'PASS' if dit_sf.gain_loss_asymmetry_pass else 'FAIL'}")

dit_events = run_emergence_analysis(dit_prices)
for k, v in dit_events.items():
    if v: print(f"  {k}: {len(v)} events")

# ============================================================
# COMPARISON TABLE
# ============================================================
print("\n" + "="*60)
print("  COMPARISON TABLE")
print("="*60)
print(f"{'Metric':<30} {'LeWM v3.1':>15} {'DiT v3':>15}")
print("-"*60)
print(f"{'Stylized Facts':<30} {lewm_sf.total_passed:>12}/6  {dit_sf.total_passed:>12}/6")
print(f"{'Fat tail alpha':<30} {lewm_sf.fat_tail_alpha:>15.2f} {dit_sf.fat_tail_alpha:>15.2f}")
print(f"{'Leverage effect':<30} {lewm_sf.leverage_effect_corr:>15.4f} {dit_sf.leverage_effect_corr:>15.4f}")
print(f"{'Price std (1000 steps)':<30} {lewm_prices.std():>15.6f} {dit_prices.std():>15.6f}")
print(f"{'Speed (ms/step)':<30} {lewm_speed:>15.2f} {dit_speed:>15.2f}")
print(f"{'Total params':<30} {'42M':>15} {'~250M':>15}")

print("\n=== EVALUATION COMPLETE ===")
PYEOF
