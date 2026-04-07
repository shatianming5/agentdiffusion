#!/usr/bin/env bash
# ============================================================================
#  Causal Intervention Ground-Truth Validation
#  -------------------------------------------
#  Compare our Video DiT model's causal intervention predictions against
#  ABIDES simulator ground truth for agent removal experiments.
#
#  Experiments:
#    1. ABIDES baseline (all agents)
#    2. ABIDES counterfactual (remove MM / momentum / noise)
#    3. Model counterfactual (zero out agent grid positions, generate)
#    4. Compare model prediction vs ABIDES ground truth
#
#  Uses existing data in data/abides_real/ (34x33 grid, 128-dim states)
#  and trains/loads a Video DiT model.
# ============================================================================
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}

PYTHON=".venv/bin/python3"

echo "================================================================"
echo "  Causal Intervention Ground-Truth Validation"
echo "  Video DiT Model vs ABIDES Simulator"
echo "================================================================"
echo ""

# ============================================================================
# Phase 1: Run ABIDES simulations -- baseline + counterfactuals
# ============================================================================
$PYTHON -u << 'PYEOF'
import os, sys, math, time, logging, json
import numpy as np
import torch
from pathlib import Path
from collections import Counter

logging.disable(logging.WARNING)

# ============================================================================
# Configuration
# ============================================================================
NUM_SIMS        = 30       # simulations per scenario (enough for statistics)
NUM_SNAPSHOTS   = 20       # snapshots per sim
END_TIME        = "10:00:00"
SEED_BASE       = 7777     # fixed base seed for reproducibility

BASE_DIR = Path("data/causal_groundtruth")
BASE_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = {
    "baseline":     {"remove": None,               "label": "All agents (baseline)"},
    "no_mm":        {"remove": "market_makers",     "label": "No market makers"},
    "no_momentum":  {"remove": "momentum",          "label": "No momentum traders"},
    "no_noise":     {"remove": "noise",             "label": "No noise traders (50% removed)"},
}

# ============================================================================
# Helpers
# ============================================================================
from abides_core import abides
from abides_markets.configs import rmsc04

PRICE_SCALE = 100_000.0

def _get_l1_prices(L1, snap_idx):
    """Extract bid/ask/mid/spread from L1 snapshots at given index."""
    bids = L1.get("best_bids", [])
    asks = L1.get("best_asks", [])
    bid = bids[snap_idx][1] if snap_idx < len(bids) and bids[snap_idx][1] is not None else 100000
    ask = asks[snap_idx][1] if snap_idx < len(asks) and asks[snap_idx][1] is not None else 100000
    mid = (bid + ask) / 2.0
    spread = ask - bid
    spread_bps = spread / mid * 10000.0 if mid > 0 else 0.0
    return {"bid": bid, "ask": ask, "mid": mid, "spread": spread, "spread_bps": spread_bps}


def _type_map(agent):
    """Map ABIDES agent class to our type enum."""
    name = type(agent).__name__
    mapping = {
        "NoiseAgent": 3,              # NOISE_TRADER
        "ValueAgent": 2,              # FUNDAMENTALIST
        "AdaptiveMarketMakerAgent": 0, # MARKET_MAKER
        "MomentumAgent": 1,           # TREND_FOLLOWER
    }
    return mapping.get(name, 3)


def _extract_state(agent, l1_prices, step_idx, total_steps):
    """Extract 128-dim state vector (matches abides_generator.py logic)."""
    state = np.zeros(128, dtype=np.float32)
    h = agent.holdings
    mid_price = l1_prices.get("mid", PRICE_SCALE) or PRICE_SCALE
    starting = max(getattr(agent, "starting_cash", 10_000_000), 1)
    cash_raw = h.get("CASH", 0)
    position = h.get("ABM", 0)

    state[0] = np.sign(position) * np.log1p(abs(position))
    state[1] = np.clip(cash_raw / starting, -5, 5)
    state[2] = np.clip(position * mid_price / max(abs(cash_raw), starting * 0.01), -5, 5)
    state[32] = state[1]
    state[33] = np.clip((cash_raw - starting) / starting, -5, 5)
    state[34] = np.clip(abs(state[2]), 0, 5)
    state[35] = np.tanh((cash_raw - starting) / starting)

    atype = _type_map(agent)
    state[48 + atype] = 1.0

    state[96] = l1_prices.get("bid", mid_price) / PRICE_SCALE
    state[97] = l1_prices.get("ask", mid_price) / PRICE_SCALE
    state[98] = mid_price / PRICE_SCALE
    state[99] = (l1_prices.get("ask", mid_price) - l1_prices.get("bid", mid_price)) / mid_price * 1000
    state[100] = (mid_price - PRICE_SCALE) / PRICE_SCALE
    state[112] = float(atype) / 3.0
    state[82] = step_idx / max(total_steps, 1)

    np.clip(state, -5, 5, out=state)
    return state


def run_scenario(scenario_name, scenario_cfg, num_sims, seed_base):
    """Run ABIDES simulations for a scenario, save grid data + spread series."""
    out_dir = BASE_DIR / scenario_name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_spreads = []       # list of spread time series (one per sim)
    all_mids = []          # list of mid-price time series
    all_volatilities = []  # realized vol per sim
    sample_idx = 0

    for sim in range(num_sims):
        seed = seed_base + sim
        remove = scenario_cfg["remove"]

        # Build ABIDES config with appropriate agent removal
        kwargs = {"seed": seed, "end_time": END_TIME}
        if remove == "market_makers":
            # rmsc04 has 2 MMs; set to 0 is not directly supported,
            # so we use a very short simulation where MMs never wake up.
            # Instead, set num_noise=1000, num_value=102, num_momentum=12
            # and manually build without MMs by setting mm params to have 0 MMs.
            # The rmsc04 config creates MMs from MM_PARAMS list. We rebuild with empty list.
            pass  # handled below
        elif remove == "momentum":
            kwargs["num_momentum_agents"] = 0
        elif remove == "noise":
            kwargs["num_noise_agents"] = 500  # reduce by 50% (removing all would crash the sim)

        try:
            if remove == "market_makers":
                # Custom config: build rmsc04 but override to 0 MMs
                # We do this by patching: build normally then remove MM agents
                config = rmsc04.build_config(seed=seed, end_time=END_TIME)
                # Remove AdaptiveMarketMakerAgent from agents list
                original_agents = config["agents"]
                filtered_agents = [a for a in original_agents
                                   if not (hasattr(a, 'type') and 'MarketMaker' in getattr(a, 'type', ''))]
                # Re-id remaining agents
                for i, a in enumerate(filtered_agents):
                    pass  # keep original IDs (exchange agent needs id=0)
                config["agents"] = filtered_agents
                # Rebuild latency model for fewer agents
                from abides_markets.utils import generate_latency_model
                config["agent_latency_model"] = generate_latency_model(len(filtered_agents))
            else:
                config = rmsc04.build_config(**kwargs)

            end_state = abides.run(config)
        except Exception as e:
            print(f"  [WARN] Sim {sim} scenario={scenario_name} failed: {e}")
            continue

        # Extract L1 data
        ob = end_state["agents"][0].order_books["ABM"]
        L1 = ob.get_L1_snapshots()
        n_l1 = len(L1.get("best_bids", []))

        if n_l1 < 2:
            continue

        # Get trading agents
        trading_agents = [a for a in end_state["agents"]
                          if hasattr(a, "holdings") and a.type != "ExchangeAgent"]
        num_agents = len(trading_agents)

        # Compute grid dimensions
        grid_h = int(math.ceil(math.sqrt(num_agents)))
        grid_w = int(math.ceil(num_agents / grid_h))

        # Extract spread time series
        snap_indices = np.linspace(0, n_l1 - 1, min(NUM_SNAPSHOTS, n_l1), dtype=int)
        spreads = []
        mids = []
        for si in snap_indices:
            prices = _get_l1_prices(L1, si)
            spreads.append(prices["spread_bps"])
            mids.append(prices["mid"])

        all_spreads.append(np.array(spreads, dtype=np.float64))
        all_mids.append(np.array(mids, dtype=np.float64))

        # Compute realized volatility
        mids_arr = np.array(mids, dtype=np.float64)
        mids_arr = np.clip(mids_arr, 1, None)
        if len(mids_arr) > 1:
            log_rets = np.diff(np.log(mids_arr))
            all_volatilities.append(log_rets.std())
        else:
            all_volatilities.append(0.0)

        # Save grid frames for model inference (same format as training data)
        for i in range(len(snap_indices) - 1):
            t_idx = snap_indices[i]
            t1_idx = snap_indices[i + 1]
            l1_t = _get_l1_prices(L1, t_idx)
            l1_t1 = _get_l1_prices(L1, t1_idx)

            states_t = np.stack([_extract_state(a, l1_t, t_idx, n_l1) for a in trading_agents])
            states_t1 = np.stack([_extract_state(a, l1_t1, t1_idx, n_l1) for a in trading_agents])

            types = torch.tensor([_type_map(a) for a in trading_agents])
            capitals = torch.tensor([a.holdings.get("CASH", 0) for a in trading_agents], dtype=torch.float32)

            # Sort by (type, capital) and reshape into grid -- mirrors AgentGrid.arrange()
            sort_key = types.float() * 1e12 - capitals
            sort_indices = sort_key.argsort()

            sorted_t = torch.from_numpy(states_t)[sort_indices]
            sorted_t1 = torch.from_numpy(states_t1)[sort_indices]
            sorted_types = types[sort_indices]

            grid_size = grid_h * grid_w
            pad_n = grid_size - num_agents
            if pad_n > 0:
                sorted_t = torch.nn.functional.pad(sorted_t, (0, 0, 0, pad_n))
                sorted_t1 = torch.nn.functional.pad(sorted_t1, (0, 0, 0, pad_n))
                sorted_types = torch.nn.functional.pad(sorted_types, (0, pad_n), value=-1)

            grid_t = sorted_t.view(grid_h, grid_w, 128)
            grid_t1 = sorted_t1.view(grid_h, grid_w, 128)
            grid_types = sorted_types.view(grid_h, grid_w)

            # Market condition
            market_cond = torch.zeros(32)
            market_cond[0] = l1_t.get("mid", 100000) / 100000.0
            market_cond[1] = l1_t.get("spread", 0) / 100000.0
            market_cond[7] = np.clip(l1_t.get("spread_bps", 0.0) / 100.0, 0.0, 5.0)

            torch.save({
                "state_t": grid_t,
                "state_t1": grid_t1,
                "market_cond": market_cond,
                "agent_types": grid_types,
                "sim_id": sim,
                "time_index": int(t_idx),
            }, out_dir / f"sample_{sample_idx:06d}.pt")
            sample_idx += 1

    # Save summary statistics
    summary = {
        "scenario": scenario_name,
        "label": scenario_cfg["label"],
        "num_sims": len(all_spreads),
        "num_samples": sample_idx,
    }
    if all_spreads:
        # Compute mean spread series (pad shorter ones with NaN)
        max_len = max(len(s) for s in all_spreads)
        padded = np.full((len(all_spreads), max_len), np.nan)
        for i, s in enumerate(all_spreads):
            padded[i, :len(s)] = s
        mean_spread = np.nanmean(padded, axis=0)
        std_spread = np.nanstd(padded, axis=0)
        mean_vol = np.mean(all_volatilities)
        std_vol = np.std(all_volatilities)

        summary["mean_spread_series"] = mean_spread.tolist()
        summary["std_spread_series"] = std_spread.tolist()
        summary["mean_vol"] = float(mean_vol)
        summary["std_vol"] = float(std_vol)
        summary["mean_spread_overall"] = float(np.nanmean(mean_spread))
        summary["std_spread_overall"] = float(np.nanmean(std_spread))

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  [{scenario_name}] {len(all_spreads)} sims, {sample_idx} samples, "
          f"mean_spread={summary.get('mean_spread_overall', 'N/A'):.2f} bps")
    return summary


# ============================================================================
# Run all scenarios
# ============================================================================
print("=" * 64)
print("  Phase 1: ABIDES Ground-Truth Simulations")
print("=" * 64)

summaries = {}
for name, cfg in SCENARIOS.items():
    print(f"\n--- Running scenario: {cfg['label']} ---")
    t0 = time.time()
    summaries[name] = run_scenario(name, cfg, NUM_SIMS, SEED_BASE)
    elapsed = time.time() - t0
    print(f"    Time: {elapsed:.1f}s")

# Save all summaries
with open(BASE_DIR / "all_summaries.json", "w") as f:
    json.dump(summaries, f, indent=2)

print("\n--- Phase 1 Complete ---")
for name, s in summaries.items():
    print(f"  {name}: spread={s.get('mean_spread_overall', 'N/A'):.2f} bps, "
          f"vol={s.get('mean_vol', 'N/A'):.6f}")
PYEOF

echo ""
echo "================================================================"
echo "  Phase 2: Model Inference -- Causal Interventions"
echo "================================================================"
echo ""

# ============================================================================
# Phase 2: Load model, run causal interventions, compare to ground truth
# ============================================================================
$PYTHON -u << 'PYEOF'
import os, sys, math, time, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy import stats as sp_stats

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = Path("data/causal_groundtruth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================================
# Load ABIDES ground truth summaries
# ============================================================================
print("\n--- Loading ABIDES ground truth ---")
summaries_path = BASE_DIR / "all_summaries.json"
if not summaries_path.exists():
    print("[ERROR] Phase 1 summaries not found. Run Phase 1 first.")
    sys.exit(1)

with open(summaries_path) as f:
    gt_summaries = json.load(f)

for name, s in gt_summaries.items():
    print(f"  {name}: spread={s.get('mean_spread_overall', 'N/A'):.2f} bps, "
          f"vol={s.get('mean_vol', 'N/A'):.6f}, "
          f"sims={s.get('num_sims', 0)}")

# ============================================================================
# Detect available model checkpoint
# ============================================================================
print("\n--- Detecting model checkpoint ---")

# Priority order for checkpoint discovery
CKPT_SEARCH_PATHS = [
    "outputs/vdit_10k/",
    "outputs/video_dit_10k_noae/",
    "outputs/video_dit_10k/",
    "outputs/vdit_lob_4x4/",
    "outputs/video_dit/",
    "outputs/stage2_dit/",
]

vdit_ckpt_path = None
model_config = None

for search_dir in CKPT_SEARCH_PATHS:
    p = Path(search_dir)
    if p.exists():
        ckpts = sorted(p.glob("*.pt"))
        if ckpts:
            vdit_ckpt_path = str(ckpts[-1])
            print(f"  Found checkpoint: {vdit_ckpt_path}")
            break

NO_AE_MODE = False  # whether to skip autoencoder
USE_RAW_DIM = False

if vdit_ckpt_path is None:
    print("  [WARN] No pre-trained Video DiT checkpoint found.")
    print("  Will train a minimal model on baseline data.")
    TRAIN_MINIMAL = True
else:
    TRAIN_MINIMAL = False
    # Probe checkpoint to detect config
    ckpt_data = torch.load(vdit_ckpt_path, map_location="cpu", weights_only=True)
    if "config" in ckpt_data:
        model_config = ckpt_data["config"]
        print(f"  Config from checkpoint: {model_config}")

# ============================================================================
# Load baseline data to determine grid dimensions and agent types
# ============================================================================
print("\n--- Loading baseline data for grid info ---")

baseline_dir = BASE_DIR / "baseline"
baseline_files = sorted(baseline_dir.glob("*.pt"))
if not baseline_files:
    # Fall back to existing training data
    baseline_dir = Path("data/abides_real")
    baseline_files = sorted(baseline_dir.glob("*.pt"))
    if not baseline_files:
        print("[ERROR] No baseline data found.")
        sys.exit(1)
    print(f"  Using existing training data from {baseline_dir}")

sample0 = torch.load(str(baseline_files[0]), map_location="cpu", weights_only=True)
GRID_H, GRID_W, RAW_DIM = sample0["state_t"].shape
agent_types_grid = sample0["agent_types"]  # [H, W]

print(f"  Grid: {GRID_H}x{GRID_W}, RAW_DIM={RAW_DIM}")
types_counter = {}
for t_val in sorted(agent_types_grid.unique().tolist()):
    cnt = (agent_types_grid == t_val).sum().item()
    if t_val >= 0:
        label = {0: "MARKET_MAKER", 1: "TREND_FOLLOWER", 2: "FUNDAMENTALIST", 3: "NOISE_TRADER"}.get(int(t_val), f"type_{t_val}")
        types_counter[int(t_val)] = cnt
        print(f"    {label} (type {int(t_val)}): {cnt} agents")

# Build masks
valid_mask = (agent_types_grid >= 0)
mm_mask = (agent_types_grid == 0)
momentum_mask = (agent_types_grid == 1)
noise_mask = (agent_types_grid == 3)

print(f"  Valid agents: {valid_mask.sum().item()}")
print(f"  Market makers: {mm_mask.sum().item()}")
print(f"  Momentum: {momentum_mask.sum().item()}")
print(f"  Noise: {noise_mask.sum().item()}")

# ============================================================================
# Build or load model
# ============================================================================
print("\n--- Building Video DiT model ---")

from agentdiffusion.models.video_dit import VideoDiT, VideoDDIMSampler
from agentdiffusion.diffusion.scheduler import NoiseScheduler

# Grid must be patch-size divisible
PATCH_SIZE = 4
PAD_H = ((GRID_H + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
PAD_W = ((GRID_W + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE

# Determine model dimensions from checkpoint or defaults
if model_config and "d_model" in model_config:
    D_MODEL = model_config["d_model"]
    DEPTH = model_config.get("depth", 6)
    HEADS = model_config.get("heads", 4)
    NUM_FRAMES = model_config.get("num_frames", 20)
    NUM_COND = model_config.get("num_cond_frames", 4)
else:
    D_MODEL = 256
    DEPTH = 6
    HEADS = 4
    NUM_FRAMES = 20
    NUM_COND = 4

NUM_GEN = NUM_FRAMES - NUM_COND

# Check if checkpoint uses raw dim as latent (no AE)
if vdit_ckpt_path and not TRAIN_MINIMAL:
    # Detect d_latent from checkpoint weights
    state_dict = ckpt_data.get("ema", ckpt_data.get("model", {}))
    if "patchify.proj.weight" in state_dict:
        # patchify.proj.weight shape: [d_model, patch_size^2 * d_latent]
        proj_in = state_dict["patchify.proj.weight"].shape[1]
        D_LATENT = proj_in // (PATCH_SIZE * PATCH_SIZE)
        print(f"  Detected d_latent={D_LATENT} from checkpoint")
        if D_LATENT == RAW_DIM:
            NO_AE_MODE = True
            USE_RAW_DIM = True
            print(f"  NO-AE mode: d_latent=RAW_DIM={RAW_DIM}")
        elif D_LATENT == 16 and RAW_DIM == 128:
            NO_AE_MODE = False
            print(f"  AE mode: d_latent=16, RAW_DIM=128")
    else:
        D_LATENT = 16
else:
    # Training minimal model: use raw 16-dim features from the causal groundtruth data
    # (which we generated with 128-dim states)
    # For quick training, we use raw dim as latent with no AE
    D_LATENT = RAW_DIM
    NO_AE_MODE = True
    USE_RAW_DIM = True
    print(f"  Minimal mode: d_latent=RAW_DIM={RAW_DIM}")

# Detect grid_h/grid_w from checkpoint (may differ from data)
if vdit_ckpt_path and not TRAIN_MINIMAL and "spatial_pos_embed" in state_dict:
    num_patches_ckpt = state_dict["spatial_pos_embed"].shape[1]
    # num_patches = (grid_h/patch) * (grid_w/patch)
    # Try to infer grid size
    side_patches = int(math.sqrt(num_patches_ckpt))
    if side_patches * side_patches == num_patches_ckpt:
        CKPT_GRID_H = side_patches * PATCH_SIZE
        CKPT_GRID_W = side_patches * PATCH_SIZE
    else:
        CKPT_GRID_H = PAD_H
        CKPT_GRID_W = PAD_W
    print(f"  Checkpoint grid: {CKPT_GRID_H}x{CKPT_GRID_W}")
    # Use checkpoint grid for model, pad/crop data to match
    MODEL_GRID_H = CKPT_GRID_H
    MODEL_GRID_W = CKPT_GRID_W
else:
    MODEL_GRID_H = PAD_H
    MODEL_GRID_W = PAD_W

print(f"  Model grid: {MODEL_GRID_H}x{MODEL_GRID_W}, d_latent={D_LATENT}")
print(f"  Architecture: d_model={D_MODEL}, depth={DEPTH}, heads={HEADS}")
print(f"  Frames: {NUM_COND} cond + {NUM_GEN} gen = {NUM_FRAMES} total")

model = VideoDiT(
    d_latent=D_LATENT,
    d_model=D_MODEL,
    depth=DEPTH,
    heads=HEADS,
    mlp_ratio=4.0,
    patch_size=PATCH_SIZE,
    grid_h=MODEL_GRID_H,
    grid_w=MODEL_GRID_W,
    num_frames=NUM_FRAMES,
    num_cond_frames=NUM_COND,
    market_cond_dim=32,
    dropout=0.0,
).to(device)

scheduler = NoiseScheduler(timesteps=1000, schedule="cosine").to(device)

# ============================================================================
# Load or train model
# ============================================================================
if not TRAIN_MINIMAL:
    print(f"\n--- Loading pretrained weights from {vdit_ckpt_path} ---")
    state_dict = ckpt_data.get("ema", ckpt_data.get("model", {}))
    try:
        model.load_state_dict(state_dict, strict=True)
        print("  Loaded weights (strict)")
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
        print("  Loaded weights (non-strict, some keys may differ)")
else:
    # Quick minimal training on baseline data so the model at least captures
    # the data distribution (not random noise)
    print("\n--- Training minimal Video DiT (1000 steps) ---")
    from agentdiffusion.data.video_dataset import AgentVideoDataset, SyntheticVideoDataset
    from torch.utils.data import DataLoader

    try:
        dataset = AgentVideoDataset(
            data_dir=str(baseline_dir),
            total_frames=NUM_FRAMES,
            cond_frames=NUM_COND,
            pad_to=(MODEL_GRID_H, MODEL_GRID_W),
            market_cond_dim=32,
        )
        print(f"  Loaded {len(dataset)} sequences from {baseline_dir}")
    except (FileNotFoundError, ValueError) as e:
        print(f"  [WARN] Could not load real data ({e}), using synthetic")
        dataset = SyntheticVideoDataset(
            num_samples=200,
            total_frames=NUM_FRAMES,
            cond_frames=NUM_COND,
            grid_h=MODEL_GRID_H,
            grid_w=MODEL_GRID_W,
            raw_dim=RAW_DIM,
            market_cond_dim=32,
        )

    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)

    # Optionally load AE for encoding
    ae = None
    if not NO_AE_MODE:
        from agentdiffusion.models.autoencoder import AgentAutoencoder
        ae_ckpts = sorted(Path("outputs/stage1_ae").glob("*.pt"))
        if ae_ckpts:
            ae = AgentAutoencoder(raw_dim=RAW_DIM, latent_dim=D_LATENT).to(device)
            ae_state = torch.load(str(ae_ckpts[-1]), map_location=device, weights_only=True)
            ae.load_state_dict(ae_state["model"])
            ae.eval()
            for p in ae.parameters():
                p.requires_grad_(False)
            print(f"  Loaded AE from {ae_ckpts[-1]}")
        else:
            print("  [WARN] No AE checkpoint, falling back to NO-AE mode")
            NO_AE_MODE = True
            USE_RAW_DIM = True
            # Rebuild model with raw_dim as latent
            D_LATENT = RAW_DIM
            model = VideoDiT(
                d_latent=D_LATENT, d_model=D_MODEL, depth=DEPTH, heads=HEADS,
                mlp_ratio=4.0, patch_size=PATCH_SIZE,
                grid_h=MODEL_GRID_H, grid_w=MODEL_GRID_W,
                num_frames=NUM_FRAMES, num_cond_frames=NUM_COND,
                market_cond_dim=32, dropout=0.0,
            ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    model.train()

    TRAIN_STEPS = 1000
    step = 0
    running_loss = 0.0
    while step < TRAIN_STEPS:
        for batch in loader:
            if step >= TRAIN_STEPS:
                break
            frames = batch["frames"].to(device)  # [B, T, H, W, C]
            B, T, H, W, C = frames.shape

            # Encode if needed
            if NO_AE_MODE:
                latents = frames
            else:
                with torch.no_grad():
                    flat = frames.reshape(B * T, H, W, C)
                    latents = ae.encode(flat).reshape(B, T, H, W, -1)

            z_cond = latents[:, :NUM_COND]
            z_gen = latents[:, NUM_COND:]

            t_diff = torch.randint(0, scheduler.timesteps, (B,), device=device)
            noise = torch.randn_like(z_gen)

            N_gen = z_gen.shape[1]
            z_gen_flat = z_gen.reshape(B * N_gen, H, W, -1)
            noise_flat = noise.reshape(B * N_gen, H, W, -1)
            t_exp = t_diff.unsqueeze(1).expand(B, N_gen).reshape(B * N_gen)
            z_noisy_flat = scheduler.q_sample(z_gen_flat, t_exp, noise_flat)
            z_noisy = z_noisy_flat.reshape(B, N_gen, H, W, -1)

            v_pred = model(z_cond, z_noisy, t_diff)
            v_target_flat = scheduler.v_target(z_gen_flat, noise_flat, t_exp)
            v_target = v_target_flat.reshape(B, N_gen, H, W, -1)

            loss = F.mse_loss(v_pred, v_target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            step += 1
            if step % 100 == 0:
                print(f"    Step {step}/{TRAIN_STEPS}, loss={running_loss/100:.4f}")
                running_loss = 0.0

    print("  Minimal training done.")

model.eval()
sampler = VideoDDIMSampler(
    model=model, scheduler=scheduler,
    prediction_type="v_prediction", ddim_steps=50, eta=0.0,
)
print("  DDIM sampler ready (50 steps)")

# ============================================================================
# Load AE for decoding if needed
# ============================================================================
ae_decode = None
if not NO_AE_MODE:
    from agentdiffusion.models.autoencoder import AgentAutoencoder
    ae_ckpts = sorted(Path("outputs/stage1_ae").glob("*.pt"))
    if ae_ckpts:
        ae_decode = AgentAutoencoder(raw_dim=RAW_DIM, latent_dim=D_LATENT).to(device)
        ae_state = torch.load(str(ae_ckpts[-1]), map_location=device, weights_only=True)
        ae_decode.load_state_dict(ae_state["model"])
        ae_decode.eval()
        print(f"  AE decoder loaded from {ae_ckpts[-1]}")

# ============================================================================
# Helper functions
# ============================================================================

def _pad_to_model_grid(tensor, target_h, target_w):
    """Pad [H, W, ...] or [T, H, W, ...] to model grid size."""
    if tensor.dim() == 3:
        h, w = tensor.shape[:2]
        if h < target_h or w < target_w:
            pad_w = max(0, target_w - w)
            pad_h = max(0, target_h - h)
            tensor = F.pad(tensor, (0, 0, 0, pad_w, 0, pad_h))
        return tensor[:target_h, :target_w]
    elif tensor.dim() == 4:
        t, h, w, c = tensor.shape
        if h < target_h or w < target_w:
            pad_w = max(0, target_w - w)
            pad_h = max(0, target_h - h)
            tensor = F.pad(tensor, (0, 0, 0, pad_w, 0, pad_h))
        return tensor[:, :target_h, :target_w]
    return tensor


def load_cond_frames(scenario_dir, start_idx=0, num_cond=NUM_COND):
    """Load condition frames from a scenario's saved data."""
    files = sorted(Path(scenario_dir).glob("*.pt"))
    if not files:
        return None, None

    raw_frames = []
    for i in range(num_cond):
        fidx = min(start_idx + i, len(files) - 1)
        data = torch.load(str(files[fidx]), map_location="cpu", weights_only=True)
        if i == 0:
            raw_frames.append(data["state_t"])
        raw_frames.append(data["state_t1"])

    raw_frames = raw_frames[:num_cond]

    # Pad to model grid
    padded = [_pad_to_model_grid(f, MODEL_GRID_H, MODEL_GRID_W) for f in raw_frames]

    # Encode if AE mode
    if NO_AE_MODE:
        cond = torch.stack(padded).unsqueeze(0).to(device)  # [1, K, H, W, RAW_DIM]
    else:
        with torch.no_grad():
            encoded = []
            for f in padded:
                enc = ae_decode.encode(f.unsqueeze(0).to(device))  # [1, H, W, d_latent]
                encoded.append(enc.squeeze(0))
            cond = torch.stack(encoded).unsqueeze(0).to(device)  # [1, K, H, W, d_latent]

    return cond, raw_frames


def generate_and_extract_spread(x_cond, num_gen=NUM_GEN, n_runs=5):
    """Generate frames and extract spread proxy from model output.

    Returns mean spread (position dispersion) across runs.
    """
    spreads_all = []
    for run in range(n_runs):
        gen_shape = (1, num_gen, MODEL_GRID_H, MODEL_GRID_W, D_LATENT)
        with torch.no_grad():
            gen = sampler.sample(x_cond, gen_shape, device=device)

        # Decode if needed
        if not NO_AE_MODE and ae_decode is not None:
            B, N, H, W, C = gen.shape
            gen_flat = gen.reshape(B * N, H, W, C)
            gen_raw = ae_decode.decode(gen_flat).reshape(B, N, H, W, -1)
        else:
            gen_raw = gen

        # Extract spread proxy: std of observation dim (dim 99 = spread in permille)
        # and position dim (dim 0)
        # Crop to original data grid
        gen_crop = gen_raw[:, :, :GRID_H, :GRID_W, :]  # [1, N, GRID_H, GRID_W, C]

        # Use market observation dims as spread proxy:
        # dim 99 = (ask-bid)/mid * 1000 (spread in permille)
        vm = valid_mask.to(device)
        if gen_crop.shape[-1] > 99:
            spread_vals = gen_crop[0, :, :, :, 99]  # [N, H, W]
            per_frame_spread = []
            for t in range(spread_vals.shape[0]):
                frame_spread = spread_vals[t][vm[:spread_vals.shape[1], :spread_vals.shape[2]]].mean().item()
                per_frame_spread.append(frame_spread)
            spreads_all.append(per_frame_spread)
        else:
            # Fallback: use position dispersion as liquidity proxy
            pos_vals = gen_crop[0, :, :, :, 0]  # [N, H, W]
            per_frame_spread = []
            for t in range(pos_vals.shape[0]):
                frame_std = pos_vals[t][vm[:pos_vals.shape[1], :pos_vals.shape[2]]].std().item()
                per_frame_spread.append(frame_std)
            spreads_all.append(per_frame_spread)

    # Average across runs
    max_len = max(len(s) for s in spreads_all)
    padded = np.full((len(spreads_all), max_len), np.nan)
    for i, s in enumerate(spreads_all):
        padded[i, :len(s)] = s
    mean_spread = np.nanmean(padded, axis=0)
    return mean_spread


# ============================================================================
# Run model interventions for each scenario
# ============================================================================
print("\n" + "=" * 64)
print("  Phase 2a: Model Baseline Generation")
print("=" * 64)

# Load baseline condition frames
x_cond_baseline, raw_baseline = load_cond_frames(BASE_DIR / "baseline", start_idx=0)
if x_cond_baseline is None:
    x_cond_baseline, raw_baseline = load_cond_frames(Path("data/abides_real"), start_idx=0)

print(f"  Condition frames shape: {x_cond_baseline.shape}")

# Normal generation
print("  Generating baseline (all agents)...")
model_baseline_spread = generate_and_extract_spread(x_cond_baseline, n_runs=10)
print(f"  Model baseline spread (mean): {np.mean(model_baseline_spread):.4f}")

# ============================================================================
# Intervention: Remove Market Makers
# ============================================================================
print("\n--- Intervention: Remove Market Makers ---")
x_cond_no_mm = x_cond_baseline.clone()
mm_mask_dev = mm_mask.to(device)

# Pad mask to model grid
mm_mask_padded = F.pad(mm_mask.float(), (0, MODEL_GRID_W - GRID_W, 0, MODEL_GRID_H - GRID_H)).bool().to(device)
mm_5d = mm_mask_padded.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand_as(x_cond_no_mm)
x_cond_no_mm[mm_5d] = 0.0

print("  Generating with MM removed (zeroed in condition)...")
model_no_mm_spread = generate_and_extract_spread(x_cond_no_mm, n_runs=10)
print(f"  Model no-MM spread (mean): {np.mean(model_no_mm_spread):.4f}")

# ============================================================================
# Intervention: Remove Momentum Traders
# ============================================================================
print("\n--- Intervention: Remove Momentum Traders ---")
x_cond_no_mom = x_cond_baseline.clone()
mom_mask_padded = F.pad(momentum_mask.float(), (0, MODEL_GRID_W - GRID_W, 0, MODEL_GRID_H - GRID_H)).bool().to(device)
mom_5d = mom_mask_padded.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand_as(x_cond_no_mom)
x_cond_no_mom[mom_5d] = 0.0

print("  Generating with momentum removed...")
model_no_mom_spread = generate_and_extract_spread(x_cond_no_mom, n_runs=10)
print(f"  Model no-momentum spread (mean): {np.mean(model_no_mom_spread):.4f}")

# ============================================================================
# Intervention: Remove Noise Traders
# ============================================================================
print("\n--- Intervention: Remove Noise Traders ---")
x_cond_no_noise = x_cond_baseline.clone()
noise_mask_padded = F.pad(noise_mask.float(), (0, MODEL_GRID_W - GRID_W, 0, MODEL_GRID_H - GRID_H)).bool().to(device)
noise_5d = noise_mask_padded.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand_as(x_cond_no_noise)
# Zero out half of noise traders (matching ABIDES counterfactual which removes 50%)
noise_positions = noise_mask_padded.nonzero(as_tuple=False)
n_noise = noise_positions.shape[0]
torch.manual_seed(42)
remove_idx = torch.randperm(n_noise)[:n_noise // 2]
for idx in remove_idx:
    r, c = noise_positions[idx]
    x_cond_no_noise[:, :, r, c, :] = 0.0

print("  Generating with 50% noise traders removed...")
model_no_noise_spread = generate_and_extract_spread(x_cond_no_noise, n_runs=10)
print(f"  Model no-noise spread (mean): {np.mean(model_no_noise_spread):.4f}")

# ============================================================================
# Phase 3: Comparison -- Model vs ABIDES Ground Truth
# ============================================================================
print("\n" + "=" * 64)
print("  Phase 3: Comparison -- Model vs ABIDES Ground Truth")
print("=" * 64)

# ABIDES ground truth spread changes (relative to baseline)
gt_baseline_spread = gt_summaries["baseline"].get("mean_spread_overall", 1.0)
gt_no_mm_spread = gt_summaries["no_mm"].get("mean_spread_overall", 1.0)
gt_no_mom_spread = gt_summaries["no_momentum"].get("mean_spread_overall", 1.0)
gt_no_noise_spread = gt_summaries["no_noise"].get("mean_spread_overall", 1.0)

# Relative changes (percentage)
gt_mm_change = (gt_no_mm_spread - gt_baseline_spread) / max(abs(gt_baseline_spread), 1e-10) * 100
gt_mom_change = (gt_no_mom_spread - gt_baseline_spread) / max(abs(gt_baseline_spread), 1e-10) * 100
gt_noise_change = (gt_no_noise_spread - gt_baseline_spread) / max(abs(gt_baseline_spread), 1e-10) * 100

# Model spread changes
m_baseline = float(np.mean(model_baseline_spread))
m_no_mm = float(np.mean(model_no_mm_spread))
m_no_mom = float(np.mean(model_no_mom_spread))
m_no_noise = float(np.mean(model_no_noise_spread))

m_mm_change = (m_no_mm - m_baseline) / max(abs(m_baseline), 1e-10) * 100
m_mom_change = (m_no_mom - m_baseline) / max(abs(m_baseline), 1e-10) * 100
m_noise_change = (m_no_noise - m_baseline) / max(abs(m_baseline), 1e-10) * 100

# ============================================================================
# Statistical tests
# ============================================================================
print("\n--- Statistical Analysis ---")

# Correlation between model-predicted and ABIDES ground-truth spread changes
gt_changes = np.array([gt_mm_change, gt_mom_change, gt_noise_change])
model_changes = np.array([m_mm_change, m_mom_change, m_noise_change])

# Direction agreement
direction_match = np.sign(gt_changes) == np.sign(model_changes)
direction_accuracy = direction_match.mean()

# Pearson correlation (only meaningful with >2 points, but we report it)
if len(gt_changes) >= 3:
    corr, p_corr = sp_stats.pearsonr(gt_changes, model_changes)
else:
    corr, p_corr = 0.0, 1.0

# Spearman rank correlation
if len(gt_changes) >= 3:
    rho, p_rho = sp_stats.spearmanr(gt_changes, model_changes)
else:
    rho, p_rho = 0.0, 1.0

# Per-intervention spread series correlation (if available)
series_correlations = {}
for scenario, model_series in [("no_mm", model_no_mm_spread),
                                ("no_momentum", model_no_mom_spread),
                                ("no_noise", model_no_noise_spread)]:
    gt_series = gt_summaries[scenario].get("mean_spread_series", [])
    if gt_series and len(gt_series) > 2 and len(model_series) > 2:
        # Align lengths
        min_len = min(len(gt_series), len(model_series))
        gt_s = np.array(gt_series[:min_len])
        m_s = np.array(model_series[:min_len])
        # Remove NaN
        valid = ~(np.isnan(gt_s) | np.isnan(m_s))
        if valid.sum() >= 3:
            r, p = sp_stats.pearsonr(gt_s[valid], m_s[valid])
            series_correlations[scenario] = (r, p)
        else:
            series_correlations[scenario] = (float('nan'), float('nan'))
    else:
        series_correlations[scenario] = (float('nan'), float('nan'))

# ============================================================================
# T-tests: is the model's spread change significantly different from ABIDES?
# ============================================================================
# Two-sided t-test: model change vs GT change (treated as point estimates here)
# More meaningful test: paired test on spread series
paired_tests = {}
for scenario, model_series in [("no_mm", model_no_mm_spread),
                                ("no_momentum", model_no_mom_spread),
                                ("no_noise", model_no_noise_spread)]:
    gt_series = gt_summaries[scenario].get("mean_spread_series", [])
    gt_base_series = gt_summaries["baseline"].get("mean_spread_series", [])
    if gt_series and gt_base_series:
        min_len = min(len(gt_series), len(gt_base_series), len(model_series))
        if min_len >= 3:
            gt_delta = np.array(gt_series[:min_len]) - np.array(gt_base_series[:min_len])
            m_delta_raw = np.array(model_series[:min_len]) - np.array(model_baseline_spread[:min_len])
            # Normalize both to comparable scale
            gt_norm = gt_delta / max(np.std(gt_delta), 1e-10)
            m_norm = m_delta_raw / max(np.std(m_delta_raw), 1e-10)
            t_stat, p_val = sp_stats.ttest_ind(gt_norm, m_norm)
            paired_tests[scenario] = (t_stat, p_val)
        else:
            paired_tests[scenario] = (float('nan'), float('nan'))
    else:
        paired_tests[scenario] = (float('nan'), float('nan'))

# ============================================================================
# Output Results
# ============================================================================
print("\n" + "=" * 72)
print("  COMPARISON TABLE: Model Prediction vs ABIDES Ground Truth")
print("=" * 72)
print("")
print(f"  {'':20s}  {'ABIDES GT':>12s}  {'Model Pred':>12s}  {'Direction':>10s}")
print(f"  {'Scenario':<20s}  {'Spread(bps)':>12s}  {'Spread':>12s}  {'Match?':>10s}")
print(f"  {'-' * 60}")
print(f"  {'Baseline':<20s}  {gt_baseline_spread:>12.2f}  {m_baseline:>12.4f}  {'---':>10s}")
print(f"  {'No MM':<20s}  {gt_no_mm_spread:>12.2f}  {m_no_mm:>12.4f}  {'---':>10s}")
print(f"  {'No Momentum':<20s}  {gt_no_mom_spread:>12.2f}  {m_no_mom:>12.4f}  {'---':>10s}")
print(f"  {'No Noise (50%)':<20s}  {gt_no_noise_spread:>12.2f}  {m_no_noise:>12.4f}  {'---':>10s}")
print("")

print(f"  {'':20s}  {'ABIDES GT':>12s}  {'Model Pred':>12s}  {'Direction':>10s}")
print(f"  {'Intervention':<20s}  {'Change(%)':>12s}  {'Change(%)':>12s}  {'Match?':>10s}")
print(f"  {'-' * 60}")
print(f"  {'Remove MM':<20s}  {gt_mm_change:>+12.2f}  {m_mm_change:>+12.2f}  "
      f"{'YES' if direction_match[0] else 'NO':>10s}")
print(f"  {'Remove Momentum':<20s}  {gt_mom_change:>+12.2f}  {m_mom_change:>+12.2f}  "
      f"{'YES' if direction_match[1] else 'NO':>10s}")
print(f"  {'Remove Noise(50%)':<20s}  {gt_noise_change:>+12.2f}  {m_noise_change:>+12.2f}  "
      f"{'YES' if direction_match[2] else 'NO':>10s}")
print("")

print(f"  {'=' * 72}")
print(f"  STATISTICAL TESTS")
print(f"  {'=' * 72}")
print(f"  Direction accuracy:           {direction_accuracy:.1%} ({int(direction_accuracy * 3)}/3)")
print(f"  Pearson correlation (r):      {corr:+.4f}  (p={p_corr:.4f})")
print(f"  Spearman rank corr (rho):     {rho:+.4f}  (p={p_rho:.4f})")
print("")

print(f"  {'Series Correlations':}")
print(f"  {'Scenario':<20s}  {'Pearson r':>10s}  {'p-value':>10s}")
print(f"  {'-' * 45}")
for scenario, (r, p) in series_correlations.items():
    r_str = f"{r:+.4f}" if not np.isnan(r) else "N/A"
    p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"
    print(f"  {scenario:<20s}  {r_str:>10s}  {p_str:>10s}")
print("")

print(f"  {'Paired T-Tests (normalized spread change: model vs ABIDES):':}")
print(f"  {'Scenario':<20s}  {'t-stat':>10s}  {'p-value':>10s}")
print(f"  {'-' * 45}")
for scenario, (t, p) in paired_tests.items():
    t_str = f"{t:+.4f}" if not np.isnan(t) else "N/A"
    p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"
    print(f"  {scenario:<20s}  {t_str:>10s}  {p_str:>10s}")
print("")

# ============================================================================
# Interpretation
# ============================================================================
print(f"  {'=' * 72}")
print(f"  INTERPRETATION")
print(f"  {'=' * 72}")
if direction_accuracy >= 2/3:
    print("  The model correctly predicts the DIRECTION of spread change")
    print("  for the majority of causal interventions.")
else:
    print("  The model does NOT reliably predict the direction of spread")
    print("  change across interventions.")

if corr > 0.5:
    sig_str = "significant" if p_corr < 0.05 else "not significant"
    print(f"  Positive correlation ({corr:+.3f}) between model and GT changes ({sig_str}).")
elif corr < -0.5:
    print(f"  NEGATIVE correlation ({corr:+.3f}) -- model predicts OPPOSITE of ground truth.")
else:
    print(f"  Weak correlation ({corr:+.3f}) -- model does not strongly track GT changes.")

# Economic intuition checks
print("")
print("  Expected economic intuition:")
if gt_mm_change > 0:
    mm_expected = "CORRECT (removing MM should widen spreads)"
else:
    mm_expected = "UNEXPECTED (removing MM narrowed spreads in ABIDES)"
print(f"    Remove MM -> spread change: ABIDES={gt_mm_change:+.1f}%, {mm_expected}")
print(f"    Remove MM -> model spread change: {m_mm_change:+.1f}%")

print("")
print("  Model fidelity summary:")
print(f"    Direction accuracy: {direction_accuracy:.0%}")
print(f"    Rank correlation:   {rho:+.3f}")
print(f"    Pearson r:          {corr:+.3f}")

# ============================================================================
# Save full results to JSON
# ============================================================================
results = {
    "abides_ground_truth": {
        "baseline_spread_bps": gt_baseline_spread,
        "no_mm_spread_bps": gt_no_mm_spread,
        "no_momentum_spread_bps": gt_no_mom_spread,
        "no_noise_spread_bps": gt_no_noise_spread,
        "mm_change_pct": gt_mm_change,
        "momentum_change_pct": gt_mom_change,
        "noise_change_pct": gt_noise_change,
    },
    "model_prediction": {
        "baseline_spread": m_baseline,
        "no_mm_spread": m_no_mm,
        "no_momentum_spread": m_no_mom,
        "no_noise_spread": m_no_noise,
        "mm_change_pct": m_mm_change,
        "momentum_change_pct": m_mom_change,
        "noise_change_pct": m_noise_change,
    },
    "statistics": {
        "direction_accuracy": direction_accuracy,
        "pearson_r": corr,
        "pearson_p": p_corr,
        "spearman_rho": rho,
        "spearman_p": p_rho,
        "series_correlations": {k: {"r": v[0], "p": v[1]} for k, v in series_correlations.items()},
        "paired_ttests": {k: {"t": v[0], "p": v[1]} for k, v in paired_tests.items()},
    },
    "config": {
        "model_grid": f"{MODEL_GRID_H}x{MODEL_GRID_W}",
        "d_latent": D_LATENT,
        "d_model": D_MODEL,
        "depth": DEPTH,
        "no_ae_mode": NO_AE_MODE,
        "num_abides_sims": gt_summaries["baseline"].get("num_sims", 0),
        "num_model_runs": 10,
        "checkpoint": vdit_ckpt_path or "minimal_trained",
    },
}

results_path = BASE_DIR / "causal_validation_results.json"
# Handle NaN for JSON serialization
class NanEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return super().default(obj)

with open(results_path, "w") as f:
    json.dump(results, f, indent=2, cls=NanEncoder)

print(f"\n  Full results saved to: {results_path}")
print("=" * 72)
PYEOF

echo ""
echo "================================================================"
echo "  Causal Ground-Truth Validation Complete"
echo "================================================================"
