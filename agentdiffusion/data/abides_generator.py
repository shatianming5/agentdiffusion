"""Generate training data from ABIDES financial market simulations."""

from __future__ import annotations

import math
import os
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

from .agent_state import AgentGrid, AgentType, STATE_SLICES


# Agent type mapping from ABIDES class names to our enum
_TYPE_MAP = {
    "NoiseAgent": AgentType.NOISE_TRADER,
    "ValueAgent": AgentType.FUNDAMENTALIST,
    "AdaptiveMarketMakerAgent": AgentType.MARKET_MAKER,
    "AdaptivePOVMarketMakerAgent": AgentType.MARKET_MAKER,
    "MomentumAgent": AgentType.TREND_FOLLOWER,
}


# ---- Normalization constants ----
_PRICE_SCALE = 100_000.0  # r_bar = 100000 cents = $1000
_CLIP = 5.0               # hard clamp all dims to [-5, 5]


def _safe_div(a: float, b: float) -> float:
    """Division with safe denominator, clipped to [-_CLIP, _CLIP]."""
    return np.clip(a / max(abs(b), 1e-6), -_CLIP, _CLIP)


def _extract_agent_state(agent, L1_prices: dict, step_idx: int, total_steps: int) -> np.ndarray:
    """Extract 128-dim normalized state vector from a single ABIDES TradingAgent.

    ALL dimensions are hard-clipped to [-5, 5]. No unbounded values.
    """
    state = np.zeros(128, dtype=np.float32)
    h = agent.holdings
    mid_price = L1_prices.get("mid", _PRICE_SCALE) or _PRICE_SCALE
    starting = max(getattr(agent, "starting_cash", 10_000_000), 1)
    cash_raw = h.get("CASH", 0)
    position = h.get("ABM", 0)

    # [0:32] 持仓信息 — signed log scale for positions, relative for cash
    state[0] = np.sign(position) * np.log1p(abs(position))  # signed log, ~[-7, 7] then clip
    state[1] = _safe_div(cash_raw, starting)                 # relative cash ~[0.5, 1.5]
    state[2] = _safe_div(position * mid_price, max(abs(cash_raw), starting * 0.01))  # leverage

    # [32:48] 资金状态
    state[32] = state[1]                                   # relative cash (same as [1])
    state[33] = _safe_div(cash_raw - starting, starting)   # PnL ratio
    state[34] = np.clip(abs(state[2]), 0, _CLIP)           # |leverage|
    state[35] = np.tanh((cash_raw - starting) / starting)  # squashed PnL

    # [48:64] 策略参数 — one-hot + squashed params
    atype = _TYPE_MAP.get(type(agent).__name__, AgentType.NOISE_TRADER)
    state[48 + int(atype)] = 1.0  # one-hot [48:52]

    if hasattr(agent, "lambda_a"):
        state[52] = np.tanh(agent.lambda_a * 1e12)
    if hasattr(agent, "kappa"):
        state[53] = np.tanh(agent.kappa * 1e15)
    if hasattr(agent, "r_bar"):
        state[54] = agent.r_bar / _PRICE_SCALE  # ~1.0

    # [64:80] 历史统计
    exec_orders = getattr(agent, "executed_orders", [])
    if exec_orders:
        prices_exec = [o.fill_price for o in exec_orders if hasattr(o, "fill_price") and o.fill_price]
        if len(prices_exec) >= 2:
            returns = np.diff(np.log(np.array(prices_exec, dtype=float).clip(min=1)))
            state[64] = np.clip(returns.mean(), -1, 1)
            state[65] = np.clip(returns.std(), 0, 1)
            state[66] = np.clip(state[64] / (state[65] + 1e-8), -3, 3)
        state[67] = np.tanh(len(exec_orders) / 50.0)  # squashed count

    # [80:96] 行为特征
    state[80] = np.tanh(len(getattr(agent, "orders", {})) / 5.0)
    state[81] = state[67]  # same as exec count
    state[82] = step_idx / max(total_steps, 1)  # time [0, 1]

    # [96:112] 市场观察 — centered around 1.0
    bid = L1_prices.get("bid", mid_price) or mid_price
    ask = L1_prices.get("ask", mid_price) or mid_price
    state[96] = bid / _PRICE_SCALE              # ~1.0
    state[97] = ask / _PRICE_SCALE              # ~1.0
    state[98] = mid_price / _PRICE_SCALE        # ~1.0
    state[99] = (ask - bid) / mid_price * 1000  # spread in ‰ (per mille)
    state[100] = (mid_price - _PRICE_SCALE) / _PRICE_SCALE  # deviation from par

    # [112:128] 社交/信息
    state[112] = float(atype) / 3.0  # type [0, 1]

    # HARD CLIP everything
    np.clip(state, -_CLIP, _CLIP, out=state)

    return state


def _get_l1_prices(L1: dict, snapshot_idx: int) -> dict:
    """Extract price info from a specific L1 snapshot index."""
    prices = {}
    bids = L1.get("best_bids", [])
    asks = L1.get("best_asks", [])

    if snapshot_idx < len(bids) and bids[snapshot_idx][1] is not None:
        prices["bid"] = bids[snapshot_idx][1]
    if snapshot_idx < len(asks) and asks[snapshot_idx][1] is not None:
        prices["ask"] = asks[snapshot_idx][1]

    bid = prices.get("bid", 100000) or 100000
    ask = prices.get("ask", 100000) or 100000
    prices["mid"] = (bid + ask) / 2
    prices["spread"] = ask - bid
    prices["last_trade"] = prices["mid"]

    return prices


def generate_abides_dataset(
    output_dir: str = "data/abides_real",
    num_simulations: int = 100,
    seed_start: int = 0,
    end_time: str = "10:00:00",
    num_snapshots: int = 20,
):
    """Run multiple ABIDES simulations and save agent state transition pairs.

    Args:
        output_dir: Where to save .pt files
        num_simulations: Number of independent simulations to run
        seed_start: Starting random seed
        end_time: Market close time (e.g. "10:00:00" for 30-min sim)
        num_snapshots: Number of time slices to extract per simulation
    """
    from abides_core import abides
    from abides_core.utils import parse_logs_df
    from abides_markets.configs import rmsc04

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sample_idx = 0
    stats = {"total_agents": 0, "active_agents": 0, "total_samples": 0}

    for sim in tqdm(range(num_simulations), desc="ABIDES simulations"):
        seed = seed_start + sim

        config = rmsc04.build_config(seed=seed, end_time=end_time)
        end_state = abides.run(config)

        # Extract order book data
        ob = end_state["agents"][0].order_books["ABM"]
        L1 = ob.get_L1_snapshots()
        n_l1 = len(L1.get("best_bids", []))

        if n_l1 < 2:
            continue  # skip sims with no trading activity

        # Get all trading agents
        trading_agents = [
            a for a in end_state["agents"]
            if hasattr(a, "holdings") and a.type != "ExchangeAgent"
        ]
        num_agents = len(trading_agents)
        stats["total_agents"] += num_agents

        # Build agent grid
        grid = AgentGrid(num_agents)
        H, W = grid.grid_h, grid.grid_w

        # Determine snapshot time indices (evenly spaced across L1 snapshots)
        snap_indices = np.linspace(0, n_l1 - 1, min(num_snapshots, n_l1), dtype=int)

        # For each consecutive pair of snapshots, generate a training sample
        for i in range(len(snap_indices) - 1):
            t_idx = snap_indices[i]
            t1_idx = snap_indices[i + 1]

            l1_prices_t = _get_l1_prices(L1, t_idx)
            l1_prices_t1 = _get_l1_prices(L1, t1_idx)

            # Extract state vectors for all agents
            states_t = np.stack([
                _extract_agent_state(a, l1_prices_t, t_idx, n_l1)
                for a in trading_agents
            ])  # [N, 128]

            states_t1 = np.stack([
                _extract_agent_state(a, l1_prices_t1, t1_idx, n_l1)
                for a in trading_agents
            ])  # [N, 128]

            # Agent types and capitals for grid arrangement
            types = torch.tensor([
                int(_TYPE_MAP.get(type(a).__name__, AgentType.NOISE_TRADER))
                for a in trading_agents
            ])
            capitals = torch.tensor([
                a.holdings.get("CASH", 0) for a in trading_agents
            ], dtype=torch.float32)

            states_t_tensor = torch.from_numpy(states_t)
            states_t1_tensor = torch.from_numpy(states_t1)

            # Arrange into grid
            grid_t, grid_types, sort_idx = grid.arrange(states_t_tensor, types, capitals)
            grid_t1, _, _ = grid.arrange(states_t1_tensor, types, capitals)

            # Market condition vector [32]
            market_cond = torch.zeros(32)
            market_cond[0] = l1_prices_t.get("mid", 100000) / 100000.0
            market_cond[1] = l1_prices_t.get("spread", 0) / 100000.0
            market_cond[2] = l1_prices_t.get("bid", 100000) / 100000.0
            market_cond[3] = l1_prices_t.get("ask", 100000) / 100000.0
            market_cond[4] = float(t_idx) / max(n_l1, 1)  # time progress

            # Save
            torch.save({
                "state_t": grid_t,
                "state_t1": grid_t1,
                "market_cond": market_cond,
                "agent_types": grid_types,
                "sim_id": sim,
                "time_index": int(t_idx),
            }, out / f"sample_{sample_idx:06d}.pt")

            sample_idx += 1

        stats["active_agents"] += len([
            a for a in trading_agents if a.holdings.get("ABM", 0) != 0
        ])

    stats["total_samples"] = sample_idx
    print(f"\nGenerated {sample_idx} samples from {num_simulations} simulations")
    print(f"Avg agents/sim: {stats['total_agents'] / max(num_simulations, 1):.0f}")
    print(f"Avg active agents/sim: {stats['active_agents'] / max(num_simulations, 1):.0f}")
    print(f"Output: {output_dir}")

    return sample_idx
