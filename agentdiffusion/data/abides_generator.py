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


def _extract_agent_state(agent, L1_prices: dict, step_idx: int, total_steps: int) -> np.ndarray:
    """Extract 128-dim state vector from a single ABIDES TradingAgent."""
    state = np.zeros(128, dtype=np.float32)
    h = agent.holdings

    # [0:32] 持仓信息
    state[0] = h.get("ABM", 0)
    state[1] = h.get("CASH", 0) / 100.0  # cents -> dollars

    # [32:48] 资金状态
    state[32] = h.get("CASH", 0) / 100.0
    state[33] = getattr(agent, "starting_cash", 100000) / 100.0
    # 杠杆 = |持仓价值| / 现金
    pos_val = abs(state[0]) * L1_prices.get("mid", 1000.0)
    cash = max(abs(state[32]), 1.0)
    state[34] = pos_val / cash

    # [48:64] 策略参数（类型编码 + agent 特有参数）
    atype = _TYPE_MAP.get(type(agent).__name__, AgentType.NOISE_TRADER)
    state[48 + int(atype)] = 1.0  # one-hot encoding

    if hasattr(agent, "lambda_a"):
        state[52] = agent.lambda_a * 1e12  # scale to reasonable range
    if hasattr(agent, "kappa"):
        state[53] = agent.kappa * 1e15
    if hasattr(agent, "r_bar"):
        state[54] = agent.r_bar / 100000.0  # normalize

    # [64:80] 历史统计
    exec_orders = getattr(agent, "executed_orders", [])
    if exec_orders:
        prices_exec = [o.fill_price for o in exec_orders if hasattr(o, "fill_price") and o.fill_price]
        if len(prices_exec) >= 2:
            returns = np.diff(np.log(np.array(prices_exec, dtype=float).clip(min=1)))
            state[64] = returns.mean() if len(returns) > 0 else 0  # mean return
            state[65] = returns.std() if len(returns) > 1 else 0   # volatility
            state[66] = state[64] / (state[65] + 1e-10)           # sharpe
        state[67] = len(exec_orders)  # total executions

    # [80:96] 行为特征
    state[80] = len(getattr(agent, "orders", {}))     # active orders
    state[81] = len(exec_orders)                       # executed orders
    state[82] = step_idx / max(total_steps, 1)         # time progress

    # [96:112] 市场观察
    state[96] = L1_prices.get("bid", 0) / 100000.0
    state[97] = L1_prices.get("ask", 0) / 100000.0
    state[98] = L1_prices.get("mid", 0) / 100000.0
    state[99] = L1_prices.get("spread", 0) / 100000.0
    state[100] = L1_prices.get("last_trade", 0) / 100000.0

    # [112:128] 社交/信息 (基础版本使用噪声填充)
    state[112] = float(atype) / 4.0  # type as continuous

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
