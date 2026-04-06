"""A-Share Level-3 dataset: tick-by-tick orders + trades + 10-level snapshots.

Parses the Chinese A-share L3 data format (Wind/CSMAR style):
  - 逐笔委托.csv: order-by-order with order_id
  - 逐笔成交.csv: trade-by-trade with buyer/seller order refs
  - 行情.csv: 10-level order book snapshots

Constructs agent state grids from order_id clustering for Video DiT training.

Data format (GBK encoded CSV):
  逐笔委托: 万得代码,交易所代码,自然日,时间,委托编号,交易所委托号,委托类型,委托代码,委托价格,委托数量
  逐笔成交: 万得代码,交易所代码,自然日,时间,成交编号,成交代码,委托代码,BS标志,成交价格,成交数量,叫卖序号,叫买序号
  行情:     万得代码,...,申卖价1-10,申卖量1-10,申买价1-10,申买量1-10
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def parse_time_ms(t: int) -> float:
    """Convert HHMMSSMMM integer to seconds since midnight.
    e.g. 93000123 → 9*3600 + 30*60 + 0 + 0.123 = 34200.123
    """
    ms = t % 1000
    t = t // 1000
    ss = t % 100
    t = t // 100
    mm = t % 100
    hh = t // 100
    return hh * 3600 + mm * 60 + ss + ms / 1000.0


def load_stock_l3(
    stock_dir: str | Path,
    encoding: str = "gbk",
) -> dict[str, pd.DataFrame]:
    """Load all three CSV files for one stock.

    Returns dict with keys 'orders', 'trades', 'snapshots'.
    """
    stock_dir = Path(stock_dir)

    # 逐笔委托
    orders = pd.read_csv(
        stock_dir / "逐笔委托.csv", encoding=encoding,
        names=["code", "exch_code", "date", "time", "order_num",
               "exch_order_id", "order_type", "side", "price", "size", "_"],
        header=0,
    )
    orders["timestamp"] = orders["time"].apply(parse_time_ms)
    orders["price"] = orders["price"] / 10000.0  # 万分 → 元
    orders["direction"] = orders["side"].map({"B": 1, "S": -1}).fillna(0).astype(int)

    # 逐笔成交
    trades = pd.read_csv(
        stock_dir / "逐笔成交.csv", encoding=encoding,
        names=["code", "exch_code", "date", "time", "trade_id",
               "trade_type", "order_type", "bs_flag", "price", "size",
               "ask_order_id", "bid_order_id", "_"],
        header=0,
    )
    trades["timestamp"] = trades["time"].apply(parse_time_ms)
    trades["price"] = trades["price"] / 10000.0

    # 行情 (10档快照)
    snap_cols = (
        ["code", "exch_code", "date", "time",
         "last_price", "volume", "amount", "num_trades", "iopv",
         "trade_flag", "bs_flag", "cum_volume", "cum_amount",
         "high", "low", "open", "prev_close"]
        + [f"ask_p{i}" for i in range(1, 11)]
        + [f"ask_v{i}" for i in range(1, 11)]
        + [f"bid_p{i}" for i in range(1, 11)]
        + [f"bid_v{i}" for i in range(1, 11)]
        + ["wavg_ask", "wavg_bid", "total_ask_vol", "total_bid_vol",
           "index", "total_stocks", "up_stocks", "down_stocks", "flat_stocks", "_"]
    )
    snapshots = pd.read_csv(
        stock_dir / "行情.csv", encoding=encoding,
        names=snap_cols, header=0,
    )
    snapshots["timestamp"] = snapshots["time"].apply(parse_time_ms)
    # Convert prices
    for col in [c for c in snapshots.columns if "_p" in c or c in ["last_price", "high", "low", "open", "prev_close"]]:
        snapshots[col] = snapshots[col] / 10000.0

    return {"orders": orders, "trades": trades, "snapshots": snapshots}


def build_agent_states_from_orders(
    orders: pd.DataFrame,
    window_seconds: float = 1.0,
    n_agent_types: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster orders into agent types and build state grid over time.

    Clustering heuristic (from L3 order features):
      - By order size bucket (4 buckets: micro/small/medium/large)
      - By side (buy/sell) → 2
      - By aggressiveness (close to mid vs far) → 2
      Total: 4 × 2 × 2 = 16 agent types

    Returns:
        agent_states: [T, N_agents, d_state] where d_state includes:
            0: net_position (cumulative signed volume)
            1: order_rate (orders per window)
            2: avg_price (volume-weighted)
            3: cancel_rate
            4: avg_size
            5: aggressiveness (fraction near best price)
        time_edges: [T+1] window boundaries in seconds
    """
    # Filter to trading hours
    orders = orders[(orders["timestamp"] >= 34200) & (orders["timestamp"] <= 54000)].copy()
    if len(orders) == 0:
        return np.zeros((0, n_agent_types, 6), dtype=np.float32), np.array([])

    # Mid price estimate (running median of order prices)
    valid_prices = orders[orders["price"] > 0]["price"]
    if len(valid_prices) == 0:
        return np.zeros((0, n_agent_types, 6), dtype=np.float32), np.array([])
    mid_price = valid_prices.median()

    # Classify each order into agent type
    # Size buckets: 0=micro(<100), 1=small(100-500), 2=medium(500-5000), 3=large(>5000)
    sizes = orders["size"].values
    size_bucket = np.zeros(len(orders), dtype=int)
    size_bucket[sizes >= 100] = 1
    size_bucket[sizes >= 500] = 2
    size_bucket[sizes >= 5000] = 3

    # Side: 0=buy, 1=sell
    side_bucket = (orders["direction"].values == -1).astype(int)

    # Aggressiveness: 0=passive (far from mid), 1=aggressive (close to mid)
    prices = orders["price"].values
    price_dist = np.abs(prices - mid_price) / max(mid_price, 1e-8)
    aggr_bucket = (price_dist < 0.002).astype(int)  # within 0.2% of mid

    agent_type = size_bucket * 4 + side_bucket * 2 + aggr_bucket
    agent_type = np.clip(agent_type, 0, n_agent_types - 1)
    orders = orders.copy()
    orders["agent_type"] = agent_type

    # Time windows
    t_min = orders["timestamp"].min()
    t_max = orders["timestamp"].max()
    time_edges = np.arange(t_min, t_max + window_seconds, window_seconds)
    T = len(time_edges) - 1

    d_state = 6
    agent_states = np.zeros((T, n_agent_types, d_state), dtype=np.float32)

    for t in range(T):
        mask = (orders["timestamp"] >= time_edges[t]) & (orders["timestamp"] < time_edges[t + 1])
        window = orders[mask]
        if len(window) == 0:
            continue

        for a in range(n_agent_types):
            a_mask = window["agent_type"] == a
            a_orders = window[a_mask]
            if len(a_orders) == 0:
                continue

            signed_vol = (a_orders["size"] * a_orders["direction"]).sum()
            n_orders = len(a_orders)
            avg_price = (a_orders["price"] * a_orders["size"]).sum() / max(a_orders["size"].sum(), 1)
            cancel_count = (a_orders["order_type"] == 3).sum()  # type 3 = cancel
            avg_size = a_orders["size"].mean()
            aggr_frac = a_orders["agent_type"].apply(lambda x: x % 2).mean()

            agent_states[t, a, 0] = signed_vol
            agent_states[t, a, 1] = n_orders
            agent_states[t, a, 2] = avg_price
            agent_states[t, a, 3] = cancel_count / max(n_orders, 1)
            agent_states[t, a, 4] = avg_size
            agent_states[t, a, 5] = aggr_frac

    return agent_states, time_edges


class AShareL3VideoDataset(Dataset):
    """A-Share L3 data as video sequences for Video DiT.

    Each sample = T consecutive windows of agent states, reshaped to [T, H, W, C].

    Args:
        data_dir: directory containing stock subdirectories (e.g. data/external/20220601/)
        stock_codes: list of stock codes to include (None = all)
        total_frames: frames per sequence
        cond_frames: condition frames
        window_seconds: aggregation window
        grid_shape: (H, W) for agent grid (H*W >= n_agent_types)
        max_stocks: limit number of stocks to load
    """

    def __init__(
        self,
        data_dir: str,
        stock_codes: list[str] | None = None,
        total_frames: int = 20,
        cond_frames: int = 4,
        window_seconds: float = 1.0,
        grid_shape: tuple[int, int] = (4, 4),
        max_stocks: int = 50,
    ):
        self.total_frames = total_frames
        self.cond_frames = cond_frames
        self.grid_shape = grid_shape

        data_dir = Path(data_dir)
        if stock_codes is None:
            stock_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])[:max_stocks]
        else:
            stock_dirs = [data_dir / c for c in stock_codes if (data_dir / c).exists()]

        logger.info(f"Loading {len(stock_dirs)} stocks from {data_dir}")

        # Build all sequences
        self.all_sequences = []  # list of [T_total, H, W, C] tensors
        H, W = grid_shape
        n_agents = H * W

        for sd in stock_dirs:
            try:
                data = load_stock_l3(sd)
                states, _ = build_agent_states_from_orders(
                    data["orders"], window_seconds=window_seconds,
                    n_agent_types=n_agents,
                )
            except Exception as e:
                logger.warning(f"Failed to load {sd.name}: {e}")
                continue

            if len(states) < total_frames:
                continue

            # Normalize per-feature
            d_state = states.shape[-1]
            flat = states.reshape(-1, d_state)
            mean = flat.mean(axis=0, keepdims=True)
            std = flat.std(axis=0, keepdims=True).clip(min=1e-8)
            states = (states - mean) / std
            states = states.clip(-5, 5)

            # Reshape [T, N_agents, d_state] → [T, H, W, d_state]
            T_total = states.shape[0]
            grid = states.reshape(T_total, H, W, d_state)

            # Split into sequences
            for i in range(0, T_total - total_frames + 1, total_frames // 2):
                seq = grid[i:i + total_frames]
                if len(seq) == total_frames:
                    self.all_sequences.append(torch.from_numpy(seq).float())

        logger.info(
            f"AShareL3VideoDataset: {len(stock_dirs)} stocks → "
            f"{len(self.all_sequences)} sequences of {total_frames} frames, "
            f"grid={grid_shape}, d_state=6"
        )

    def __len__(self) -> int:
        return len(self.all_sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        frames = self.all_sequences[idx]  # [T, H, W, C]
        return {
            "frames": frames,
            "market_conds": torch.zeros(self.total_frames, 32),
        }
