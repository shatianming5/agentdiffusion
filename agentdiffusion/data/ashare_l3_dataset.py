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


MARKET_COND_DIM = 8  # number of market conditioning features


def build_market_conditions(
    snapshots: pd.DataFrame,
    time_edges: np.ndarray,
) -> np.ndarray:
    """Extract market-level conditioning from 10-level snapshots.

    Returns: [T, MARKET_COND_DIM] with features:
        0: mid_price_return (log return of mid price)
        1: spread (best ask - best bid, normalized)
        2: depth_imbalance (bid_vol - ask_vol) / (bid_vol + ask_vol)
        3: trade_intensity (trades per second proxy from snapshot frequency)
        4: volatility (rolling std of mid returns)
        5: price_momentum (cumulative return over 5 windows)
        6: total_depth (log total volume at all levels)
        7: time_of_day (normalized 0=open, 1=close)
    """
    T = len(time_edges) - 1
    conds = np.zeros((T, MARKET_COND_DIM), dtype=np.float32)

    # Align snapshots to time windows
    snap_ts = snapshots["timestamp"].values
    ask_p1 = snapshots.get("ask_p1", pd.Series(dtype=float)).values
    bid_p1 = snapshots.get("bid_p1", pd.Series(dtype=float)).values
    ask_v1 = snapshots.get("ask_v1", pd.Series(dtype=float)).values
    bid_v1 = snapshots.get("bid_v1", pd.Series(dtype=float)).values

    if len(ask_p1) == 0 or np.all(ask_p1 == 0):
        return conds

    mid = (ask_p1 + bid_p1) / 2.0
    mid[mid <= 0] = np.nan

    # --- Vectorized: assign snapshots to time bins, aggregate ---
    time_bin = np.digitize(snap_ts, time_edges) - 1
    time_bin = np.clip(time_bin, 0, T - 1)

    # Per-bin aggregates via bincount
    bc_count = np.bincount(time_bin, minlength=T).astype(np.float64)
    bc_mid = np.bincount(time_bin, weights=np.nan_to_num(mid), minlength=T)
    bc_ask = np.bincount(time_bin, weights=np.nan_to_num(ask_p1), minlength=T)
    bc_bid = np.bincount(time_bin, weights=np.nan_to_num(bid_p1), minlength=T)
    bc_askv = np.bincount(time_bin, weights=np.nan_to_num(ask_v1), minlength=T)
    bc_bidv = np.bincount(time_bin, weights=np.nan_to_num(bid_v1), minlength=T)

    safe_count = np.where(bc_count > 0, bc_count, 1)
    w_mid_arr = bc_mid[:T] / safe_count[:T]
    w_ask_arr = bc_ask[:T] / safe_count[:T]
    w_bid_arr = bc_bid[:T] / safe_count[:T]
    w_askv_arr = bc_askv[:T] / safe_count[:T]
    w_bidv_arr = bc_bidv[:T] / safe_count[:T]

    # Total depth (sum all 10 levels)
    total_depth = np.zeros(T, dtype=np.float64)
    for lv in range(1, 11):
        av = snapshots.get(f"ask_v{lv}")
        bv = snapshots.get(f"bid_v{lv}")
        if av is not None:
            total_depth += np.bincount(time_bin, weights=np.nan_to_num(av.values), minlength=T)[:T] / safe_count[:T]
        if bv is not None:
            total_depth += np.bincount(time_bin, weights=np.nan_to_num(bv.values), minlength=T)[:T] / safe_count[:T]

    # Compute features vectorized
    # 0: mid return
    w_mid_safe = np.where(w_mid_arr > 0, w_mid_arr, np.nan)
    returns = np.zeros(T)
    returns[1:] = np.diff(np.log(np.nan_to_num(w_mid_safe, nan=1.0) + 1e-12))
    returns = np.nan_to_num(returns, nan=0.0)
    conds[:, 0] = np.clip(returns * 1000, -5, 5)

    # 1: spread (bps)
    spread = (w_ask_arr - w_bid_arr) / np.maximum(w_mid_arr, 1e-8) * 10000
    conds[:, 1] = np.clip(spread / 50, -5, 5)

    # 2: depth imbalance
    total_bav = w_bidv_arr + w_askv_arr
    conds[:, 2] = np.where(total_bav > 0, (w_bidv_arr - w_askv_arr) / total_bav, 0)

    # 3: trade intensity
    conds[:, 3] = np.clip(bc_count[:T] / 10.0, 0, 5)

    # 4: volatility (rolling std of returns, window=20)
    from numpy.lib.stride_tricks import sliding_window_view
    if T > 20:
        windowed = sliding_window_view(returns, 20)
        vol = np.std(windowed, axis=1)
        conds[19:, 4] = np.clip(vol * 1000, 0, 5)

    # 5: momentum (5-window cumsum)
    if T > 5:
        cum = np.cumsum(returns)
        mom = np.zeros(T)
        mom[5:] = cum[5:] - cum[:-5]
        conds[:, 5] = np.clip(mom * 1000, -5, 5)

    # 6: total depth
    conds[:, 6] = np.clip(np.log1p(total_depth) / 15, 0, 5)

    # 7: time of day
    conds[:, 7] = np.clip((time_edges[:T] - 34200) / (54000 - 34200), 0, 1)

    # Forward-fill empty windows
    for t in range(1, T):
        if bc_count[t] == 0:
            conds[t] = conds[t - 1]

    return conds


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

    # --- Vectorized aggregation (replaces double for-loop) ---
    d_state = 6
    agent_states = np.zeros((T, n_agent_types, d_state), dtype=np.float32)

    # Assign each order to a time bin
    ts = orders["timestamp"].values
    time_bin = np.digitize(ts, time_edges) - 1  # [0, T-1]
    time_bin = np.clip(time_bin, 0, T - 1)

    # Pre-extract arrays for speed
    at = agent_type  # already computed above
    sz = orders["size"].values.astype(np.float64)
    dr = orders["direction"].values.astype(np.float64)
    pr = orders["price"].values.astype(np.float64)
    ot = orders["order_type"].values

    signed_vol = sz * dr
    price_x_size = pr * sz
    is_cancel = (ot == 3).astype(np.float64)
    is_aggr = (at % 2).astype(np.float64)

    # Composite key: (time_bin, agent_type) → single int for np.bincount
    key = time_bin * n_agent_types + at
    n_bins = T * n_agent_types

    # Aggregate with bincount (one pass, no loops)
    sum_signed_vol = np.bincount(key, weights=signed_vol, minlength=n_bins)
    sum_count = np.bincount(key, minlength=n_bins).astype(np.float64)
    sum_price_x_size = np.bincount(key, weights=price_x_size, minlength=n_bins)
    sum_size = np.bincount(key, weights=sz, minlength=n_bins)
    sum_cancel = np.bincount(key, weights=is_cancel, minlength=n_bins)
    sum_aggr = np.bincount(key, weights=is_aggr, minlength=n_bins)

    # Reshape to [T, n_agent_types]
    sum_signed_vol = sum_signed_vol[:n_bins].reshape(T, n_agent_types)
    sum_count = sum_count[:n_bins].reshape(T, n_agent_types)
    sum_price_x_size = sum_price_x_size[:n_bins].reshape(T, n_agent_types)
    sum_size = sum_size[:n_bins].reshape(T, n_agent_types)
    sum_cancel = sum_cancel[:n_bins].reshape(T, n_agent_types)
    sum_aggr = sum_aggr[:n_bins].reshape(T, n_agent_types)

    # Fill agent_states
    agent_states[:, :, 0] = sum_signed_vol                                     # net_position
    agent_states[:, :, 1] = sum_count                                          # order_rate
    safe_size = np.where(sum_size > 0, sum_size, 1)
    agent_states[:, :, 2] = sum_price_x_size / safe_size                       # avg_price
    safe_count = np.where(sum_count > 0, sum_count, 1)
    agent_states[:, :, 3] = sum_cancel / safe_count                            # cancel_rate
    agent_states[:, :, 4] = sum_size / safe_count                              # avg_size
    agent_states[:, :, 5] = sum_aggr / safe_count                              # aggressiveness

    return agent_states.astype(np.float32), time_edges


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
        self.all_sequences = []   # list of [T, H, W, C] tensors
        self.all_market_conds = []  # list of [T, MARKET_COND_DIM] tensors
        H, W = grid_shape
        n_agents = H * W

        for sd in stock_dirs:
            try:
                data = load_stock_l3(sd)
                states, time_edges = build_agent_states_from_orders(
                    data["orders"], window_seconds=window_seconds,
                    n_agent_types=n_agents,
                )
                # Build market conditions aligned to same time windows
                market_conds = build_market_conditions(data["snapshots"], time_edges)
            except Exception as e:
                logger.warning(f"Failed to load {sd.name}: {e}")
                continue

            if len(states) < total_frames:
                continue

            # Normalize agent states per-feature
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
                mc = market_conds[i:i + total_frames]
                if len(seq) == total_frames and len(mc) == total_frames:
                    self.all_sequences.append(torch.from_numpy(seq).float())
                    self.all_market_conds.append(torch.from_numpy(mc).float())

        logger.info(
            f"AShareL3VideoDataset: {len(stock_dirs)} stocks → "
            f"{len(self.all_sequences)} sequences of {total_frames} frames, "
            f"grid={grid_shape}, d_state=6"
        )

    def __len__(self) -> int:
        return len(self.all_sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        frames = self.all_sequences[idx]  # [T, H, W, C]
        market_conds = self.all_market_conds[idx]  # [T, MARKET_COND_DIM]
        return {
            "frames": frames,
            "market_conds": market_conds,
        }
