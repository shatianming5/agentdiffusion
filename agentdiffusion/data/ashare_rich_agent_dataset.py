"""A-Share Rich Agent Dataset: stores RAW ORDER SEQUENCES per stock per window.

Unlike AShareStockAgentDataset which stores 6-dim aggregated features,
this dataset stores the raw order sequence as [N_max, 10] feature tensors
per stock per window, enabling the RichAgentEncoder to encode them at
training time into 200-dim latent vectors.

Grid layout: 10x10 (100 stocks), sector-grouped (by stock code prefix).
Data source: L3 tick-by-tick data from data/external/20240619/

Per-order features (d_raw_order=10, via extract_rich_order_features):
    0: relative_price      (distance from mid, in bps)
    1: log_size
    2: direction           (+1 buy, -1 sell)
    3: order_type          (1=new, 2=modify, 3=cancel)
    4: relative_time       (within window, 0-1 normalized)
    5: is_executed         (did this order get filled?)
    6: cancel_flag         (was this order cancelled?)
    7: price_aggressiveness(how close to best price)
    8: size_percentile     (relative to typical size)
    9: time_since_last_order (normalized)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .ashare_l3_dataset import parse_time_ms
from .ashare_stock_agents import (
    OPEN_SEC,
    CLOSE_SEC,
    sort_stocks_by_sector,
)
from ..models.rich_agent_encoder import (
    D_RAW_ORDER,
    extract_rich_order_features,
)

logger = logging.getLogger(__name__)

MARKET_COND_DIM = 8


def _load_orders_with_direction(
    stock_dir: Path,
    encoding: str = "gbk",
) -> pd.DataFrame | None:
    """Load 逐笔委托.csv and add 'direction' column required by extract_rich_order_features.

    Returns a DataFrame with columns: timestamp, price, size, direction,
    order_type, plus the originals. Returns None on failure.
    """
    csv_path = stock_dir / "逐笔委托.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(
            csv_path, encoding=encoding,
            names=["code", "exch_code", "date", "time", "order_num",
                   "exch_order_id", "order_type", "side", "price", "size", "_"],
            header=0, low_memory=False,
        )
        df["timestamp"] = df["time"].apply(parse_time_ms)
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0) / 10000.0
        df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0)
        # Add direction column: B=+1, S=-1, else 0
        df["direction"] = df["side"].map({"B": 1, "S": -1}).fillna(0).astype(int)
        # Filter trading hours and valid prices
        df = df[(df["timestamp"] >= OPEN_SEC) & (df["timestamp"] <= CLOSE_SEC)]
        df = df[df["price"] > 0]
        return df if len(df) > 0 else None
    except Exception as e:
        logger.warning("Failed loading orders from %s: %s", stock_dir.name, e)
        return None


def _estimate_mid_prices(
    orders_df: pd.DataFrame,
    time_edges: np.ndarray,
) -> np.ndarray:
    """Estimate mid-price per window from order VWAP.

    Returns: [T] float64 array of mid-price estimates.
    """
    T = len(time_edges) - 1
    ts = orders_df["timestamp"].values.astype(np.float64)
    pr = orders_df["price"].values.astype(np.float64)
    sz = orders_df["size"].values.astype(np.float64)

    time_bin = np.digitize(ts, time_edges) - 1
    time_bin = np.clip(time_bin, 0, T - 1)

    px_sz = pr * sz
    sum_px_sz = np.bincount(time_bin, weights=px_sz, minlength=T)[:T]
    sum_sz = np.bincount(time_bin, weights=sz, minlength=T)[:T]
    safe_sz = np.where(sum_sz > 0, sum_sz, 1.0)
    vwap = sum_px_sz / safe_sz

    # Forward-fill windows with no orders
    last_valid = 0.0
    for t in range(T):
        if vwap[t] > 0:
            last_valid = vwap[t]
        else:
            vwap[t] = last_valid

    # If still zero (no orders at all), use global median
    if vwap.max() == 0:
        med = np.median(pr[pr > 0]) if (pr > 0).any() else 1.0
        vwap[:] = med

    return vwap


class AShareRichAgentDataset(Dataset):
    """A-Share rich agent dataset: raw order sequences per stock per window.

    Pre-computes all raw order features during __init__ and stores them.
    Each sequence is [T, 100, N_max, 10] for orders plus [T, 100, N_max]
    for masks.

    Args:
        data_dir: directory containing stock subdirectories
        total_frames: frames per sequence
        cond_frames: condition frames
        window_seconds: aggregation window (default 60s)
        grid_h: grid height (default 10)
        grid_w: grid width (default 10)
        max_stocks: maximum number of stocks (default 100)
        n_max_orders: max orders per window per stock (default 64)
    """

    def __init__(
        self,
        data_dir: str,
        total_frames: int = 20,
        cond_frames: int = 4,
        window_seconds: float = 60.0,
        grid_h: int = 10,
        grid_w: int = 10,
        max_stocks: int = 100,
        n_max_orders: int = 64,
    ):
        self.total_frames = total_frames
        self.cond_frames = cond_frames
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_max_orders = n_max_orders

        data_dir = Path(data_dir)
        stock_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        logger.info("Found %d stock directories in %s", len(stock_dirs), data_dir)

        # --- Phase 1: find top-N most active stocks by order count ---
        activity = []
        for sd in stock_dirs:
            csv_path = sd / "逐笔委托.csv"
            if csv_path.exists():
                try:
                    with open(csv_path, "r", encoding="gbk", errors="replace") as f:
                        n_lines = sum(1 for _ in f) - 1
                    activity.append((n_lines, sd))
                except Exception:
                    pass

        activity.sort(key=lambda x: -x[0])
        selected_dirs = [sd for _, sd in activity[:max_stocks]]
        logger.info("Selected top %d stocks by order count", len(selected_dirs))

        # --- Phase 2: sort by sector for grid layout ---
        selected_dirs, sector_labels = sort_stocks_by_sector(selected_dirs)
        n_cells = grid_h * grid_w
        n_stocks = min(len(selected_dirs), n_cells)
        selected_dirs = selected_dirs[:n_stocks]
        sector_labels = sector_labels[:n_stocks]

        self.stock_codes = [sd.name for sd in selected_dirs]
        self.sector_labels = sector_labels
        self.grid_positions = {}
        for idx, code in enumerate(self.stock_codes):
            r, c = idx // grid_w, idx % grid_w
            self.grid_positions[code] = (r, c)

        logger.info(
            "Grid layout: %d stocks on %dx%d grid", n_stocks, grid_h, grid_w
        )

        # --- Phase 3: build time edges ---
        time_edges = np.arange(OPEN_SEC, CLOSE_SEC + window_seconds, window_seconds)
        T_total = len(time_edges) - 1
        logger.info(
            "Time windows: %d windows of %.0fs",
            T_total, window_seconds,
        )

        # --- Phase 4: load raw order features for each stock ---
        # Full grid: [T_total, n_cells, n_max_orders, D_RAW_ORDER]
        all_orders = np.zeros(
            (T_total, n_cells, n_max_orders, D_RAW_ORDER), dtype=np.float32
        )
        all_masks = np.zeros(
            (T_total, n_cells, n_max_orders), dtype=np.bool_
        )
        loaded = 0

        for i, sd in enumerate(selected_dirs):
            orders_df = _load_orders_with_direction(sd)
            if orders_df is None:
                continue

            # Estimate mid prices for normalization
            mid_prices = _estimate_mid_prices(orders_df, time_edges)

            # Extract features for each window
            for t in range(T_total):
                feat, msk = extract_rich_order_features(
                    orders_df,
                    window_start=float(time_edges[t]),
                    window_end=float(time_edges[t + 1]),
                    mid_price=float(mid_prices[t]),
                    n_max=n_max_orders,
                )
                all_orders[t, i] = feat.numpy()
                all_masks[t, i] = msk.numpy()

            loaded += 1
            if loaded % 20 == 0:
                logger.info("  Loaded %d / %d stocks...", loaded, n_stocks)

        logger.info("Loaded %d / %d stocks successfully", loaded, n_stocks)

        # --- Phase 5: split into sequences ---
        self.all_sequences = []
        stride = max(total_frames // 2, 1)

        for start in range(0, T_total - total_frames + 1, stride):
            end = start + total_frames
            seq_orders = torch.from_numpy(
                all_orders[start:end].copy()
            ).float()  # [T, n_cells, n_max, 10]
            seq_masks = torch.from_numpy(
                all_masks[start:end].copy()
            )  # [T, n_cells, n_max]
            self.all_sequences.append((seq_orders, seq_masks))

        logger.info(
            "AShareRichAgentDataset: %d stocks on %dx%d grid, "
            "%d windows -> %d sequences of %d frames, "
            "n_max_orders=%d",
            loaded, grid_h, grid_w,
            T_total, len(self.all_sequences), total_frames,
            n_max_orders,
        )

    def __len__(self) -> int:
        return len(self.all_sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_orders, seq_masks = self.all_sequences[idx]
        # seq_orders: [T, H*W, N_max, 10]
        # seq_masks:  [T, H*W, N_max]
        return {
            "raw_orders": seq_orders,
            "order_masks": seq_masks,
            "market_conds": torch.zeros(self.total_frames, MARKET_COND_DIM),
        }
