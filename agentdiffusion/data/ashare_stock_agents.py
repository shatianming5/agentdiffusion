"""A-Share stock-level agent dataset: each grid cell = one real stock.

Each stock is treated as an autonomous agent whose state evolves over
1-minute time windows.  Stocks are arranged on the grid by industry
sector (proxied by stock code prefix).

Data source: L3 tick-by-tick data from data/external/20240619/
  - 逐笔委托.csv  (order-by-order)
  - 行情.csv       (10-level snapshots)

Per-stock per-window features (d_state=6):
  0: return        — log return of VWAP within the window
  1: log_volume    — log(1 + total volume)
  2: spread        — (ask_p1 - bid_p1) / mid, normalised
  3: imbalance     — (bid_vol - ask_vol) / total
  4: volatility    — std of intra-window order prices
  5: order_intensity — number of orders / 100
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .ashare_l3_dataset import parse_time_ms

logger = logging.getLogger(__name__)

D_STATE = 6
# Trading hours in seconds since midnight
OPEN_SEC = 34200.0   # 09:30:00
CLOSE_SEC = 54000.0  # 15:00:00

# Sector classification by stock code prefix
SECTOR_ORDER = [
    "SZ_MAIN",   # 000xxx, 001xxx
    "SZ_SME",    # 002xxx
    "SZ_CHINEXT", # 003xxx, 300xxx
    "SH_MAIN",   # 600xxx, 601xxx, 603xxx
    "SH_STAR",   # 688xxx
    "OTHER",
]


def classify_sector(stock_code: str) -> str:
    """Map a stock code (directory name like '000001.SZ') to sector label."""
    # Extract numeric prefix (first 3 digits)
    digits = "".join(c for c in stock_code if c.isdigit())
    if len(digits) < 3:
        return "OTHER"
    prefix3 = int(digits[:3])
    prefix6 = int(digits[:6]) if len(digits) >= 6 else prefix3 * 1000

    if prefix3 in (0, 1):          # 000xxx, 001xxx
        return "SZ_MAIN"
    if prefix3 == 2:               # 002xxx
        return "SZ_SME"
    if prefix3 == 3 or prefix3 == 300:  # 003xxx, 300xxx
        return "SZ_CHINEXT"
    if prefix6 >= 300000 and prefix6 < 310000:
        return "SZ_CHINEXT"
    if prefix3 in (600, 601, 603):
        return "SH_MAIN"
    if prefix3 == 688:
        return "SH_STAR"
    # Fallback: check 6-digit prefix
    if prefix6 >= 0 and prefix6 < 2000:
        return "SZ_MAIN"
    if prefix6 >= 2000 and prefix6 < 3000:
        return "SZ_SME"
    if prefix6 >= 3000 and prefix6 < 4000:
        return "SZ_CHINEXT"
    if prefix6 >= 600000 and prefix6 < 606000:
        return "SH_MAIN"
    if prefix6 >= 688000 and prefix6 < 689000:
        return "SH_STAR"
    return "OTHER"


def sort_stocks_by_sector(stock_dirs: list[Path]) -> tuple[list[Path], list[str]]:
    """Sort stock directories by sector, then by code within sector.

    Returns:
        sorted_dirs: stock directories in sector-grouped order
        sector_labels: sector label for each stock
    """
    items = []
    for sd in stock_dirs:
        sector = classify_sector(sd.name)
        items.append((SECTOR_ORDER.index(sector) if sector in SECTOR_ORDER else 999,
                       sd.name, sd, sector))
    items.sort(key=lambda x: (x[0], x[1]))
    sorted_dirs = [it[2] for it in items]
    sector_labels = [it[3] for it in items]
    return sorted_dirs, sector_labels


def load_stock_orders(
    stock_dir: Path,
    encoding: str = "gbk",
) -> pd.DataFrame | None:
    """Load 逐笔委托.csv for one stock. Returns None on failure."""
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
        # Filter trading hours and valid prices
        df = df[(df["timestamp"] >= OPEN_SEC) & (df["timestamp"] <= CLOSE_SEC)]
        df = df[df["price"] > 0]
        return df if len(df) > 0 else None
    except Exception as e:
        logger.warning("Failed loading orders from %s: %s", stock_dir.name, e)
        return None


def load_stock_snapshots(
    stock_dir: Path,
    encoding: str = "gbk",
) -> pd.DataFrame | None:
    """Load 行情.csv snapshots for one stock. Returns None on failure."""
    csv_path = stock_dir / "行情.csv"
    if not csv_path.exists():
        return None
    try:
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
               "index", "total_stocks", "up_stocks", "down_stocks",
               "flat_stocks", "_"]
        )
        df = pd.read_csv(
            csv_path, encoding=encoding,
            names=snap_cols, header=0, low_memory=False,
        )
        df["timestamp"] = df["time"].apply(parse_time_ms)
        # Convert prices from 万分 to 元
        for col in [c for c in df.columns
                    if "_p" in c or c in ["last_price", "high", "low",
                                           "open", "prev_close"]]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0) / 10000.0
        df = df[(df["timestamp"] >= OPEN_SEC) & (df["timestamp"] <= CLOSE_SEC)]
        return df if len(df) > 0 else None
    except Exception as e:
        logger.warning("Failed loading snapshots from %s: %s", stock_dir.name, e)
        return None


def aggregate_stock_features(
    orders: pd.DataFrame,
    snapshots: pd.DataFrame | None,
    time_edges: np.ndarray,
) -> np.ndarray:
    """Aggregate one stock's L3 data into per-window features.

    Returns: [T, D_STATE] float32 array.

    Uses vectorised np.digitize + np.bincount (no Python loops over windows).
    """
    T = len(time_edges) - 1
    features = np.zeros((T, D_STATE), dtype=np.float32)

    # --- Orders-based features ---
    ts = orders["timestamp"].values.astype(np.float64)
    pr = orders["price"].values.astype(np.float64)
    sz = orders["size"].values.astype(np.float64)

    time_bin = np.digitize(ts, time_edges) - 1
    time_bin = np.clip(time_bin, 0, T - 1)

    # Volume-weighted average price per window
    px_sz = pr * sz
    sum_px_sz = np.bincount(time_bin, weights=px_sz, minlength=T)[:T]
    sum_sz = np.bincount(time_bin, weights=sz, minlength=T)[:T]
    safe_sz = np.where(sum_sz > 0, sum_sz, 1.0)
    vwap = sum_px_sz / safe_sz

    # Feature 0: log return of VWAP
    vwap_safe = np.where(vwap > 0, vwap, np.nan)
    log_vwap = np.log(np.nan_to_num(vwap_safe, nan=1.0) + 1e-12)
    returns = np.zeros(T, dtype=np.float64)
    returns[1:] = np.diff(log_vwap)
    # Where vwap was zero (no orders), return stays 0
    returns = np.nan_to_num(returns, nan=0.0)
    features[:, 0] = np.clip(returns * 100, -5, 5)

    # Feature 1: log volume
    features[:, 1] = np.clip(np.log1p(sum_sz) / 15, 0, 5)

    # Feature 4: volatility (std of order prices within window)
    # Use Welford-style: Var = E[X^2] - E[X]^2
    order_count = np.bincount(time_bin, minlength=T)[:T].astype(np.float64)
    sum_pr = np.bincount(time_bin, weights=pr, minlength=T)[:T]
    sum_pr2 = np.bincount(time_bin, weights=pr * pr, minlength=T)[:T]
    safe_cnt = np.where(order_count > 1, order_count, 1.0)
    mean_pr = sum_pr / safe_cnt
    var_pr = sum_pr2 / safe_cnt - mean_pr ** 2
    var_pr = np.clip(var_pr, 0, None)
    std_pr = np.sqrt(var_pr)
    # Normalise by mean price to get relative volatility
    features[:, 4] = np.clip(
        std_pr / np.maximum(mean_pr, 1e-8) * 100, 0, 5
    )

    # Feature 5: order intensity
    features[:, 5] = np.clip(order_count / 100.0, 0, 5)

    # --- Snapshot-based features (spread, imbalance) ---
    if snapshots is not None and len(snapshots) > 0:
        snap_ts = snapshots["timestamp"].values.astype(np.float64)
        ask_p1 = snapshots.get("ask_p1", pd.Series(dtype=float)).values.astype(np.float64)
        bid_p1 = snapshots.get("bid_p1", pd.Series(dtype=float)).values.astype(np.float64)
        ask_v1 = snapshots.get("ask_v1", pd.Series(dtype=float)).values.astype(np.float64)
        bid_v1 = snapshots.get("bid_v1", pd.Series(dtype=float)).values.astype(np.float64)

        snap_bin = np.digitize(snap_ts, time_edges) - 1
        snap_bin = np.clip(snap_bin, 0, T - 1)

        snap_cnt = np.bincount(snap_bin, minlength=T)[:T].astype(np.float64)
        safe_snap = np.where(snap_cnt > 0, snap_cnt, 1.0)

        mid = (ask_p1 + bid_p1) / 2.0
        spread_raw = ask_p1 - bid_p1

        # Feature 2: normalised spread (bps / 50)
        sum_spread = np.bincount(snap_bin, weights=np.nan_to_num(spread_raw), minlength=T)[:T]
        sum_mid = np.bincount(snap_bin, weights=np.nan_to_num(mid), minlength=T)[:T]
        avg_spread = sum_spread / safe_snap
        avg_mid = sum_mid / safe_snap
        spread_bps = avg_spread / np.maximum(avg_mid, 1e-8) * 10000
        features[:, 2] = np.clip(spread_bps / 50, -5, 5)

        # Feature 3: imbalance
        sum_bidv = np.bincount(snap_bin, weights=np.nan_to_num(bid_v1), minlength=T)[:T]
        sum_askv = np.bincount(snap_bin, weights=np.nan_to_num(ask_v1), minlength=T)[:T]
        avg_bidv = sum_bidv / safe_snap
        avg_askv = sum_askv / safe_snap
        total_vol = avg_bidv + avg_askv
        features[:, 3] = np.where(
            total_vol > 0,
            (avg_bidv - avg_askv) / total_vol,
            0.0,
        )

    # --- Forward-fill empty windows ---
    active = order_count > 0
    if not active.all() and active.any():
        last_seen = np.maximum.accumulate(
            np.where(active, np.arange(T), -1)
        )
        valid = last_seen >= 0
        features[valid] = features[last_seen[valid]]

    return features


class AShareStockAgentDataset(Dataset):
    """A-Share stock-level agent dataset: each grid cell = one real stock.

    Loads L3 data for the top-N most active stocks, aggregates per-window
    features, and arranges stocks on a 2D grid grouped by sector.

    Args:
        data_dir: directory containing stock subdirectories (e.g. data/external/20240619/)
        total_frames: frames per sequence
        cond_frames: condition frames
        window_seconds: aggregation window (default 60s = 1 minute)
        grid_h: grid height
        grid_w: grid width
        max_stocks: maximum number of stocks to load
    """

    def __init__(
        self,
        data_dir: str,
        total_frames: int = 20,
        cond_frames: int = 4,
        window_seconds: float = 60.0,
        grid_h: int = 32,
        grid_w: int = 32,
        max_stocks: int = 1000,
    ):
        self.total_frames = total_frames
        self.cond_frames = cond_frames
        self.grid_h = grid_h
        self.grid_w = grid_w

        data_dir = Path(data_dir)
        stock_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        logger.info("Found %d stock directories in %s", len(stock_dirs), data_dir)

        # --- Phase 1: quick scan to find top-N most active stocks ---
        activity = []
        for sd in stock_dirs:
            csv_path = sd / "逐笔委托.csv"
            if csv_path.exists():
                try:
                    # Count lines as activity proxy (fast, no full parse)
                    with open(csv_path, "r", encoding="gbk", errors="replace") as f:
                        n_lines = sum(1 for _ in f) - 1  # minus header
                    activity.append((n_lines, sd))
                except Exception:
                    pass

        # Sort by activity (descending), take top max_stocks
        activity.sort(key=lambda x: -x[0])
        selected_dirs = [sd for _, sd in activity[:max_stocks]]
        logger.info("Selected top %d stocks by order count", len(selected_dirs))

        # --- Phase 2: sort by sector for grid layout ---
        selected_dirs, sector_labels = sort_stocks_by_sector(selected_dirs)

        # Build stock_code → grid position mapping
        n_stocks = min(len(selected_dirs), grid_h * grid_w)
        selected_dirs = selected_dirs[:n_stocks]
        sector_labels = sector_labels[:n_stocks]

        self.stock_codes = [sd.name for sd in selected_dirs]
        self.sector_labels = sector_labels
        self.grid_positions = {}  # stock_code → (row, col)
        for idx, code in enumerate(self.stock_codes):
            r, c = idx // grid_w, idx % grid_w
            self.grid_positions[code] = (r, c)

        logger.info("Grid layout: %d stocks on %dx%d grid", n_stocks, grid_h, grid_w)
        # Log sector distribution
        from collections import Counter
        sector_counts = Counter(sector_labels)
        for sec in SECTOR_ORDER:
            if sec in sector_counts:
                logger.info("  %s: %d stocks", sec, sector_counts[sec])

        # --- Phase 3: build time edges (shared across all stocks) ---
        time_edges = np.arange(OPEN_SEC, CLOSE_SEC + window_seconds, window_seconds)
        T_total = len(time_edges) - 1
        logger.info("Time windows: %d windows of %.0fs (%.1f hours)",
                     T_total, window_seconds,
                     (CLOSE_SEC - OPEN_SEC) / 3600)

        # --- Phase 4: load and aggregate each stock ---
        grid = np.zeros((T_total, grid_h, grid_w, D_STATE), dtype=np.float32)
        loaded = 0

        for i, sd in enumerate(selected_dirs):
            orders = load_stock_orders(sd)
            if orders is None:
                continue

            snapshots = load_stock_snapshots(sd)

            try:
                feats = aggregate_stock_features(orders, snapshots, time_edges)
            except Exception as e:
                logger.warning("Aggregation failed for %s: %s", sd.name, e)
                continue

            r, c = i // grid_w, i % grid_w
            grid[:, r, c, :] = feats
            loaded += 1

            if loaded % 50 == 0:
                logger.info("  Loaded %d / %d stocks...", loaded, n_stocks)

        logger.info("Loaded %d / %d stocks successfully", loaded, n_stocks)

        # --- Phase 5: normalise per-feature across all stocks and windows ---
        flat = grid.reshape(-1, D_STATE)
        mean = flat.mean(axis=0, keepdims=True)
        std = flat.std(axis=0, keepdims=True).clip(min=1e-8)
        grid = ((grid - mean) / std).clip(-5, 5).astype(np.float32)

        # --- Phase 6: split into sequences ---
        self.all_sequences = []
        stride = total_frames // 2
        for i in range(0, T_total - total_frames + 1, stride):
            seq = grid[i:i + total_frames]
            self.all_sequences.append(torch.from_numpy(seq).float())

        logger.info(
            "AShareStockAgentDataset: %d stocks on %dx%d grid, "
            "%d windows (%.0fs) -> %d sequences of %d frames",
            loaded, grid_h, grid_w,
            T_total, window_seconds,
            len(self.all_sequences), total_frames,
        )

    def __len__(self) -> int:
        return len(self.all_sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "frames": self.all_sequences[idx],
            "market_conds": torch.zeros(self.total_frames, 32),
        }

    def get_sector_mask(self, sector: str) -> np.ndarray:
        """Return a boolean mask [grid_h, grid_w] for cells belonging to a sector."""
        mask = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        for code, (r, c) in self.grid_positions.items():
            if classify_sector(code) == sector:
                mask[r, c] = True
        return mask

    def get_stock_position(self, stock_code: str) -> tuple[int, int] | None:
        """Return (row, col) for a stock code, or None if not on grid."""
        return self.grid_positions.get(stock_code)
