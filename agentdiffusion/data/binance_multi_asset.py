"""Multi-asset Binance aggTrades dataset.

Each "pixel" in the grid is a cryptocurrency (BTC, ETH, SOL, ...).
Per-asset features are aggregated per time window from aggTrades data.

Grid layout (4x4, 13 assets + 3 padding):
  ┌─────┬─────┬─────┬─────┐
  │ BTC │ ETH │ BNB │ SOL │
  ├─────┼─────┼─────┼─────┤
  │ XRP │DOGE │ ADA │AVAX │
  ├─────┼─────┼─────┼─────┤
  │ DOT │MATIC│LINK │ ARB │
  ├─────┼─────┼─────┼─────┤
  │  OP │ pad │ pad │ pad │
  └─────┴─────┴─────┴─────┘

Per-asset features (d_state=8):
  0: log_return (mean return in window)
  1: log_volume (total volume)
  2: buy_ratio (fraction of buyer-maker trades)
  3: trade_count (number of trades)
  4: volatility (std of returns within window)
  5: vwap_deviation (VWAP vs simple mean price)
  6: max_trade_size (largest single trade)
  7: price_momentum (cumulative return over last 5 windows)
"""

from __future__ import annotations

import logging
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT",
    "DOTUSDT", "MATICUSDT", "LINKUSDT", "ARBUSDT",
    "OPUSDT",
]
GRID_H, GRID_W = 4, 4
D_STATE = 8


def load_aggtrades_zip(zip_path: str | Path) -> pd.DataFrame:
    """Load a single Binance aggTrades zip file."""
    zip_path = Path(zip_path)
    with ZipFile(zip_path) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(
                f,
                header=None,
                usecols=[1, 2, 5, 6],
                dtype={1: np.float32, 2: np.float32, 5: np.int64, 6: np.int8},
            )
    df.columns = ["price", "qty", "timestamp", "is_buyer_maker"]
    return df


def _forward_fill_features(features: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    """Carry the previous window forward for empty windows."""
    if active_mask.all() or not active_mask.any():
        return features

    last_seen = np.maximum.accumulate(
        np.where(active_mask, np.arange(len(active_mask)), -1)
    )
    valid = last_seen >= 0
    features[valid] = features[last_seen[valid]]
    return features


def aggregate_per_window(
    df: pd.DataFrame,
    window_seconds: float = 60.0,
) -> np.ndarray:
    """Aggregate aggTrades into per-window features.

    Returns: [T, D_STATE] array of features.
    """
    if df.empty:
        return np.zeros((0, D_STATE), dtype=np.float32)

    timestamps = df["timestamp"].to_numpy(dtype=np.int64, copy=False)
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    prices = df["price"].to_numpy(dtype=np.float64, copy=False)[order]
    qtys = df["qty"].to_numpy(dtype=np.float64, copy=False)[order]
    is_buyer = df["is_buyer_maker"].to_numpy(dtype=np.float64, copy=False)[order]

    window_ms = max(int(window_seconds * 1000), 1)
    time_bin = ((timestamps - timestamps[0]) // window_ms).astype(np.int64)
    T = int(time_bin[-1]) + 1

    features = np.zeros((T, D_STATE), dtype=np.float32)
    counts = np.bincount(time_bin, minlength=T).astype(np.float64)
    active = counts > 0
    if not active.any():
        return features

    sum_price = np.bincount(time_bin, weights=prices, minlength=T)
    sum_qty = np.bincount(time_bin, weights=qtys, minlength=T)
    sum_buy = np.bincount(time_bin, weights=is_buyer, minlength=T)
    sum_px_qty = np.bincount(time_bin, weights=prices * qtys, minlength=T)

    mean_price = np.divide(sum_price, counts, out=np.zeros(T, dtype=np.float64), where=active)
    vwap = np.divide(sum_px_qty, sum_qty, out=np.zeros(T, dtype=np.float64), where=sum_qty > 0)

    starts = np.flatnonzero(np.r_[True, time_bin[1:] != time_bin[:-1]])
    bins = time_bin[starts]
    ends = np.r_[starts[1:] - 1, len(time_bin) - 1]

    first_price = np.zeros(T, dtype=np.float64)
    last_price = np.zeros(T, dtype=np.float64)
    first_price[bins] = prices[starts]
    last_price[bins] = prices[ends]

    max_trade_size = np.zeros(T, dtype=np.float64)
    np.maximum.at(max_trade_size, time_bin, qtys)

    features[:, 0] = np.clip(
        np.divide(
            np.log(np.divide(last_price, first_price, out=np.ones(T), where=first_price > 0)),
            1.0,
            out=np.zeros(T),
            where=first_price > 0,
        ) * 100,
        -5,
        5,
    )
    features[:, 1] = np.clip(np.log1p(sum_qty) / 10, 0, 5)
    features[:, 2] = np.divide(sum_buy, counts, out=np.full(T, 0.5, dtype=np.float64), where=active)
    features[:, 3] = np.clip(counts / 100.0, 0, 5)

    if len(prices) > 1:
        log_prices = np.log(prices + 1e-10)
        diff_bins_mask = time_bin[1:] == time_bin[:-1]
        diff_bins = time_bin[1:][diff_bins_mask]
        diff_vals = np.diff(log_prices)[diff_bins_mask]
        if len(diff_vals) > 0:
            diff_counts = np.bincount(diff_bins, minlength=T).astype(np.float64)
            diff_sum = np.bincount(diff_bins, weights=diff_vals, minlength=T)
            diff_sq_sum = np.bincount(diff_bins, weights=diff_vals * diff_vals, minlength=T)
            diff_mean = np.divide(diff_sum, diff_counts, out=np.zeros(T), where=diff_counts > 0)
            diff_var = np.divide(diff_sq_sum, diff_counts, out=np.zeros(T), where=diff_counts > 0) - diff_mean ** 2
            features[:, 4] = np.clip(np.sqrt(np.clip(diff_var, 0, None)) * 100, 0, 5)

    features[:, 5] = np.clip(
        np.divide(vwap - mean_price, mean_price + 1e-10, out=np.zeros(T), where=active) * 1000,
        -5,
        5,
    )
    features[:, 6] = np.clip(np.log1p(max_trade_size) / 5, 0, 5)

    features = _forward_fill_features(features, active)

    # Momentum (5-window cumulative return)
    cum_ret = np.cumsum(features[:, 0])
    features[5:, 7] = np.clip(cum_ret[5:] - cum_ret[:-5], -5, 5)

    return features


class BinanceMultiAssetDataset(Dataset):
    """Multi-asset video dataset: 13 crypto assets as a 4x4 agent grid.

    Args:
        data_dir: directory containing aggTrades subdirectories per symbol
        total_frames: frames per sequence
        cond_frames: condition frames
        window_seconds: aggregation window (seconds)
        max_months: max monthly zip files to load per symbol
    """

    def __init__(
        self,
        data_dir: str,
        total_frames: int = 20,
        cond_frames: int = 4,
        window_seconds: float = 60.0,
        max_months: int = 3,
    ):
        self.total_frames = total_frames
        self.cond_frames = cond_frames

        data_dir = Path(data_dir)
        logger.info("Loading Binance multi-asset data from %s", data_dir)

        # Load each symbol's aggTrades
        all_symbol_features = {}  # symbol → [T, D_STATE]
        min_T = float("inf")

        for sym in SYMBOLS:
            sym_dir = data_dir / sym
            if not sym_dir.exists():
                logger.warning("Missing %s, will zero-pad", sym)
                continue

            zips = sorted(sym_dir.glob("*.zip"))[:max_months]
            if not zips:
                continue

            feature_blocks = []
            for z in zips:
                try:
                    block = aggregate_per_window(load_aggtrades_zip(z), window_seconds)
                    if len(block) > 0:
                        feature_blocks.append(block)
                except Exception as e:
                    logger.warning("Failed to load %s: %s", z, e)

            if not feature_blocks:
                continue

            features = np.concatenate(feature_blocks, axis=0)

            # Normalize per-feature
            mean = features.mean(axis=0, keepdims=True)
            std = features.std(axis=0, keepdims=True).clip(min=1e-8)
            features = ((features - mean) / std).clip(-5, 5)

            all_symbol_features[sym] = features
            min_T = min(min_T, len(features))
            logger.info("  %s: %d windows from %d files", sym, len(features), len(zips))

        if not all_symbol_features:
            logger.error("No data loaded!")
            self.all_sequences = []
            return

        # Truncate all to same length
        min_T = int(min_T)

        # Build grid: [T, H, W, D_STATE]
        grid = np.zeros((min_T, GRID_H, GRID_W, D_STATE), dtype=np.float32)
        for i, sym in enumerate(SYMBOLS):
            r, c = i // GRID_W, i % GRID_W
            if sym in all_symbol_features:
                grid[:, r, c, :] = all_symbol_features[sym][:min_T]

        # Split into sequences
        self.all_sequences = []
        stride = total_frames // 2
        for i in range(0, min_T - total_frames + 1, stride):
            seq = grid[i:i + total_frames]
            self.all_sequences.append(torch.from_numpy(seq).float())

        logger.info(
            "BinanceMultiAssetDataset: %d symbols, %d windows → %d sequences, grid=(%d,%d), d_state=%d",
            len(all_symbol_features), min_T, len(self.all_sequences), GRID_H, GRID_W, D_STATE,
        )

    def __len__(self) -> int:
        return len(self.all_sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        frames = self.all_sequences[idx]
        return {
            "frames": frames,
            "market_conds": torch.zeros(self.total_frames, 32),
        }
