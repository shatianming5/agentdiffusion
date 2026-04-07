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
import io

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
                names=["agg_id", "price", "qty", "first_id", "last_id", "timestamp", "is_buyer_maker", "best_match"],
                dtype={"price": float, "qty": float, "timestamp": int, "is_buyer_maker": bool},
            )
    df["timestamp_s"] = df["timestamp"] / 1000.0  # ms → seconds
    return df


def aggregate_per_window(
    df: pd.DataFrame,
    window_seconds: float = 60.0,
) -> np.ndarray:
    """Aggregate aggTrades into per-window features.

    Returns: [T, D_STATE] array of features.
    """
    ts = df["timestamp_s"].values
    t_min, t_max = ts.min(), ts.max()
    edges = np.arange(t_min, t_max + window_seconds, window_seconds)
    T = len(edges) - 1

    features = np.zeros((T, D_STATE), dtype=np.float32)
    prices = df["price"].values
    qtys = df["qty"].values
    is_buyer = df["is_buyer_maker"].values

    time_bin = np.digitize(ts, edges) - 1
    time_bin = np.clip(time_bin, 0, T - 1)

    # Vectorized aggregates
    for t in range(T):
        mask = time_bin == t
        if mask.sum() == 0:
            if t > 0:
                features[t] = features[t - 1]
            continue

        p = prices[mask]
        q = qtys[mask]
        b = is_buyer[mask]
        n = len(p)

        mean_p = p.mean()
        ret = np.log(p[-1] / p[0]) if p[0] > 0 else 0.0

        features[t, 0] = np.clip(ret * 100, -5, 5)  # log return scaled
        features[t, 1] = np.clip(np.log1p(q.sum()) / 10, 0, 5)  # log volume
        features[t, 2] = b.mean() if n > 0 else 0.5  # buy ratio
        features[t, 3] = np.clip(n / 100.0, 0, 5)  # trade count
        features[t, 4] = np.clip(np.std(np.diff(np.log(p + 1e-10))) * 100 if n > 1 else 0, 0, 5)  # vol
        vwap = (p * q).sum() / (q.sum() + 1e-10)
        features[t, 5] = np.clip((vwap - mean_p) / (mean_p + 1e-10) * 1000, -5, 5)  # vwap dev
        features[t, 6] = np.clip(np.log1p(q.max()) / 5, 0, 5)  # max trade size
        features[t, 7] = 0  # momentum filled below

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

            dfs = []
            for z in zips:
                try:
                    dfs.append(load_aggtrades_zip(z))
                except Exception as e:
                    logger.warning("Failed to load %s: %s", z, e)

            if not dfs:
                continue

            combined = pd.concat(dfs, ignore_index=True).sort_values("timestamp_s")
            features = aggregate_per_window(combined, window_seconds)

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
