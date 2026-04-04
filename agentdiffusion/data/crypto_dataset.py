"""PyTorch Dataset for Binance kline (1-minute candlestick) data.

Loads compressed CSV files from data/external/crypto/, computes per-bar technical
features, normalises to ~[-3, 3], and yields sliding-window pairs for next-step
prediction training with LeWorldModel.

Each sample is a dict:
    state_t:     [H, W, feature_dim]   current window reshaped as 2-D grid
    state_t1:    [H, W, feature_dim]   next window (shifted by stride)
    market_cond: [cond_dim]            multi-scale context vector
"""

from __future__ import annotations

import io
import csv
import math
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _load_klines_from_zips(data_dir: str | Path, symbol_filter: str | None = None) -> np.ndarray:
    """Load and concatenate all kline CSVs from zip archives.

    Args:
        data_dir: Directory containing *.zip kline archives.
        symbol_filter: If set, only load zips whose name starts with this (e.g. 'BTCUSDT').

    Returns:
        raw: [N, 12] float64 array with columns:
            open_time, open, high, low, close, volume,
            close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore
    """
    data_dir = Path(data_dir)
    zips = sorted(data_dir.glob("*.zip"))
    if symbol_filter:
        zips = [z for z in zips if z.name.startswith(symbol_filter)]
    if not zips:
        raise FileNotFoundError(
            f"No zip files found in {data_dir}"
            + (f" matching '{symbol_filter}*'" if symbol_filter else "")
        )

    all_rows: list[np.ndarray] = []
    for zpath in zips:
        with zipfile.ZipFile(zpath, "r") as zf:
            for name in zf.namelist():
                if not name.endswith(".csv"):
                    continue
                with zf.open(name) as f:
                    text = f.read().decode("utf-8")
                lines = text.strip().splitlines()
                rows = []
                for line in lines:
                    parts = line.split(",")
                    if len(parts) >= 12:
                        try:
                            rows.append([float(p) for p in parts[:12]])
                        except ValueError:
                            continue  # skip header / malformed
                if rows:
                    all_rows.append(np.array(rows, dtype=np.float64))

    if not all_rows:
        raise ValueError(f"No valid kline rows found in {data_dir}")

    raw = np.concatenate(all_rows, axis=0)
    # Sort by open_time
    raw = raw[raw[:, 0].argsort()]
    return raw


def _compute_features(raw: np.ndarray) -> np.ndarray:
    """Compute per-bar feature matrix from raw klines.

    Args:
        raw: [N, 12] kline array.

    Returns:
        features: [N, F] normalised feature matrix (F ~ 16 base features).
    """
    N = raw.shape[0]
    opens   = raw[:, 1]
    highs   = raw[:, 2]
    lows    = raw[:, 3]
    closes  = raw[:, 4]
    volumes = raw[:, 5]
    quote_volumes = raw[:, 7]
    trades  = raw[:, 8]
    open_times = raw[:, 0]

    # --- Derived signals ---
    # 1) log return
    log_return = np.zeros(N)
    log_return[1:] = np.log(closes[1:] / np.maximum(closes[:-1], 1e-10))

    # 2) log volume
    log_volume = np.log(volumes + 1.0)

    # 3) rolling volatility (5 and 20 bars)
    vol5 = _rolling_std(log_return, 5)
    vol20 = _rolling_std(log_return, 20)

    # 4) momentum (5 and 20 bars)
    mom5 = _rolling_sum(log_return, 5)
    mom20 = _rolling_sum(log_return, 20)

    # 5) spread (high - low) / close
    spread = (highs - lows) / np.maximum(closes, 1e-10)

    # 6) VWAP deviation: (vwap - close) / close
    vwap = quote_volumes / np.maximum(volumes, 1e-10)
    vwap_dev = (vwap - closes) / np.maximum(closes, 1e-10)

    # 7) RSI proxy (14-bar)
    rsi_proxy = _rolling_rsi(log_return, 14)

    # 8) signed volume
    sign_ret = np.sign(log_return)
    signed_volume = log_volume * sign_ret

    # 9) trade intensity: trades / rolling mean(trades, 20)
    mean_trades = _rolling_mean(trades, 20)
    trade_intensity = trades / np.maximum(mean_trades, 1.0)

    # 10) OHLC normalised range features
    body = (closes - opens) / np.maximum(closes, 1e-10)
    upper_wick = (highs - np.maximum(opens, closes)) / np.maximum(closes, 1e-10)
    lower_wick = (np.minimum(opens, closes) - lows) / np.maximum(closes, 1e-10)

    # 11) hour-of-day and day-of-week (sin/cos encoding from open_time ms)
    seconds = open_times / 1000.0
    hour_frac = (seconds % 86400) / 86400.0  # fraction of day
    dow_frac = ((seconds / 86400.0 + 4) % 7) / 7.0  # approx day-of-week (epoch was Thu)
    hour_sin = np.sin(2 * np.pi * hour_frac)
    hour_cos = np.cos(2 * np.pi * hour_frac)
    dow_sin = np.sin(2 * np.pi * dow_frac)
    dow_cos = np.cos(2 * np.pi * dow_frac)

    # Stack all features [N, 20]
    features = np.stack([
        log_return,         # 0
        log_volume,         # 1
        vol5,               # 2
        vol20,              # 3
        mom5,               # 4
        mom20,              # 5
        spread,             # 6
        vwap_dev,           # 7
        rsi_proxy,          # 8
        signed_volume,      # 9
        trade_intensity,    # 10
        body,               # 11
        upper_wick,         # 12
        lower_wick,         # 13
        hour_sin,           # 14
        hour_cos,           # 15
        dow_sin,            # 16
        dow_cos,            # 17
        closes,             # 18 (raw close - will be normalised)
        volumes,            # 19 (raw volume - will be normalised)
    ], axis=1)  # [N, 20]

    return features


def _normalise_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust normalisation: median / IQR, clipped to [-3, 3].

    Returns:
        normalised features, medians, iqr_scales (for inverse transform).
    """
    medians = np.median(features, axis=0)
    q25 = np.percentile(features, 25, axis=0)
    q75 = np.percentile(features, 75, axis=0)
    iqr = q75 - q25
    iqr = np.maximum(iqr, 1e-8)

    normed = (features - medians) / iqr
    normed = np.clip(normed, -3.0, 3.0)
    return normed, medians, iqr


def _pad_features(features: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad or truncate feature dim to target_dim.

    Args:
        features: [N, F]
        target_dim: desired feature dimension.

    Returns:
        [N, target_dim] array.
    """
    F = features.shape[1]
    if F >= target_dim:
        return features[:, :target_dim]
    pad_width = target_dim - F
    return np.concatenate([features, np.zeros((features.shape[0], pad_width))], axis=1)


def _build_market_cond(features: np.ndarray, idx: int, cond_dim: int = 32) -> np.ndarray:
    """Build market condition vector at position idx.

    Encodes multi-scale context: returns and volatility at 1, 5, 20, 60 bars,
    volume trend, spread trend, and time-of-day encoding.

    Args:
        features: [N, F] normalised feature matrix.
        idx: current bar index.
        cond_dim: target condition dimension.

    Returns:
        cond: [cond_dim] float32 vector.
    """
    cond = np.zeros(cond_dim, dtype=np.float32)
    N = features.shape[0]

    # Multi-scale returns (feature col 0 = log_return)
    for i, lookback in enumerate([1, 5, 20, 60]):
        start = max(0, idx - lookback)
        cond[i] = features[start:idx + 1, 0].sum() if idx > 0 else 0.0

    # Multi-scale volatility (feature col 0 = log_return)
    for i, lookback in enumerate([5, 20, 60]):
        start = max(0, idx - lookback)
        window = features[start:idx + 1, 0]
        cond[4 + i] = window.std() if len(window) > 1 else 0.0

    # Previous return (used by leverage effect in loss)
    cond[5] = features[max(0, idx - 1), 0]  # Note: overlaps with vol_20 slot
    # Fix: put previous return explicitly at slot 7
    cond[7] = features[max(0, idx - 1), 0]

    # Volume trend: mean(log_vol[-5:]) - mean(log_vol[-20:])
    vol_col = 1  # log_volume
    short_vol = features[max(0, idx - 5):idx + 1, vol_col].mean() if idx > 0 else 0.0
    long_vol = features[max(0, idx - 20):idx + 1, vol_col].mean() if idx > 0 else 0.0
    cond[8] = short_vol - long_vol

    # Spread trend: mean(spread[-5:]) - mean(spread[-20:])
    spread_col = 6
    short_sp = features[max(0, idx - 5):idx + 1, spread_col].mean() if idx > 0 else 0.0
    long_sp = features[max(0, idx - 20):idx + 1, spread_col].mean() if idx > 0 else 0.0
    cond[9] = short_sp - long_sp

    # Time encoding (from features)
    for i, col in enumerate([14, 15, 16, 17]):  # hour_sin, hour_cos, dow_sin, dow_cos
        if col < features.shape[1]:
            cond[10 + i] = features[idx, col]

    # Trade intensity recent
    ti_col = 10
    if ti_col < features.shape[1]:
        cond[14] = features[idx, ti_col]

    # Fill remaining slots with momentum/vol at different scales
    cond[15] = features[idx, 4] if features.shape[1] > 4 else 0.0  # mom5
    cond[16] = features[idx, 5] if features.shape[1] > 5 else 0.0  # mom20
    cond[17] = features[idx, 2] if features.shape[1] > 2 else 0.0  # vol5
    cond[18] = features[idx, 3] if features.shape[1] > 3 else 0.0  # vol20
    cond[19] = features[idx, 8] if features.shape[1] > 8 else 0.0  # rsi_proxy

    # Slots 20-31: higher-order context (60-bar momentum, volatility change, etc.)
    for j, lookback in enumerate([120, 240, 480]):
        start = max(0, idx - lookback)
        w = features[start:idx + 1, 0]
        cond[20 + j] = w.sum()
        cond[23 + j] = w.std() if len(w) > 1 else 0.0

    # Clip condition to [-3, 3]
    cond = np.clip(cond, -3.0, 3.0)
    return cond


# ---------------------------------------------------------------------------
# Rolling helpers (pure numpy, no pandas)
# ---------------------------------------------------------------------------

def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Vectorised rolling standard deviation with edge handling."""
    N = len(x)
    out = np.zeros(N)
    # Compute using cumulative sums for O(N) complexity
    cumsum = np.cumsum(x)
    cumsum2 = np.cumsum(x ** 2)
    # Prepend zero for easier indexing
    cs = np.concatenate([[0.0], cumsum])
    cs2 = np.concatenate([[0.0], cumsum2])
    for i in range(N):
        start = max(0, i - window + 1)
        n = i - start + 1
        if n < 2:
            out[i] = 0.0
        else:
            s = cs[i + 1] - cs[start]
            s2 = cs2[i + 1] - cs2[start]
            var = s2 / n - (s / n) ** 2
            out[i] = np.sqrt(max(var, 0.0))
    return out


def _rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    """Vectorised rolling sum."""
    N = len(x)
    cumsum = np.cumsum(x)
    cs = np.concatenate([[0.0], cumsum])
    out = np.zeros(N)
    starts = np.maximum(np.arange(N) - window + 1, 0)
    out = cs[np.arange(N) + 1] - cs[starts]
    return out


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Vectorised rolling mean."""
    N = len(x)
    cumsum = np.cumsum(x)
    cs = np.concatenate([[0.0], cumsum])
    starts = np.maximum(np.arange(N) - window + 1, 0)
    counts = np.arange(N) - starts + 1
    out = (cs[np.arange(N) + 1] - cs[starts]) / counts
    return out


def _rolling_rsi(returns: np.ndarray, window: int = 14) -> np.ndarray:
    """Vectorised RSI proxy: mean(positive returns) / mean(abs(returns))."""
    N = len(returns)
    pos = np.maximum(returns, 0.0)
    absr = np.abs(returns)
    cumpos = np.concatenate([[0.0], np.cumsum(pos)])
    cumabs = np.concatenate([[0.0], np.cumsum(absr)])
    starts = np.maximum(np.arange(N) - window + 1, 0)
    idx1 = np.arange(N) + 1
    sum_pos = cumpos[idx1] - cumpos[starts]
    sum_abs = cumabs[idx1] - cumabs[starts]
    counts = idx1 - starts
    mean_pos = sum_pos / counts
    mean_abs = sum_abs / counts
    out = np.where(mean_abs < 1e-10, 0.5, mean_pos / (mean_abs + 1e-8))
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CryptoKlineDataset(Dataset):
    """Binance kline sliding-window dataset for LeWorldModel training.

    Loads kline CSVs from zip archives, computes technical features,
    and yields (state_t, state_t1, market_cond) pairs where each state
    is a [grid_h, grid_w, feature_dim] tensor reshaped from the sliding
    window for compatibility with the patchify encoder.

    Args:
        data_dir:      Directory with kline zip files.
        symbol:        Symbol filter (e.g. 'BTCUSDT', None for all).
        window_size:   Number of bars per state window (default 64 -> 8x8 grid).
        feature_dim:   Padded feature dimension per bar.
        stride:        Bars to advance between consecutive samples.
        cond_dim:      Market condition vector dimension.
        grid_h:        Grid height for reshape (must satisfy grid_h * grid_w == window_size).
        grid_w:        Grid width for reshape.
    """

    def __init__(
        self,
        data_dir: str | Path,
        symbol: str | None = "BTCUSDT",
        window_size: int = 64,
        feature_dim: int = 32,
        stride: int = 1,
        cond_dim: int = 32,
        grid_h: int = 8,
        grid_w: int = 8,
    ):
        assert grid_h * grid_w == window_size, (
            f"grid_h * grid_w ({grid_h}*{grid_w}={grid_h * grid_w}) "
            f"must equal window_size ({window_size})"
        )

        self.window_size = window_size
        self.feature_dim = feature_dim
        self.stride = stride
        self.cond_dim = cond_dim
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Load and process data
        raw = _load_klines_from_zips(data_dir, symbol_filter=symbol)
        print(f"[CryptoKlineDataset] Loaded {len(raw)} bars for {symbol or 'all symbols'}")

        features = _compute_features(raw)
        features_padded = _pad_features(features, feature_dim)
        self.features_normed, self.medians, self.iqr = _normalise_features(features_padded)
        self.features_raw = features  # keep un-normalised for market_cond and price extraction

        # Raw close prices for evaluation
        self.close_prices = raw[:, 4].copy()
        self.volumes = raw[:, 5].copy()

        # Valid sample indices: each sample needs window at t and window at t+stride
        min_start = 60  # skip first 60 bars (warmup for rolling features)
        max_start = len(self.features_normed) - window_size - stride
        self.indices = list(range(min_start, max_start, stride))
        print(f"[CryptoKlineDataset] {len(self.indices)} samples "
              f"(window={window_size}, stride={stride}, features={feature_dim})")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.indices[idx]

        # Window t: [window_size, feature_dim]
        win_t = self.features_normed[start:start + self.window_size]
        # Window t+1: shifted by stride
        win_t1 = self.features_normed[start + self.stride:start + self.stride + self.window_size]

        # Reshape to [grid_h, grid_w, feature_dim]
        state_t = win_t.reshape(self.grid_h, self.grid_w, self.feature_dim)
        state_t1 = win_t1.reshape(self.grid_h, self.grid_w, self.feature_dim)

        # Market condition at the end of window t
        cond_idx = start + self.window_size - 1
        market_cond = _build_market_cond(self.features_normed, cond_idx, self.cond_dim)

        return {
            "state_t": torch.from_numpy(state_t).float(),
            "state_t1": torch.from_numpy(state_t1).float(),
            "market_cond": torch.from_numpy(market_cond).float(),
        }

    def get_close_prices(self, start_idx: int, length: int) -> np.ndarray:
        """Get raw close prices starting from a sample index."""
        bar_start = self.indices[start_idx]
        return self.close_prices[bar_start:bar_start + length]

    def get_volumes(self, start_idx: int, length: int) -> np.ndarray:
        """Get raw volumes starting from a sample index."""
        bar_start = self.indices[start_idx]
        return self.volumes[bar_start:bar_start + length]

    def get_normalisation_params(self) -> dict:
        """Return normalisation parameters for inverse transform."""
        return {
            "medians": self.medians.copy(),
            "iqr": self.iqr.copy(),
        }


class CryptoSequenceDataset(Dataset):
    """Sequence variant: returns K consecutive windows for multi-step rollout training.

    Each __getitem__ returns:
        seq_states:  [K+1, grid_h, grid_w, feature_dim]
        seq_conds:   [K, cond_dim]
    """

    def __init__(
        self,
        data_dir: str | Path,
        symbol: str | None = "BTCUSDT",
        window_size: int = 64,
        feature_dim: int = 32,
        seq_len: int = 8,
        stride: int = 1,
        cond_dim: int = 32,
        grid_h: int = 8,
        grid_w: int = 8,
    ):
        assert grid_h * grid_w == window_size

        self.window_size = window_size
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.stride = stride
        self.cond_dim = cond_dim
        self.grid_h = grid_h
        self.grid_w = grid_w

        raw = _load_klines_from_zips(data_dir, symbol_filter=symbol)
        features = _compute_features(raw)
        features_padded = _pad_features(features, feature_dim)
        self.features_normed, self.medians, self.iqr = _normalise_features(features_padded)
        self.close_prices = raw[:, 4].copy()
        self.volumes = raw[:, 5].copy()

        # Each sequence needs (seq_len + 1) consecutive windows
        min_start = 60
        total_bars_needed = window_size + seq_len * stride
        max_start = len(self.features_normed) - total_bars_needed
        self.indices = list(range(min_start, max_start, stride))
        print(f"[CryptoSequenceDataset] {len(self.indices)} sequences "
              f"(seq_len={seq_len}, window={window_size})")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        base = self.indices[idx]
        states = []
        conds = []

        for k in range(self.seq_len + 1):
            start = base + k * self.stride
            win = self.features_normed[start:start + self.window_size]
            state = win.reshape(self.grid_h, self.grid_w, self.feature_dim)
            states.append(torch.from_numpy(state).float())

            if k < self.seq_len:
                cond_idx = start + self.window_size - 1
                cond = _build_market_cond(self.features_normed, cond_idx, self.cond_dim)
                conds.append(torch.from_numpy(cond).float())

        return {
            "seq_states": torch.stack(states),   # [K+1, H, W, C]
            "seq_conds": torch.stack(conds),     # [K, cond_dim]
        }
