"""Limit Order Book dataset for Video DiT training on real NASDAQ data.

Loads LOBSTER orderbook + message files, normalizes features, and produces
sliding windows of consecutive LOB snapshots as "video frames".

Each frame = one LOB snapshot with 10 levels of (ask_price, ask_vol, bid_price, bid_vol)
plus derived features (mid_price, spread, imbalance, returns, etc.)

A "video" = T consecutive LOB snapshots for spatiotemporal diffusion.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def load_lobster_data(
    orderbook_path: str,
    message_path: str,
    num_levels: int = 10,
    subsample: int = 1,
) -> dict[str, np.ndarray]:
    """Load and parse LOBSTER orderbook + message files.

    Returns dict with:
        raw_ob:     [N, 4*num_levels] raw orderbook
        timestamps: [N] seconds after midnight
        msg_types:  [N] message types (1=submit, 4=exec, etc.)
        msg_sizes:  [N] order sizes
        msg_dirs:   [N] order directions (-1=sell, 1=buy)
    """
    ob = np.loadtxt(orderbook_path, delimiter=",")
    msg = np.loadtxt(message_path, delimiter=",")

    if subsample > 1:
        ob = ob[::subsample]
        msg = msg[::subsample]

    return {
        "raw_ob": ob,
        "timestamps": msg[:, 0],
        "msg_types": msg[:, 1].astype(int),
        "msg_sizes": msg[:, 3].astype(int),
        "msg_prices": msg[:, 4],
        "msg_dirs": msg[:, 5].astype(int),
    }


def compute_lob_features(raw_ob: np.ndarray, num_levels: int = 10) -> np.ndarray:
    """Compute normalized features from raw orderbook data.

    Input:  [N, 4*num_levels] — ask1_p, ask1_v, bid1_p, bid1_v, ask2_p, ...
    Output: [N, feature_dim] — normalized features per snapshot

    Features per level (4 each):
        - relative_ask_price (distance from mid in bps)
        - log_ask_volume
        - relative_bid_price
        - log_bid_volume

    Global features (appended):
        - log_return (vs previous snapshot)
        - spread_bps
        - mid_price_normalized
        - volume_imbalance (bid_vol - ask_vol) / (bid_vol + ask_vol)
        - order_flow_imbalance (rolling)
        - volatility (rolling std of returns)
        - depth_ratio (total bid vol / total ask vol)
        - price_range (high - low over recent window)
    """
    N = raw_ob.shape[0]

    # Extract best bid/ask
    ask1_price = raw_ob[:, 0]   # ask price × 10000
    ask1_vol = raw_ob[:, 1]
    bid1_price = raw_ob[:, 2]   # bid price × 10000
    bid1_vol = raw_ob[:, 3]

    mid_price = (ask1_price + bid1_price) / 2.0
    spread = ask1_price - bid1_price

    # Relative features per level
    features_per_level = []
    for lev in range(num_levels):
        col = lev * 4
        ask_p = raw_ob[:, col]
        ask_v = raw_ob[:, col + 1]
        bid_p = raw_ob[:, col + 2]
        bid_v = raw_ob[:, col + 3]

        # Relative price (distance from mid in basis points)
        rel_ask = (ask_p - mid_price) / np.maximum(mid_price, 1) * 10000
        rel_bid = (mid_price - bid_p) / np.maximum(mid_price, 1) * 10000

        # Log volume
        log_ask_v = np.log1p(ask_v)
        log_bid_v = np.log1p(bid_v)

        features_per_level.extend([rel_ask, log_ask_v, rel_bid, log_bid_v])

    level_features = np.stack(features_per_level, axis=1)  # [N, 4*num_levels]

    # Global features
    log_return = np.zeros(N)
    log_return[1:] = np.log(mid_price[1:] / np.maximum(mid_price[:-1], 1))

    spread_bps = spread / np.maximum(mid_price, 1) * 10000

    mid_norm = mid_price / mid_price[0] - 1.0  # relative to first

    total_bid_vol = sum(raw_ob[:, lev * 4 + 3] for lev in range(num_levels))
    total_ask_vol = sum(raw_ob[:, lev * 4 + 1] for lev in range(num_levels))
    vol_imbalance = (total_bid_vol - total_ask_vol) / np.maximum(total_bid_vol + total_ask_vol, 1)

    depth_ratio = np.log1p(total_bid_vol) - np.log1p(total_ask_vol)

    # Rolling volatility (20-step window)
    volatility = np.zeros(N)
    window = 20
    for i in range(window, N):
        volatility[i] = log_return[i - window:i].std()

    # Rolling momentum (5-step cumulative return)
    momentum = np.zeros(N)
    for i in range(5, N):
        momentum[i] = log_return[i - 5:i].sum()

    global_features = np.stack([
        log_return,
        spread_bps,
        mid_norm,
        vol_imbalance,
        depth_ratio,
        volatility * 1000,  # scale up for visibility
        momentum * 1000,
        np.log1p(total_bid_vol + total_ask_vol),  # total depth
    ], axis=1)  # [N, 8]

    all_features = np.concatenate([level_features, global_features], axis=1)  # [N, 4*L + 8]

    return all_features.astype(np.float32)


def normalize_lob_features(features: np.ndarray) -> tuple[np.ndarray, dict]:
    """Normalize features to roughly [-3, 3] using median/IQR (robust to outliers)."""
    medians = np.median(features, axis=0)
    q75 = np.percentile(features, 75, axis=0)
    q25 = np.percentile(features, 25, axis=0)
    iqr = (q75 - q25).clip(min=1e-8)

    normed = (features - medians) / iqr
    normed = normed.clip(-5, 5)

    stats = {"medians": medians, "iqr": iqr}
    return normed, stats


class LOBVideoDataset(Dataset):
    """LOBSTER order book data as video sequences for Video DiT.

    Each sample is a sliding window of T consecutive LOB snapshots,
    reshaped to [T, H, W, C] to match Video DiT's input format.

    The LOB features are arranged into a 2D grid:
    - 10 levels as "height" (H=10)
    - 4 features per level as part of "width"
    - Global features appended as extra row

    For simplicity, we reshape [T, feature_dim] → [T, H, W, C] where
    H*W*C = feature_dim (padded if needed).

    Args:
        orderbook_path: Path to LOBSTER orderbook CSV
        message_path: Path to LOBSTER message CSV
        total_frames: Frames per sequence (default 20 = 4 cond + 16 gen)
        cond_frames: Condition frames (default 4)
        subsample: Take every N-th snapshot (default 10 for ~27K frames)
        grid_shape: (H, W) for reshaping features into spatial grid
    """

    def __init__(
        self,
        orderbook_path: str,
        message_path: str,
        total_frames: int = 20,
        cond_frames: int = 4,
        subsample: int = 10,
        grid_shape: tuple[int, int] = (6, 8),
    ):
        self.total_frames = total_frames
        self.cond_frames = cond_frames

        # Load and process
        logger.info(f"Loading LOBSTER data from {orderbook_path}")
        data = load_lobster_data(orderbook_path, message_path, subsample=subsample)
        raw_features = compute_lob_features(data["raw_ob"])
        self.feature_dim = raw_features.shape[1]
        logger.info(f"Raw features: {raw_features.shape} ({self.feature_dim} dims)")

        # Normalize
        normed, self.norm_stats = normalize_lob_features(raw_features)

        # Pad feature_dim to H*W*C (auto-compute C = ceil(feature_dim / (H*W)))
        import math
        H, W = grid_shape
        C = max(1, math.ceil(self.feature_dim / (H * W)))
        target_dim = H * W * C
        if self.feature_dim < target_dim:
            pad_width = target_dim - self.feature_dim
            normed = np.pad(normed, ((0, 0), (0, pad_width)))
        elif self.feature_dim > target_dim:
            normed = normed[:, :target_dim]
        self.padded_dim = target_dim
        self.grid_shape = grid_shape

        # Store as tensor
        self.features = torch.from_numpy(normed)  # [N, H*W]
        self.num_snapshots = self.features.shape[0]
        self.num_sequences = self.num_snapshots - total_frames + 1

        # Store raw mid prices for evaluation
        self.mid_prices = (data["raw_ob"][:, 0] + data["raw_ob"][:, 2]) / 2.0
        if subsample > 1:
            self.mid_prices = self.mid_prices  # already subsampled in load

        logger.info(
            f"LOBVideoDataset: {self.num_snapshots} snapshots → "
            f"{self.num_sequences} sequences of {total_frames} frames, "
            f"grid={grid_shape}, feature_dim={self.feature_dim}→{self.padded_dim}"
        )

    def __len__(self) -> int:
        return max(0, self.num_sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return T consecutive frames reshaped to [T, H, W, C].

        C = padded_dim // (H * W).  When grid fills features exactly, C>1
        gives the model richer per-cell information (e.g. grid=(4,4), C=3
        for 48 LOB features → zero padding).
        """
        H, W = self.grid_shape
        C = self.padded_dim // (H * W)
        window = self.features[idx : idx + self.total_frames]  # [T, H*W*C]
        frames = window.view(self.total_frames, H, W, C)  # [T, H, W, C]

        return {
            "frames": frames,
            "market_conds": torch.zeros(self.total_frames, 32),  # placeholder
        }

    def get_mid_prices(self, start: int, length: int) -> np.ndarray:
        """Get raw mid prices for evaluation."""
        end = min(start + length, len(self.mid_prices))
        return self.mid_prices[start:end]

    def get_returns(self, start: int, length: int) -> np.ndarray:
        """Get log returns for evaluation."""
        prices = self.get_mid_prices(start, length + 1)
        return np.diff(np.log(prices + 1e-10))
