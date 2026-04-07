"""A-Share L3 → 10K agent clusters on a 100×100 grid.

Strategy:
  1. Pool orders from top-N liquid stocks
  2. Extract per-order behavioral features
  3. MiniBatchKMeans with K=10,000 → 10K agent clusters
  4. Arrange clusters in 100×100 grid (2D PCA of cluster centers)
  5. Per-time-window, aggregate each cluster's activity → d_state
  6. Result: [T, 100, 100, d_state] video for Video DiT

Each "agent" is a behavioral archetype (e.g., "small aggressive buyer near mid on
tech stocks") that persists across time and has evolving state.
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

N_CLUSTERS = 10000
GRID_H, GRID_W = 100, 100
D_STATE = 6


def load_orders_multi_stock(
    data_dir: str | Path,
    max_stocks: int = 30,
    encoding: str = "gbk",
) -> pd.DataFrame:
    """Load and pool orders from multiple stocks."""
    data_dir = Path(data_dir)
    stock_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])[:max_stocks]

    all_orders = []
    for sd in stock_dirs:
        try:
            df = pd.read_csv(
                sd / "逐笔委托.csv", encoding=encoding,
                names=["code", "exch_code", "date", "time", "order_num",
                       "exch_order_id", "order_type", "side", "price", "size", "_"],
                header=0, low_memory=False,
            )
            # Parse time
            t = df["time"].values
            ms = t % 1000; t = t // 1000
            ss = t % 100; t = t // 100
            mm = t % 100; hh = t // 100
            df["timestamp"] = hh * 3600 + mm * 60 + ss + ms / 1000.0

            df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0) / 10000.0
            df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0)
            df["direction"] = df["side"].map({"B": 1, "S": -1}).fillna(0).astype(int)
            df["stock_idx"] = len(all_orders)  # stock index for feature

            # Filter trading hours
            df = df[(df["timestamp"] >= 34200) & (df["timestamp"] <= 54000)]
            df = df[df["price"] > 0]

            if len(df) > 0:
                all_orders.append(df[["timestamp", "price", "size", "direction",
                                      "order_type", "stock_idx", "exch_order_id"]])
        except Exception as e:
            logger.warning("Failed %s: %s", sd.name, e)

    if not all_orders:
        return pd.DataFrame()

    combined = pd.concat(all_orders, ignore_index=True)
    logger.info("Pooled %d orders from %d stocks", len(combined), len(all_orders))
    return combined


def extract_order_features(orders: pd.DataFrame) -> np.ndarray:
    """Extract behavioral features per order for clustering.

    Returns: [N_orders, n_features] array.
    Features:
      0: log_size (order size)
      1: relative_price (distance from stock's median price, in bps)
      2: direction (1=buy, -1=sell)
      3: order_type_encoded
      4: time_of_day (normalized 0-1)
      5: stock_idx (normalized)
    """
    n = len(orders)
    features = np.zeros((n, 6), dtype=np.float32)

    sizes = orders["size"].values.astype(float)
    features[:, 0] = np.log1p(sizes) / 10.0  # normalize

    # Relative price per stock
    prices = orders["price"].values.astype(float)
    stock_idx = orders["stock_idx"].values
    for si in np.unique(stock_idx):
        mask = stock_idx == si
        med = np.median(prices[mask])
        features[mask, 1] = np.clip((prices[mask] - med) / (med + 1e-8) * 10000 / 100, -5, 5)

    features[:, 2] = orders["direction"].values.astype(float)

    ot = pd.to_numeric(orders["order_type"], errors="coerce").fillna(1).values
    features[:, 3] = np.clip(ot / 5.0, 0, 1)

    ts = orders["timestamp"].values.astype(float)
    features[:, 4] = (ts - 34200) / (54000 - 34200)  # 0=open, 1=close

    si = orders["stock_idx"].values.astype(float)
    features[:, 5] = si / (si.max() + 1)

    return features


def cluster_agents(
    features: np.ndarray,
    n_clusters: int = N_CLUSTERS,
    sample_size: int = 500_000,
) -> tuple[np.ndarray, np.ndarray]:
    """MiniBatchKMeans to get 10K agent clusters.

    Returns:
        labels: [N_orders] cluster assignment
        centers: [n_clusters, n_features] cluster centers
    """
    from sklearn.cluster import MiniBatchKMeans

    n = len(features)
    if n > sample_size:
        # Subsample for faster clustering
        idx = np.random.choice(n, sample_size, replace=False)
        sample = features[idx]
    else:
        sample = features

    logger.info("Clustering %d samples into %d clusters...", len(sample), n_clusters)
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, batch_size=10000,
        max_iter=50, n_init=1, random_state=42,
    )
    kmeans.fit(sample)

    # Predict all
    labels = kmeans.predict(features)
    centers = kmeans.cluster_centers_

    logger.info("Clustering done. Largest cluster: %d orders",
                np.bincount(labels).max())
    return labels, centers


def arrange_grid_2d(centers: np.ndarray, H: int = GRID_H, W: int = GRID_W) -> np.ndarray:
    """Arrange cluster centers on a 2D grid using PCA → Hilbert-like mapping.

    Returns: grid_assignment[cluster_id] → (row, col)
    """
    from sklearn.decomposition import PCA

    n = len(centers)
    assert n <= H * W

    # PCA to 2D
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(centers)  # [n, 2]

    # Normalize to [0, H-1] and [0, W-1]
    for dim in range(2):
        mn, mx = coords_2d[:, dim].min(), coords_2d[:, dim].max()
        if mx > mn:
            coords_2d[:, dim] = (coords_2d[:, dim] - mn) / (mx - mn)
        else:
            coords_2d[:, dim] = 0.5

    # Greedy assignment: each cluster to nearest unoccupied grid cell
    grid_pos = np.zeros((n, 2), dtype=int)
    target_r = (coords_2d[:, 0] * (H - 1)).astype(int).clip(0, H - 1)
    target_c = (coords_2d[:, 1] * (W - 1)).astype(int).clip(0, W - 1)

    occupied = set()
    for i in range(n):
        r, c = target_r[i], target_c[i]
        # Find nearest unoccupied
        for radius in range(max(H, W)):
            found = False
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in occupied:
                        grid_pos[i] = [nr, nc]
                        occupied.add((nr, nc))
                        found = True
                        break
                if found:
                    break
            if found:
                break

    return grid_pos  # [n_clusters, 2]


def build_agent_grid(
    orders: pd.DataFrame,
    labels: np.ndarray,
    grid_pos: np.ndarray,
    window_seconds: float = 5.0,
    H: int = GRID_H,
    W: int = GRID_W,
) -> tuple[np.ndarray, np.ndarray]:
    """Build [T, H, W, D_STATE] agent grid from clustered orders.

    d_state per cell:
      0: net_signed_volume
      1: order_count
      2: avg_price (volume-weighted)
      3: cancel_rate
      4: avg_size
      5: buy_ratio
    """
    ts = orders["timestamp"].values.astype(float)
    t_min, t_max = ts.min(), ts.max()
    time_edges = np.arange(t_min, t_max + window_seconds, window_seconds)
    T = len(time_edges) - 1

    grid = np.zeros((T, H, W, D_STATE), dtype=np.float32)

    # Vectorized: assign orders to time bins
    time_bin = np.digitize(ts, time_edges) - 1
    time_bin = np.clip(time_bin, 0, T - 1)

    # Pre-extract arrays
    sz = orders["size"].values.astype(np.float64)
    dr = orders["direction"].values.astype(np.float64)
    pr = orders["price"].values.astype(np.float64)
    ot = pd.to_numeric(orders["order_type"], errors="coerce").fillna(1).values

    signed_vol = sz * dr
    px_sz = pr * sz
    is_cancel = (ot == 3).astype(np.float64)
    is_buy = (dr > 0).astype(np.float64)

    # Map cluster → grid position
    rows = grid_pos[labels, 0]
    cols = grid_pos[labels, 1]

    # Composite key: (time_bin, row, col) → flat index
    flat_key = time_bin * (H * W) + rows * W + cols
    n_bins = T * H * W

    # Bincount aggregation
    sum_sv = np.bincount(flat_key, weights=signed_vol, minlength=n_bins)[:n_bins]
    sum_cnt = np.bincount(flat_key, minlength=n_bins)[:n_bins].astype(np.float64)
    sum_pxsz = np.bincount(flat_key, weights=px_sz, minlength=n_bins)[:n_bins]
    sum_sz = np.bincount(flat_key, weights=sz, minlength=n_bins)[:n_bins]
    sum_cancel = np.bincount(flat_key, weights=is_cancel, minlength=n_bins)[:n_bins]
    sum_buy = np.bincount(flat_key, weights=is_buy, minlength=n_bins)[:n_bins]

    # Reshape and fill
    safe_cnt = np.where(sum_cnt > 0, sum_cnt, 1)
    safe_sz = np.where(sum_sz > 0, sum_sz, 1)

    grid[:, :, :, 0] = sum_sv.reshape(T, H, W)
    grid[:, :, :, 1] = sum_cnt.reshape(T, H, W)
    grid[:, :, :, 2] = (sum_pxsz / safe_sz).reshape(T, H, W)
    grid[:, :, :, 3] = (sum_cancel / safe_cnt).reshape(T, H, W)
    grid[:, :, :, 4] = (sum_sz / safe_cnt).reshape(T, H, W)
    grid[:, :, :, 5] = (sum_buy / safe_cnt).reshape(T, H, W)

    return grid, time_edges


class AShare10KAgentDataset(Dataset):
    """A-Share L3 data with configurable agent clusters on an H x W grid.

    Args:
        data_dir: directory containing stock subdirectories
        total_frames: frames per sequence
        cond_frames: condition frames
        window_seconds: aggregation window
        max_stocks: number of stocks to pool
        n_clusters: number of agent clusters (default 10000)
        grid_h: grid height (default GRID_H=100)
        grid_w: grid width (default GRID_W=100)
    """

    def __init__(
        self,
        data_dir: str,
        total_frames: int = 20,
        cond_frames: int = 4,
        window_seconds: float = 5.0,
        max_stocks: int = 30,
        n_clusters: int = N_CLUSTERS,
        grid_h: int = GRID_H,
        grid_w: int = GRID_W,
    ):
        self.total_frames = total_frames
        self.cond_frames = cond_frames

        # Step 1: Load pooled orders
        orders = load_orders_multi_stock(data_dir, max_stocks)
        if orders.empty:
            self.all_sequences = []
            logger.error("No orders loaded!")
            return

        # Step 2: Extract features
        logger.info("Extracting order features...")
        features = extract_order_features(orders)

        # Step 3: Cluster into K agents
        labels, centers = cluster_agents(features, n_clusters)

        # Step 4: Arrange on grid_h x grid_w grid
        H, W = grid_h, grid_w
        logger.info("Arranging %d clusters on %dx%d grid...", n_clusters, H, W)
        grid_pos = arrange_grid_2d(centers, H, W)

        # Step 5: Build temporal grid
        logger.info("Building agent grid (window=%.1fs)...", window_seconds)
        grid, time_edges = build_agent_grid(
            orders, labels, grid_pos, window_seconds, H, W)
        T_total = grid.shape[0]
        logger.info("Grid shape: %s", grid.shape)

        # Normalize per-feature
        flat = grid.reshape(-1, D_STATE)
        mean = flat.mean(axis=0, keepdims=True)
        std = flat.std(axis=0, keepdims=True).clip(min=1e-8)
        grid = ((grid - mean) / std).clip(-5, 5)

        # Split into sequences
        self.all_sequences = []
        stride = total_frames // 2
        for i in range(0, T_total - total_frames + 1, stride):
            seq = grid[i:i + total_frames]
            self.all_sequences.append(torch.from_numpy(seq).float())

        logger.info(
            "AShare10KAgentDataset: %d stocks, %d clusters, %dx%d grid, %d windows -> %d sequences",
            max_stocks, n_clusters, H, W, T_total, len(self.all_sequences),
        )

    def __len__(self) -> int:
        return len(self.all_sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "frames": self.all_sequences[idx],
            "market_conds": torch.zeros(self.total_frames, 32),
        }
