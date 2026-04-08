"""A-share 100x100 Rich-order grid built on behavioral archetype clusters.

Cell semantics:
    Each grid cell is a persistent behavioral archetype cluster learned from
    pooled L3 orders across multiple stocks. For every time window, the cell
    stores up to N_max raw orders emitted by that archetype during the window.

This differs from:
    - AShareRichAgentDataset: one cell = one stock
    - AShare10KAgentDataset: one cell = one cluster with 6-d aggregated state

Here we keep the 10k cluster layout, but each cell carries a rich raw-order
micro-sequence that can be encoded by RichAgentEncoder into a latent state.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .ashare_10k_agents import arrange_grid_2d, cluster_agents, extract_order_features
from .ashare_stock_agents import CLOSE_SEC, OPEN_SEC
from ..models.rich_agent_encoder import D_RAW_ORDER, _map_order_type

logger = logging.getLogger(__name__)

GRID_H = 100
GRID_W = 100
N_CLUSTERS = 10000
MARKET_COND_DIM = 40


def _dataset_cache_path(
    data_dir: str | Path,
    total_frames: int,
    cond_frames: int,
    window_seconds: float,
    max_stocks: int,
    n_clusters: int,
    grid_h: int,
    grid_w: int,
    n_max_orders: int,
    cache_dir: str | Path,
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_tag = str(Path(data_dir).resolve())
    key = (
        f"{data_tag}|tf={total_frames}|cf={cond_frames}|ws={window_seconds}|"
        f"stocks={max_stocks}|k={n_clusters}|gh={grid_h}|gw={grid_w}|nmax={n_max_orders}"
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    name = (
        f"{Path(data_dir).name}_rich10k_tf{total_frames}_cf{cond_frames}_ws{window_seconds:g}_"
        f"stocks{max_stocks}_k{n_clusters}_{grid_h}x{grid_w}_nmax{n_max_orders}_{digest}.pt"
    )
    return cache_dir / name


def load_orders_multi_stock_with_codes(
    data_dir: str | Path,
    max_stocks: int = 30,
    encoding: str = "gbk",
) -> tuple[pd.DataFrame, list[str]]:
    """Load pooled orders from multiple stocks and preserve stock-code metadata."""
    data_dir = Path(data_dir)
    stock_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])[:max_stocks]
    stock_codes = [sd.name for sd in stock_dirs]

    all_orders = []
    for stock_idx, sd in enumerate(stock_dirs):
        try:
            df = pd.read_csv(
                sd / "逐笔委托.csv",
                encoding=encoding,
                names=[
                    "code",
                    "exch_code",
                    "date",
                    "time",
                    "order_num",
                    "exch_order_id",
                    "order_type",
                    "side",
                    "price",
                    "size",
                    "_",
                ],
                header=0,
                low_memory=False,
            )
            t = df["time"].values
            ms = t % 1000
            t = t // 1000
            ss = t % 100
            t = t // 100
            mm = t % 100
            hh = t // 100
            df["timestamp"] = hh * 3600 + mm * 60 + ss + ms / 1000.0

            df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0) / 10000.0
            df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0)
            df["direction"] = df["side"].map({"B": 1, "S": -1}).fillna(0).astype(int)
            df["stock_idx"] = stock_idx
            df["stock_code"] = sd.name

            df = df[(df["timestamp"] >= OPEN_SEC) & (df["timestamp"] <= CLOSE_SEC)]
            df = df[df["price"] > 0]

            if len(df) > 0:
                all_orders.append(
                    df[
                        [
                            "timestamp",
                            "price",
                            "size",
                            "direction",
                            "order_type",
                            "stock_idx",
                            "stock_code",
                            "exch_order_id",
                        ]
                    ]
                )
        except Exception as exc:
            logger.warning("Failed %s: %s", sd.name, exc)

    if not all_orders:
        return pd.DataFrame(), stock_codes

    combined = pd.concat(all_orders, ignore_index=True)
    logger.info("Pooled %d orders from %d stocks", len(combined), len(all_orders))
    return combined, stock_codes


def _compute_per_order_mid_prices(
    orders: pd.DataFrame,
    time_edges: np.ndarray,
    n_stocks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a stock-local mid/VWAP for every order's time window."""
    ts = orders["timestamp"].values.astype(np.float64)
    prices = orders["price"].values.astype(np.float64)
    sizes = orders["size"].values.astype(np.float64)
    stock_idx = orders["stock_idx"].values.astype(np.int64)
    t_bins = np.digitize(ts, time_edges) - 1
    t_bins = np.clip(t_bins, 0, len(time_edges) - 2)

    n_windows = len(time_edges) - 1
    flat_key = stock_idx * n_windows + t_bins
    n_keys = n_stocks * n_windows

    sum_px_sz = np.bincount(flat_key, weights=prices * sizes, minlength=n_keys)
    sum_sz = np.bincount(flat_key, weights=sizes, minlength=n_keys)

    sum_px_sz = sum_px_sz.reshape(n_stocks, n_windows)
    sum_sz = sum_sz.reshape(n_stocks, n_windows)
    mids = sum_px_sz / np.where(sum_sz > 0, sum_sz, 1.0)

    stock_medians = (
        orders.groupby("stock_idx")["price"].median().reindex(range(n_stocks), fill_value=1.0)
    )
    stock_medians = stock_medians.to_numpy(dtype=np.float64)

    for stock_id in range(n_stocks):
        last_valid = stock_medians[stock_id] if stock_medians[stock_id] > 0 else 1.0
        for window_id in range(n_windows):
            if sum_sz[stock_id, window_id] > 0:
                last_valid = mids[stock_id, window_id]
            else:
                mids[stock_id, window_id] = last_valid

    per_order_mid = mids[stock_idx, t_bins]
    per_order_mid = np.where(per_order_mid > 0, per_order_mid, 1.0)
    return t_bins, per_order_mid


def _build_rich_cluster_order_grid(
    orders: pd.DataFrame,
    labels: np.ndarray,
    grid_pos: np.ndarray,
    time_edges: np.ndarray,
    n_stocks: int,
    n_max_orders: int,
    grid_h: int,
    grid_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build [T, H, W, N_max, 10] rich-order grid for clustered behavioral cells."""
    t_bins, per_order_mid = _compute_per_order_mid_prices(orders, time_edges, n_stocks)
    n_windows = len(time_edges) - 1
    n_cells = grid_h * grid_w
    window_seconds = float(time_edges[1] - time_edges[0]) if len(time_edges) > 1 else 1.0

    rows = grid_pos[labels, 0]
    cols = grid_pos[labels, 1]
    cell_id = t_bins * n_cells + rows * grid_w + cols

    selected = pd.DataFrame(
        {
            "timestamp": orders["timestamp"].values.astype(np.float64),
            "price": orders["price"].values.astype(np.float64),
            "size": orders["size"].values.astype(np.float64),
            "direction": orders["direction"].values.astype(np.float64),
            "order_type": orders["order_type"].values,
            "time_bin": t_bins.astype(np.int64),
            "row": rows.astype(np.int64),
            "col": cols.astype(np.int64),
            "cell_id": cell_id.astype(np.int64),
            "mid_price": per_order_mid.astype(np.float64),
        }
    )
    selected = (
        selected.sort_values(["cell_id", "timestamp"], kind="mergesort")
        .groupby("cell_id", sort=False, group_keys=False)
        .tail(n_max_orders)
        .copy()
    )
    selected["seq_pos"] = selected.groupby("cell_id", sort=False).cumcount().astype(np.int64)

    timestamps = selected["timestamp"].to_numpy(dtype=np.float64)
    prices = selected["price"].to_numpy(dtype=np.float64)
    sizes = selected["size"].to_numpy(dtype=np.float64)
    directions = selected["direction"].to_numpy(dtype=np.float64)
    order_types = _map_order_type(selected["order_type"].to_numpy())
    mids = selected["mid_price"].to_numpy(dtype=np.float64)

    rel_price = (prices - mids) / np.maximum(mids, 1e-8) * 10000.0
    rel_price = np.clip(rel_price, -500.0, 500.0)
    log_size = np.log1p(np.clip(sizes, 0.0, None))
    rel_time = np.clip(
        (timestamps - time_edges[selected["time_bin"].to_numpy(dtype=np.int64)]) / max(window_seconds, 1e-6),
        0.0,
        1.0,
    )

    is_exec = np.where(
        (directions > 0) & (prices >= mids * 0.9999),
        1.0,
        np.where((directions < 0) & (prices <= mids * 1.0001), 1.0, 0.0),
    )
    is_exec[order_types == 3] = 0.0
    cancel_flag = (order_types == 3).astype(np.float64)
    aggressiveness = np.clip(1.0 - np.abs(rel_price) / 100.0, 0.0, 1.0)

    group_counts = (
        selected.groupby("cell_id", sort=False)["size"].transform("count").to_numpy(dtype=np.float64)
    )
    size_ranks = (
        selected.groupby("cell_id", sort=False)["size"]
        .rank(method="first")
        .to_numpy(dtype=np.float64)
    )
    size_pct = (size_ranks - 1.0) / np.maximum(group_counts - 1.0, 1.0)
    size_pct[group_counts <= 1] = 0.5

    dt = (
        selected.groupby("cell_id", sort=False)["timestamp"].diff().fillna(0.0).to_numpy(dtype=np.float64)
    )
    positive_dt = pd.Series(dt).where(dt > 0)
    median_dt = (
        positive_dt.groupby(selected["cell_id"], sort=False).transform("median").fillna(window_seconds)
    )
    median_dt = median_dt.to_numpy(dtype=np.float64)
    dt_norm = np.clip(dt / np.maximum(median_dt, 1e-6), 0.0, 10.0)

    features = np.stack(
        [
            rel_price,
            log_size,
            directions,
            order_types,
            rel_time,
            is_exec,
            cancel_flag,
            aggressiveness,
            size_pct,
            dt_norm,
        ],
        axis=-1,
    ).astype(np.float16)

    order_grid = np.zeros((n_windows, grid_h, grid_w, n_max_orders, D_RAW_ORDER), dtype=np.float16)
    mask_grid = np.zeros((n_windows, grid_h, grid_w, n_max_orders), dtype=np.bool_)

    t_idx = selected["time_bin"].to_numpy(dtype=np.int64)
    r_idx = selected["row"].to_numpy(dtype=np.int64)
    c_idx = selected["col"].to_numpy(dtype=np.int64)
    p_idx = selected["seq_pos"].to_numpy(dtype=np.int64)

    order_grid[t_idx, r_idx, c_idx, p_idx] = features
    mask_grid[t_idx, r_idx, c_idx, p_idx] = True
    return order_grid, mask_grid


class AShare10KRichOrderDataset(Dataset):
    """100x100 rich-order cluster dataset for cached-latent Stage 2."""

    def __init__(
        self,
        data_dir: str,
        total_frames: int = 20,
        cond_frames: int = 4,
        window_seconds: float = 60.0,
        max_stocks: int = 30,
        n_clusters: int = N_CLUSTERS,
        grid_h: int = GRID_H,
        grid_w: int = GRID_W,
        n_max_orders: int = 16,
        cache_dir: str | Path = "outputs/cache/ashare_10k_rich_orders",
        use_cache: bool = True,
    ):
        self.total_frames = total_frames
        self.cond_frames = cond_frames
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_max_orders = n_max_orders
        self.market_conds = torch.zeros(self.total_frames, MARKET_COND_DIM)
        self.raw_order_grid = torch.empty(
            0, grid_h, grid_w, n_max_orders, D_RAW_ORDER, dtype=torch.float16
        )
        self.mask_grid = torch.empty(0, grid_h, grid_w, n_max_orders, dtype=torch.bool)
        self.sequence_starts: list[int] = []
        self.stock_codes: list[str] = []
        self.cache_path = _dataset_cache_path(
            data_dir,
            total_frames,
            cond_frames,
            window_seconds,
            max_stocks,
            n_clusters,
            grid_h,
            grid_w,
            n_max_orders,
            cache_dir,
        )

        if use_cache and self.cache_path.exists():
            logger.info("Loading cached 10K rich-order grid from %s", self.cache_path)
            cached = torch.load(self.cache_path, map_location="cpu", weights_only=False)
            self.raw_order_grid = cached["raw_order_grid"]
            self.mask_grid = cached["mask_grid"].bool()
            self.sequence_starts = list(cached["sequence_starts"])
            self.stock_codes = list(cached.get("stock_codes", []))
            logger.info(
                "Loaded cached rich-order grid: shape=%s, sequences=%d",
                tuple(self.raw_order_grid.shape),
                len(self.sequence_starts),
            )
            return

        orders, stock_codes = load_orders_multi_stock_with_codes(data_dir, max_stocks=max_stocks)
        self.stock_codes = stock_codes
        if orders.empty:
            logger.error("No orders loaded for AShare10KRichOrderDataset")
            return

        logger.info("Extracting cluster features for %d pooled orders...", len(orders))
        cluster_features = extract_order_features(orders)
        labels, centers = cluster_agents(cluster_features, n_clusters=n_clusters)

        logger.info("Arranging %d rich clusters on %dx%d grid...", n_clusters, grid_h, grid_w)
        grid_pos = arrange_grid_2d(centers, grid_h, grid_w)

        time_edges = np.arange(OPEN_SEC, CLOSE_SEC + window_seconds, window_seconds)
        n_windows = len(time_edges) - 1
        logger.info(
            "Building rich cluster order grid: %d windows of %.0fs, n_max_orders=%d",
            n_windows,
            window_seconds,
            n_max_orders,
        )
        order_grid, mask_grid = _build_rich_cluster_order_grid(
            orders,
            labels,
            grid_pos,
            time_edges,
            n_stocks=len(stock_codes),
            n_max_orders=n_max_orders,
            grid_h=grid_h,
            grid_w=grid_w,
        )

        self.raw_order_grid = torch.from_numpy(order_grid)
        self.mask_grid = torch.from_numpy(mask_grid)

        stride = max(total_frames // 2, 1)
        self.sequence_starts = list(range(0, n_windows - total_frames + 1, stride))
        logger.info(
            "AShare10KRichOrderDataset: %d stocks -> %d clusters on %dx%d grid, %d windows -> %d sequences",
            len(stock_codes),
            n_clusters,
            grid_h,
            grid_w,
            n_windows,
            len(self.sequence_starts),
        )

        if use_cache:
            torch.save(
                {
                    "raw_order_grid": self.raw_order_grid,
                    "mask_grid": self.mask_grid,
                    "sequence_starts": self.sequence_starts,
                    "stock_codes": self.stock_codes,
                    "config": {
                        "data_dir": str(data_dir),
                        "total_frames": total_frames,
                        "cond_frames": cond_frames,
                        "window_seconds": window_seconds,
                        "max_stocks": max_stocks,
                        "n_clusters": n_clusters,
                        "grid_h": grid_h,
                        "grid_w": grid_w,
                        "n_max_orders": n_max_orders,
                        "cell_semantics": "behavioral_archetype_cluster",
                    },
                },
                self.cache_path,
            )
            logger.info("Saved cached 10K rich-order grid to %s", self.cache_path)

    def __len__(self) -> int:
        return len(self.sequence_starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.sequence_starts[idx]
        end = start + self.total_frames
        raw_orders = self.raw_order_grid[start:end].float().reshape(
            self.total_frames,
            self.grid_h * self.grid_w,
            self.n_max_orders,
            D_RAW_ORDER,
        )
        order_masks = self.mask_grid[start:end].reshape(
            self.total_frames,
            self.grid_h * self.grid_w,
            self.n_max_orders,
        )
        return {
            "raw_orders": raw_orders,
            "order_masks": order_masks,
            "market_conds": self.market_conds.clone(),
        }
