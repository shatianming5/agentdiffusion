"""LOBSTER Order Flow dataset for Order-Agent-Order self-supervised training.

Converts LOBSTER message + orderbook data into:
  - order_windows: [T, N_max, d_order] windowed order features
  - order_stats:   [T, H, W, d_order_out] per-cell aggregated order statistics

d_order features per raw order:
  0: relative_time     (seconds within window, normalised to [0,1])
  1: relative_price    (price - mid_price, normalised by spread)
  2: log_size          (log(1 + size))
  3: direction         (+1 buy, -1 sell)
  4: is_limit          (1 if limit order, 0 otherwise)
  5: is_cancel         (1 if cancel, 0 otherwise)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# LOBSTER message types
MSG_SUBMIT_LIMIT = 1
MSG_CANCEL_PARTIAL = 2
MSG_CANCEL_FULL = 3
MSG_EXEC_VISIBLE = 4
MSG_EXEC_HIDDEN = 5


def load_and_prepare(
    orderbook_path: str,
    message_path: str,
    num_levels: int = 10,
) -> dict[str, np.ndarray]:
    """Load LOBSTER files and compute derived features."""
    ob = np.loadtxt(orderbook_path, delimiter=",")
    msg = np.loadtxt(message_path, delimiter=",")

    N = min(len(ob), len(msg))
    ob, msg = ob[:N], msg[:N]

    # Mid price and spread from best bid/ask
    ask_p = ob[:, 0]    # best ask price
    bid_p = ob[:, 2]    # best bid price
    mid = (ask_p + bid_p) / 2.0
    spread = (ask_p - bid_p).clip(min=1)

    return {
        "timestamps": msg[:, 0],
        "msg_types": msg[:, 1].astype(int),
        "order_ids": msg[:, 2].astype(int),
        "sizes": msg[:, 3].astype(int),
        "prices": msg[:, 4].astype(int),
        "directions": msg[:, 5].astype(int),  # -1=sell, 1=buy
        "mid_prices": mid,
        "spreads": spread,
        "raw_ob": ob,
    }


class LOBOrderFlowDataset(Dataset):
    """Sliding-window dataset of order flow for Encoder-Decoder training.

    Each sample contains T consecutive time windows of order flow.
    Within each window, orders are zero-padded to N_max.
    """

    def __init__(
        self,
        orderbook_path: str,
        message_path: str,
        window_seconds: float = 1.0,
        total_windows: int = 20,
        n_max_orders: int = 256,
        d_order: int = 6,
        stride_windows: int = 1,
    ):
        super().__init__()
        self.n_max = n_max_orders
        self.d_order = d_order
        self.total_windows = total_windows

        data = load_and_prepare(orderbook_path, message_path)

        # Split timeline into fixed-duration windows
        ts = data["timestamps"]
        t_start = ts[0]
        t_end = ts[-1]
        window_edges = np.arange(t_start, t_end, window_seconds)
        if len(window_edges) < total_windows + 1:
            window_seconds = (t_end - t_start) / (total_windows * 10)
            window_edges = np.arange(t_start, t_end, window_seconds)

        # Pre-compute per-window order features
        self.windows = []  # list of [n_orders_in_window, d_order] arrays
        self.window_mids = []  # mid-price at window start

        for i in range(len(window_edges) - 1):
            mask = (ts >= window_edges[i]) & (ts < window_edges[i + 1])
            if mask.sum() == 0:
                self.windows.append(np.zeros((0, d_order), dtype=np.float32))
                self.window_mids.append(data["mid_prices"][0])
                continue

            idx = np.where(mask)[0]
            n = len(idx)
            mid = data["mid_prices"][idx[0]]
            sp = max(data["spreads"][idx[0]], 1.0)

            feats = np.zeros((n, d_order), dtype=np.float32)
            # 0: relative time within window [0,1]
            wt = data["timestamps"][idx]
            feats[:, 0] = (wt - wt[0]) / max(wt[-1] - wt[0], 1e-6)
            # 1: relative price (normalised by spread)
            feats[:, 1] = (data["prices"][idx] - mid) / sp
            # 2: log size
            feats[:, 2] = np.log1p(np.abs(data["sizes"][idx]))
            # 3: direction
            feats[:, 3] = data["directions"][idx].astype(np.float32)
            # 4: is_limit
            feats[:, 4] = (data["msg_types"][idx] == MSG_SUBMIT_LIMIT).astype(np.float32)
            # 5: is_cancel
            feats[:, 5] = ((data["msg_types"][idx] == MSG_CANCEL_PARTIAL) |
                           (data["msg_types"][idx] == MSG_CANCEL_FULL)).astype(np.float32)

            self.windows.append(feats)
            self.window_mids.append(mid)

        self.num_windows = len(self.windows)
        # Number of valid sequences
        self.n_sequences = max(0, (self.num_windows - total_windows) // stride_windows)
        self.stride = stride_windows
        logger.info(
            "LOBOrderFlowDataset: %d windows (%.1fs each), %d sequences of %d",
            self.num_windows, window_seconds, self.n_sequences, total_windows,
        )

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.stride
        # Stack T windows, zero-pad each to n_max
        order_seq = torch.zeros(self.total_windows, self.n_max, self.d_order)
        order_counts = torch.zeros(self.total_windows, dtype=torch.long)

        for t in range(self.total_windows):
            w = self.windows[start + t]
            n = min(len(w), self.n_max)
            if n > 0:
                order_seq[t, :n] = torch.from_numpy(w[:n])
            order_counts[t] = n

        return {
            "orders": order_seq,        # [T, N_max, d_order]
            "order_counts": order_counts,  # [T]
        }
