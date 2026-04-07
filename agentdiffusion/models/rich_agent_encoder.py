"""Rich Agent Encoder: convert raw order sequences into 200-dim agent state vectors.

Maps a variable-length window of raw L3 orders for one agent/stock into a
dense d_state=200 latent vector that captures position state, behavioral
signature, market perception, current intention, and strategy fingerprint.

Architecture:
    1. Per-order embedding:  Linear(d_raw_order=10, 64) + GELU + LayerNorm
    2. Temporal attention:   TransformerEncoder(2 layers, 4 heads, d=64)
    3. Aggregation:          learned attention-weighted pooling -> 200-dim output
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_RAW_ORDER = 10   # number of per-order features
D_EMBED = 64       # internal embedding dimension
D_STATE = 200      # output latent dimension
N_MAX_ORDERS = 128 # default max orders per window


# ===================================================================
# Module: RichAgentEncoder
# ===================================================================

class _AttentionPool(nn.Module):
    """Learned attention-weighted pooling over the sequence dimension.

    Given input [B, N, D], produces [B, D] via:
        score_i = v^T tanh(W h_i + b)
        alpha   = softmax(score, dim=N)   (masked)
        out     = sum_i alpha_i * h_i
    """

    def __init__(self, d_in: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_in, bias=True)
        self.score = nn.Linear(d_in, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    [B, N, D]
            mask: [B, N] bool, True = valid token, False = padding
        Returns:
            pooled: [B, D]
        """
        # [B, N, 1]
        scores = self.score(torch.tanh(self.proj(x)))
        if mask is not None:
            # mask out padding positions with -inf before softmax
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(scores, dim=1)  # [B, N, 1]
        # Handle all-padding edge case: softmax(-inf, ..., -inf) = nan
        alpha = torch.nan_to_num(alpha, nan=0.0)
        return (alpha * x).sum(dim=1)  # [B, D]


class RichAgentEncoder(nn.Module):
    """Encode a window of raw orders for one agent/stock into a rich d_state=200 latent vector.

    The 200-dim vector captures:
    - Position state (net position, cash proxy, unrealized P&L direction)
    - Behavioral signature (order size distribution, cancel rate, aggressiveness)
    - Market perception (local spread, depth, price momentum)
    - Current intention (pending order direction, urgency, price target)
    - Strategy fingerprint (MM-like? momentum? mean-reversion? noise?)

    Input:
        orders: [B, N_max_orders, d_raw_order] where d_raw_order=10
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
        mask: [B, N_max_orders] bool, True = real order, False = padding

    Output:
        state: [B, d_state]  where d_state=200
    """

    def __init__(
        self,
        d_raw_order: int = D_RAW_ORDER,
        d_embed: int = D_EMBED,
        d_state: int = D_STATE,
        n_heads: int = 4,
        n_layers: int = 2,
        n_max_orders: int = N_MAX_ORDERS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_raw_order = d_raw_order
        self.d_embed = d_embed
        self.d_state = d_state
        self.n_max_orders = n_max_orders

        # --- 1) Per-order embedding ---
        self.order_embed = nn.Sequential(
            nn.Linear(d_raw_order, d_embed),
            nn.GELU(),
            nn.Linear(d_embed, d_embed),
            nn.LayerNorm(d_embed),
        )

        # --- Learnable positional encoding (by position index) ---
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_max_orders, d_embed) * 0.02
        )

        # --- 2) Temporal TransformerEncoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_embed,
            nhead=n_heads,
            dim_feedforward=d_embed * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # --- 3) Attention-weighted pooling ---
        self.attn_pool = _AttentionPool(d_embed)

        # --- 4) Projection to d_state ---
        self.state_proj = nn.Sequential(
            nn.Linear(d_embed, d_embed * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_embed * 2, d_state),
            nn.LayerNorm(d_state),
        )

        # --- Learnable "empty" state for fully-empty windows ---
        self.empty_state = nn.Parameter(torch.zeros(d_state))

        self._init_weights()

    # -----------------------------------------------------------------
    def _init_weights(self) -> None:
        """Xavier-uniform for linear layers, zero for biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------
    def forward(
        self,
        orders: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            orders: [B, N, d_raw_order]
            mask:   [B, N] bool — True = valid order, False = padding.
                    If None, all positions are treated as valid.
        Returns:
            state: [B, d_state]
        """
        B, N, _ = orders.shape

        # If mask is None, treat everything as valid
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=orders.device)

        # --- Handle fully-empty sequences ---
        any_valid = mask.any(dim=1)  # [B]
        if not any_valid.any():
            return self.empty_state.unsqueeze(0).expand(B, -1)

        # --- 1) Per-order embedding + positional encoding ---
        x = self.order_embed(orders)  # [B, N, d_embed]
        x = x + self.pos_embed[:, :N, :]

        # --- 2) Transformer (src_key_padding_mask expects True=ignore) ---
        src_key_padding_mask = ~mask  # True where padded
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # --- 3) Attention-weighted pooling ---
        pooled = self.attn_pool(x, mask)  # [B, d_embed]

        # --- 4) Project to d_state ---
        state = self.state_proj(pooled)  # [B, d_state]

        # --- Replace fully-empty samples with learned empty state ---
        if not any_valid.all():
            empty_expanded = self.empty_state.unsqueeze(0).expand(B, -1)
            state = torch.where(
                any_valid.unsqueeze(-1),
                state,
                empty_expanded,
            )

        return state


# ===================================================================
# Feature extraction from A-share L3 逐笔委托 DataFrame
# ===================================================================

def extract_rich_order_features(
    orders_df: pd.DataFrame,
    window_start: float,
    window_end: float,
    mid_price: float,
    n_max: int = N_MAX_ORDERS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract d_raw_order=10 features from raw L3 orders in a time window.

    Operates on a 逐笔委托 DataFrame (already parsed by load_stock_l3 or
    load_stock_orders) whose columns include at minimum:
        timestamp, price, size, direction, order_type

    Args:
        orders_df:    Full orders DataFrame for one stock (pre-parsed).
        window_start: Window start in seconds since midnight.
        window_end:   Window end in seconds since midnight.
        mid_price:    Current mid-price (元) for normalization.
                      If <= 0, relative_price and aggressiveness default to 0.
        n_max:        Maximum number of orders to retain (pad/truncate).

    Returns:
        features: [n_max, 10] float tensor
        mask:     [n_max]     bool tensor (True = valid order)
    """
    features = torch.zeros(n_max, D_RAW_ORDER, dtype=torch.float32)
    mask = torch.zeros(n_max, dtype=torch.bool)

    # --- Filter to time window ---
    ts = orders_df["timestamp"].values
    in_window = (ts >= window_start) & (ts < window_end)
    window_df = orders_df.loc[in_window]

    if len(window_df) == 0:
        return features, mask

    # Sort by timestamp (should already be, but be safe)
    window_df = window_df.sort_values("timestamp", ignore_index=True)

    # Truncate if too many orders (keep the most recent ones)
    if len(window_df) > n_max:
        window_df = window_df.iloc[-n_max:].reset_index(drop=True)

    n = len(window_df)
    mask[:n] = True

    # --- Extract raw arrays ---
    prices = window_df["price"].values.astype(np.float64)
    sizes = window_df["size"].values.astype(np.float64)
    timestamps = window_df["timestamp"].values.astype(np.float64)
    directions = window_df["direction"].values.astype(np.float64)

    # order_type: original column is string or int; map to 1/2/3
    ot_raw = window_df["order_type"].values
    order_types = _map_order_type(ot_raw)

    # --- Feature 0: relative_price (distance from mid, in bps) ---
    safe_mid = max(mid_price, 1e-8)
    rel_price = (prices - mid_price) / safe_mid * 10000.0  # bps
    rel_price = np.clip(rel_price, -500, 500)
    features[:n, 0] = torch.from_numpy(rel_price.astype(np.float32))

    # --- Feature 1: log_size ---
    log_size = np.log1p(np.clip(sizes, 0, None))
    features[:n, 1] = torch.from_numpy(log_size.astype(np.float32))

    # --- Feature 2: direction (+1 buy, -1 sell) ---
    features[:n, 2] = torch.from_numpy(directions.astype(np.float32))

    # --- Feature 3: order_type (1=new, 2=modify, 3=cancel) ---
    features[:n, 3] = torch.from_numpy(order_types.astype(np.float32))

    # --- Feature 4: relative_time (within window, 0-1) ---
    window_dur = max(window_end - window_start, 1e-6)
    rel_time = (timestamps - window_start) / window_dur
    rel_time = np.clip(rel_time, 0.0, 1.0)
    features[:n, 4] = torch.from_numpy(rel_time.astype(np.float32))

    # --- Feature 5: is_executed ---
    # Heuristic: if order_type is not cancel and price is aggressive
    # (within 1 bps of mid), treat as likely executed.  If a real
    # execution flag column exists, prefer it.
    if "is_executed" in window_df.columns:
        is_exec = window_df["is_executed"].values.astype(np.float64)
    else:
        # proxy: aggressive buy (price >= mid) or aggressive sell (price <= mid)
        is_exec = np.where(
            (directions > 0) & (prices >= mid_price * 0.9999),
            1.0,
            np.where(
                (directions < 0) & (prices <= mid_price * 1.0001),
                1.0,
                0.0,
            ),
        )
        # Cancel orders never execute
        is_exec[order_types == 3] = 0.0
    features[:n, 5] = torch.from_numpy(is_exec.astype(np.float32))

    # --- Feature 6: cancel_flag ---
    cancel_flag = (order_types == 3).astype(np.float64)
    features[:n, 6] = torch.from_numpy(cancel_flag.astype(np.float32))

    # --- Feature 7: price_aggressiveness ---
    # How close to best price: 1 = at mid, 0 = far away
    abs_dist_bps = np.abs(rel_price)
    aggressiveness = np.clip(1.0 - abs_dist_bps / 100.0, 0.0, 1.0)
    features[:n, 7] = torch.from_numpy(aggressiveness.astype(np.float32))

    # --- Feature 8: size_percentile ---
    # Rank of this order's size relative to all orders in the window
    if n > 1:
        ranks = np.argsort(np.argsort(sizes)).astype(np.float64)
        size_pct = ranks / (n - 1)  # 0 = smallest, 1 = largest
    else:
        size_pct = np.array([0.5])
    features[:n, 8] = torch.from_numpy(size_pct.astype(np.float32))

    # --- Feature 9: time_since_last_order (normalized) ---
    dt = np.zeros(n, dtype=np.float64)
    if n > 1:
        dt[1:] = np.diff(timestamps)
    # Normalize: divide by median non-zero delta (or window duration)
    nonzero_dt = dt[dt > 0]
    if len(nonzero_dt) > 0:
        median_dt = np.median(nonzero_dt)
    else:
        median_dt = window_dur
    safe_median = max(median_dt, 1e-6)
    dt_norm = np.clip(dt / safe_median, 0.0, 10.0)
    features[:n, 9] = torch.from_numpy(dt_norm.astype(np.float32))

    return features, mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_order_type(ot_raw: np.ndarray) -> np.ndarray:
    """Map raw order_type column to numeric {1=new, 2=modify, 3=cancel}.

    The L3 逐笔委托 order_type can be:
      - Numeric int: 1 / 2 / 3 (already mapped)
      - String codes: 'A'=new, 'D'=cancel, etc. (varies by exchange)

    We handle both forms.
    """
    result = np.ones(len(ot_raw), dtype=np.float64)  # default: new

    try:
        # Try numeric conversion first
        numeric = pd.to_numeric(ot_raw, errors="coerce")
        valid_num = ~np.isnan(numeric)
        if valid_num.any():
            result[valid_num] = np.clip(numeric[valid_num], 1, 3)
            if valid_num.all():
                return result
    except Exception:
        pass

    # String mapping for common L3 codes
    str_vals = np.asarray(ot_raw, dtype=str)
    # A / 1 = new order
    result[(str_vals == "A") | (str_vals == "1")] = 1.0
    # U / 2 = modify
    result[(str_vals == "U") | (str_vals == "2")] = 2.0
    # D / 3 = cancel
    result[(str_vals == "D") | (str_vals == "3")] = 3.0

    return result


def extract_rich_order_features_batch(
    orders_df: pd.DataFrame,
    time_edges: np.ndarray,
    mid_prices: np.ndarray,
    n_max: int = N_MAX_ORDERS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch extraction across multiple time windows.

    Args:
        orders_df:  Full orders DataFrame for one stock.
        time_edges: [T+1] window boundary times in seconds.
        mid_prices: [T] mid-price for each window.
        n_max:      Max orders per window.

    Returns:
        features: [T, n_max, 10] float tensor
        masks:    [T, n_max]     bool tensor
    """
    T = len(time_edges) - 1
    all_features = torch.zeros(T, n_max, D_RAW_ORDER, dtype=torch.float32)
    all_masks = torch.zeros(T, n_max, dtype=torch.bool)

    for t in range(T):
        feat, msk = extract_rich_order_features(
            orders_df,
            window_start=float(time_edges[t]),
            window_end=float(time_edges[t + 1]),
            mid_price=float(mid_prices[t]),
            n_max=n_max,
        )
        all_features[t] = feat
        all_masks[t] = msk

    return all_features, all_masks
