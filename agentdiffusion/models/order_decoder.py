"""Agent→Order Decoder: convert agent state changes to order flow.

Given two consecutive agent grids, predict the order flow that would
produce the observed state transitions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentToOrderDecoder(nn.Module):
    """Decode agent state transitions into order flow predictions.

    For each grid cell, the state change between consecutive frames implies
    trading activity. This module predicts per-cell order parameters and
    aggregates them into an order flow tensor.

    Two output modes:
        - 'per_cell': [B, H, W, d_order_out] per-cell order prediction
        - 'flow':     [B, N_max_orders, d_order_out] pooled order list
    """

    def __init__(
        self,
        d_state: int = 16,
        d_hidden: int = 128,
        d_order_out: int = 6,
    ):
        """
        Args:
            d_state: agent state dimensionality.
            d_hidden: hidden layer size.
            d_order_out: output order feature dim.
                         Typically: (price_offset, log_size, direction_logit,
                                     type_logit_limit, type_logit_market, activity_logit)
        """
        super().__init__()
        # Input: concat of (state_t, state_t+1, delta) = 3 * d_state
        self.net = nn.Sequential(
            nn.Linear(d_state * 3, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_order_out),
        )

    def forward(
        self,
        state_t: torch.Tensor,
        state_t1: torch.Tensor,
    ) -> torch.Tensor:
        """Predict per-cell order features from state transition.

        Args:
            state_t:  [B, H, W, d_state] agent grid at time t.
            state_t1: [B, H, W, d_state] agent grid at time t+1.

        Returns:
            orders: [B, H, W, d_order_out] per-cell order prediction.
                    - dim 0: price_offset (relative to mid-price)
                    - dim 1: log_size (order size in log scale)
                    - dim 2: direction logit (>0 = buy, <0 = sell)
                    - dim 3: is_limit logit
                    - dim 4: is_market logit
                    - dim 5: activity logit (>0 = active order, <0 = no order)
        """
        delta = state_t1 - state_t
        x = torch.cat([state_t, state_t1, delta], dim=-1)  # [B, H, W, 3*d_state]
        return self.net(x)  # [B, H, W, d_order_out]

    def decode_sequence(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Decode a full sequence of agent grids to order flow.

        Args:
            states: [B, T, H, W, d_state] agent state sequence.

        Returns:
            orders: [B, T-1, H, W, d_order_out] per-cell orders for each transition.
        """
        B, T, H, W, D = states.shape
        s_t = states[:, :-1].reshape(B * (T - 1), H, W, D)
        s_t1 = states[:, 1:].reshape(B * (T - 1), H, W, D)
        orders = self.forward(s_t, s_t1)  # [B*(T-1), H, W, d_order_out]
        return orders.reshape(B, T - 1, H, W, -1)


class OrderFlowLoss(nn.Module):
    """Loss for matching predicted order flow to real order flow.

    Real order flow is represented as per-cell aggregated statistics
    (not individual orders), making it directly comparable to decoder output.
    """

    def __init__(self, lambda_size: float = 1.0, lambda_dir: float = 1.0,
                 lambda_activity: float = 1.0):
        super().__init__()
        self.lambda_size = lambda_size
        self.lambda_dir = lambda_dir
        self.lambda_activity = lambda_activity

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred:   [B, T, H, W, d_order_out] predicted orders.
            target: [B, T, H, W, d_order_out] ground truth order stats.
            valid_mask: [H, W] bool mask of valid cells.

        Returns:
            dict with 'total', 'size', 'direction', 'activity' losses.
        """
        if valid_mask is not None:
            vm = valid_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1,1,H,W,1]
            pred = pred * vm
            target = target * vm

        # Size (dim 1): MSE on log_size
        loss_size = F.mse_loss(pred[..., 1], target[..., 1])

        # Direction (dim 2): BCE on direction logit
        loss_dir = F.binary_cross_entropy_with_logits(
            pred[..., 2], (target[..., 2] > 0).float()
        )

        # Activity (dim 5): BCE on activity logit
        loss_activity = F.binary_cross_entropy_with_logits(
            pred[..., 5], (target[..., 5] > 0).float()
        )

        total = (self.lambda_size * loss_size
                 + self.lambda_dir * loss_dir
                 + self.lambda_activity * loss_activity)

        return {
            "total": total,
            "size": loss_size,
            "direction": loss_dir,
            "activity": loss_activity,
        }
