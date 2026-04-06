"""Agent→Order Decoder: convert agent state changes to order flow.

Given two consecutive agent grids, predict the order flow that would
produce the observed state transitions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentToOrderDecoder(nn.Module):
    """DETR-style Transformer decoder: agent grid → variable-length orders.

    Uses learnable order query slots that cross-attend to the flattened agent
    grid. Each slot predicts one order (or no-order).

    Args:
        d_state: agent state dim.
        d_model: transformer hidden dim.
        n_queries: max orders per transition (like DETR's object queries).
        n_layers: number of decoder layers.
        n_heads: attention heads.
        d_order_out: output dim per order.
    """

    def __init__(
        self,
        d_state: int = 6,
        d_model: int = 128,
        n_queries: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        d_order_out: int = 6,
        d_hidden: int = 128,
    ):
        super().__init__()
        self.n_queries = n_queries
        self.d_model = d_model

        # Project agent states to d_model
        self.agent_proj = nn.Linear(d_state * 3, d_model)  # (state_t, state_t1, delta)

        # Learnable order query slots
        self.order_queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)

        # Transformer decoder layers (queries attend to agent grid keys/values)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output heads
        self.order_head = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_order_out),
        )

    def forward(
        self,
        state_t: torch.Tensor,
        state_t1: torch.Tensor,
    ) -> torch.Tensor:
        """Predict orders from agent state transition.

        Args:
            state_t:  [B, H, W, d_state]
            state_t1: [B, H, W, d_state]

        Returns:
            orders: [B, n_queries, d_order_out]
        """
        B = state_t.shape[0]
        delta = state_t1 - state_t
        x = torch.cat([state_t, state_t1, delta], dim=-1)  # [B, H, W, 3*d_state]

        # Flatten spatial → sequence of agent tokens
        H, W = x.shape[1], x.shape[2]
        memory = self.agent_proj(x.reshape(B, H * W, -1))  # [B, H*W, d_model]

        # Expand query slots for batch
        queries = self.order_queries.unsqueeze(0).expand(B, -1, -1)  # [B, n_queries, d_model]

        # Cross-attention: queries attend to agent memory
        decoded = self.decoder(queries, memory)  # [B, n_queries, d_model]

        return self.order_head(decoded)  # [B, n_queries, d_order_out]

    def decode_sequence(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Decode a full sequence.

        Args:
            states: [B, T, H, W, d_state]

        Returns:
            orders: [B, T-1, n_queries, d_order_out]
        """
        B, T, H, W, D = states.shape
        s_t = states[:, :-1].reshape(B * (T - 1), H, W, D)
        s_t1 = states[:, 1:].reshape(B * (T - 1), H, W, D)
        orders = self.forward(s_t, s_t1)  # [B*(T-1), n_queries, d_order_out]
        return orders.reshape(B, T - 1, self.n_queries, -1)


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
