"""Order→Agent Encoder: map unordered order flow to agent state grid via Slot Attention.

Converts raw order stream [N_orders, d_order] into agent state grid [H, W, d_state],
discovering latent agent identities through self-supervised training.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrderEmbedding(nn.Module):
    """Embed raw order features into a dense vector."""

    def __init__(self, d_order: int = 6, d_embed: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_order, d_embed),
            nn.GELU(),
            nn.Linear(d_embed, d_embed),
            nn.LayerNorm(d_embed),
        )

    def forward(self, orders: torch.Tensor) -> torch.Tensor:
        """orders: [B, N_orders, d_order] -> [B, N_orders, d_embed]"""
        return self.net(orders)


class SlotAttention(nn.Module):
    """Slot Attention (Locatello et al. 2020) adapted for order→agent mapping.

    K slots compete to explain the input orders. Each slot corresponds to
    one cell in the agent grid (K = H * W).
    """

    def __init__(self, num_slots: int, d_embed: int = 128, n_iters: int = 3):
        super().__init__()
        self.num_slots = num_slots
        self.d_embed = d_embed
        self.n_iters = n_iters

        # Learnable slot initialisation (mu + sigma for sampling)
        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, d_embed) * 0.02)
        self.slot_sigma = nn.Parameter(torch.ones(1, num_slots, d_embed) * 0.02)

        # Projections for cross-attention (slots query, orders are key/value)
        self.to_q = nn.Linear(d_embed, d_embed, bias=False)
        self.to_k = nn.Linear(d_embed, d_embed, bias=False)
        self.to_v = nn.Linear(d_embed, d_embed, bias=False)

        # GRU update for slots
        self.gru = nn.GRUCell(d_embed, d_embed)

        # Layer norms
        self.norm_slots = nn.LayerNorm(d_embed)
        self.norm_inputs = nn.LayerNorm(d_embed)

        self.scale = d_embed ** -0.5

    def forward(self, order_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            order_embeds: [B, N_orders, d_embed]

        Returns:
            slots: [B, K, d_embed] where K = num_slots
        """
        B, N, D = order_embeds.shape

        # Initialise slots with learned distribution
        slots = self.slot_mu + self.slot_sigma * torch.randn(
            B, self.num_slots, D, device=order_embeds.device
        )

        inputs = self.norm_inputs(order_embeds)
        k = self.to_k(inputs)  # [B, N, D]
        v = self.to_v(inputs)  # [B, N, D]

        for _ in range(self.n_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.to_q(slots)  # [B, K, D]

            # Attention: slots attend to orders
            attn = torch.einsum("bkd,bnd->bkn", q, k) * self.scale  # [B, K, N]
            # Softmax over SLOTS (competition: each order goes to one slot)
            attn = F.softmax(attn, dim=1)  # normalize over K
            # Weighted mean normalisation
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

            updates = torch.einsum("bkn,bnd->bkd", attn, v)  # [B, K, D]

            # GRU update
            slots = self.gru(
                updates.reshape(B * self.num_slots, D),
                slots_prev.reshape(B * self.num_slots, D),
            ).reshape(B, self.num_slots, D)

        return slots


class OrderToAgentEncoder(nn.Module):
    """Full pipeline: order flow -> agent state grid.

    order stream [B, N_orders, d_order]
        -> OrderEmbedding -> [B, N_orders, d_embed]
        -> SlotAttention  -> [B, K, d_embed]
        -> StateProjection -> [B, K, d_state]
        -> reshape         -> [B, H, W, d_state]
    """

    def __init__(
        self,
        d_order: int = 6,
        d_embed: int = 128,
        d_state: int = 16,
        grid_h: int = 108,
        grid_w: int = 108,
        n_slot_iters: int = 3,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_slots = grid_h * grid_w

        self.order_embed = OrderEmbedding(d_order, d_embed)
        self.slot_attn = SlotAttention(self.num_slots, d_embed, n_slot_iters)
        self.state_proj = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.GELU(),
            nn.Linear(d_embed, d_state),
        )

    def forward(self, orders: torch.Tensor) -> torch.Tensor:
        """
        Args:
            orders: [B, N_orders, d_order] raw order features per time window.
                    d_order typically: (relative_time, relative_price, log_size,
                                       direction, is_limit, is_cancel)

        Returns:
            agent_grid: [B, H, W, d_state]
        """
        B = orders.shape[0]
        embeds = self.order_embed(orders)            # [B, N, d_embed]
        slots = self.slot_attn(embeds)               # [B, K, d_embed]
        states = self.state_proj(slots)              # [B, K, d_state]
        return states.reshape(B, self.grid_h, self.grid_w, -1)
