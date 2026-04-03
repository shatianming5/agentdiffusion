"""Learnable market state tokens for global information aggregation."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention, CrossAttention


class MarketTokenModule(nn.Module):
    """Manages learnable market tokens and their interaction with agent tokens.

    Information flow per block:
    1. Agent tokens cross-attend TO market tokens (read global info)
    2. Market tokens self-attend (aggregate)
    3. Market tokens cross-attend TO agent tokens (collect updates)
    """

    def __init__(
        self,
        d_model: int,
        num_market_tokens: int = 128,
        heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.market_tokens = nn.Parameter(torch.randn(1, num_market_tokens, d_model) * 0.02)

        # Agent → Market: agent reads from market
        self.agent_reads_market = CrossAttention(d_model, heads, dropout)
        self.norm_agent_read = nn.LayerNorm(d_model)

        # Market self-attention
        self.market_self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.norm_market_self = nn.LayerNorm(d_model)

        # Market ← Agent: market collects from agents
        self.market_reads_agent = CrossAttention(d_model, heads, dropout)
        self.norm_market_read = nn.LayerNorm(d_model)

    def forward(self, agent_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        agent_tokens: [B, N, D]
        Returns: (updated_agent_tokens, updated_market_tokens)
        """
        B = agent_tokens.shape[0]
        mt = self.market_tokens.expand(B, -1, -1)  # [B, M, D]

        # 1) Agent reads from market tokens
        agent_tokens = agent_tokens + self.agent_reads_market(
            self.norm_agent_read(agent_tokens), mt
        )

        # 2) Market tokens self-attend
        mt = mt + self.market_self_attn(self.norm_market_self(mt))

        # 3) Market reads from agent tokens (aggregate agent info)
        mt = mt + self.market_reads_agent(
            self.norm_market_read(mt), agent_tokens
        )

        return agent_tokens, mt
