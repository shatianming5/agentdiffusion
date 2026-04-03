"""Timestep, position, and condition embeddings."""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion timestep t."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] -> [B, dim]"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.cos(), args.sin()], dim=-1)


class ConditionEmbedding(nn.Module):
    """Combine timestep + market condition + scenario into a single conditioning vector.

    Output drives adaLN-Zero modulation in each DiT block.
    """

    def __init__(self, d_model: int, market_cond_dim: int = 32, num_scenarios: int = 8):
        super().__init__()
        self.t_embed = SinusoidalTimestepEmbedding(d_model)
        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.market_mlp = nn.Sequential(
            nn.Linear(market_cond_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.scenario_embed = nn.Embedding(num_scenarios, d_model)
        self.out_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        t: torch.Tensor,                          # [B]
        market_cond: torch.Tensor | None = None,   # [B, market_cond_dim]
        scenario: torch.Tensor | None = None,      # [B] int
    ) -> torch.Tensor:
        """Returns [B, d_model] conditioning vector."""
        c = self.t_mlp(self.t_embed(t))
        if market_cond is not None:
            c = c + self.market_mlp(market_cond)
        if scenario is not None:
            c = c + self.scenario_embed(scenario)
        return self.out_mlp(c)
