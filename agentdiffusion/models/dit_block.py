"""DiT Block with adaLN-Zero conditioning and market token cross-attention."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import LocalWindowAttention
from .market_tokens import MarketTokenModule


class AdaLNZero(nn.Module):
    """Adaptive Layer Norm Zero — produces (gamma, beta, alpha) from condition c."""

    def __init__(self, d_model: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(d_model, d_model * 6)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """c: [B, D] -> 6 × [B, 1, D] modulation params."""
        params = self.linear(self.silu(c))
        return params.unsqueeze(1).chunk(6, dim=-1)


class DiTBlock(nn.Module):
    """Single DiT block: local attn -> market cross-attn -> FFN, all with adaLN-Zero."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        mlp_ratio: float = 4.0,
        num_market_tokens: int = 128,
        local_window_size: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        # adaLN-Zero modulation
        self.adaln = AdaLNZero(d_model)

        # Local window self-attention
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.local_attn = LocalWindowAttention(d_model, heads, local_window_size, dropout)

        # Market token interaction
        self.market_module = MarketTokenModule(d_model, num_market_tokens, heads, dropout)

        # FFN
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        mlp_hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,        # [B, N, D]
        c: torch.Tensor,        # [B, D]  conditioning
        Hp: int,
        Wp: int,
    ) -> torch.Tensor:
        # adaLN-Zero: 6 modulation vectors
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln(c)

        # 1) Local window self-attention
        h = self.norm1(x) * (1 + gamma1) + beta1
        h = self.local_attn(h, Hp, Wp)
        x = x + alpha1 * h

        # 2) Market token cross-attention
        x, _market_tokens = self.market_module(x)

        # 3) FFN
        h = self.norm2(x) * (1 + gamma2) + beta2
        h = self.ffn(h)
        x = x + alpha2 * h

        return x


class FinalLayer(nn.Module):
    """Final adaLN-Zero + linear projection to output patch tokens."""

    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2),
        )
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)
        self.proj = nn.Linear(d_model, out_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.adaln(c).unsqueeze(1).chunk(2, dim=-1)
        x = self.norm(x) * (1 + gamma) + beta
        return self.proj(x)
