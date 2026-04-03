"""LeWorldModel Transformer predictor with AdaLN conditioning.

Takes current latent z_t [B, d_latent] and market condition [B, d_cond],
predicts next-step latent z_{t+1} [B, d_latent].

Architecture (~10M params):
    z_t [B, d_latent] -> token sequence [B, L, d_pred]
    market_cond [B, d_cond] -> AdaLN modulation per layer
    6 Transformer layers with AdaLN conditioning
    -> z_{t+1} prediction [B, d_latent]

Supports autoregressive rollout on sequences of latents with causal masking.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLN(nn.Module):
    """Adaptive Layer Normalization: modulates normalized features
    using condition-derived scale (gamma) and shift (beta).

    Unlike adaLN-Zero (which also has gating alpha), this is the
    standard adaLN used in the Le World Model predictor.
    """

    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # Condition -> (gamma, beta) modulation
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, d_model * 2),
        )
        # Initialize to identity transform: gamma=1, beta=0
        nn.init.zeros_(self.cond_proj[-1].weight)
        nn.init.zeros_(self.cond_proj[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Apply adaptive layer norm.

        Args:
            x: Input features [B, L, d_model] or [B, d_model].
            c: Condition vector [B, d_cond].

        Returns:
            Modulated features, same shape as x.
        """
        params = self.cond_proj(c)  # [B, d_model*2]
        if x.ndim == 3:
            params = params.unsqueeze(1)  # [B, 1, d_model*2]
        gamma, beta = params.chunk(2, dim=-1)
        return self.norm(x) * (1 + gamma) + beta


class PredictorBlock(nn.Module):
    """Single Transformer predictor block with AdaLN conditioning.

    Pre-norm architecture with condition-modulated layer norms.
    Supports optional causal masking for autoregressive sequence prediction.
    """

    def __init__(
        self,
        d_pred: int,
        d_cond: int,
        heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.adaln1 = AdaLN(d_pred, d_cond)
        self.attn = nn.MultiheadAttention(
            d_pred, heads, dropout=dropout, batch_first=True,
        )
        self.adaln2 = AdaLN(d_pred, d_cond)
        mlp_hidden = int(d_pred * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_pred, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_pred),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with AdaLN conditioning.

        Args:
            x: Token sequence [B, L, d_pred].
            c: Condition vector [B, d_cond].
            attn_mask: Optional causal mask [L, L] for autoregressive mode.

        Returns:
            Updated token sequence [B, L, d_pred].
        """
        # Self-attention with AdaLN
        h = self.adaln1(x, c)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h

        # FFN with AdaLN
        x = x + self.ffn(self.adaln2(x, c))
        return x


class LeWMPredictor(nn.Module):
    """Transformer predictor for LeWorldModel.

    Takes latent z_t and market condition, predicts z_{t+1}.
    Supports both single-step prediction and autoregressive
    sequence rollout with causal masking.

    Args:
        d_latent: Encoder output / latent dimension (256).
        d_pred: Predictor hidden dimension (512).
        d_cond: Market condition dimension (32).
        depth: Number of predictor transformer layers (6).
        heads: Number of attention heads (8).
        mlp_ratio: FFN expansion ratio (4.0).
        max_seq_len: Maximum sequence length for positional encoding (256).
        dropout: Dropout rate (0.0).
    """

    def __init__(
        self,
        d_latent: int = 256,
        d_pred: int = 512,
        d_cond: int = 32,
        depth: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.d_pred = d_pred
        self.d_cond = d_cond

        # Project latent to predictor dimension
        self.input_proj = nn.Linear(d_latent, d_pred)

        # Project condition to internal conditioning dimension
        self.cond_proj = nn.Sequential(
            nn.Linear(d_cond, d_pred),
            nn.GELU(),
            nn.Linear(d_pred, d_pred),
        )

        # Learnable positional embeddings for sequence positions
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_pred))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Predictor transformer blocks
        self.blocks = nn.ModuleList([
            PredictorBlock(d_pred, d_pred, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Output projection back to latent space
        self.norm = nn.LayerNorm(d_pred)
        self.output_proj = nn.Linear(d_pred, d_latent)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init output projection for stable training start
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal (upper-triangular) attention mask.

        Returns:
            mask: [seq_len, seq_len] float tensor with -inf for masked positions.
        """
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(
        self,
        z_t: torch.Tensor,
        market_cond: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        """Predict next-step latent(s).

        Args:
            z_t: Current latent(s). Either:
                - Single step: [B, d_latent]
                - Sequence: [B, L, d_latent]
            market_cond: Market condition [B, d_cond].
            causal: If True and z_t is a sequence, apply causal masking
                    for autoregressive rollout training.

        Returns:
            z_pred: Predicted next latent(s), same shape as z_t.
        """
        single_step = z_t.ndim == 2
        if single_step:
            z_t = z_t.unsqueeze(1)  # [B, 1, d_latent]

        B, L, _ = z_t.shape

        # Project to predictor dimension
        x = self.input_proj(z_t)  # [B, L, d_pred]
        x = x + self.pos_embed[:, :L]

        # Condition embedding
        c = self.cond_proj(market_cond)  # [B, d_pred]

        # Causal mask for autoregressive mode
        attn_mask = self._causal_mask(L, x.device) if (causal and L > 1) else None

        # Transformer blocks with AdaLN conditioning
        for block in self.blocks:
            x = block(x, c, attn_mask)

        # Project back to latent space
        z_pred = self.output_proj(self.norm(x))  # [B, L, d_latent]

        if single_step:
            z_pred = z_pred.squeeze(1)  # [B, d_latent]

        return z_pred
