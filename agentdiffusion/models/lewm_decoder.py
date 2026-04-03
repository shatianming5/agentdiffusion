"""LeWorldModel Transformer decoder: latent vector -> agent grid.

Maps [B, d_latent] latent vector back to [B, H, W, d_agent] agent grid,
reversing the encoder's patchify + ViT + CLS pooling pipeline.

Architecture:
    z [B, d_latent]
    -> MLP projects to (Hp * Wp) * d_dec token grid  [B, N_tokens, d_dec]
    -> Add learned position embeddings
    -> L_dec Transformer decoder layers (self-attention, pre-norm)
    -> Linear projects each token: d_dec -> patch_size^2 * d_agent
    -> Unpatchify: reshape to [B, H, W, d_agent]

With H=W=36 (padded), patch_size=4:
    Hp = Wp = 9, so N_tokens = 81
    d_dec = 256, depth = 4, heads = 4
    Final linear: 256 -> 4*4*128 = 2048 per token
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class DecoderBlock(nn.Module):
    """Standard pre-norm Transformer block for decoder."""

    def __init__(self, d_dec: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_dec)
        self.attn = nn.MultiheadAttention(d_dec, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_dec)
        mlp_hidden = int(d_dec * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_dec, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_dec),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))
        return x


class LeWMDecoder(nn.Module):
    """Transformer decoder for LeWorldModel.

    Reverses the encoder: takes a global latent vector z and reconstructs
    the full [B, H, W, d_agent] agent grid through token expansion,
    Transformer self-attention, and unpatchification.

    Args:
        d_latent: Input latent dimension from encoder (256).
        d_agent: Per-agent raw feature dimension (128).
        d_dec: Decoder hidden / token dimension (256).
        patch_size: Spatial patch size, must match encoder (4).
        grid_h: Padded grid height (36).
        grid_w: Padded grid width (36).
        depth: Number of Transformer layers (4).
        heads: Number of attention heads (4).
        mlp_ratio: FFN expansion ratio (4.0).
        dropout: Dropout rate (0.0).
    """

    def __init__(
        self,
        d_latent: int = 256,
        d_agent: int = 128,
        d_dec: int = 256,
        patch_size: int = 4,
        grid_h: int = 36,
        grid_w: int = 36,
        depth: int = 4,
        heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.d_agent = d_agent
        self.d_dec = d_dec
        self.patch_size = patch_size
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Number of spatial tokens = (H/p) * (W/p)
        self.Hp = grid_h // patch_size   # 9
        self.Wp = grid_w // patch_size   # 9
        self.num_tokens = self.Hp * self.Wp  # 81

        # MLP to project latent -> token grid
        # d_latent -> intermediate -> num_tokens * d_dec
        self.latent_to_tokens = nn.Sequential(
            nn.Linear(d_latent, d_dec * 4),
            nn.GELU(),
            nn.Linear(d_dec * 4, self.num_tokens * d_dec),
        )

        # Learned position embeddings for the token grid
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, d_dec))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(d_dec, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_dec)

        # Project each token to a full patch: d_dec -> patch_size^2 * d_agent
        patch_dim = patch_size * patch_size * d_agent  # 4*4*128 = 2048
        self.token_to_patch = nn.Linear(d_dec, patch_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert [B, N_tokens, patch_dim] -> [B, H, W, d_agent].

        Each token covers a patch_size x patch_size spatial region with
        d_agent channels. Rearranges the flat patches back into the grid.
        """
        p = self.patch_size
        return rearrange(
            tokens,
            "b (hp wp) (p1 p2 c) -> b (hp p1) (wp p2) c",
            hp=self.Hp,
            wp=self.Wp,
            p1=p,
            p2=p,
            c=self.d_agent,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to agent grid.

        Args:
            z: Latent vector [B, d_latent].

        Returns:
            state: Reconstructed agent grid [B, grid_h, grid_w, d_agent].
        """
        B = z.shape[0]

        # Project latent to token grid
        tokens = self.latent_to_tokens(z)  # [B, num_tokens * d_dec]
        tokens = tokens.view(B, self.num_tokens, self.d_dec)  # [B, 81, 256]

        # Add positional embeddings
        tokens = tokens + self.pos_embed  # [B, 81, 256]

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        # Final norm
        tokens = self.norm(tokens)  # [B, 81, 256]

        # Project each token to patch
        patches = self.token_to_patch(tokens)  # [B, 81, 2048]

        # Unpatchify: reshape to spatial grid
        state = self.unpatchify(patches)  # [B, 36, 36, 128]

        return state
