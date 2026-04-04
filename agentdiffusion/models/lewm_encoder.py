"""LeWorldModel ViT-tiny encoder for agent grid states.

Encodes [B, H, W, 128] agent grid into a global latent vector z [B, d_latent]
using patchification, CLS token pooling, and a lightweight ViT backbone.

Architecture (~5M params):
    Input [B, H, W, 128]
    -> Patchify (patch_size=4) -> [B, N_patches, d_enc]
    -> Prepend CLS token -> [B, 1+N_patches, d_enc]
    -> Add positional embeddings
    -> L_enc Transformer encoder layers
    -> CLS token output -> Linear -> z [B, d_latent]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class EncoderBlock(nn.Module):
    """Standard pre-norm Transformer encoder block."""

    def __init__(self, d_enc: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_enc)
        self.attn = nn.MultiheadAttention(d_enc, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_enc)
        mlp_hidden = int(d_enc * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_enc, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_enc),
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


class LeWMEncoder(nn.Module):
    """ViT-tiny encoder for LeWorldModel.

    Takes agent grid [B, H, W, d_agent], patchifies it, runs through
    transformer layers, and outputs a global latent z via CLS token.

    Args:
        d_agent: Per-agent feature dimension (128).
        d_enc: Encoder hidden dimension (256).
        d_latent: Output latent dimension (256).
        patch_size: Spatial patch size (4).
        depth: Number of transformer layers (6).
        heads: Number of attention heads (8).
        mlp_ratio: FFN expansion ratio (4.0).
        max_patches: Maximum number of patches for positional embedding (1024).
        dropout: Dropout rate (0.0).
    """

    def __init__(
        self,
        d_agent: int = 128,
        d_enc: int = 256,
        d_latent: int = 256,
        patch_size: int = 4,
        depth: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        max_patches: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_enc = d_enc
        self.d_latent = d_latent

        # Patch projection: flatten patch_size^2 * d_agent -> d_enc
        patch_dim = patch_size * patch_size * d_agent
        self.patch_proj = nn.Linear(patch_dim, d_enc)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_enc))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional embedding: +1 for CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches + 1, d_enc))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            EncoderBlock(d_enc, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Final layer norm + projection to latent
        self.norm = nn.LayerNorm(d_enc)
        self.head = nn.Linear(d_enc, d_latent)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [B, H, W, d_agent] -> [B, N_patches, patch_dim]."""
        p = self.patch_size
        # rearrange into patches
        x = rearrange(x, "b (hp p1) (wp p2) c -> b (hp wp) (p1 p2 c)", p1=p, p2=p)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode agent grid to latent vector.

        Args:
            x: Agent grid state [B, H, W, d_agent]. H and W must be
               divisible by patch_size.

        Returns:
            z: Global latent vector [B, d_latent].
        """
        B = x.shape[0]

        # Patchify and project
        tokens = self.patch_proj(self.patchify(x))  # [B, N, d_enc]
        N = tokens.shape[1]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d_enc]
        tokens = torch.cat([cls, tokens], dim=1)  # [B, 1+N, d_enc]

        # Add positional embeddings
        tokens = tokens + self.pos_embed[:, : N + 1]

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        # Mean pooling over patch tokens (exclude CLS at position 0)
        patch_out = self.norm(tokens[:, 1:])  # [B, N, d_enc]
        pooled = patch_out.mean(dim=1)  # [B, d_enc]
        z = self.head(pooled)  # [B, d_latent]
        return z

    def forward_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and return all patch tokens (no CLS pooling).

        Useful for sequence-level operations or visualization.

        Args:
            x: Agent grid state [B, H, W, d_agent].

        Returns:
            tokens: Patch token representations [B, N_patches, d_enc].
        """
        B = x.shape[0]

        tokens = self.patch_proj(self.patchify(x))
        N = tokens.shape[1]

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : N + 1]

        for block in self.blocks:
            tokens = block(tokens)

        # Return patch tokens (exclude CLS)
        return self.norm(tokens[:, 1:])  # [B, N, d_enc]
