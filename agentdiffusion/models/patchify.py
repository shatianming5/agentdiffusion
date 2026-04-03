"""Spatial patchification: convert agent grid to/from patch tokens."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class Patchify(nn.Module):
    """Convert [B, H, W, d_agent] grid into [B, N_patches, d_model] tokens."""

    def __init__(self, patch_size: int, d_agent: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.d_agent = d_agent
        self.d_model = d_model
        self.proj = nn.Linear(patch_size * patch_size * d_agent, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, W, d_agent] -> [B, Hp*Wp, d_model]"""
        p = self.patch_size
        x = rearrange(x, "b (hp p1) (wp p2) c -> b (hp wp) (p1 p2 c)", p1=p, p2=p)
        return self.proj(x)

    def num_patches(self, H: int, W: int) -> int:
        return (H // self.patch_size) * (W // self.patch_size)


class Unpatchify(nn.Module):
    """Convert [B, N_patches, d_model] tokens back to [B, H, W, d_agent]."""

    def __init__(self, patch_size: int, d_agent: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.d_agent = d_agent
        self.proj = nn.Linear(d_model, patch_size * patch_size * d_agent)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """x: [B, Hp*Wp, d_model] -> [B, H, W, d_agent]"""
        p = self.patch_size
        Hp, Wp = H // p, W // p
        x = self.proj(x)
        return rearrange(
            x, "b (hp wp) (p1 p2 c) -> b (hp p1) (wp p2) c",
            hp=Hp, wp=Wp, p1=p, p2=p, c=self.d_agent,
        )


class PatchEmbedding(nn.Module):
    """Patchify + learnable 2D position embedding."""

    def __init__(self, patch_size: int, d_agent: int, d_model: int, max_patches: int = 65536):
        super().__init__()
        self.patchify = Patchify(patch_size, d_agent, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, W, d_agent] -> [B, N, d_model]"""
        tokens = self.patchify(x)
        N = tokens.shape[1]
        return tokens + self.pos_embed[:, :N]
