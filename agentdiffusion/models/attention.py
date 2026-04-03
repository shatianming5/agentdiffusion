"""Attention modules: local window self-attention and cross-attention."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with optional FlashAttention."""

    def __init__(self, d_model: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D] -> [B, N, D]"""
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each [B, H, N, d]

        # Try scaled_dot_product_attention (supports FlashAttention)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """Cross-attention: Q from x, K/V from context."""

    def __init__(self, d_model: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D], context: [B, M, D] -> [B, N, D]"""
        B, N, _ = x.shape
        M = context.shape[1]

        q = self.q_proj(x).reshape(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(context).reshape(B, M, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.out_proj(out)


class LocalWindowAttention(nn.Module):
    """Window-based local self-attention (Swin-style, no shift for simplicity)."""

    def __init__(self, d_model: int, heads: int, window_size: int = 8, dropout: float = 0.0):
        super().__init__()
        self.window_size = window_size
        self.attn = MultiHeadAttention(d_model, heads, dropout)

    def forward(self, x: torch.Tensor, Hp: int, Wp: int) -> torch.Tensor:
        """x: [B, Hp*Wp, D] with 2D layout Hp×Wp -> [B, Hp*Wp, D]"""
        B, _, D = x.shape
        w = self.window_size

        # 将 token 排列成 2D grid 再划分窗口
        x = x.view(B, Hp, Wp, D)

        # pad 到 window_size 的倍数
        pad_h = (w - Hp % w) % w
        pad_w = (w - Wp % w) % w
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp_pad, Wp_pad = x.shape[1], x.shape[2]

        # 划分窗口
        x = rearrange(
            x, "b (nh wh) (nw ww) d -> (b nh nw) (wh ww) d",
            wh=w, ww=w,
        )

        # 窗口内 attention
        x = self.attn(x)

        # 合并窗口
        nh, nw = Hp_pad // w, Wp_pad // w
        x = rearrange(
            x, "(b nh nw) (wh ww) d -> b (nh wh) (nw ww) d",
            b=B, nh=nh, nw=nw, wh=w, ww=w,
        )

        # 去除 padding
        x = x[:, :Hp, :Wp, :].contiguous()
        return x.view(B, Hp * Wp, D)
