"""Video DiT: Factorized Spatiotemporal Diffusion Transformer for agent grid sequences.

Architecture overview:
    K=8 condition frames (clean) + N=32 generation frames (noisy) = 40 total frames.
    Each frame [36,36,d_latent] -> patchify -> [81, d_model] tokens.
    Per block:
        1. Spatial self-attention within each frame: [B*T, 81, D]
        2. Temporal self-attention across frames per spatial position: [B*81, T, D]
        3. FFN with adaLN-Zero conditioning on diffusion timestep
    Output: v-prediction for the N=32 generation frames only.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .patchify import Patchify
from .embeddings import SinusoidalTimestepEmbedding


# ---------------------------------------------------------------------------
# adaLN-Zero modulation (produces 6 modulation parameters per sub-layer pair)
# ---------------------------------------------------------------------------

class AdaLNZeroModulation(nn.Module):
    """Produce (gamma1, beta1, alpha1, gamma2, beta2, alpha2) from conditioning c."""

    def __init__(self, d_model: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(d_model, d_model * 6)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """c: [B, D] -> 6 x [B, 1, D] modulation vectors."""
        return self.linear(self.silu(c)).unsqueeze(1).chunk(6, dim=-1)


# ---------------------------------------------------------------------------
# Spatial attention: self-attention within each frame
# ---------------------------------------------------------------------------

class SpatialAttention(nn.Module):
    """Self-attention within each frame.

    Input:  [B*T, N_patches, D]
    Output: [B*T, N_patches, D]
    """

    def __init__(self, d_model: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each [B, H, N, d]
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Temporal attention: self-attention across frames for each spatial position
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """Self-attention across frames for each spatial position.

    Supports causal masking (each frame only attends to current + past frames)
    and ALiBi-style relative position bias for temporal locality.

    Input:  [B*N_patches, T, D]
    Output: [B*N_patches, T, D]
    """

    def __init__(self, d_model: int, heads: int, dropout: float = 0.0,
                 causal: bool = False, alibi: bool = False):
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.causal = causal
        self.alibi = alibi

        if alibi:
            # ALiBi slopes: geometric sequence for each head
            slopes = 2.0 ** (-8.0 * torch.arange(1, heads + 1) / heads)
            self.register_buffer("alibi_slopes", slopes)  # [H]

    def _build_alibi_bias(self, T: int, device: torch.device) -> torch.Tensor:
        """Build ALiBi position bias: [1, H, T, T]."""
        pos = torch.arange(T, device=device)
        rel = pos.unsqueeze(0) - pos.unsqueeze(1)  # [T, T], rel[i,j] = j - i
        bias = rel.float().unsqueeze(0) * self.alibi_slopes.unsqueeze(1).unsqueeze(2)
        return bias.unsqueeze(0)  # [1, H, T, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Build attention mask
        attn_mask = None
        if self.causal or self.alibi:
            # Start with zero bias
            bias = torch.zeros(1, 1, T, T, device=x.device)
            if self.alibi:
                bias = bias + self._build_alibi_bias(T, x.device)
            if self.causal:
                causal_mask = torch.triu(
                    torch.full((T, T), float("-inf"), device=x.device), diagonal=1
                )
                bias = bias + causal_mask.unsqueeze(0).unsqueeze(0)
            attn_mask = bias.expand(B, self.heads, T, T)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# VideoDiT block: spatial -> temporal -> FFN, all with adaLN-Zero
# ---------------------------------------------------------------------------

class VideoDiTBlock(nn.Module):
    """Single factorized spatiotemporal DiT block.

    Processing order:
        1. adaLN + spatial self-attention (within each frame)
        2. adaLN + temporal self-attention (across frames per position)
        3. adaLN + FFN

    Uses adaLN-Zero: first 6 params for spatial+temporal, next 6 for FFN pair.
    We use two separate AdaLNZero modules: one for spatial+temporal, one for FFN.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        causal_temporal: bool = False,
        alibi_temporal: bool = False,
    ):
        super().__init__()
        # --- Spatial attention sub-block ---
        self.adaln_spatial = AdaLNZeroModulation(d_model)
        self.norm_spatial = nn.LayerNorm(d_model, elementwise_affine=False)
        self.spatial_attn = SpatialAttention(d_model, heads, dropout)

        # --- Temporal attention sub-block ---
        self.adaln_temporal = AdaLNZeroModulation(d_model)
        self.norm_temporal = nn.LayerNorm(d_model, elementwise_affine=False)
        self.temporal_attn = TemporalAttention(
            d_model, heads, dropout, causal=causal_temporal, alibi=alibi_temporal,
        )

        # --- FFN sub-block ---
        self.adaln_ffn = AdaLNZeroModulation(d_model)
        self.norm_ffn = nn.LayerNorm(d_model, elementwise_affine=False)
        mlp_hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, T: int) -> torch.Tensor:
        """
        Args:
            x: [B, T*N, D] where N = num_patches per frame (81 for 36x36 grid, patch=4).
            c: [B, D] conditioning vector (from timestep + optional market cond).
            T: number of frames (e.g. 40).

        Returns:
            [B, T*N, D]
        """
        B_orig = x.shape[0]
        N = x.shape[1] // T  # patches per frame

        # --- 1) Spatial self-attention ---
        gamma1, beta1, alpha1, _g, _b, _a = self.adaln_spatial(c)
        h = self.norm_spatial(x) * (1 + gamma1) + beta1
        # Reshape for spatial: [B*T, N, D]
        h = rearrange(h, "b (t n) d -> (b t) n d", t=T)
        h = self.spatial_attn(h)
        h = rearrange(h, "(b t) n d -> b (t n) d", b=B_orig)
        x = x + alpha1 * h

        # --- 2) Temporal self-attention ---
        gamma2, beta2, alpha2, _g2, _b2, _a2 = self.adaln_temporal(c)
        h = self.norm_temporal(x) * (1 + gamma2) + beta2
        # Reshape for temporal: [B*N, T, D]
        h = rearrange(h, "b (t n) d -> (b n) t d", t=T)
        h = self.temporal_attn(h)
        h = rearrange(h, "(b n) t d -> b (t n) d", b=B_orig)
        x = x + alpha2 * h

        # --- 3) FFN ---
        gamma3, beta3, alpha3, _g3, _b3, _a3 = self.adaln_ffn(c)
        h = self.norm_ffn(x) * (1 + gamma3) + beta3
        h = self.ffn(h)
        x = x + alpha3 * h

        return x


# ---------------------------------------------------------------------------
# Final projection layer (adaLN-Zero + linear)
# ---------------------------------------------------------------------------

class VideoFinalLayer(nn.Module):
    """Final adaLN-Zero + linear projection back to patch-token space."""

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


# ---------------------------------------------------------------------------
# Timestep + optional market conditioning
# ---------------------------------------------------------------------------

class VideoConditionEmbedding(nn.Module):
    """Combine diffusion timestep + optional market condition into a single vector.

    Drives adaLN-Zero modulation in each VideoDiTBlock.
    """

    def __init__(self, d_model: int, market_cond_dim: int = 32):
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

    def forward(
        self,
        t: torch.Tensor,
        market_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            t: [B] diffusion timestep.
            market_cond: [B, market_cond_dim] optional market conditions.
        Returns:
            [B, d_model] conditioning vector.
        """
        c = self.t_mlp(self.t_embed(t))
        if market_cond is not None:
            c = c + self.market_mlp(market_cond)
        return c


# ---------------------------------------------------------------------------
# Main model: VideoDiT
# ---------------------------------------------------------------------------

class VideoDiT(nn.Module):
    """Video Diffusion Transformer for agent grid sequences.

    Processes K condition frames (clean) + N generation frames (noisy) together
    via factorized spatiotemporal attention, predicting v for the N generation
    frames only.

    Architecture:
        For each frame: [H, W, d_latent] -> Patchify -> [Np, d_model]
        Add spatial position embedding (shared across all frames)
        Add temporal position embedding (per-frame index)
        Add frame type embedding (0=condition, 1=generation)
        Concat all frames: [B, T*Np, d_model]
        N x VideoDiTBlock (spatial attn -> temporal attn -> FFN)
        VideoFinalLayer -> Unpatchify -> extract generation frames
    """

    def __init__(
        self,
        d_latent: int = 16,
        d_model: int = 512,
        depth: int = 12,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_size: int = 4,
        grid_h: int = 36,
        grid_w: int = 36,
        num_frames: int = 40,
        num_cond_frames: int = 8,
        market_cond_dim: int = 32,
        dropout: float = 0.0,
        causal_temporal: bool = False,
        alibi_temporal: bool = False,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.d_model = d_model
        self.patch_size = patch_size
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_frames = num_frames
        self.num_cond_frames = num_cond_frames
        self.num_gen_frames = num_frames - num_cond_frames

        # Derived spatial dims
        self.Hp = grid_h // patch_size  # 9
        self.Wp = grid_w // patch_size  # 9
        self.num_patches = self.Hp * self.Wp  # 81

        # --- Patch embedding (spatial: shared across frames) ---
        self.patchify = Patchify(patch_size, d_latent, d_model)
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)

        # --- Temporal position embedding [T, D] ---
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, num_frames, d_model)
        )
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # --- Frame type embedding: 0=condition, 1=generation ---
        self.frame_type_embed = nn.Embedding(2, d_model)
        nn.init.trunc_normal_(self.frame_type_embed.weight, std=0.02)

        # --- Conditioning (diffusion timestep + optional market cond) ---
        self.cond_embed = VideoConditionEmbedding(d_model, market_cond_dim)

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            VideoDiTBlock(
                d_model=d_model,
                heads=heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                causal_temporal=causal_temporal,
                alibi_temporal=alibi_temporal,
            )
            for _ in range(depth)
        ])

        # --- Final layer: projects from d_model to patch_size^2 * d_latent ---
        out_dim = patch_size * patch_size * d_latent
        self.final_layer = VideoFinalLayer(d_model, out_dim)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)

    def _patchify_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Convert [B, T, H, W, d_latent] frames to [B, T, Np, d_model] tokens.

        Applies the shared spatial patchifier to each frame independently.
        """
        B, T, H, W, C = frames.shape
        # Flatten batch and time for patchification
        flat = frames.reshape(B * T, H, W, C)       # [B*T, H, W, d_latent]
        tokens = self.patchify(flat)                  # [B*T, Np, d_model]
        return tokens.reshape(B, T, self.num_patches, self.d_model)

    def forward(
        self,
        x_cond: torch.Tensor,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        market_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: predict v for generation frames.

        Args:
            x_cond:  [B, K, H, W, d_latent] -- K clean condition frames.
            x_noisy: [B, N, H, W, d_latent] -- N noised generation frames.
            t:       [B] -- diffusion timestep.
            market_cond: [B, market_cond_dim] -- optional market conditions.

        Returns:
            [B, N, H, W, d_latent] -- predicted v for generation frames only.
        """
        B, K = x_cond.shape[:2]
        N = x_noisy.shape[1]
        T = K + N  # total frames (e.g. 40)
        H, W = self.grid_h, self.grid_w

        # --- Patchify all frames ---
        # Concat along time dimension: [B, T, H, W, d_latent]
        all_frames = torch.cat([x_cond, x_noisy], dim=1)
        tokens = self._patchify_frames(all_frames)  # [B, T, Np, D]

        # --- Add spatial position embedding (shared across frames) ---
        tokens = tokens + self.spatial_pos_embed.unsqueeze(1)  # broadcast over T

        # --- Add temporal position embedding ---
        # temporal_pos_embed: [1, T, D] -> [1, T, 1, D] broadcast over Np
        tokens = tokens + self.temporal_pos_embed[:, :T].unsqueeze(2)

        # --- Add frame type embedding ---
        # frame_types: 0 for condition, 1 for generation
        frame_types = torch.cat([
            torch.zeros(K, dtype=torch.long, device=t.device),
            torch.ones(N, dtype=torch.long, device=t.device),
        ])  # [T]
        frame_type_emb = self.frame_type_embed(frame_types)  # [T, D]
        tokens = tokens + frame_type_emb.unsqueeze(0).unsqueeze(2)  # [1, T, 1, D]

        # --- Flatten to sequence: [B, T*Np, D] ---
        x = rearrange(tokens, "b t n d -> b (t n) d")

        # --- Conditioning vector ---
        c = self.cond_embed(t, market_cond)  # [B, D]

        # --- Transformer blocks ---
        for block in self.blocks:
            x = block(x, c, T)

        # --- Final layer ---
        x = self.final_layer(x, c)  # [B, T*Np, patch_size^2 * d_latent]

        # --- Extract generation frames only ---
        # x is [B, T*Np, out_dim]; generation frames start at index K*Np
        gen_start = K * self.num_patches
        gen_end = T * self.num_patches
        x_gen = x[:, gen_start:gen_end]  # [B, N*Np, out_dim]

        # --- Unpatchify generation frames manually ---
        # x_gen: [B, N*Np, patch_size^2 * d_latent]
        # Reshape to [B, N, Hp, Wp, patch_size, patch_size, d_latent]
        p = self.patch_size
        Hp, Wp = self.Hp, self.Wp
        x_gen = rearrange(
            x_gen,
            "b (n hp wp) (p1 p2 c) -> b n (hp p1) (wp p2) c",
            n=N, hp=Hp, wp=Wp, p1=p, p2=p, c=self.d_latent,
        )
        out = x_gen  # [B, N, H, W, d_latent]

        return out


# ---------------------------------------------------------------------------
# DDIM sampler adapted for video: only denoise generation frames
# ---------------------------------------------------------------------------

class VideoDDIMSampler:
    """DDIM sampler for Video DiT: denoises generation frames conditioned on clean frames.

    At each denoising step, the condition frames remain clean (not noised).
    Only the generation frames are iteratively denoised.
    """

    def __init__(
        self,
        model: VideoDiT,
        scheduler,  # NoiseScheduler
        prediction_type: str = "v_prediction",
        ddim_steps: int = 50,
        eta: float = 0.0,
    ):
        self.model = model
        self.scheduler = scheduler
        self.prediction_type = prediction_type
        self.ddim_steps = ddim_steps
        self.eta = eta

        # Build sub-sequence of timesteps (descending)
        total_T = scheduler.timesteps
        step_size = total_T // ddim_steps
        self.timestep_seq = list(range(0, total_T, step_size))[:ddim_steps]
        self.timestep_seq.reverse()

    @torch.no_grad()
    def sample(
        self,
        x_cond: torch.Tensor,
        gen_shape: tuple[int, ...],
        market_cond: torch.Tensor | None = None,
        device: torch.device | None = None,
        zero_sum_proj: bool = False,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run DDIM denoising on generation frames while keeping condition frames clean.

        Args:
            x_cond:    [B, K, H, W, d_latent] -- clean condition frames (latent).
            gen_shape: (B, N, H, W, d_latent)  -- shape of generation frames.
            market_cond: [B, market_cond_dim]   -- optional market conditions.
            device: target device.
            zero_sum_proj: if True, project position (dim 0) changes to zero-sum
                           after each denoising step.
            valid_mask: [H, W] bool mask of valid agents (for zero-sum projection).

        Returns:
            [B, N, H, W, d_latent] -- denoised generation frames.
        """
        if device is None:
            device = next(self.model.parameters()).device

        B = gen_shape[0]
        x_cond = x_cond.to(device)

        # Start from pure noise for generation frames
        x_gen = torch.randn(gen_shape, device=device)
        seq = self.timestep_seq

        for i, t_cur in enumerate(seq):
            t_tensor = torch.full((B,), t_cur, device=device, dtype=torch.long)

            # Model predicts v for generation frames
            v_pred = self.model(x_cond, x_gen, t_tensor, market_cond)

            # Recover x0 and epsilon from v-prediction
            if self.prediction_type == "v_prediction":
                x0_pred = self.scheduler.predict_x0_from_v(x_gen, t_tensor, v_pred)
                eps_pred = self.scheduler.predict_noise_from_v(x_gen, t_tensor, v_pred)
            elif self.prediction_type == "epsilon":
                eps_pred = v_pred
                sqrt_alpha = self.scheduler._extract(
                    self.scheduler.sqrt_alphas_cumprod, t_tensor, x_gen.shape
                )
                sqrt_one_minus = self.scheduler._extract(
                    self.scheduler.sqrt_one_minus_alphas_cumprod, t_tensor, x_gen.shape
                )
                x0_pred = (x_gen - sqrt_one_minus * eps_pred) / sqrt_alpha
            else:
                x0_pred = v_pred
                sqrt_alpha = self.scheduler._extract(
                    self.scheduler.sqrt_alphas_cumprod, t_tensor, x_gen.shape
                )
                sqrt_one_minus = self.scheduler._extract(
                    self.scheduler.sqrt_one_minus_alphas_cumprod, t_tensor, x_gen.shape
                )
                eps_pred = (x_gen - sqrt_alpha * x0_pred) / sqrt_one_minus

            # DDIM update step
            if i < len(seq) - 1:
                t_next = seq[i + 1]
                t_next_tensor = torch.full((B,), t_next, device=device, dtype=torch.long)
                alpha_cur = self.scheduler._extract(
                    self.scheduler.alphas_cumprod, t_tensor, x_gen.shape
                )
                alpha_next = self.scheduler._extract(
                    self.scheduler.alphas_cumprod, t_next_tensor, x_gen.shape
                )
            else:
                alpha_cur = self.scheduler._extract(
                    self.scheduler.alphas_cumprod, t_tensor, x_gen.shape
                )
                alpha_next = torch.ones_like(alpha_cur)

            sigma = self.eta * (
                (1 - alpha_next) / (1 - alpha_cur) * (1 - alpha_cur / alpha_next)
            ).sqrt()

            x_gen = (
                alpha_next.sqrt() * x0_pred
                + (1 - alpha_next - sigma ** 2).clamp(min=0).sqrt() * eps_pred
            )
            if self.eta > 0:
                x_gen = x_gen + sigma * torch.randn_like(x_gen)

            # --- Zero-sum projection on position (dim 0) ---
            if zero_sum_proj:
                # x_gen: [B, N, H, W, d_latent], dim 0 of last axis = position
                pos = x_gen[:, :, :, :, 0]  # [B, N, H, W]
                if valid_mask is not None:
                    # Mean correction over valid agents only
                    vm = valid_mask.to(pos.device)  # [H, W]
                    n_valid = vm.sum().float().clamp(min=1)
                    net = (pos * vm).sum(dim=(-2, -1), keepdim=True) / n_valid
                    x_gen[:, :, :, :, 0] = pos - net * vm
                else:
                    H, W = pos.shape[-2], pos.shape[-1]
                    net = pos.mean(dim=(-2, -1), keepdim=True)
                    x_gen[:, :, :, :, 0] = pos - net

        return x_gen


# ---------------------------------------------------------------------------
# Builder function
# ---------------------------------------------------------------------------

def build_video_dit(cfg) -> VideoDiT:
    """Build VideoDiT from OmegaConf config."""
    return VideoDiT(
        d_latent=cfg.agent.latent_dim,
        d_model=cfg.model.d_model,
        depth=cfg.model.depth,
        heads=cfg.model.heads,
        mlp_ratio=cfg.model.mlp_ratio,
        patch_size=cfg.patch.patch_size,
        grid_h=cfg.patch.grid_h,
        grid_w=cfg.patch.grid_w,
        num_frames=cfg.video.num_frames,
        num_cond_frames=cfg.video.num_cond_frames,
        market_cond_dim=getattr(cfg.model, "market_cond_dim", 32),
        dropout=cfg.model.dropout,
        causal_temporal=getattr(cfg.model, "causal_temporal", False),
        alibi_temporal=getattr(cfg.model, "alibi_temporal", False),
    )
