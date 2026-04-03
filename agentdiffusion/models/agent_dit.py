"""AgentDiT: Market-Aware Diffusion Transformer for Agent-Based Simulation."""

from __future__ import annotations

import torch
import torch.nn as nn

from .autoencoder import AgentEncoder, AgentDecoder
from .patchify import PatchEmbedding, Unpatchify
from .embeddings import ConditionEmbedding
from .dit_block import DiTBlock, FinalLayer


class AgentDiT(nn.Module):
    """Full diffusion model: encodes agent grid, denoises in latent space, decodes back.

    Architecture:
        Raw Grid [H,W,128]
         -> Per-agent encode [H,W,d_agent]
         -> Patchify + pos embed [N_patches, d_model]
         -> N × DiTBlock (local attn + market cross-attn + FFN)
         -> FinalLayer
         -> Unpatchify [H,W,d_agent]
         -> Per-agent decode [H,W,128]
    """

    def __init__(
        self,
        raw_dim: int = 128,
        latent_dim: int = 16,
        d_model: int = 512,
        depth: int = 12,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_size: int = 4,
        num_market_tokens: int = 128,
        local_window_size: int = 8,
        dropout: float = 0.0,
        market_cond_dim: int = 32,
        num_scenarios: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size

        # Condition embedding
        self.cond_embed = ConditionEmbedding(d_model, market_cond_dim, num_scenarios)

        # Patch embedding (operates on latent space)
        self.patch_embed = PatchEmbedding(patch_size, latent_dim, d_model)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                d_model=d_model,
                heads=heads,
                mlp_ratio=mlp_ratio,
                num_market_tokens=num_market_tokens,
                local_window_size=local_window_size,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # Final layer -> unpatchify output dimension
        out_dim = patch_size * patch_size * latent_dim
        self.final_layer = FinalLayer(d_model, out_dim)

        # Unpatchify
        self.unpatchify = Unpatchify(patch_size, latent_dim, d_model)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)

    def forward(
        self,
        z_noisy: torch.Tensor,                    # [B, H, W, latent_dim]  noised latent
        t: torch.Tensor,                           # [B]  diffusion timestep
        market_cond: torch.Tensor | None = None,   # [B, market_cond_dim]
        scenario: torch.Tensor | None = None,      # [B]  int
    ) -> torch.Tensor:
        """Predict v (or noise/x0) given noisy latent + conditions.

        Returns: [B, H, W, latent_dim]
        """
        B, H, W, _ = z_noisy.shape
        p = self.patch_size
        Hp, Wp = H // p, W // p

        # Condition vector
        c = self.cond_embed(t, market_cond, scenario)  # [B, d_model]

        # Patchify + position embed
        x = self.patch_embed(z_noisy)  # [B, Hp*Wp, d_model]

        # DiT blocks
        for block in self.blocks:
            x = block(x, c, Hp, Wp)

        # Final layer + unpatchify
        x = self.final_layer(x, c)  # [B, Hp*Wp, p*p*latent_dim]

        # Reshape back to grid
        x = x.view(B, Hp, Wp, p, p, self.latent_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, self.latent_dim)

        return x


def build_agent_dit(cfg) -> AgentDiT:
    """Build AgentDiT from config object."""
    return AgentDiT(
        raw_dim=cfg.agent.raw_dim,
        latent_dim=cfg.agent.latent_dim,
        d_model=cfg.model.d_model,
        depth=cfg.model.depth,
        heads=cfg.model.heads,
        mlp_ratio=cfg.model.mlp_ratio,
        patch_size=cfg.patch.patch_size,
        num_market_tokens=cfg.model.num_market_tokens,
        local_window_size=cfg.model.local_window_size,
        dropout=cfg.model.dropout,
    )
