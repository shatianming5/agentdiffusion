"""Per-Agent MLP Autoencoder for state compression."""

from __future__ import annotations

import torch
import torch.nn as nn


class AgentEncoder(nn.Module):
    """Compress per-agent state vector from raw_dim to latent_dim."""

    def __init__(self, raw_dim: int = 128, latent_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(raw_dim),
            nn.Linear(raw_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., raw_dim] -> [..., latent_dim]"""
        return self.net(x)


class AgentDecoder(nn.Module):
    """Decompress per-agent latent back to raw_dim."""

    def __init__(self, latent_dim: int = 16, raw_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, raw_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [..., latent_dim] -> [..., raw_dim]"""
        return self.net(z)


class AgentAutoencoder(nn.Module):
    """Full autoencoder: encode + decode with optional KL regularization (VAE mode)."""

    def __init__(
        self,
        raw_dim: int = 128,
        latent_dim: int = 16,
        hidden: int = 64,
        vae: bool = False,
    ):
        super().__init__()
        self.vae = vae
        self.encoder = AgentEncoder(raw_dim, latent_dim * (2 if vae else 1), hidden)
        self.decoder = AgentDecoder(latent_dim, raw_dim, hidden)
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        if not self.vae:
            return h
        mu, log_var = h.chunk(2, dim=-1)
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.vae:
            z, mu, log_var = self.encode(x)
            x_recon = self.decode(z)
            recon_loss = nn.functional.mse_loss(x_recon, x)
            kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()
            return {"recon": x_recon, "z": z, "recon_loss": recon_loss, "kl_loss": kl_loss}
        else:
            z = self.encode(x)
            x_recon = self.decode(z)
            recon_loss = nn.functional.mse_loss(x_recon, x)
            return {"recon": x_recon, "z": z, "recon_loss": recon_loss}
