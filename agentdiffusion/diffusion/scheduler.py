"""Noise schedules for diffusion: cosine, linear."""

from __future__ import annotations

import math
import torch


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal 2021."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(max=0.999).float()


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


class NoiseScheduler:
    """Precomputes and caches all diffusion coefficients."""

    def __init__(self, timesteps: int = 1000, schedule: str = "cosine"):
        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.timesteps = timesteps
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas = alphas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])

        # Precompute useful quantities
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).sqrt()
        self.sqrt_recip_alphas_cumprod = (1.0 / self.alphas_cumprod).sqrt()
        self.sqrt_recip_alphas_cumprod_minus_one = (1.0 / self.alphas_cumprod - 1).sqrt()

        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device: torch.device) -> "NoiseScheduler":
        for attr in (
            "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "sqrt_recip_alphas_cumprod", "sqrt_recip_alphas_cumprod_minus_one",
            "posterior_variance",
        ):
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Index into tensor a at timestep t, broadcast to shape."""
        out = a.gather(0, t)
        return out.view(-1, *([1] * (len(shape) - 1)))

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward process: q(x_t | x_0) = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def predict_x0_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Recover x_0 from v-prediction: x_0 = sqrt(ᾱ_t)*x_t - sqrt(1-ᾱ_t)*v."""
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_alpha * x_t - sqrt_one_minus * v

    def predict_noise_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Recover ε from v-prediction: ε = sqrt(ᾱ_t)*v + sqrt(1-ᾱ_t)*x_t."""
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_alpha * v + sqrt_one_minus * x_t

    def v_target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute v-prediction target: v = sqrt(ᾱ_t)*ε - sqrt(1-ᾱ_t)*x_0."""
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alpha * noise - sqrt_one_minus * x0
