"""DDPM training logic with v-prediction support."""

from __future__ import annotations

import torch
import torch.nn as nn

from .scheduler import NoiseScheduler


class DDPMTrainer:
    """Handles the forward diffusion + loss computation for training."""

    def __init__(
        self,
        model: nn.Module,
        scheduler: NoiseScheduler,
        prediction_type: str = "v_prediction",
    ):
        self.model = model
        self.scheduler = scheduler
        self.prediction_type = prediction_type

    def compute_loss(
        self,
        z0: torch.Tensor,                          # [B, H, W, d_agent]  clean latent
        market_cond: torch.Tensor | None = None,
        scenario: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Single training step: sample t, add noise, predict, compute loss."""
        B = z0.shape[0]
        device = z0.device

        # Sample random timesteps
        t = torch.randint(0, self.scheduler.timesteps, (B,), device=device)

        # Sample noise
        noise = torch.randn_like(z0)

        # Forward diffusion
        z_t = self.scheduler.q_sample(z0, t, noise)

        # Model prediction
        pred = self.model(z_t, t, market_cond, scenario)

        # Compute target and loss
        if self.prediction_type == "v_prediction":
            target = self.scheduler.v_target(z0, noise, t)
        elif self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "x0":
            target = z0
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        loss = nn.functional.mse_loss(pred, target)

        return {
            "loss": loss,
            "pred": pred,
            "target": target,
            "z_t": z_t,
            "t": t,
            "noise": noise,
        }
