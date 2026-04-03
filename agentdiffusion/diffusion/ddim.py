"""DDIM sampler for accelerated inference."""

from __future__ import annotations

import torch
import torch.nn as nn

from .scheduler import NoiseScheduler


class DDIMSampler:
    """Deterministic DDIM sampling with optional eta (stochastic when eta > 0)."""

    def __init__(
        self,
        model: nn.Module,
        scheduler: NoiseScheduler,
        prediction_type: str = "v_prediction",
        ddim_steps: int = 50,
        eta: float = 0.0,
    ):
        self.model = model
        self.scheduler = scheduler
        self.prediction_type = prediction_type
        self.ddim_steps = ddim_steps
        self.eta = eta

        # Build sub-sequence of timesteps
        total_T = scheduler.timesteps
        step_size = total_T // ddim_steps
        self.timestep_seq = list(range(0, total_T, step_size))[:ddim_steps]
        self.timestep_seq.reverse()  # descending: T-1 ... 0

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],                    # (B, H, W, d_agent)
        market_cond: torch.Tensor | None = None,
        scenario: torch.Tensor | None = None,
        device: torch.device | None = None,
        guidance_fn: callable | None = None,
        guidance_kwargs: dict | None = None,
    ) -> torch.Tensor:
        """Run full DDIM denoising loop.

        Args:
            guidance_fn: optional constraint guidance function(x_t, t, **kwargs) -> x_t_guided
        """
        if device is None:
            device = next(self.model.parameters()).device

        x = torch.randn(shape, device=device)
        seq = self.timestep_seq

        for i, t_cur in enumerate(seq):
            t_tensor = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)

            # Model prediction
            pred = self.model(x, t_tensor, market_cond, scenario)

            # Recover x_0 prediction
            if self.prediction_type == "v_prediction":
                x0_pred = self.scheduler.predict_x0_from_v(x, t_tensor, pred)
                eps_pred = self.scheduler.predict_noise_from_v(x, t_tensor, pred)
            elif self.prediction_type == "epsilon":
                eps_pred = pred
                sqrt_alpha = self.scheduler._extract(
                    self.scheduler.sqrt_alphas_cumprod, t_tensor, x.shape
                )
                sqrt_one_minus = self.scheduler._extract(
                    self.scheduler.sqrt_one_minus_alphas_cumprod, t_tensor, x.shape
                )
                x0_pred = (x - sqrt_one_minus * eps_pred) / sqrt_alpha
            else:  # x0
                x0_pred = pred
                sqrt_alpha = self.scheduler._extract(
                    self.scheduler.sqrt_alphas_cumprod, t_tensor, x.shape
                )
                sqrt_one_minus = self.scheduler._extract(
                    self.scheduler.sqrt_one_minus_alphas_cumprod, t_tensor, x.shape
                )
                eps_pred = (x - sqrt_alpha * x0_pred) / sqrt_one_minus

            # DDIM step
            if i < len(seq) - 1:
                t_next = seq[i + 1]
                t_next_tensor = torch.full((shape[0],), t_next, device=device, dtype=torch.long)
                alpha_cur = self.scheduler._extract(
                    self.scheduler.alphas_cumprod, t_tensor, x.shape
                )
                alpha_next = self.scheduler._extract(
                    self.scheduler.alphas_cumprod, t_next_tensor, x.shape
                )
            else:
                alpha_cur = self.scheduler._extract(
                    self.scheduler.alphas_cumprod, t_tensor, x.shape
                )
                alpha_next = torch.ones_like(alpha_cur)

            sigma = self.eta * ((1 - alpha_next) / (1 - alpha_cur) * (1 - alpha_cur / alpha_next)).sqrt()

            x = (
                alpha_next.sqrt() * x0_pred
                + (1 - alpha_next - sigma ** 2).clamp(min=0).sqrt() * eps_pred
            )
            if self.eta > 0:
                x = x + sigma * torch.randn_like(x)

            # Apply constraint guidance
            if guidance_fn is not None:
                progress = 1.0 - i / len(seq)  # 1 at start, 0 at end
                x = guidance_fn(x, t_cur, progress=progress, **(guidance_kwargs or {}))

        return x

    @torch.no_grad()
    def sample_trajectory(
        self,
        initial_state: torch.Tensor,               # [B, H, W, d_agent]
        num_steps: int,
        market_conds: torch.Tensor | None = None,  # [num_steps, B, market_cond_dim]
        guidance_fn: callable | None = None,
        guidance_kwargs: dict | None = None,
    ) -> list[torch.Tensor]:
        """Autoregressively generate a trajectory of states."""
        states = [initial_state]
        device = initial_state.device
        shape = initial_state.shape

        for k in range(num_steps):
            mc = market_conds[k] if market_conds is not None else None
            next_state = self.sample(
                shape, mc, device=device,
                guidance_fn=guidance_fn, guidance_kwargs=guidance_kwargs,
            )
            states.append(next_state)

        return states
