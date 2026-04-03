"""Single-step inference: generate S_{t+1} from S_t."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..models.autoencoder import AgentAutoencoder
from ..models.agent_dit import AgentDiT
from ..diffusion.scheduler import NoiseScheduler
from ..diffusion.ddim import DDIMSampler
from ..constraints.guidance import constraint_guidance_fn
from ..constraints.projection import apply_all_projections


class SingleStepInference:
    """Generate next agent state from current state using trained diffusion model."""

    def __init__(
        self,
        model: AgentDiT,
        ae: AgentAutoencoder,
        scheduler: NoiseScheduler,
        ddim_steps: int = 50,
        guidance_config: dict | None = None,
    ):
        self.model = model
        self.ae = ae
        self.scheduler = scheduler
        self.guidance_config = guidance_config or {}

        self.sampler = DDIMSampler(
            model=model,
            scheduler=scheduler,
            prediction_type="v_prediction",
            ddim_steps=ddim_steps,
        )

    @torch.no_grad()
    def generate(
        self,
        state_t: torch.Tensor,                    # [B, H, W, C]  raw
        market_cond: torch.Tensor | None = None,
        target_totals: torch.Tensor | None = None,
        use_guidance: bool = True,
        use_projection: bool = True,
    ) -> torch.Tensor:
        """Generate S_{t+1} given S_t.

        Returns: [B, H, W, C] raw state
        """
        device = state_t.device
        B, H, W, C = state_t.shape

        # Encode current state to get latent shape
        z_shape = (B, H, W, self.ae.latent_dim)

        # Build guidance function
        guidance_fn = None
        if use_guidance and self.guidance_config:
            def guidance_fn(x_t, t, progress=0.0):
                return constraint_guidance_fn(
                    x_t, t,
                    progress=progress,
                    decoder=self.ae.decoder,
                    prev_state=state_t,
                    **self.guidance_config,
                )

        # DDIM sampling in latent space
        z_pred = self.sampler.sample(
            shape=z_shape,
            market_cond=market_cond,
            device=device,
            guidance_fn=guidance_fn,
        )

        # Decode to raw space
        state_t1 = self.ae.decode(z_pred)

        # Apply hard constraint projections
        if use_projection:
            state_t1 = apply_all_projections(state_t, state_t1, target_totals)

        return state_t1
