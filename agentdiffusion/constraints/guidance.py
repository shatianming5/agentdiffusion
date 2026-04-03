"""Inference-time constraint guidance for the diffusion denoising loop."""

from __future__ import annotations

import torch

from ..data.agent_state import STATE_SLICES


def constraint_guidance_fn(
    x_t: torch.Tensor,              # [B, H, W, d_agent] — latent (or decoded) state
    t: int,                          # current diffusion timestep
    *,
    progress: float = 0.0,          # 1.0 at t=T, 0.0 at t=0
    guidance_scale_clearing: float = 2.0,
    guidance_scale_budget: float = 1.0,
    guidance_start_ratio: float = 0.5,
    decoder=None,                    # optional: decode latent to raw for constraint eval
    prev_state: torch.Tensor | None = None,  # [B, H, W, C] previous raw state
) -> torch.Tensor:
    """Apply constraint gradients to guide denoising toward feasible region.

    Only active in the latter portion of denoising (when progress < guidance_start_ratio),
    where the signal is less noisy and gradients are more meaningful.
    """
    # Only guide in the latter half of denoising
    if progress > guidance_start_ratio:
        return x_t

    # Scale guidance strength: stronger as we approach t=0
    scale_factor = 1.0 - progress / guidance_start_ratio

    # If a decoder is provided, compute constraints in raw space
    # Otherwise assume x_t is already in interpretable space
    x_eval = x_t
    needs_grad = True

    if needs_grad:
        x_eval = x_t.detach().requires_grad_(True)

    # --- Market clearing gradient ---
    if prev_state is not None and decoder is not None:
        raw_pred = decoder(x_eval)
        pos_slice = STATE_SLICES["positions"]
        delta = raw_pred[..., pos_slice] - prev_state[..., pos_slice]
        clearing_violation = (delta.sum(dim=(1, 2)) ** 2).sum()
    else:
        # Simplified: operate directly on latent (assume first dims correspond to positions)
        pos_dims = min(x_eval.shape[-1], 8)  # first 8 latent dims ≈ position info
        delta = x_eval[..., :pos_dims].sum(dim=(1, 2))
        clearing_violation = (delta ** 2).sum()

    grad_clearing = torch.autograd.grad(clearing_violation, x_eval, retain_graph=True)[0]

    # --- Budget gradient ---
    if decoder is not None:
        raw_pred = decoder(x_eval)
        cash = raw_pred[..., STATE_SLICES["funds"].start]
        budget_violation = (torch.relu(-cash) ** 2).sum()
    else:
        budget_violation = torch.tensor(0.0, device=x_t.device)

    if budget_violation.requires_grad:
        grad_budget = torch.autograd.grad(budget_violation, x_eval)[0]
    else:
        grad_budget = torch.zeros_like(x_t)

    # Apply gradients
    guided = x_t.detach() - (
        guidance_scale_clearing * scale_factor * grad_clearing
        + guidance_scale_budget * scale_factor * grad_budget
    )

    return guided
