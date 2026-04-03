"""Post-processing projections for hard constraint satisfaction."""

from __future__ import annotations

import torch

from ..data.agent_state import STATE_SLICES


def project_market_clearing(
    state_t: torch.Tensor,       # [B, H, W, C]  previous state
    state_t1: torch.Tensor,      # [B, H, W, C]  predicted next state (mutated in-place)
) -> torch.Tensor:
    """Ensure net position change across all agents is exactly zero per asset.

    Adjustment: Δposition_i → Δposition_i - mean(Δposition)
    """
    pos = STATE_SLICES["positions"]
    delta = state_t1[..., pos] - state_t[..., pos]
    # mean over agent dimensions (H, W)
    mean_delta = delta.mean(dim=(1, 2), keepdim=True)  # [B, 1, 1, num_assets]
    state_t1 = state_t1.clone()
    state_t1[..., pos] = state_t[..., pos] + (delta - mean_delta)
    return state_t1


def project_conservation(
    state: torch.Tensor,         # [B, H, W, C]
    target_totals: torch.Tensor, # [B, num_assets]
) -> torch.Tensor:
    """Ensure total holdings per asset equal target."""
    pos = STATE_SLICES["positions"]
    positions = state[..., pos]
    current_total = positions.sum(dim=(1, 2))  # [B, num_assets]
    N = positions.shape[1] * positions.shape[2]
    correction = ((target_totals - current_total) / N).unsqueeze(1).unsqueeze(1)
    state = state.clone()
    state[..., pos] = positions + correction
    return state


def project_budget(state: torch.Tensor) -> torch.Tensor:
    """Clamp cash to be non-negative."""
    state = state.clone()
    cash_idx = STATE_SLICES["funds"].start
    state[..., cash_idx] = state[..., cash_idx].clamp(min=0)
    return state


def project_leverage(state: torch.Tensor, max_leverage: float = 10.0) -> torch.Tensor:
    """Clamp leverage to be within limit."""
    state = state.clone()
    # Leverage is typically the second element of funds block
    lev_idx = STATE_SLICES["funds"].start + 1
    state[..., lev_idx] = state[..., lev_idx].clamp(max=max_leverage)
    return state


def apply_all_projections(
    state_t: torch.Tensor,
    state_t1: torch.Tensor,
    target_totals: torch.Tensor | None = None,
    max_leverage: float = 10.0,
) -> torch.Tensor:
    """Apply all hard constraint projections in sequence."""
    out = project_market_clearing(state_t, state_t1)
    out = project_budget(out)
    out = project_leverage(out, max_leverage)
    if target_totals is not None:
        out = project_conservation(out, target_totals)
    return out
