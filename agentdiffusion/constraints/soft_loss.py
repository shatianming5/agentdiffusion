"""Training-time soft constraint losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data.agent_state import STATE_SLICES


def market_clearing_loss(
    state_t: torch.Tensor,      # [B, H, W, C]  current state
    state_t1: torch.Tensor,     # [B, H, W, C]  predicted next state
) -> torch.Tensor:
    """Sum of position changes across all agents should be zero (market clearing).

    L_clearing = ‖Σ_i (position_t1_i - position_t_i)‖²
    """
    pos_slice = STATE_SLICES["positions"]
    delta_positions = state_t1[..., pos_slice] - state_t[..., pos_slice]
    # Sum across agents (H, W dims), per asset (last dim)
    net_delta = delta_positions.sum(dim=(1, 2))  # [B, num_assets]
    return (net_delta ** 2).sum(dim=-1).mean()


def budget_constraint_loss(
    state: torch.Tensor,        # [B, H, W, C]
) -> torch.Tensor:
    """Budget (cash) should remain non-negative.

    L_budget = Σ_i max(0, -cash_i)²
    """
    funds_slice = STATE_SLICES["funds"]
    cash = state[..., funds_slice.start]  # first element of funds = cash
    violations = F.relu(-cash)
    return (violations ** 2).mean()


def conservation_loss(
    state: torch.Tensor,        # [B, H, W, C]
    target_totals: torch.Tensor | None = None,  # [B, num_assets]
) -> torch.Tensor:
    """Total holdings per asset should be conserved.

    L_conservation = ‖Σ_i position_i - S_target‖²
    """
    pos_slice = STATE_SLICES["positions"]
    positions = state[..., pos_slice]
    total = positions.sum(dim=(1, 2))  # [B, num_assets]
    if target_totals is None:
        return torch.tensor(0.0, device=state.device)
    return ((total - target_totals) ** 2).sum(dim=-1).mean()


class ConstraintLoss(torch.nn.Module):
    """Combined soft constraint loss for training."""

    def __init__(
        self,
        lambda_clearing: float = 1.0,
        lambda_budget: float = 0.5,
        lambda_conservation: float = 1.0,
    ):
        super().__init__()
        self.lambda_clearing = lambda_clearing
        self.lambda_budget = lambda_budget
        self.lambda_conservation = lambda_conservation

    def forward(
        self,
        state_t: torch.Tensor,
        state_t1_pred: torch.Tensor,
        target_totals: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        l_clear = market_clearing_loss(state_t, state_t1_pred)
        l_budget = budget_constraint_loss(state_t1_pred)
        l_conserve = conservation_loss(state_t1_pred, target_totals)

        total = (
            self.lambda_clearing * l_clear
            + self.lambda_budget * l_budget
            + self.lambda_conservation * l_conserve
        )

        return {
            "constraint_total": total,
            "clearing_loss": l_clear,
            "budget_loss": l_budget,
            "conservation_loss": l_conserve,
        }
