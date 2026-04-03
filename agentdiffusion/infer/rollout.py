"""Multi-step rollout: generate full simulation trajectory."""

from __future__ import annotations

import torch
from tqdm import tqdm

from .single_step import SingleStepInference
from ..constraints.projection import apply_all_projections


class TrajectoryRollout:
    """Autoregressively generate a sequence of agent states."""

    def __init__(
        self,
        inference: SingleStepInference,
        full_check_interval: int = 10,
    ):
        self.inference = inference
        self.full_check_interval = full_check_interval

    @torch.no_grad()
    def rollout(
        self,
        initial_state: torch.Tensor,               # [B, H, W, C]
        num_steps: int,
        market_conds: torch.Tensor | None = None,   # [num_steps, B, market_cond_dim] or None
        target_totals: torch.Tensor | None = None,
        use_guidance: bool = True,
    ) -> torch.Tensor:
        """Generate full trajectory.

        Returns: [num_steps+1, B, H, W, C]
        """
        states = [initial_state]

        for k in tqdm(range(num_steps), desc="Rollout"):
            mc = None
            if market_conds is not None:
                mc = market_conds[k]

            next_state = self.inference.generate(
                state_t=states[-1],
                market_cond=mc,
                target_totals=target_totals,
                use_guidance=use_guidance,
            )

            # Periodic full constraint check
            if (k + 1) % self.full_check_interval == 0 and target_totals is not None:
                next_state = apply_all_projections(
                    states[-1], next_state, target_totals
                )

            states.append(next_state)

        return torch.stack(states, dim=0)

    @torch.no_grad()
    def rollout_batch_scenarios(
        self,
        initial_states: list[torch.Tensor],         # list of [1, H, W, C]
        num_steps: int,
        market_conds_list: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        """Run multiple scenario rollouts (different initial conditions)."""
        results = []
        for i, init_state in enumerate(initial_states):
            mc = market_conds_list[i] if market_conds_list else None
            trajectory = self.rollout(init_state, num_steps, mc)
            results.append(trajectory)
        return results
