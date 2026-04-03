"""Data preprocessing: normalization, grid layout, data generation pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from tqdm import tqdm

from .agent_state import AgentGrid, AgentType, normalize_states


def generate_synthetic_dataset(
    output_dir: str,
    num_simulations: int = 100,
    num_steps_per_sim: int = 50,
    num_agents: int = 10000,
    raw_dim: int = 128,
    market_cond_dim: int = 32,
    seed: int = 42,
):
    """Generate synthetic training data with structurally valid agent transitions.

    This is a stand-in for ABIDES data generation. Each simulation produces
    a sequence of agent states, and we save consecutive (S_t, S_{t+1}) pairs.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    grid = AgentGrid(num_agents)
    H, W = grid.grid_h, grid.grid_w
    gen = torch.Generator().manual_seed(seed)

    sample_idx = 0
    for sim in tqdm(range(num_simulations), desc="Generating data"):
        # Random agent types based on distribution
        type_counts = {
            AgentType.MARKET_MAKER: int(num_agents * 0.02),
            AgentType.TREND_FOLLOWER: int(num_agents * 0.30),
            AgentType.FUNDAMENTALIST: int(num_agents * 0.20),
        }
        type_counts[AgentType.NOISE_TRADER] = num_agents - sum(type_counts.values())

        types = torch.cat([
            torch.full((count,), int(atype))
            for atype, count in type_counts.items()
        ])

        # Initial state: structured random
        state = torch.randn(num_agents, raw_dim, generator=gen)
        # Make positions positive (holdings)
        state[:, :32] = state[:, :32].abs() * 10
        # Cash should be positive
        state[:, 32] = state[:, 32].abs() * 1000

        capitals = state[:, 32].clone()  # sort by initial cash

        # Arrange into grid
        grid_state, grid_types, sort_idx = grid.arrange(state, types, capitals)

        market_cond = torch.randn(market_cond_dim, generator=gen)

        for step in range(num_steps_per_sim):
            # Simple dynamics: mean-reverting + noise
            delta = torch.randn(H, W, raw_dim, generator=gen) * 0.05
            # Trend followers amplify recent moves
            tf_mask = (grid_types == int(AgentType.TREND_FOLLOWER)).unsqueeze(-1).float()
            delta = delta + tf_mask * delta * 0.5

            # Market makers dampen
            mm_mask = (grid_types == int(AgentType.MARKET_MAKER)).unsqueeze(-1).float()
            delta = delta - mm_mask * grid_state * 0.01

            grid_state_next = grid_state + delta

            # Enforce basic positivity on cash
            grid_state_next[..., 32] = grid_state_next[..., 32].clamp(min=0)

            # Market condition evolves
            market_cond = market_cond + torch.randn_like(market_cond, generator=gen) * 0.1

            # Save pair
            torch.save({
                "state_t": grid_state.clone(),
                "state_t1": grid_state_next.clone(),
                "market_cond": market_cond.clone(),
                "agent_types": grid_types.clone(),
            }, out / f"sample_{sample_idx:06d}.pt")

            grid_state = grid_state_next
            sample_idx += 1

    print(f"Generated {sample_idx} samples in {output_dir}")
    return sample_idx
