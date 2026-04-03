"""ABM fidelity metrics: compare generated data against ground-truth ABM simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from .stylized_facts import acf


@dataclass
class FidelityReport:
    wasserstein_positions: float
    wasserstein_cash: float
    wasserstein_returns: float
    acf_l2_returns: float
    acf_l2_volatility: float
    mse_1step: float
    mse_10step: float
    mse_100step: float
    clearing_violation: float
    budget_violation: float
    conservation_violation: float


def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    """1D Wasserstein distance between two distributions."""
    return wasserstein_distance(x.flatten(), y.flatten())


def acf_l2_distance(x: np.ndarray, y: np.ndarray, max_lag: int = 100) -> float:
    """L2 distance between autocorrelation functions."""
    acf_x = acf(x, max_lag)
    acf_y = acf(y, max_lag)
    return float(np.sqrt(np.mean((acf_x - acf_y) ** 2)))


def compute_fidelity(
    gen_states: torch.Tensor,        # [T, B, H, W, C] generated trajectory
    abm_states: torch.Tensor,        # [T, B, H, W, C] ground-truth trajectory
) -> FidelityReport:
    """Compute comprehensive fidelity metrics."""
    gen_np = gen_states.cpu().numpy()
    abm_np = abm_states.cpu().numpy()

    # Position distribution (dim 0:32)
    w_pos = wasserstein_1d(gen_np[..., :32], abm_np[..., :32])

    # Cash distribution (dim 32)
    w_cash = wasserstein_1d(gen_np[..., 32], abm_np[..., 32])

    # Returns from aggregated price proxy
    gen_price = gen_np[..., :32].mean(axis=(1, 2, 3))  # [T]
    abm_price = abm_np[..., :32].mean(axis=(1, 2, 3))
    gen_returns = np.diff(gen_price)
    abm_returns = np.diff(abm_price)
    min_len = min(len(gen_returns), len(abm_returns))
    w_returns = wasserstein_1d(gen_returns[:min_len], abm_returns[:min_len])

    # ACF distances
    acf_ret = acf_l2_distance(gen_returns[:min_len], abm_returns[:min_len])
    gen_vol = np.abs(gen_returns[:min_len])
    abm_vol = np.abs(abm_returns[:min_len])
    acf_vol = acf_l2_distance(gen_vol, abm_vol)

    # Multi-step MSE
    T = min(gen_states.shape[0], abm_states.shape[0])
    mse_1 = (gen_states[1] - abm_states[1]).pow(2).mean().item() if T > 1 else 0
    mse_10 = (gen_states[min(10, T-1)] - abm_states[min(10, T-1)]).pow(2).mean().item() if T > 10 else 0
    mse_100 = (gen_states[min(100, T-1)] - abm_states[min(100, T-1)]).pow(2).mean().item() if T > 100 else 0

    # Constraint violations
    # Clearing: sum of position deltas
    if T > 1:
        delta_pos = gen_states[1:, ..., :32] - gen_states[:-1, ..., :32]
        clearing = delta_pos.sum(dim=(2, 3)).abs().mean().item()
    else:
        clearing = 0.0

    # Budget: negative cash
    budget = torch.relu(-gen_states[..., 32]).mean().item()

    # Conservation: total positions should be constant
    total_pos = gen_states[..., :32].sum(dim=(2, 3))  # [T, B, 32]
    conservation = (total_pos[1:] - total_pos[:-1]).abs().mean().item() if T > 1 else 0

    return FidelityReport(
        wasserstein_positions=w_pos,
        wasserstein_cash=w_cash,
        wasserstein_returns=w_returns,
        acf_l2_returns=acf_ret,
        acf_l2_volatility=acf_vol,
        mse_1step=mse_1,
        mse_10step=mse_10,
        mse_100step=mse_100,
        clearing_violation=clearing,
        budget_violation=budget,
        conservation_violation=conservation,
    )
