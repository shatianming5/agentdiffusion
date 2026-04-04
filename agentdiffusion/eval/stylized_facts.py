"""Stylized facts evaluation for generated financial time series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import stats


@dataclass
class StylizedFactsReport:
    fat_tail_alpha: float
    fat_tail_pass: bool
    volatility_clustering_acf: np.ndarray
    volatility_clustering_pass: bool
    leverage_effect_corr: float
    leverage_effect_pass: bool
    volume_volatility_corr: float
    volume_volatility_pass: bool
    return_autocorr: np.ndarray
    return_autocorr_pass: bool
    gain_loss_asymmetry_pvalue: float
    gain_loss_asymmetry_pass: bool

    @property
    def total_passed(self) -> int:
        return sum([
            self.fat_tail_pass,
            self.volatility_clustering_pass,
            self.leverage_effect_pass,
            self.volume_volatility_pass,
            self.return_autocorr_pass,
            self.gain_loss_asymmetry_pass,
        ])

    @property
    def summary(self) -> str:
        return f"{self.total_passed}/6 stylized facts passed"


def compute_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from price series."""
    return np.diff(np.log(prices + 1e-10))


def return_distribution_wasserstein(
    gen_returns: np.ndarray,
    real_returns: np.ndarray,
) -> float:
    """1-D Wasserstein (Earth Mover's) distance between generated and real return distributions.

    Lower is better -- measures how much "work" is needed to transform
    the generated return distribution into the real one.

    Args:
        gen_returns: Generated log-returns [T_gen].
        real_returns: Real log-returns [T_real].

    Returns:
        Wasserstein-1 distance (float).
    """
    return float(stats.wasserstein_distance(gen_returns, real_returns))


def hill_estimator(returns: np.ndarray, k: int | None = None) -> float:
    """Hill estimator for tail index α.

    A value of 2 < α < 5 indicates fat tails (leptokurtic).
    """
    abs_returns = np.abs(returns)
    abs_returns = abs_returns[abs_returns > 0]
    sorted_r = np.sort(abs_returns)[::-1]

    if k is None:
        k = max(int(len(sorted_r) * 0.05), 10)  # top 5%

    k = min(k, len(sorted_r) - 1)
    if k < 2:
        return float("nan")

    log_sorted = np.log(sorted_r[:k])
    threshold = np.log(sorted_r[k])
    alpha = 1.0 / (np.mean(log_sorted) - threshold)
    return alpha


def acf(x: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """Compute autocorrelation function."""
    x = np.asarray(x).flatten()
    n = len(x)
    x = x - x.mean()
    var = np.sum(x ** 2) / n
    if var < 1e-12:
        return np.zeros(max_lag)
    result = np.correlate(x, x, mode="full")
    result = result[n - 1:] / (var * n)
    return result[:max_lag]


def check_fat_tails(returns: np.ndarray) -> tuple[float, bool]:
    """Check if return distribution has fat tails (α ∈ [2, 5])."""
    alpha = hill_estimator(returns)
    passed = 2.0 < alpha < 5.0
    return alpha, passed


def check_volatility_clustering(returns: np.ndarray, lag_threshold: int = 100) -> tuple[np.ndarray, bool]:
    """Check if |returns| show slow ACF decay (volatility clustering)."""
    abs_returns = np.abs(returns)
    acf_vals = acf(abs_returns, max_lag=lag_threshold)
    # Pass if ACF at lag=100 is still > 0.05 (slow decay)
    passed = len(acf_vals) > lag_threshold - 1 and acf_vals[lag_threshold - 1] > 0.05
    return acf_vals, passed


def check_leverage_effect(returns: np.ndarray, max_k: int = 20) -> tuple[float, bool]:
    """Check negative correlation between returns and future volatility."""
    vol = np.abs(returns)
    if len(returns) < max_k + 10:
        return 0.0, False

    # Compute corr(r_t, |r_{t+k}|) for k=1..max_k
    corrs = []
    for k in range(1, max_k + 1):
        r = returns[:-k]
        v = vol[k:]
        if len(r) > 10:
            corrs.append(np.corrcoef(r, v)[0, 1])

    if not corrs:
        return 0.0, False

    avg_corr = np.mean(corrs[:5])  # focus on small k
    passed = avg_corr < -0.05  # should be negative
    return avg_corr, passed


def check_volume_volatility_correlation(
    volumes: np.ndarray, returns: np.ndarray
) -> tuple[float, bool]:
    """Check positive correlation between volume and |returns|."""
    if len(volumes) != len(returns):
        min_len = min(len(volumes), len(returns))
        volumes = volumes[:min_len]
        returns = returns[:min_len]

    corr = np.corrcoef(volumes, np.abs(returns))[0, 1]
    passed = corr > 0.3
    return corr, passed


def check_return_autocorrelation(returns: np.ndarray) -> tuple[np.ndarray, bool]:
    """Check that raw returns have near-zero autocorrelation."""
    acf_vals = acf(returns, max_lag=20)
    # Pass if |ACF| < 0.05 for lag > 1
    passed = all(abs(acf_vals[i]) < 0.05 for i in range(2, min(20, len(acf_vals))))
    return acf_vals, passed


def check_gain_loss_asymmetry(returns: np.ndarray) -> tuple[float, bool]:
    """Check asymmetry between gain and loss durations."""
    gains = []
    losses = []
    current_streak = 0
    for r in returns:
        if r > 0:
            if current_streak < 0:
                losses.append(-current_streak)
                current_streak = 1
            else:
                current_streak += 1
        elif r < 0:
            if current_streak > 0:
                gains.append(current_streak)
                current_streak = -1
            else:
                current_streak -= 1

    if len(gains) < 5 or len(losses) < 5:
        return 1.0, False

    _, p_value = stats.ks_2samp(gains, losses)
    passed = p_value < 0.05  # distributions should differ
    return p_value, passed


def evaluate_stylized_facts(
    prices: np.ndarray,
    volumes: np.ndarray | None = None,
) -> StylizedFactsReport:
    """Run all stylized fact checks on a price series."""
    returns = compute_returns(prices)

    alpha, fat_pass = check_fat_tails(returns)
    vol_acf, vol_pass = check_volatility_clustering(returns)
    lev_corr, lev_pass = check_leverage_effect(returns)
    ret_acf, ret_pass = check_return_autocorrelation(returns)
    gl_pval, gl_pass = check_gain_loss_asymmetry(returns)

    if volumes is not None:
        vv_corr, vv_pass = check_volume_volatility_correlation(volumes, returns)
    else:
        vv_corr, vv_pass = 0.0, False  # auto-FAIL when no volume data provided

    return StylizedFactsReport(
        fat_tail_alpha=alpha,
        fat_tail_pass=fat_pass,
        volatility_clustering_acf=vol_acf,
        volatility_clustering_pass=vol_pass,
        leverage_effect_corr=lev_corr,
        leverage_effect_pass=lev_pass,
        volume_volatility_corr=vv_corr,
        volume_volatility_pass=vv_pass,
        return_autocorr=ret_acf,
        return_autocorr_pass=ret_pass,
        gain_loss_asymmetry_pvalue=gl_pval,
        gain_loss_asymmetry_pass=gl_pass,
    )
