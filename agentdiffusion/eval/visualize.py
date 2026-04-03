"""Visualization utilities for evaluation results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .stylized_facts import StylizedFactsReport, acf, compute_returns
from .fidelity import FidelityReport
from .emergence import EmergenceEvent


def plot_stylized_facts(
    report: StylizedFactsReport,
    prices: np.ndarray,
    save_dir: str,
    prefix: str = "sf",
):
    """Generate stylized facts diagnostic plots."""
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    returns = compute_returns(prices)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Return distribution (fat tails)
    ax = axes[0, 0]
    ax.hist(returns, bins=100, density=True, alpha=0.7, label="Generated")
    x = np.linspace(returns.min(), returns.max(), 200)
    ax.plot(x, np.exp(-x**2 / (2 * returns.var())) / np.sqrt(2 * np.pi * returns.var()),
            "r--", label="Gaussian")
    ax.set_title(f"Return Distribution (α={report.fat_tail_alpha:.2f})")
    ax.set_yscale("log")
    ax.legend()

    # 2. Volatility clustering ACF
    ax = axes[0, 1]
    ax.bar(range(len(report.volatility_clustering_acf)),
           report.volatility_clustering_acf, alpha=0.7)
    ax.axhline(y=0.05, color="r", linestyle="--", label="Threshold")
    ax.set_title("ACF of |Returns| (Volatility Clustering)")
    ax.set_xlabel("Lag")
    ax.legend()

    # 3. Leverage effect
    ax = axes[0, 2]
    max_k = 20
    corrs = []
    vol = np.abs(returns)
    for k in range(1, max_k + 1):
        if len(returns) > k + 10:
            corrs.append(np.corrcoef(returns[:-k], vol[k:])[0, 1])
    ax.bar(range(1, len(corrs) + 1), corrs, alpha=0.7)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.set_title(f"Leverage Effect (avg={report.leverage_effect_corr:.3f})")
    ax.set_xlabel("Lag k")

    # 4. Return autocorrelation
    ax = axes[1, 0]
    ax.bar(range(len(report.return_autocorr)), report.return_autocorr, alpha=0.7)
    ax.axhline(y=0.05, color="r", linestyle="--")
    ax.axhline(y=-0.05, color="r", linestyle="--")
    ax.set_title("ACF of Returns")
    ax.set_xlabel("Lag")

    # 5. QQ plot
    ax = axes[1, 1]
    from scipy.stats import probplot
    probplot(returns, dist="norm", plot=ax)
    ax.set_title("QQ Plot vs Normal")

    # 6. Summary
    ax = axes[1, 2]
    ax.axis("off")
    summary_text = (
        f"Fat Tails (α={report.fat_tail_alpha:.2f}): {'PASS' if report.fat_tail_pass else 'FAIL'}\n"
        f"Vol Clustering: {'PASS' if report.volatility_clustering_pass else 'FAIL'}\n"
        f"Leverage Effect ({report.leverage_effect_corr:.3f}): {'PASS' if report.leverage_effect_pass else 'FAIL'}\n"
        f"Vol-Volume Corr ({report.volume_volatility_corr:.3f}): {'PASS' if report.volume_volatility_pass else 'FAIL'}\n"
        f"Return ACF: {'PASS' if report.return_autocorr_pass else 'FAIL'}\n"
        f"Gain/Loss Asym (p={report.gain_loss_asymmetry_pvalue:.3f}): {'PASS' if report.gain_loss_asymmetry_pass else 'FAIL'}\n"
        f"\nTotal: {report.total_passed}/6"
    )
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment="center", fontfamily="monospace")

    plt.tight_layout()
    plt.savefig(out / f"{prefix}_stylized_facts.png", dpi=150)
    plt.close()


def plot_fidelity_comparison(
    report: FidelityReport,
    gen_prices: np.ndarray,
    abm_prices: np.ndarray,
    save_dir: str,
    prefix: str = "fidelity",
):
    """Plot generated vs ABM ground-truth comparison."""
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Price series overlay
    ax = axes[0, 0]
    ax.plot(gen_prices, label="Generated", alpha=0.8)
    ax.plot(abm_prices, label="ABM Ground Truth", alpha=0.8)
    ax.set_title("Price Series")
    ax.legend()

    # 2. Return distributions
    ax = axes[0, 1]
    gen_ret = compute_returns(gen_prices)
    abm_ret = compute_returns(abm_prices)
    ax.hist(gen_ret, bins=80, density=True, alpha=0.5, label="Generated")
    ax.hist(abm_ret, bins=80, density=True, alpha=0.5, label="ABM")
    ax.set_title(f"Return Dist (W-dist={report.wasserstein_returns:.4f})")
    ax.legend()

    # 3. ACF comparison
    ax = axes[1, 0]
    acf_gen = acf(np.abs(gen_ret), 50)
    acf_abm = acf(np.abs(abm_ret), 50)
    ax.plot(acf_gen, label="Generated", alpha=0.8)
    ax.plot(acf_abm, label="ABM", alpha=0.8)
    ax.set_title(f"Volatility ACF (L2={report.acf_l2_volatility:.4f})")
    ax.legend()

    # 4. Multi-step MSE
    ax = axes[1, 1]
    steps = [1, 10, 100]
    mses = [report.mse_1step, report.mse_10step, report.mse_100step]
    ax.bar(range(len(steps)), mses)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([f"Step {s}" for s in steps])
    ax.set_title("Multi-step Prediction MSE")
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(out / f"{prefix}_fidelity.png", dpi=150)
    plt.close()


def plot_emergence_events(
    prices: np.ndarray,
    events: dict[str, list[EmergenceEvent]],
    save_dir: str,
    prefix: str = "emergence",
):
    """Plot price series with annotated emergence events."""
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(prices, "k-", linewidth=0.5, alpha=0.8)

    colors = {
        "flash_crashes": "red",
        "bubbles": "orange",
        "liquidity_crises": "purple",
        "herding": "blue",
    }

    for etype, evts in events.items():
        color = colors.get(etype, "gray")
        for e in evts:
            ax.axvspan(e.start_step, e.end_step, alpha=0.2, color=color, label=etype)

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left")
    ax.set_title("Price Series with Emergent Events")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Price")

    plt.tight_layout()
    plt.savefig(out / f"{prefix}_emergence.png", dpi=150)
    plt.close()
