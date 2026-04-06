"""Compare TRADES-LOB vs Video DiT on same metrics."""
import numpy as np
import pandas as pd
from agentdiffusion.eval.stylized_facts import evaluate_stylized_facts, compute_returns, acf
from scipy.stats import kurtosis as scipy_kurtosis

print("=" * 64)
print("  TRADES-LOB vs Video DiT: Same-metric Comparison")
print("=" * 64)

# TRADES-LOB synthetic data
trades_files = [
    "vendor/DeepMarket/data/TRADES-LOB/TSLA_2015-01-29.csv",
    "vendor/DeepMarket/data/TRADES-LOB/TSLA_2015-01-30.csv",
]
trades_prices = []
for f in trades_files:
    df = pd.read_csv(f)
    mid = df["MID_PRICE"].values
    mid = mid[mid > 0]
    trades_prices.append(mid)
trades_all = np.concatenate(trades_prices)
trades_ret = compute_returns(trades_all)

# Real LOBSTER (use pandas for speed instead of np.loadtxt)
ob = pd.read_csv(
    "data/external/lobster/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv",
    header=None,
).values
real_mid = (ob[:, 0] + ob[:, 2]) / 2.0
real_ret = compute_returns(real_mid[real_mid > 0])

# Compute metrics
def get_metrics(ret):
    kurt = scipy_kurtosis(ret, fisher=False)
    ac = acf(np.abs(ret), 10)
    ac1 = ac[1] if len(ac) > 1 else 0
    ac5 = ac[5] if len(ac) > 5 else 0
    return kurt, ac1, ac5, ret.std() * 1000

r_k, r_a1, r_a5, r_s = get_metrics(real_ret)
t_k, t_a1, t_a5, t_s = get_metrics(trades_ret)
o_k, o_a1, o_a5, o_s = 7.46, 0.1397, 0.0062, 98.52  # Our results from earlier

fmt = "{:<25s} {:<15s} {:<15s} {:<15s}"
print(fmt.format("Metric", "Real (AMZN)", "TRADES (TSLA)", "Video DiT"))
print("-" * 70)
print("{:<25s} {:<15.2f} {:<15.2f} {:<15.2f}".format("Kurtosis", r_k, t_k, o_k))
print("{:<25s} {:<15.4f} {:<15.4f} {:<15.4f}".format("|Ret| ACF(1)", r_a1, t_a1, o_a1))
print("{:<25s} {:<15.4f} {:<15.4f} {:<15.4f}".format("|Ret| ACF(5)", r_a5, t_a5, o_a5))
print("{:<25s} {:<15.2f} {:<15.2f} {:<15.2f}".format("Ret std (x1000)", r_s, t_s, o_s))
print()

# TRADES stylized facts
trades_results = [evaluate_stylized_facts(p) for p in trades_prices]
trades_avg = np.mean([r.total_passed for r in trades_results])
print("TRADES-LOB Stylized Facts: {:.1f}/6".format(trades_avg))
for attr, name in [("fat_tail_pass", "fat_tail"), ("volatility_clustering_pass", "vol_cluster"),
                   ("leverage_effect_pass", "leverage"), ("return_autocorr_pass", "ret_autocorr"),
                   ("gain_loss_asymmetry_pass", "gl_asymm")]:
    val = np.mean([getattr(r, attr) for r in trades_results]) * 100
    print("  {}: {:.0f}%".format(name, val))

print()
print("Video DiT Stylized Facts: 0.2/6 (from A-Share eval)")
print("=" * 64)
