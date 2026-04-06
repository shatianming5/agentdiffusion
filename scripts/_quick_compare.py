"""Quick TRADES vs Video DiT comparison (truncated data for speed)."""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis

def quick_metrics(mid, label):
    mid = mid[mid > 0][:10000]
    ret = np.diff(np.log(mid + 1e-10))
    kurt = scipy_kurtosis(ret, fisher=False)
    abs_ret = np.abs(ret)
    x = abs_ret - abs_ret.mean()
    n = len(x)
    ac1 = np.sum(x[:-1] * x[1:]) / (np.sum(x**2) + 1e-12)
    ac5 = np.sum(x[:-5] * x[5:]) / (np.sum(x**2) + 1e-12) if n > 5 else 0
    std = ret.std() * 1000
    print("  {}: N={}, kurt={:.2f}, ACF1={:.4f}, ACF5={:.4f}, std={:.2f}".format(
        label, len(mid), kurt, ac1, ac5, std))
    return kurt, ac1, ac5, std

print("=" * 64)
print("  TRADES-LOB vs Video DiT: Quick Comparison")
print("=" * 64)

# TRADES-LOB
print("\nTRADES-LOB generated data:")
trades_metrics = []
for fname in ["vendor/DeepMarket/data/TRADES-LOB/TSLA_2015-01-29.csv",
              "vendor/DeepMarket/data/TRADES-LOB/TSLA_2015-01-30.csv"]:
    df = pd.read_csv(fname)
    m = quick_metrics(df["MID_PRICE"].values, fname.split("/")[-1])
    trades_metrics.append(m)

t_k = np.mean([m[0] for m in trades_metrics])
t_a1 = np.mean([m[1] for m in trades_metrics])
t_a5 = np.mean([m[2] for m in trades_metrics])
t_s = np.mean([m[3] for m in trades_metrics])

# Real LOBSTER
print("\nReal LOBSTER:")
ob = pd.read_csv(
    "data/external/lobster/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv",
    header=None).values
real_mid = (ob[:, 0] + ob[:, 2]) / 2.0
r_k, r_a1, r_a5, r_s = quick_metrics(real_mid, "AMZN-real")

# Our results
o_k, o_a1, o_a5, o_s = 7.46, 0.1397, 0.0062, 98.52

# Table
print("\n" + "=" * 70)
fmt = "{:<20s} {:<15s} {:<15s} {:<15s}"
print(fmt.format("Metric", "Real (AMZN)", "TRADES (TSLA)", "Video DiT"))
print("-" * 70)
print("{:<20s} {:<15.2f} {:<15.2f} {:<15.2f}".format("Kurtosis", r_k, t_k, o_k))
print("{:<20s} {:<15.4f} {:<15.4f} {:<15.4f}".format("|Ret| ACF(1)", r_a1, t_a1, o_a1))
print("{:<20s} {:<15.4f} {:<15.4f} {:<15.4f}".format("|Ret| ACF(5)", r_a5, t_a5, o_a5))
print("{:<20s} {:<15.2f} {:<15.2f} {:<15.2f}".format("Ret std (x1000)", r_s, t_s, o_s))
print("=" * 70)
