"""Run LOB-Bench evaluation using our model output vs real LOBSTER data.

Instead of using Simple_Loader (complex filename convention), we directly
use lob_bench's metric functions on our data.
"""
import sys
sys.path.insert(0, "vendor/lob_bench")

import numpy as np
import pandas as pd
from pathlib import Path

# Load real LOBSTER data
print("=" * 64)
print("  LOB-Bench Evaluation")
print("=" * 64)

print("\nLoading real LOBSTER data...")
msg_df = pd.read_csv(
    "data/external/lobster/AMZN_2012-06-21_34200000_57600000_message_10.csv",
    header=None,
    names=["Time", "Type", "OrderID", "Size", "Price", "Direction"],
)
book_df = pd.read_csv(
    "data/external/lobster/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv",
    header=None,
)
print("  Messages: {}  Book snapshots: {}".format(len(msg_df), len(book_df)))

# Import lob_bench metrics
try:
    from eval import (
        spread_from_book,
        compute_imbalance,
    )
    HAS_EVAL = True
except ImportError:
    HAS_EVAL = False
    print("  [WARN] Could not import eval functions, using manual metrics")

try:
    from metrics import wasserstein_distance, l1_distance
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False

# Compute real LOB metrics
print("\n--- Real LOBSTER (AMZN) Metrics ---")

# Spread
ask_p1 = book_df.iloc[:, 0].values.astype(float)
bid_p1 = book_df.iloc[:, 2].values.astype(float)
mid = (ask_p1 + bid_p1) / 2.0
spread = (ask_p1 - bid_p1)
spread_bps = spread / (mid + 1e-8) * 10000

print("  Spread (bps):   mean={:.2f}, std={:.2f}, median={:.2f}".format(
    spread_bps.mean(), spread_bps.std(), np.median(spread_bps)))

# Imbalance
ask_v1 = book_df.iloc[:, 1].values.astype(float)
bid_v1 = book_df.iloc[:, 3].values.astype(float)
imbalance = (bid_v1 - ask_v1) / (bid_v1 + ask_v1 + 1e-8)
print("  Imbalance:      mean={:.4f}, std={:.4f}".format(
    imbalance.mean(), imbalance.std()))

# Interarrival times
times = msg_df["Time"].values
interarrival = np.diff(times)
interarrival = interarrival[interarrival > 0]
print("  Interarrival:   mean={:.6f}s, std={:.6f}s".format(
    interarrival.mean(), interarrival.std()))

# Returns
returns = np.diff(np.log(mid[mid > 0] + 1e-10))
print("  Returns:        mean={:.6f}, std={:.6f}, kurtosis={:.2f}".format(
    returns.mean(), returns.std(),
    float(pd.Series(returns).kurtosis() + 3)))

# Volume
volumes = msg_df["Size"].values.astype(float)
print("  Order size:     mean={:.1f}, median={:.1f}".format(
    volumes.mean(), np.median(volumes)))

# Message type distribution
type_counts = msg_df["Type"].value_counts().sort_index()
print("  Message types:  {}".format(dict(type_counts)))

# Now load TRADES-LOB for comparison
print("\n--- TRADES-LOB (TSLA) Metrics ---")
trades_df = pd.read_csv("vendor/DeepMarket/data/TRADES-LOB/TSLA_2015-01-29.csv")
t_mid = trades_df["MID_PRICE"].values
t_spread = trades_df["SPREAD"].values
t_imbalance = trades_df["ORDER_VOLUME_IMBALANCE"].values
t_returns = np.diff(np.log(t_mid[t_mid > 0] + 1e-10))

print("  Spread:         mean={:.2f}, std={:.2f}".format(t_spread.mean(), t_spread.std()))
print("  Imbalance:      mean={:.4f}, std={:.4f}".format(t_imbalance.mean(), t_imbalance.std()))
print("  Returns:        mean={:.6f}, std={:.6f}, kurtosis={:.2f}".format(
    t_returns.mean(), t_returns.std(),
    float(pd.Series(t_returns).kurtosis() + 3)))

# Summary comparison table
print("\n" + "=" * 64)
print("  LOB-Bench Summary")
print("=" * 64)
fmt = "{:<25s} {:<20s} {:<20s}"
print(fmt.format("Metric", "Real (AMZN)", "TRADES-LOB (TSLA)"))
print("-" * 65)
print("{:<25s} {:<20.2f} {:<20.2f}".format("Spread mean (bps)", spread_bps.mean(), t_spread.mean()))
print("{:<25s} {:<20.4f} {:<20.4f}".format("Imbalance mean", imbalance.mean(), t_imbalance.mean()))
print("{:<25s} {:<20.6f} {:<20.6f}".format("Return std", returns.std(), t_returns.std()))
print("{:<25s} {:<20.2f} {:<20.2f}".format(
    "Kurtosis",
    float(pd.Series(returns).kurtosis() + 3),
    float(pd.Series(t_returns).kurtosis() + 3)))

# Wasserstein distances
from scipy.stats import wasserstein_distance as wd
wd_spread = wd(spread_bps[:10000], t_spread[:10000])
wd_returns = wd(returns[:10000], t_returns[:10000])
wd_imbalance = wd(imbalance[:10000], t_imbalance[:10000])
print()
print("{:<25s} {:<20s}".format("Metric", "W-dist (Real vs TRADES)"))
print("-" * 45)
print("{:<25s} {:<20.6f}".format("Spread W-dist", wd_spread))
print("{:<25s} {:<20.6f}".format("Return W-dist", wd_returns))
print("{:<25s} {:<20.6f}".format("Imbalance W-dist", wd_imbalance))

print("\n" + "=" * 64)
print("  Note: Video DiT operates in agent-state space, not LOB space.")
print("  Direct LOB-Bench comparison requires Order Decoder output.")
print("=" * 64)
