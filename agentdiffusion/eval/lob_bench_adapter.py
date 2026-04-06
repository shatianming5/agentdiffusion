"""Adapter to convert our A-Share L3 data and model outputs to LOB-Bench format.

LOB-Bench expects LOBSTER-format CSV files:
  message.csv: Time, Type, OrderID, Size, Price, Direction
  orderbook.csv: AskPrice1, AskSize1, BidPrice1, BidSize1, ...

This module converts:
  1. Real A-Share 逐笔委托 → LOBSTER message format
  2. Model-generated agent grids → synthetic LOBSTER messages
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# LOBSTER message types
LOBSTER_NEW = 1       # new limit order
LOBSTER_CANCEL = 3    # full cancellation
LOBSTER_EXECUTE = 4   # execution (visible)
LOBSTER_EXECUTE_H = 5 # execution (hidden)


def ashare_orders_to_lobster(
    orders_csv: str | Path,
    output_dir: str | Path,
    encoding: str = "gbk",
) -> tuple[Path, Path]:
    """Convert A-Share 逐笔委托.csv to LOBSTER message + orderbook format.

    Args:
        orders_csv: path to 逐笔委托.csv
        output_dir: directory to write LOBSTER CSVs
        encoding: CSV encoding

    Returns:
        (message_path, orderbook_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        orders_csv, encoding=encoding,
        names=["code", "exch_code", "date", "time", "order_num",
               "exch_order_id", "order_type", "side", "price", "size", "_"],
        header=0,
    )

    # Filter to trading hours and valid orders
    df = df[(df["time"] >= 93000000) & (df["time"] <= 150000000)].copy()
    df = df[df["price"] > 0].copy()

    # Convert time to seconds
    def time_to_sec(t):
        ms = t % 1000
        t = t // 1000
        ss = t % 100
        t = t // 100
        mm = t % 100
        hh = t // 100
        return hh * 3600 + mm * 60 + ss + ms / 1000.0
    df["timestamp"] = df["time"].apply(time_to_sec)

    # Map to LOBSTER format
    type_map = {1: LOBSTER_NEW, 2: LOBSTER_NEW, 3: LOBSTER_CANCEL}
    df["lobster_type"] = df["order_type"].map(type_map).fillna(LOBSTER_NEW).astype(int)
    df["direction"] = df["side"].map({"B": 1, "S": -1}).fillna(0).astype(int)

    # Write message file
    msg = pd.DataFrame({
        "Time": df["timestamp"],
        "Type": df["lobster_type"],
        "OrderID": df["exch_order_id"],
        "Size": df["size"],
        "Price": df["price"],  # already in 万分
        "Direction": df["direction"],
    })
    msg_path = output_dir / "message.csv"
    msg.to_csv(msg_path, index=False, header=False)

    # Build simple orderbook snapshots (10 levels, sampled every 100 messages)
    # This is approximate — true orderbook requires full replay
    ob_rows = []
    sample_every = 100
    for i in range(0, len(msg), sample_every):
        row = [0] * 40  # 10 levels × 4 (ask_p, ask_v, bid_p, bid_v)
        window = msg.iloc[max(0, i-1000):i+1]

        buys = window[window["Direction"] == 1].nlargest(10, "Price")
        sells = window[window["Direction"] == -1].nsmallest(10, "Price")

        for lv, (_, r) in enumerate(sells.iterrows()):
            if lv >= 10:
                break
            row[lv * 4] = int(r["Price"])
            row[lv * 4 + 1] = int(r["Size"])

        for lv, (_, r) in enumerate(buys.iterrows()):
            if lv >= 10:
                break
            row[lv * 4 + 2] = int(r["Price"])
            row[lv * 4 + 3] = int(r["Size"])

        ob_rows.append(row)

    ob_path = output_dir / "orderbook.csv"
    pd.DataFrame(ob_rows).to_csv(ob_path, index=False, header=False)

    return msg_path, ob_path


def agent_grid_to_lobster(
    agent_states: np.ndarray,
    output_dir: str | Path,
    base_price: float = 150000,
    time_start: float = 34200.0,
    time_step: float = 1.0,
) -> tuple[Path, Path]:
    """Convert generated agent state grid to LOBSTER message format.

    Args:
        agent_states: [T, H, W, d_state] generated agent grid
        output_dir: directory to write LOBSTER CSVs
        base_price: base price for order generation
        time_start: starting timestamp (seconds)
        time_step: seconds per frame

    Returns:
        (message_path, orderbook_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    T, H, W, D = agent_states.shape
    messages = []
    order_id = 1

    for t in range(T - 1):
        timestamp = time_start + t * time_step
        for h in range(H):
            for w in range(W):
                delta_pos = agent_states[t + 1, h, w, 0] - agent_states[t, h, w, 0]
                if abs(delta_pos) < 0.01:
                    continue

                direction = 1 if delta_pos > 0 else -1
                size = max(int(abs(delta_pos) * 100), 1)
                price_offset = agent_states[t, h, w, 2] * 100 if D > 2 else 0
                price = int(base_price + price_offset)

                messages.append([
                    timestamp, LOBSTER_NEW, order_id,
                    size, price, direction,
                ])
                order_id += 1

    msg_df = pd.DataFrame(messages, columns=[
        "Time", "Type", "OrderID", "Size", "Price", "Direction",
    ])
    msg_path = output_dir / "message.csv"
    msg_df.to_csv(msg_path, index=False, header=False)

    # Minimal orderbook (for LOB-Bench compatibility)
    ob_rows = []
    for t in range(T):
        row = [0] * 40
        row[0] = int(base_price + 100)   # ask_p1
        row[1] = 100                      # ask_v1
        row[2] = int(base_price - 100)   # bid_p1
        row[3] = 100                      # bid_v1
        ob_rows.append(row)
    ob_path = output_dir / "orderbook.csv"
    pd.DataFrame(ob_rows).to_csv(ob_path, index=False, header=False)

    return msg_path, ob_path


def run_lob_bench(
    real_dir: str | Path,
    generated_dir: str | Path,
    cond_dir: str | Path | None = None,
) -> dict:
    """Run LOB-Bench evaluation if installed.

    Returns dict of benchmark results or error message.
    """
    try:
        from lob_bench import data_loading, scoring, impact

        loader = data_loading.Simple_Loader(
            cond_path=str(cond_dir) if cond_dir else None,
            generated_path=str(generated_dir),
            real_path=str(real_dir),
        )

        score_cfg = {
            "Spread": {"fn": scoring.spread},
            "Interarrival": {"fn": scoring.interarrival, "Discrete": False},
            "Imbalance": {"fn": scoring.imbalance},
        }
        metric_cfg = {
            "L1": scoring.l1_distance,
            "Wasserstein": scoring.wasserstein_distance,
        }

        results = scoring.run_benchmark(loader, score_cfg, metric_cfg)
        return {"status": "ok", "results": results}

    except ImportError:
        return {"status": "error", "message": "lob_bench not installed. pip install lob_bench"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
