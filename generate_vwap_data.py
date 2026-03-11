"""
generate_vwap_data.py — VWAP Behavior-Cloning from LOBSTER LOB Data
====================================================================

Converts raw LOBSTER limit-order-book data into a behavior-cloning dataset
for a **VWAP execution policy**.

Problem framing
---------------
Task:   "Sell (or buy) Q shares over the next T bars."
Oracle: At each bar, trade proportionally to that bar's market volume,
        i.e.  child_size = Q × (bar_volume / total_horizon_volume).
DNN:    Learn to replicate the oracle schedule using only features
        observable *at decision time* (no look-ahead).

Why this works: the oracle perfectly achieves VWAP because the execution
price equals the volume-weighted average by construction.  The DNN must
learn to *predict* volume dynamics from LOB microstructure to approximate
the oracle — that is the volumetric insight captured by the model.

Pipeline
--------
1. Load LOBSTER message + orderbook files
2. Aggregate tick-level data into fixed-width time bars (default: 1 min)
3. Simulate many parent-order episodes (varying start bar, horizon)
4. For each episode, compute oracle VWAP schedule + extract features
5. Save (X, y) as .npy files for DNN training

Features  (X)  — per decision step
-----------------------------------
  Execution state (2):
    inventory_remaining_frac   remaining_qty / Q                     ∈ (0, 1]
    time_remaining_frac        remaining_bars / T                    ∈ (0, 1]

  LOB snapshot (6) — measured at bar open:
    spread_bps                 (best_ask – best_bid) / mid  × 1e4
    mid_price_return           log(mid / prev_mid)
    order_imbalance_L1         (bid_vol₁ – ask_vol₁) / (bid_vol₁ + ask_vol₁)
    depth_imbalance_5L         same, summed across all 5 levels
    bid_depth_norm             Σ bid_vol / mean(Σ bid_vol)
    ask_depth_norm             Σ ask_vol / mean(Σ ask_vol)

  Volume dynamics (2) — from recent trade history:
    volume_rate_ratio          bar_vol / rolling-mean bar_vol
    volume_ma_short_ratio      short MA(vol) / long MA(vol)

  Total: 10 features, all normalised ≈ [0, 2] range.

Target  (y)
-----------
    trade_frac : child_size / Q  ∈ [0, 1]
    Oracle:  bar_volume / horizon_volume
    Recover absolute shares at inference:  child_size = trade_frac × Q

Usage
-----
    python generate_vwap_data.py                           # defaults
    python generate_vwap_data.py --bar-sec 30 --n-orders 20000
    python generate_vwap_data.py --lobster-dir path/to/dir
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

# ───────────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────────

# LOBSTER column names
MSG_COLS = ["time", "event_type", "order_id", "size", "price", "direction"]

# Orderbook: 5 levels → 20 columns
OB_COLS = []
for lvl in range(1, 6):
    OB_COLS += [f"ask_price_{lvl}", f"ask_size_{lvl}",
                f"bid_price_{lvl}", f"bid_size_{lvl}"]

# Event types that represent actual executions (trades)
EXEC_EVENT_TYPES = {4, 5}  # 4 = visible execution, 5 = hidden execution

# Feature names (for documentation / downstream use)
FEATURE_NAMES = [
    "inventory_remaining_frac",
    "time_remaining_frac",
    "spread_bps",
    "mid_price_return",
    "order_imbalance_L1",
    "depth_imbalance_5L",
    "bid_depth_norm",
    "ask_depth_norm",
    "volume_rate_ratio",
    "volume_ma_short_ratio",
]

NUM_FEATURES = len(FEATURE_NAMES)

# Parent-order horizon bounds (in bars)
T_MIN = 5
T_MAX = 100

# ───────────────────────────────────────────────────────────────────────
# Step 1 — Load LOBSTER data
# ───────────────────────────────────────────────────────────────────────

def load_lobster(lobster_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the LOBSTER message and orderbook CSVs from *lobster_dir*.

    Returns (messages, orderbook) DataFrames aligned row-by-row.
    """
    lobster_dir = Path(lobster_dir)

    msg_files = sorted(lobster_dir.glob("*_message_*.csv"))
    ob_files = sorted(lobster_dir.glob("*_orderbook_*.csv"))

    if not msg_files or not ob_files:
        raise FileNotFoundError(
            f"No LOBSTER message/orderbook CSVs found in {lobster_dir}"
        )

    messages = pd.read_csv(msg_files[0], header=None, names=MSG_COLS)
    orderbook = pd.read_csv(ob_files[0], header=None, names=OB_COLS)

    assert len(messages) == len(orderbook), (
        f"Row count mismatch: messages={len(messages)}, orderbook={len(orderbook)}"
    )

    # Convert prices from integer (× 10 000) to dollars
    messages["price"] = messages["price"] / 1e4
    for col in OB_COLS:
        if "price" in col:
            orderbook[col] = orderbook[col] / 1e4

    print(f"Loaded {len(messages):,} events from {msg_files[0].name}")
    return messages, orderbook


# ───────────────────────────────────────────────────────────────────────
# Step 2 — Aggregate into time bars
# ───────────────────────────────────────────────────────────────────────

def aggregate_bars(
    messages: pd.DataFrame,
    orderbook: pd.DataFrame,
    bar_seconds: int = 60,
) -> pd.DataFrame:
    """Aggregate tick-level LOBSTER data into fixed-width time bars.

    Each bar contains:
      - LOB snapshot at bar *open* (first event in bar)
      - Trade volume during the bar (shares executed)
      - Mid-price at bar open

    Returns a DataFrame with one row per bar.
    """
    t_min = messages["time"].min()
    t_max = messages["time"].max()

    bar_edges = np.arange(t_min, t_max + bar_seconds, bar_seconds)
    messages = messages.copy()
    messages["bar_idx"] = np.searchsorted(bar_edges, messages["time"].values, side="right") - 1

    # — Trade volume per bar (only execution events) ———————————
    exec_mask = messages["event_type"].isin(EXEC_EVENT_TYPES)
    vol_per_bar = (
        messages.loc[exec_mask]
        .groupby("bar_idx")["size"]
        .sum()
        .rename("volume")
    )

    # — LOB snapshot at bar open (first event in each bar) ————
    first_idx = messages.groupby("bar_idx").cumcount() == 0
    ob_open = orderbook.loc[first_idx].copy()
    ob_open["bar_idx"] = messages.loc[first_idx, "bar_idx"].values

    # — Assemble bars ——————————————————————————————————————————
    n_bars = int(bar_edges.shape[0] - 1)
    bars = pd.DataFrame({"bar_idx": range(n_bars)})
    bars = bars.merge(vol_per_bar, on="bar_idx", how="left")
    bars["volume"] = bars["volume"].fillna(0).astype(np.float64)

    # Merge LOB snapshots
    bars = bars.merge(ob_open.drop_duplicates("bar_idx"), on="bar_idx", how="left")
    bars = bars.ffill()  # forward-fill if a bar has no events

    # Derived LOB features
    bars["mid_price"] = (bars["ask_price_1"] + bars["bid_price_1"]) / 2
    bars["spread"] = bars["ask_price_1"] - bars["bid_price_1"]
    bars["spread_bps"] = (bars["spread"] / bars["mid_price"]) * 1e4

    bars["mid_price_return"] = np.log(bars["mid_price"] / bars["mid_price"].shift(1)).fillna(0)

    # Order imbalance — Level 1
    bid1 = bars["bid_size_1"].astype(float)
    ask1 = bars["ask_size_1"].astype(float)
    bars["order_imbalance_L1"] = (bid1 - ask1) / (bid1 + ask1 + 1e-9)

    # Depth imbalance — all 5 levels
    total_bid = sum(bars[f"bid_size_{i}"].astype(float) for i in range(1, 6))
    total_ask = sum(bars[f"ask_size_{i}"].astype(float) for i in range(1, 6))
    bars["depth_imbalance_5L"] = (total_bid - total_ask) / (total_bid + total_ask + 1e-9)

    # Normalised depth (relative to mean)
    bars["bid_depth_total"] = total_bid
    bars["ask_depth_total"] = total_ask
    bars["bid_depth_norm"] = total_bid / (total_bid.mean() + 1e-9)
    bars["ask_depth_norm"] = total_ask / (total_ask.mean() + 1e-9)

    # Volume features
    bars["cumulative_volume"] = bars["volume"].cumsum()
    mean_vol = bars["volume"].mean()
    bars["volume_rate_ratio"] = bars["volume"] / (mean_vol + 1e-9)

    short_window = 5
    long_window = 20
    bars["vol_ma_short"] = bars["volume"].rolling(short_window, min_periods=1).mean()
    bars["vol_ma_long"] = bars["volume"].rolling(long_window, min_periods=1).mean()
    bars["volume_ma_short_ratio"] = bars["vol_ma_short"] / (bars["vol_ma_long"] + 1e-9)

    # Drop bars with zero mid-price (degenerate)
    bars = bars[bars["mid_price"] > 0].reset_index(drop=True)

    print(f"Aggregated into {len(bars)} bars ({bar_seconds}s each)")
    print(f"  Total volume: {bars['volume'].sum():,.0f} shares")
    print(f"  Avg volume/bar: {bars['volume'].mean():,.0f} shares")
    return bars


# ───────────────────────────────────────────────────────────────────────
# Step 3 — Generate behaviour-cloning episodes
# ───────────────────────────────────────────────────────────────────────

def generate_vwap_dataset(
    bars: pd.DataFrame,
    n_orders: int = 5_000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a VWAP behavior-cloning dataset from aggregated bars.

    Simulates *n_orders* parent-order episodes.  For each:
      - Sample a random start bar and horizon T ∈ [T_MIN, T_MAX]
      - Compute the oracle VWAP execution schedule (proportional to volume)
      - Record (features, target) for each bar in the horizon

    Parameters
    ----------
    bars : pd.DataFrame
        Output of ``aggregate_bars``.
    n_orders : int
        Number of parent-order episodes to simulate.
    seed : int
        Random seed.

    Returns
    -------
    X : np.ndarray, shape (N, 10), float32
    y : np.ndarray, shape (N,), float32
    """
    rng = np.random.default_rng(seed)
    n_bars = len(bars)

    # Pre-extract arrays for speed
    volume = bars["volume"].values.astype(np.float64)
    spread_bps = bars["spread_bps"].values.astype(np.float64)
    mid_ret = bars["mid_price_return"].values.astype(np.float64)
    oi_l1 = bars["order_imbalance_L1"].values.astype(np.float64)
    di_5l = bars["depth_imbalance_5L"].values.astype(np.float64)
    bid_dn = bars["bid_depth_norm"].values.astype(np.float64)
    ask_dn = bars["ask_depth_norm"].values.astype(np.float64)
    vol_rr = bars["volume_rate_ratio"].values.astype(np.float64)
    vol_ma = bars["volume_ma_short_ratio"].values.astype(np.float64)

    rows_X: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    skipped = 0

    for _ in range(n_orders):
        # Sample horizon and start bar such that the order fits in the data
        T = rng.integers(T_MIN, min(T_MAX, n_bars) + 1)
        start = rng.integers(0, n_bars - T + 1)
        end = start + T

        # Oracle VWAP schedule for this horizon
        horizon_vol = volume[start:end]
        total_horizon_vol = horizon_vol.sum()

        if total_horizon_vol <= 0:
            skipped += 1
            continue

        # Oracle: trade fraction each bar = bar_vol / horizon_vol
        oracle_frac = horizon_vol / total_horizon_vol  # shape (T,)

        # ── Build features for each bar in the episode ──────────
        inv_remaining = 1.0 - np.cumsum(oracle_frac) + oracle_frac  # at bar open
        time_remaining = np.arange(T, 0, -1, dtype=np.float64) / T

        X_episode = np.column_stack([
            inv_remaining,                     # execution state
            time_remaining,                    # execution state
            spread_bps[start:end],             # LOB
            mid_ret[start:end],                # LOB
            oi_l1[start:end],                  # LOB
            di_5l[start:end],                  # LOB
            bid_dn[start:end],                 # LOB
            ask_dn[start:end],                 # LOB
            vol_rr[start:end],                 # volume dynamics
            vol_ma[start:end],                 # volume dynamics
        ]).astype(np.float32)

        y_episode = oracle_frac.astype(np.float32)

        rows_X.append(X_episode)
        rows_y.append(y_episode)

    if skipped:
        print(f"  Skipped {skipped} zero-volume episodes")

    X = np.vstack(rows_X)
    y = np.concatenate(rows_y)

    return X, y


# ───────────────────────────────────────────────────────────────────────
# Persistence
# ───────────────────────────────────────────────────────────────────────

def save_dataset(X: np.ndarray, y: np.ndarray, out_dir: str | Path = "data") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vwap_X.npy", X)
    np.save(out / "vwap_y.npy", y)
    print(f"Saved  X → {out / 'vwap_X.npy'}  shape={X.shape}  dtype={X.dtype}")
    print(f"Saved  y → {out / 'vwap_y.npy'}  shape={y.shape}  dtype={y.dtype}")


def load_dataset(data_dir: str | Path = "data") -> Tuple[np.ndarray, np.ndarray]:
    d = Path(data_dir)
    return np.load(d / "vwap_X.npy"), np.load(d / "vwap_y.npy")


# ───────────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate VWAP behavior-cloning data from LOBSTER LOB files."
    )
    parser.add_argument(
        "--lobster-dir",
        default=str(
            Path(__file__).resolve().parent.parent
            / "LOBSTER_SampleFile_AAPL_2012-06-21_5"
        ),
        help="Directory containing LOBSTER message + orderbook CSVs",
    )
    parser.add_argument("--bar-sec", type=int, default=60, help="Bar width in seconds (default: 60)")
    parser.add_argument("--n-orders", type=int, default=5_000, help="Number of parent-order episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────
    messages, orderbook = load_lobster(args.lobster_dir)

    # ── Aggregate ─────────────────────────────────────────────
    bars = aggregate_bars(messages, orderbook, bar_seconds=args.bar_sec)

    # ── Generate dataset ──────────────────────────────────────
    print(f"\nGenerating VWAP dataset ({args.n_orders:,} episodes) …")
    X, y = generate_vwap_dataset(bars, n_orders=args.n_orders, seed=args.seed)

    print(f"\n{'─' * 50}")
    print(f"Dataset summary")
    print(f"  Samples     : {len(y):,}")
    print(f"  Features    : {X.shape[1]}  ({', '.join(FEATURE_NAMES)})")
    print(f"  X range     : [{X.min():.4f}, {X.max():.4f}]")
    print(f"  y range     : [{y.min():.6f}, {y.max():.6f}]")
    print(f"  y mean      : {y.mean():.6f}")
    print(f"  y std       : {y.std():.6f}")
    print(f"{'─' * 50}")

    # ── Save ──────────────────────────────────────────────────
    out_dir = Path(__file__).resolve().parent / "data"
    save_dataset(X, y, out_dir=out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
