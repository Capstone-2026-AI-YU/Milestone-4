"""
vwap_rnn_bc.py — VWAP Behavior-Cloning with GRU-RNN (Actor-only)
=================================================================

Replaces the feed-forward VWAPNet with an RNN architecture modelled on
ActorCriticRNN from ippo_rnn_JAXMARL.py, but with the critic head removed.

Architecture
------------
  obs → Dense(FC_DIM_SIZE) → ReLU                    [embedding]
      → GRU(GRU_HIDDEN_DIM)                           [recurrence]
      → Dense(GRU_HIDDEN_DIM) → ReLU → Dense(1) → σ  [actor head]

The GRU lets the model condition on the *history* of LOB features within
an execution episode — capturing volume patterns, spread dynamics, and
inventory trajectory — rather than treating each bar independently.

Data format
-----------
  Episodes are variable-length sequences of bars.  We pad every episode
  to MAX_SEQ_LEN and use a boolean mask so the loss ignores padding.

  X_seq : (N_episodes, MAX_SEQ_LEN, 10)   — features per bar
  y_seq : (N_episodes, MAX_SEQ_LEN)       — oracle trade_frac per bar
  mask  : (N_episodes, MAX_SEQ_LEN)       — True for real bars

Usage
-----
  # Generate data + train (default LOBSTER path)
  python vwap_rnn_bc.py

  # Custom options
  python vwap_rnn_bc.py --lobster-dir /path/to/lobster \\
                        --epochs 120 --wandb

  # Just evaluate a saved checkpoint
  python vwap_rnn_bc.py --eval-only --checkpoint checkpoints/vwap_rnn_best.pkl
"""

from __future__ import annotations

import argparse
import functools
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

# Local data generation
import generate_vwap_data as gvd

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # Data
    "BAR_SEC": 60,
    "N_ORDERS": 10_000,
    "SEED": 42,
    "VAL_SPLIT": 0.1,
    "MAX_SEQ_LEN": 100,        # pad all episodes to this length (= T_MAX)

    # Architecture (mirrors ActorCriticRNN naming)
    "FC_DIM_SIZE": 64,         # embedding dimension
    "GRU_HIDDEN_DIM": 64,      # GRU hidden state size

    # Training
    "LR": 3e-4,
    "BATCH_SIZE": 64,          # episodes per batch
    "EPOCHS": 100,
    "ANNEAL_LR": True,
    "MAX_GRAD_NORM": 0.5,

    # Features
    "N_FEATURES": 10,
}


# ═══════════════════════════════════════════════════════════════════════
# 1. RNN Model — Actor-only version of ActorCriticRNN
# ═══════════════════════════════════════════════════════════════════════

class ScannedRNN(nn.Module):
    """GRU scanned over the time axis.

    Identical to the one in ippo_rnn_JAXMARL.py: resets hidden state
    when `resets=True` (episode boundary).
    """

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        # Reset hidden state at episode boundaries (or padding start)
        rnn_state = jnp.where(
            resets[:, jnp.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )


class ActorRNN(nn.Module):
    """Actor-only RNN for VWAP behavior cloning.

    Based on ActorCriticRNN from ippo_rnn_JAXMARL.py with the critic
    head removed.  Output is a scalar trade_frac ∈ [0, 1] per timestep
    (sigmoid instead of Categorical, since this is a regression task).

    Forward signature matches the original:
        hidden, output = model.apply(params, hidden, (obs, dones))

    Where:
        obs   : (seq_len, batch, n_features)
        dones : (seq_len, batch)
    Returns:
        hidden : (batch, GRU_HIDDEN_DIM)  — updated GRU state
        pred   : (seq_len, batch)         — predicted trade_frac
    """
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        # ── Embedding (same as ActorCriticRNN) ──
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        # ── GRU (same as ActorCriticRNN) ──
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # ── Actor head (no critic) ──
        actor = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor = nn.relu(actor)

        # Single output → sigmoid for trade_frac ∈ [0, 1]
        actor = nn.Dense(
            1,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor)
        pred = jax.nn.sigmoid(actor.squeeze(-1))  # (seq_len, batch)

        return hidden, pred


# ═══════════════════════════════════════════════════════════════════════
# 2. Sequence Data Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_sequence_dataset(
    bars,
    n_orders: int,
    max_seq_len: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate padded sequence episodes for RNN training.

    Returns
    -------
    X_seq : (N_episodes, max_seq_len, 10) float32  — features
    y_seq : (N_episodes, max_seq_len)     float32  — oracle trade_frac
    mask  : (N_episodes, max_seq_len)     bool     — True for real bars
    """
    rng = np.random.default_rng(seed)
    n_bars = len(bars)

    # Pre-extract arrays
    volume = bars["volume"].values.astype(np.float64)
    spread_bps = bars["spread_bps"].values.astype(np.float64)
    mid_ret = bars["mid_price_return"].values.astype(np.float64)
    oi_l1 = bars["order_imbalance_L1"].values.astype(np.float64)
    di_5l = bars["depth_imbalance_5L"].values.astype(np.float64)
    bid_dn = bars["bid_depth_norm"].values.astype(np.float64)
    ask_dn = bars["ask_depth_norm"].values.astype(np.float64)
    vol_rr = bars["volume_rate_ratio"].values.astype(np.float64)
    vol_ma = bars["volume_ma_short_ratio"].values.astype(np.float64)

    X_list, y_list, mask_list = [], [], []
    skipped = 0

    for _ in range(n_orders):
        T = rng.integers(gvd.T_MIN, min(gvd.T_MAX, n_bars) + 1)
        start = rng.integers(0, n_bars - T + 1)
        end = start + T

        horizon_vol = volume[start:end]
        total_horizon_vol = horizon_vol.sum()
        if total_horizon_vol <= 0:
            skipped += 1
            continue

        oracle_frac = horizon_vol / total_horizon_vol
        inv_remaining = 1.0 - np.cumsum(oracle_frac) + oracle_frac
        time_remaining = np.arange(T, 0, -1, dtype=np.float64) / T

        X_ep = np.column_stack([
            inv_remaining,
            time_remaining,
            spread_bps[start:end],
            mid_ret[start:end],
            oi_l1[start:end],
            di_5l[start:end],
            bid_dn[start:end],
            ask_dn[start:end],
            vol_rr[start:end],
            vol_ma[start:end],
        ]).astype(np.float32)  # (T, 10)

        y_ep = oracle_frac.astype(np.float32)  # (T,)

        # Pad to max_seq_len
        pad_len = max_seq_len - T
        X_padded = np.pad(X_ep, ((0, pad_len), (0, 0)), mode="constant")
        y_padded = np.pad(y_ep, (0, pad_len), mode="constant")
        m = np.zeros(max_seq_len, dtype=bool)
        m[:T] = True

        X_list.append(X_padded)
        y_list.append(y_padded)
        mask_list.append(m)

    if skipped:
        print(f"  Skipped {skipped} zero-volume episodes")

    X_seq = np.stack(X_list)   # (N, max_seq_len, 10)
    y_seq = np.stack(y_list)   # (N, max_seq_len)
    mask = np.stack(mask_list)  # (N, max_seq_len)

    print(f"  Generated {len(X_list)} episodes, padded to {max_seq_len} bars")
    print(f"  X_seq shape: {X_seq.shape},  y_seq shape: {y_seq.shape}")
    return X_seq, y_seq, mask


# ═══════════════════════════════════════════════════════════════════════
# 3. Training
# ═══════════════════════════════════════════════════════════════════════

def create_train_state(
    rng: jax.Array,
    config: Dict,
    n_epochs: int,
    steps_per_epoch: int,
) -> Tuple[TrainState, jnp.ndarray]:
    """Initialise ActorRNN + optimizer (mirrors ippo_rnn_JAXMARL.py)."""

    network = ActorRNN(config=config)

    # Dummy inputs: (seq_len=1, batch=1, n_features)
    init_x = (
        jnp.zeros((1, 1, config["N_FEATURES"])),
        jnp.zeros((1, 1)),
    )
    init_hstate = ScannedRNN.initialize_carry(1, config["GRU_HIDDEN_DIM"])
    params = network.init(rng, init_hstate, init_x)

    total_steps = n_epochs * steps_per_epoch

    if config["ANNEAL_LR"]:
        lr_schedule = optax.linear_schedule(
            init_value=config["LR"],
            end_value=0.0,
            transition_steps=total_steps,
        )
    else:
        lr_schedule = config["LR"]

    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(lr_schedule, eps=1e-5),
    )

    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )

    init_hstate_batch = None  # created per-batch in train loop
    return train_state, params


def masked_mse_loss(params, apply_fn, hstate, obs, targets, mask, dones):
    """MSE loss over valid (non-padded) timesteps only.

    obs     : (seq_len, batch, n_features)
    targets : (seq_len, batch)
    mask    : (seq_len, batch)   — True for real timesteps
    dones   : (seq_len, batch)   — boundary resets (all False here)
    """
    _, pred = apply_fn(params, hstate, (obs, dones))
    # pred: (seq_len, batch)
    sq_err = (pred - targets) ** 2
    # Only count real timesteps
    masked_sq_err = sq_err * mask
    loss = masked_sq_err.sum() / (mask.sum() + 1e-8)
    return loss, pred


@jax.jit
def train_step(
    train_state: TrainState,
    hstate: jnp.ndarray,
    obs: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    dones: jnp.ndarray,
):
    """One gradient step on a minibatch of episodes."""
    (loss, pred), grads = jax.value_and_grad(masked_mse_loss, has_aux=True)(
        train_state.params, train_state.apply_fn,
        hstate, obs, targets, mask, dones,
    )
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss, pred


@jax.jit
def eval_step(
    train_state: TrainState,
    hstate: jnp.ndarray,
    obs: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    dones: jnp.ndarray,
):
    """Evaluate on a batch without gradient updates."""
    loss, pred = masked_mse_loss(
        train_state.params, train_state.apply_fn,
        hstate, obs, targets, mask, dones,
    )
    return loss, pred


def run_training(config, X_train, y_train, mask_train,
                 X_val, y_val, mask_val, use_wandb=False):
    """Main training loop."""

    max_seq_len = config["MAX_SEQ_LEN"]
    batch_size = config["BATCH_SIZE"]
    n_train = X_train.shape[0]
    n_batches = max(1, n_train // batch_size)

    rng = jax.random.PRNGKey(config["SEED"])
    rng, init_rng = jax.random.split(rng)

    train_state, _ = create_train_state(
        init_rng, config, config["EPOCHS"], n_batches,
    )

    # Standardize features (fit on train only)
    # Reshape to (N*T, F), compute stats, reshape back
    flat_X = X_train.reshape(-1, config["N_FEATURES"])
    flat_mask = mask_train.reshape(-1)
    # Only compute stats on real (non-padded) timesteps
    real_X = flat_X[flat_mask]
    X_mean = real_X.mean(axis=0)
    X_std = real_X.std(axis=0) + 1e-8

    X_train_s = (X_train - X_mean) / X_std
    X_val_s = (X_val - X_mean) / X_std

    # Save normalization params
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    np.save(data_dir / "rnn_X_mean.npy", X_mean)
    np.save(data_dir / "rnn_X_std.npy", X_std)

    # Convert to JAX arrays — shape (N, T, F) and (N, T)
    X_train_j = jnp.array(X_train_s)
    y_train_j = jnp.array(y_train)
    mask_train_j = jnp.array(mask_train.astype(np.float32))
    X_val_j = jnp.array(X_val_s)
    y_val_j = jnp.array(y_val)
    mask_val_j = jnp.array(mask_val.astype(np.float32))

    batch_rng = np.random.default_rng(config["SEED"] + 1)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    print(f"\n{'Epoch':>6}  {'Train MSE':>12}  {'Val MSE':>12}  "
          f"{'Train RMSE':>12}  {'Val RMSE':>12}  {'Time':>8}")
    print("-" * 72)

    for epoch in range(1, config["EPOCHS"] + 1):
        t0 = time.time()
        perm = batch_rng.permutation(n_train)
        epoch_loss = 0.0

        for b in range(n_batches):
            idx = perm[b * batch_size: (b + 1) * batch_size]

            # (batch, seq_len, features) → transpose to (seq_len, batch, features)
            obs_batch = jnp.transpose(X_train_j[idx], (1, 0, 2))
            tgt_batch = jnp.transpose(y_train_j[idx], (1, 0))
            msk_batch = jnp.transpose(mask_train_j[idx], (1, 0))

            # No episode boundaries within a single episode → dones = False
            dones_batch = jnp.zeros_like(msk_batch)

            # Fresh hidden state each episode
            hstate = ScannedRNN.initialize_carry(
                len(idx), config["GRU_HIDDEN_DIM"]
            )

            train_state, loss, _ = train_step(
                train_state, hstate, obs_batch, tgt_batch, msk_batch, dones_batch
            )
            epoch_loss += float(loss)

        train_mse = epoch_loss / n_batches

        # ── Validation ──
        val_obs = jnp.transpose(X_val_j, (1, 0, 2))
        val_tgt = jnp.transpose(y_val_j, (1, 0))
        val_msk = jnp.transpose(mask_val_j, (1, 0))
        val_dones = jnp.zeros_like(val_msk)
        val_hstate = ScannedRNN.initialize_carry(
            X_val_j.shape[0], config["GRU_HIDDEN_DIM"]
        )
        val_loss, val_pred = eval_step(
            train_state, val_hstate, val_obs, val_tgt, val_msk, val_dones
        )
        val_mse = float(val_loss)

        history["train_loss"].append(train_mse)
        history["val_loss"].append(val_mse)

        elapsed = time.time() - t0

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"{epoch:>6}  {train_mse:>12.6f}  {val_mse:>12.6f}  "
                f"{train_mse**0.5:>12.6f}  {val_mse**0.5:>12.6f}  "
                f"{elapsed:>7.1f}s"
            )

        # ── WANDB logging ──
        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/mse": train_mse,
                "train/rmse": train_mse ** 0.5,
                "val/mse": val_mse,
                "val/rmse": val_mse ** 0.5,
                "lr": float(
                    train_state.opt_state[1][1].mu
                    if hasattr(train_state.opt_state, '__getitem__')
                    else config["LR"]
                ),
            })

        # ── Checkpoint best ──
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            ckpt_dir = Path("checkpoints")
            ckpt_dir.mkdir(exist_ok=True)
            with open(ckpt_dir / "vwap_rnn_best.pkl", "wb") as f:
                pickle.dump({
                    "params": jax.device_get(train_state.params),
                    "config": config,
                    "X_mean": X_mean,
                    "X_std": X_std,
                    "epoch": epoch,
                    "val_mse": val_mse,
                }, f)

    print("-" * 72)
    print(f"Best val MSE: {best_val_loss:.6f}  (RMSE: {best_val_loss**0.5:.6f})")
    return train_state, history, X_mean, X_std


# ═══════════════════════════════════════════════════════════════════════
# 4. Evaluation — Execution Simulation
# ═══════════════════════════════════════════════════════════════════════

def make_rnn_predict_fn(train_state, config, X_mean, X_std):
    """Return a callable that predicts trade_frac for a single episode.

    Unlike the feed-forward VWAPNet which predicts one bar at a time,
    the RNN processes the whole episode as a sequence, maintaining
    hidden state across bars.
    """

    def predict_episode(features_seq: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        features_seq : (T, 10) float32 — raw (un-standardized) features

        Returns
        -------
        trade_frac : (T,) float32
        """
        T = features_seq.shape[0]
        features_s = (features_seq - X_mean) / X_std

        # (T, 1, 10) — seq_len, batch=1, features
        obs = jnp.array(features_s)[jnp.newaxis, :, :].transpose(1, 0, 2)
        dones = jnp.zeros((T, 1))
        hstate = ScannedRNN.initialize_carry(1, config["GRU_HIDDEN_DIM"])

        _, pred = train_state.apply_fn(train_state.params, hstate, (obs, dones))
        return np.array(pred.squeeze(-1))  # (T,)

    return predict_episode


def simulate_rnn_execution(bars_df, start_bar, horizon, parent_Q,
                           rnn_predict_fn, X_mean, X_std):
    """Simulate execution using the RNN model over a full episode at once."""
    end = start_bar + horizon
    window = bars_df.iloc[start_bar:end]

    volume = window["volume"].values.astype(np.float64)
    mid = window["mid_price"].values.astype(np.float64)
    total_vol = volume.sum()
    market_vwap = (np.sum(volume * mid) / total_vol) if total_vol > 0 else mid.mean()

    # Oracle & TWAP
    oracle_frac = volume / (total_vol + 1e-9)
    twap_frac = np.full(horizon, 1.0 / horizon)

    # Build features for the whole episode
    inv_remaining = 1.0 - np.cumsum(oracle_frac) + oracle_frac
    time_remaining = np.arange(horizon, 0, -1, dtype=np.float64) / horizon

    features = np.column_stack([
        inv_remaining,
        time_remaining,
        window["spread_bps"].values,
        window["mid_price_return"].values,
        window["order_imbalance_L1"].values,
        window["depth_imbalance_5L"].values,
        window["bid_depth_norm"].values,
        window["ask_depth_norm"].values,
        window["volume_rate_ratio"].values,
        window["volume_ma_short_ratio"].values,
    ]).astype(np.float32)

    # RNN predicts the whole sequence at once
    rnn_frac = rnn_predict_fn(features)
    rnn_frac = np.clip(rnn_frac, 0.0, 1.0)
    rnn_frac = rnn_frac / (rnn_frac.sum() + 1e-9)  # normalize to sum to 1

    def exec_price(frac):
        return np.sum(frac * parent_Q * mid) / parent_Q

    return {
        "market_vwap": market_vwap,
        "oracle_price": exec_price(oracle_frac),
        "twap_price": exec_price(twap_frac),
        "rnn_price": exec_price(rnn_frac),
        "oracle_frac": oracle_frac,
        "twap_frac": twap_frac,
        "rnn_frac": rnn_frac,
    }


def run_evaluation(train_state, config, bars, X_mean, X_std,
                   n_trials=200, parent_Q=5000, use_wandb=False):
    """Run execution simulation and print/log results."""

    predict_fn = make_rnn_predict_fn(train_state, config, X_mean, X_std)
    rng_sim = np.random.default_rng(99)
    n_bars = len(bars)

    oracle_slip, twap_slip, rnn_slip = [], [], []

    for _ in range(n_trials):
        T = rng_sim.integers(10, 40)
        start = rng_sim.integers(0, n_bars - T)
        res = simulate_rnn_execution(bars, start, T, parent_Q, predict_fn, X_mean, X_std)
        vwap = res["market_vwap"]
        oracle_slip.append((vwap - res["oracle_price"]) / vwap * 1e4)
        twap_slip.append((vwap - res["twap_price"]) / vwap * 1e4)
        rnn_slip.append((vwap - res["rnn_price"]) / vwap * 1e4)

    oracle_slip = np.array(oracle_slip)
    twap_slip = np.array(twap_slip)
    rnn_slip = np.array(rnn_slip)

    print(f"\n{'Strategy':<14} {'Mean Slip (bps)':>16} {'Std (bps)':>12} {'Median (bps)':>14}")
    print("-" * 60)
    for name, arr in [("Oracle", oracle_slip), ("TWAP", twap_slip), ("RNN", rnn_slip)]:
        print(f"{name:<14} {arr.mean():>16.4f} {arr.std():>12.4f} {np.median(arr):>14.4f}")
    print(f"\nRNN improvement over TWAP: {(twap_slip.mean() - rnn_slip.mean()):.4f} bps")

    if use_wandb:
        import wandb
        wandb.log({
            "eval/oracle_slip_mean_bps": oracle_slip.mean(),
            "eval/twap_slip_mean_bps": twap_slip.mean(),
            "eval/rnn_slip_mean_bps": rnn_slip.mean(),
            "eval/rnn_vs_twap_bps": twap_slip.mean() - rnn_slip.mean(),
        })

    # ── Plot ──
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Slippage boxplot
        data = [oracle_slip, twap_slip, rnn_slip]
        labels = ["Oracle\n(upper bound)", "Naive\nTWAP", "RNN\nVWAP"]
        bp = axes[0].boxplot(data, labels=labels, patch_artist=True, showmeans=True)
        for patch, color in zip(bp["boxes"], ["#2ecc71", "#e74c3c", "#9b59b6"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_ylabel("Slippage vs Market VWAP (bps)")
        axes[0].set_title(f"Execution Quality — RNN ({n_trials} trials)")
        axes[0].grid(True, alpha=0.3)

        # Example schedule
        ex = simulate_rnn_execution(bars, 100, 30, parent_Q, predict_fn, X_mean, X_std)
        x_bars = np.arange(30)
        w = 0.25
        axes[1].bar(x_bars - w, ex["oracle_frac"], width=w, label="Oracle", alpha=0.7, color="#2ecc71")
        axes[1].bar(x_bars,     ex["twap_frac"],   width=w, label="TWAP",   alpha=0.7, color="#e74c3c")
        axes[1].bar(x_bars + w, ex["rnn_frac"],    width=w, label="RNN",    alpha=0.7, color="#9b59b6")
        axes[1].set_xlabel("Bar within execution window")
        axes[1].set_ylabel("Fraction of order")
        axes[1].set_title("Example RNN Execution Schedule")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("rnn_execution_comparison.png", dpi=150)
        print("\nSaved rnn_execution_comparison.png")
        plt.close()

        if use_wandb:
            wandb.log({"eval/execution_comparison": wandb.Image("rnn_execution_comparison.png")})

    except ImportError:
        print("(matplotlib not available — skipping plots)")

    return {"oracle": oracle_slip, "twap": twap_slip, "rnn": rnn_slip}


# ═══════════════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VWAP Behavior Cloning with GRU-RNN (actor-only ActorCriticRNN)"
    )
    parser.add_argument(
        "--lobster-dir",
        default=str(
            Path(__file__).resolve().parent
            / "LOBSTER_SampleFile_AAPL_2012-06-21_5"
        ),
        help="Directory containing LOBSTER message + orderbook CSVs",
    )
    parser.add_argument("--bar-sec", type=int, default=60)
    parser.add_argument("--n-orders", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fc-dim", type=int, default=64)
    parser.add_argument("--gru-dim", type=int, default=64)
    parser.add_argument("--wandb", action="store_true", help="Enable WANDB logging")
    parser.add_argument("--wandb-project", default="vwap-rnn-bc")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["BAR_SEC"] = args.bar_sec
    config["N_ORDERS"] = args.n_orders
    config["SEED"] = args.seed
    config["EPOCHS"] = args.epochs
    config["BATCH_SIZE"] = args.batch_size
    config["LR"] = args.lr
    config["FC_DIM_SIZE"] = args.fc_dim
    config["GRU_HIDDEN_DIM"] = args.gru_dim

    print("=" * 60)
    print("VWAP Behavior Cloning — GRU-RNN (Actor-only)")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    for k, v in sorted(config.items()):
        print(f"  {k:<20} = {v}")
    print()

    # ── WANDB ──
    use_wandb = args.wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            config=config,
            name=f"rnn_fc{config['FC_DIM_SIZE']}_gru{config['GRU_HIDDEN_DIM']}"
                 f"_lr{config['LR']}_ep{config['EPOCHS']}",
        )
        print("WANDB initialized.")

    # ── Load LOBSTER data ──
    print("\n1. Loading LOBSTER data...")
    messages, orderbook = gvd.load_lobster(args.lobster_dir)
    bars = gvd.aggregate_bars(messages, orderbook, bar_seconds=config["BAR_SEC"])

    # ── Generate sequence dataset ──
    print("\n2. Generating sequence dataset...")
    X_seq, y_seq, mask = generate_sequence_dataset(
        bars, config["N_ORDERS"], config["MAX_SEQ_LEN"], config["SEED"]
    )

    # ── Train/val split ──
    N = X_seq.shape[0]
    rng_split = np.random.default_rng(config["SEED"])
    idx = rng_split.permutation(N)
    n_val = int(N * config["VAL_SPLIT"])
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train, y_train, mask_train = X_seq[train_idx], y_seq[train_idx], mask[train_idx]
    X_val, y_val, mask_val = X_seq[val_idx], y_seq[val_idx], mask[val_idx]

    print(f"\n  Train episodes: {len(train_idx)}")
    print(f"  Val episodes  : {len(val_idx)}")

    if args.eval_only:
        # Load checkpoint and evaluate
        ckpt_path = args.checkpoint or "checkpoints/vwap_rnn_best.pkl"
        print(f"\nLoading checkpoint: {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        config = ckpt["config"]
        X_mean, X_std = ckpt["X_mean"], ckpt["X_std"]

        rng = jax.random.PRNGKey(config["SEED"])
        train_state, _ = create_train_state(rng, config, 1, 1)
        train_state = train_state.replace(params=ckpt["params"])

        print(f"\nEvaluating (checkpoint from epoch {ckpt['epoch']}, "
              f"val MSE={ckpt['val_mse']:.6f})...")
        run_evaluation(train_state, config, bars, X_mean, X_std, use_wandb=use_wandb)
        return

    # ── Train ──
    print("\n3. Training...")
    train_state, history, X_mean, X_std = run_training(
        config, X_train, y_train, mask_train,
        X_val, y_val, mask_val, use_wandb=use_wandb,
    )

    # ── Plot training curves ──
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs_range = range(1, config["EPOCHS"] + 1)

        axes[0].plot(epochs_range, history["train_loss"], label="Train")
        axes[0].plot(epochs_range, history["val_loss"], label="Val", linestyle="--")
        axes[0].set_title("MSE Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs_range, [v**0.5 for v in history["train_loss"]], label="Train")
        axes[1].plot(epochs_range, [v**0.5 for v in history["val_loss"]], label="Val", linestyle="--")
        axes[1].set_title("RMSE (trade_frac units)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("RMSE")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("rnn_training_curves.png", dpi=150)
        print("\nSaved rnn_training_curves.png")
        plt.close()

        if use_wandb:
            import wandb
            wandb.log({"training_curves": wandb.Image("rnn_training_curves.png")})

    except ImportError:
        pass

    # ── Evaluate ──
    print("\n4. Evaluating...")
    run_evaluation(train_state, config, bars, X_mean, X_std, use_wandb=use_wandb)

    if use_wandb:
        import wandb
        wandb.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
