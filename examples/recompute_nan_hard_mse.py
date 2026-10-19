"""
recompute_nan_hard_mse.py

Handles the 2 residual NaN hard_mse records: (seed 10, z=2.0) and (seed 34, z=3.0),
both true_fixedpoint_collapse.

Step 1 (diagnosis): checks xs_hard for NaN/inf and prints per-column tail std,
   so we can tell "constant columns + float32 ridge marginality" (expected)
   apart from "the simulation itself diverged" (would need a different fix).
Step 2 (recompute): refits the readout in float64 numpy using the centered
   formulation (intercept fit separately, unregularized; relative ridge on R),
   with an lstsq fallback, and prints the resulting hard_mse.

Standalone / non-invasive: reads existing params only, writes nothing.

FILL IN get_targets() to return exactly the (T, m) targets your fit_readout
receives inside n_cycle.py (trimmed to the post-burn-in window, ~502 rows),
and match the MSE definition in apply_readout if it differs from the plain
mean squared error used below.
"""

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from support_sequence_analysis import hardened_weights, spec, x0

from oscillon.dynamics import simulate
from oscillon.graph import MotifSpec, NetworkSpec, plot_network_graph
from oscillon.model import train
from oscillon.readout import apply_readout, fit_readout
from oscillon.topology import (
    BlockSoftSpec,
    block_gate_matrix,
    extract_block_adjacency,
    init_params_asymmetric,
    make_block_param_to_model,
)

BURN_IN = 200
REF_PERIOD = 2.0 * np.pi
dt = 0.1
N_STEPS = int(BURN_IN + 8 * (REF_PERIOD / dt))
RIDGE = 1e-6

N = 5  # <-- number of nodes; change this to scale
theta_init = 0.5

RESULTS_DIR = Path("results")  # adjust to your current results dir
RECORDS = [(10, 2.0), (34, 3.0)]


def measure_period(xs, dt, rel_thresh=0.5):
    """Peak-to-peak period of neuron 0, in time units. None if no clear cycle."""
    x = np.asarray(xs[:, 0])
    hi = rel_thresh * x.max()
    peaks = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > hi))[0]
    if len(peaks) < 2:
        return None
    return float(np.mean(np.diff(peaks))) * dt

def make_targets(n, t, period, amplitudes=None, phases=None, offset=0.0):
    """n sinusoids sharing one period; per-channel amplitude + phase.
    Shared period is required for single-motif linear decodability."""
    omega = 2.0 * np.pi / period
    if phases is None:
        phases = [2.0 * np.pi * k / n for k in range(n)]  # even n-phase spread
    if amplitudes is None:
        amplitudes = [1.0 + 0.5 * (k % 3) for k in range(n)]  # some amplitude variety
    cols = [
        a * jnp.sin(omega * t + ph) + offset
        for a, ph in zip(amplitudes, phases, strict=True)
    ]
    return jnp.stack(cols, axis=1)

def get_targets(burn_in, theta_init, N):
    """Return targets of shape (T, m) with T == N_STEPS - BURN_IN (~502).
    Use the same construction as in n_cycle.py."""

    key = jax.random.PRNGKey(0)

    # --- spec: single motif of size N ---
    spec = BlockSoftSpec(
        sizes=(N,),
        cross_deltas=np.zeros((1, 1)),
        eps=0.10,
        delta=0.50,
        theta_init=theta_init,
    )

    cycle = NetworkSpec([MotifSpec.sparse(N)])
    key, sub = jax.random.split(key)

    params0 = init_params_asymmetric(spec, sub, z_std=2.0)

    p2m = make_block_param_to_model(spec)
    W0, th0 = p2m(params0)

    # probe: rough period estimate with a generous fixed rollout
    probe_steps = 4000
    xs_probe = simulate(W0, th0, x0, dt=dt, n_steps=probe_steps)[burn_in:]
    period = measure_period(xs_probe, dt)
    if period is None:
        period = 2.0 * np.pi
    n_steps = int(burn_in + 8 * (period / dt))  # real rollout sized to the cycle

    # REBUILD t and targets at the new n_steps
    t = jnp.arange(n_steps) * dt
    targets = make_targets(N, t, period)
    tgt = targets[burn_in:]

    return tgt

    raise NotImplementedError("paste your n_cycle.py targets construction here")


def fit_readout_f64(xs, targets, ridge=RIDGE):
    xs = np.asarray(xs, dtype=np.float64)
    Y = np.asarray(targets, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    x_mean, y_mean = xs.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = xs - x_mean, Y - y_mean
    G = Xc.T @ Xc
    ridge_eff = ridge * max(np.mean(np.diag(G)), 1.0)
    try:
        theta = np.linalg.solve(G + ridge_eff * np.eye(G.shape[0]), Xc.T @ Yc)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(G + ridge_eff * np.eye(G.shape[0]), Xc.T @ Yc, rcond=None)[0]
    R = theta.T
    b = y_mean - R @ x_mean
    return R, b


def mse(xs, targets, R, b):
    Y = np.asarray(targets, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    pred = np.asarray(xs, dtype=np.float64) @ R.T + b
    return float(np.mean((pred - Y) ** 2))


if __name__ == "__main__":
    targets = np.asarray(get_targets(BURN_IN, theta_init, N))

    for seed, z in RECORDS:
        pf = RESULTS_DIR / f"params_seed{seed}_z{z}.npy"
        if not pf.exists():
            print(f"[seed {seed} z={z}] missing {pf}")
            continue

        params = jnp.asarray(np.load(pf))
        W_hard, theta_hard = hardened_weights(spec, params)
        xs = np.asarray(simulate(W_hard, theta_hard, x0, dt=dt, n_steps=N_STEPS))[BURN_IN:]

        print(f"\n=== seed {seed} z={z} ===")
        print(f"xs_hard dtype={xs.dtype}, shape={xs.shape}")
        print(f"NaN in xs_hard: {np.isnan(xs).any()}   inf in xs_hard: {np.isinf(xs).any()}")
        print(f"NaN in targets: {np.isnan(targets).any()}")
        print("per-column std over window:", np.array2string(xs.std(axis=0), precision=3))
        print("per-column mean:           ", np.array2string(xs.mean(axis=0), precision=3))

        if np.isnan(xs).any() or np.isinf(xs).any():
            print("-> simulation itself is non-finite; readout fix will NOT help. Investigate dynamics.")
            continue

        R, b = fit_readout_f64(xs, targets)
        print(f"float64 centered fit: finite R/b = {np.isfinite(R).all() and np.isfinite(b).all()}")
        print(f"hard_mse (float64) = {mse(xs, targets, R, b):.6g}")
