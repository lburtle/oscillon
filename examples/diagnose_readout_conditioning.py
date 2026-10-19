"""
diagnose_readout_conditioning.py

Confirms (or rules out) the hypothesis that hard_mse NaNs come from a
numerically near-singular A = Xa.T @ Xa + diag(d) inside fit_readout,
specifically for hardened trajectories that have collapsed (some neuron
columns constant/zero over the fitting window).

Does NOT modify fit_readout, hardened_ctln_check, or anything else --
purely diagnostic. Picks one known collapsed seed and one known
full-support-oscillation seed, rebuilds A exactly as fit_readout does,
and reports condition number / dtype / where the NaN actually appears.

Adjust the `from readout import fit_readout` line below to match your
actual module path if it differs.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from support_sequence_analysis import hardened_weights, spec, p2m, x0, parse_seed_zstd
from graph_rules_check import measured_fixedpoint_support
from diagnose_boundedness import hard_len1_is_genuine_fixedpoint
from oscillon.dynamics import simulate
from oscillon.readout import fit_readout  # <-- adjust if your import path differs
from oscillon.dynamics import simulate
from oscillon.graph import MotifSpec, NetworkSpec, plot_network_graph
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
N = 5
theta_init = 0.5

key = jax.random.PRNGKey(0)

# --- spec: single motif of size N ---
spec = BlockSoftSpec(
    sizes=(N,),
    cross_deltas=np.zeros((1, 1)),
    eps=0.10,
    delta=0.50,
    theta_init=theta_init,
)

# --- warm start from the discrete N-cycle ---
cycle = NetworkSpec([MotifSpec.sparse(N)])
A_seed = cycle.to_torch_adjacency().numpy()
key, sub = jax.random.split(key)

params0 = init_params_asymmetric(spec, sub, z_std=2.0)

x0 = jnp.full((N,), 0.1)


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


def build_A(xs, ridge):
    """Exact reproduction of fit_readout's A matrix, for inspection."""
    T = xs.shape[0]
    Xa = jnp.concatenate([xs, jnp.ones((T, 1))], axis=1)
    n = xs.shape[1]
    d = ridge * jnp.ones(n + 1)
    d = d.at[-1].set(0.0)
    A = Xa.T @ Xa + jnp.diag(d)
    return A, Xa


def diagnose_one(pf, targets, label):
    params = jnp.asarray(np.load(pf))
    W_hard, theta_hard = hardened_weights(spec, params)
    xs_hard = jnp.asarray(
        np.asarray(simulate(W_hard, theta_hard, x0, dt=dt, n_steps=N_STEPS))[BURN_IN:]
    )

    print(f"\n--- {label}: {pf.name} ---")
    print(f"xs_hard dtype: {xs_hard.dtype}  shape: {xs_hard.shape}")
    print(f"per-neuron std over window: {np.asarray(jnp.std(xs_hard, axis=0))}")

    A, Xa = build_A(xs_hard, RIDGE)
    cond = np.linalg.cond(np.asarray(A))
    eigvals = np.linalg.eigvalsh(np.asarray(A))
    print(f"A shape: {A.shape}  dtype: {A.dtype}")
    print(f"cond(A) = {cond:.3e}")
    print(f"eigvals(A): min={eigvals.min():.3e}  max={eigvals.max():.3e}")

    R, b = fit_readout(xs_hard, targets, RIDGE)
    print(f"R has NaN: {bool(jnp.any(jnp.isnan(R)))}   b has NaN: {bool(jnp.any(jnp.isnan(b)))}")


if __name__ == "__main__":
    print(f"jax_enable_x64: {jax.config.jax_enable_x64}")
    print(f"default dtype check: {jnp.zeros(1).dtype}")

    results_dir = Path("results")
    param_files = sorted(results_dir.glob("params_seed*_z*.npy"))

    # find one collapsed (true fixed-point) example and one healthy example
    collapsed_pf = None
    healthy_pf = None
    for pf in param_files:
        support = measured_fixedpoint_support(pf)
        if collapsed_pf is None and len(support) < 5:
            fp_check = hard_len1_is_genuine_fixedpoint(pf)
            if max(fp_check["tail_range_rel_to_scale"]) < 1e-3:
                collapsed_pf = pf
        if healthy_pf is None and len(support) == 5:
            healthy_pf = pf
        if collapsed_pf and healthy_pf:
            break

    # NOTE: adjust this to however `targets` is actually built/loaded in
    # n_cycle.py -- placeholder shape (N_STEPS - BURN_IN, m) of zeros won't
    # give a meaningful R/b, but the cond(A) check above is dtype/scale-only
    # and doesn't depend on targets at all, so it's valid regardless.
    dt, n_steps, burn_in = 0.1, 2000, 200
    t = jnp.arange(n_steps) * dt
    p2m = make_block_param_to_model(spec)
    W0, th0 = p2m(params0)
    xs0 = simulate(W0, th0, x0, dt=dt, n_steps=n_steps)[burn_in:]
    period = measure_period(xs0, dt)
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
    targets = targets[BURN_IN:]


    if healthy_pf:
        diagnose_one(healthy_pf, targets, "HEALTHY (full support)")
    if collapsed_pf:
        diagnose_one(collapsed_pf, targets, "COLLAPSED (true fixed point)")
    else:
        print("No true-fixed-point collapsed case found in this results_dir.")
