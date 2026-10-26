"""
audit_full_support_flatness.py

Gap this closes: everywhere in the existing pipeline (check_task_fidelity.py,
tally_graph_rules.py, diagnose_boundedness.py's hard_len1_is_genuine_fixedpoint),
the flatness/genuine-oscillation check only runs when measured_support_size < 5.
Any case with measured_support_size == 5 is auto-classified as
"full_support_oscillation" with NO verification that it's actually moving --
because theory guarantees no *stable* full-support fixed point exists, but says
nothing about an *unstable* one the simulation might sit on/near for the whole
evaluation window without being perturbed off it.

This script applies the SAME flatness test used elsewhere (max tail-range
relative to scale, across neurons, tail_frac=0.3, threshold 1e-3) to every
case currently classified as full_support_oscillation, and reports how many
of those 90-ish cases are actually flat under that criterion.

Standalone / non-invasive: does not modify any existing file, does not change
results_prefix_bug/ or any .npy/.json output.
"""

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from support_sequence_analysis import hardened_weights, spec, p2m, x0, parse_seed_zstd
from graph_rules_check import measured_fixedpoint_support
from oscillon.ctln.dynamics import simulate

BURN_IN = 200
REF_PERIOD = 2.0 * np.pi
dt = 0.1
N_STEPS = int(BURN_IN + 8 * (REF_PERIOD / dt))
TAIL_FRAC = 0.3
FLAT_THRESH = 1e-3

RESULTS_DIR = Path("results")  # adjust if your current results live elsewhere


def hard_trajectory(params_path):
    params = jnp.asarray(np.load(params_path))
    W_hard, theta_hard = hardened_weights(spec, params)
    xs = np.asarray(simulate(W_hard, theta_hard, x0, dt=dt, n_steps=N_STEPS))[BURN_IN:]
    return xs


def flatness_check(xs, tail_frac=TAIL_FRAC):
    """Same criterion as hard_len1_is_genuine_fixedpoint: max per-neuron
    tail range relative to overall scale. Returns (is_flat, max_rel_range,
    per_neuron_rel_range)."""
    tail = xs[-int(tail_frac * xs.shape[0]):]
    tail_range = tail.max(axis=0) - tail.min(axis=0)
    scale = np.abs(xs).max()
    rel_range = tail_range / max(scale, 1e-12)
    return bool(rel_range.max() < FLAT_THRESH), float(rel_range.max()), rel_range


if __name__ == "__main__":
    param_files = sorted(RESULTS_DIR.glob("params_seed*_z*.npy"))
    print(f"scanning {len(param_files)} param files in {RESULTS_DIR.resolve()}\n")

    full_support_cases = []
    for pf in param_files:
        support = measured_fixedpoint_support(pf)
        if len(support) == 5:
            full_support_cases.append(pf)

    print(f"{len(full_support_cases)} cases currently classified as "
          f"full_support_oscillation\n")
    print(f"{'seed/z':>14} | {'max_rel_range':>14} | {'flat (stuck)?':>13}")

    n_flat = 0
    flat_cases = []
    for pf in full_support_cases:
        seed, z_std = parse_seed_zstd(pf)
        xs = hard_trajectory(pf)
        is_flat, max_rel, _ = flatness_check(xs)
        flag = "  <-- STUCK" if is_flat else ""
        print(f"seed {seed:>3} z={z_std:.1f} | {max_rel:>14.3e} | {str(is_flat):>13}{flag}")
        if is_flat:
            n_flat += 1
            flat_cases.append((seed, z_std, max_rel))

    print("\n" + "=" * 70)
    print(f"Of {len(full_support_cases)} cases labeled full_support_oscillation, "
          f"{n_flat} are actually flat (stuck on an unstable full-support fixed "
          f"point, not genuinely oscillating) under the same threshold used "
          f"elsewhere in the pipeline (max tail-range/scale < {FLAT_THRESH}).")
    if flat_cases:
        print("\nStuck cases:")
        for seed, z_std, max_rel in flat_cases:
            print(f"  seed {seed:>3} z={z_std:.1f}  max_rel_range={max_rel:.3e}")
    print("=" * 70)
