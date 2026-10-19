"""
verify_mismatches_extended.py

Standalone check: does NOT modify n_cycle.py, support_sequence_analysis.py,
diagnose_boundedness.py, or graph_rules_check.py, and does not change their
N_STEPS (the trained/evaluated window, ~702 steps). Those results stand as
they are.

This script separately re-simulates ONLY the 7 flagged partial-mismatch
cases from the post-fix sweep (results/) at a longer window, to check whether they were simply still settling
(not yet flat within the training window) rather than a genuine theory
disagreement. It also re-checks a small control sample of already-CLEAN
cases at the same longer window, so a shift in the mismatch cases isn't
mistaken for progress if the "clean" cases would have shifted too.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from oscillon.dynamics import simulate
from support_sequence_analysis import hardened_weights, spec, p2m, x0, parse_seed_zstd
from graph_rules_check import analyze_graph_rules

dt = 0.1
BURN_IN = 200  # same burn-in as the trained window; only the total length changes

RESULTS_DIR = "results"  # post-fix sweep

# --- the 7 flagged partial mismatches from the post-fix tally: theory predicts
#     a stable fixed point with the measured support size, but the tail was not
#     flat (<1e-3) in the trained window. Six have support_size=2, one (31, 3.0)
#     has support_size=4 ---
MISMATCH_SEEDS = [
    (14, 2.0), (28, 3.0), (33, 1.0), (41, 1.0), (42, 1.0), (4, 1.0), (31, 3.0),
]

# --- a small control sample of already-CLEAN cases (size=5 sustained
#     oscillation and size=2 genuinely-flat), to confirm the extended
#     window doesn't ALSO reclassify things that were already fine ---
CONTROL_SEEDS = [
    (0, 1.0),   # size=2, flat=True in the post-fix run
    (0, 2.0),   # size=5, genuine oscillation (not one of the 16 frozen-at-x0 seeds)
    (35, 2.0),  # size=2, flat=True
    (11, 1.0),  # size=5, genuine oscillation (not frozen at x0)
]


def measured_support_extended(params_path, n_steps, tail_frac=0.3, thresh_frac=0.01):
    params = jnp.asarray(np.load(params_path))
    W_hard, theta_hard = hardened_weights(spec, params)
    xs = np.asarray(simulate(W_hard, theta_hard, x0, dt=dt, n_steps=n_steps))[BURN_IN:]
    tail = xs[-int(tail_frac * xs.shape[0]):]
    tail_mean = tail.mean(axis=0)
    scale = np.abs(xs).max()
    support = tuple(i for i in range(len(tail_mean)) if tail_mean[i] > thresh_frac * scale)
    return support, xs


def is_flat_extended(xs, tail_frac=0.3, flat_thresh=1e-3):
    tail = xs[-int(tail_frac * xs.shape[0]):]
    tail_range = tail.max(axis=0) - tail.min(axis=0)
    overall_scale = np.abs(xs).max()
    return bool(max(tail_range / max(overall_scale, 1e-9)) < flat_thresh)


def check_one(seed, z_std, n_steps, results_dir=RESULTS_DIR):
    pf = Path(results_dir) / f"params_seed{seed}_z{z_std}.npy"
    support, xs = measured_support_extended(pf, n_steps)
    flat = is_flat_extended(xs) if len(support) < 5 else None
    gr = analyze_graph_rules(pf)  # theory prediction is window-independent -- unchanged
    return {
        "seed": seed, "z_std": z_std, "n_steps": n_steps,
        "support_size": len(support), "flat": flat,
        "n_stable_fps": gr["n_stable_fixed_points"],
    }


if __name__ == "__main__":
    TRAINED_N_STEPS = int(BURN_IN + 8 * (2.0 * np.pi / dt))  # ~702, reference only
    EXTENDED_MULTIPLIERS = [3, 6, 10]  # try progressively longer windows

    print("=" * 70)
    print("MISMATCH CASES -- does more time resolve them?")
    print("=" * 70)
    for seed, z_std in MISMATCH_SEEDS:
        print(f"\nseed {seed} z={z_std}:")
        for mult in EXTENDED_MULTIPLIERS:
            r = check_one(seed, z_std, TRAINED_N_STEPS * mult)
            print(f"  {mult}x window ({r['n_steps']:5d} steps): "
                  f"support_size={r['support_size']} flat={r['flat']} "
                  f"n_stable_fps={r['n_stable_fps']}")

    print("\n" + "=" * 70)
    print("CONTROL CASES -- do already-clean results hold at longer windows?")
    print("=" * 70)
    for seed, z_std in CONTROL_SEEDS:
        print(f"\nseed {seed} z={z_std}:")
        for mult in EXTENDED_MULTIPLIERS:
            r = check_one(seed, z_std, TRAINED_N_STEPS * mult)
            print(f"  {mult}x window ({r['n_steps']:5d} steps): "
                  f"support_size={r['support_size']} flat={r['flat']} "
                  f"n_stable_fps={r['n_stable_fps']}")
