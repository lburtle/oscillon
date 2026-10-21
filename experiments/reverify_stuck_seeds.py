"""
reverify_stuck_seeds.py

For the 15 seeds flagged by audit_full_support_flatness.py as exactly frozen
(max_rel_range == 0.0) despite being classified full_support_oscillation:

  1. Confirms the mechanism: checks whether theory (analyze_graph_rules)
     predicts the full-support fixed point is unstable for each of these --
     if so, that's consistent with "x0 happens to sit exactly on a real but
     UNSTABLE fixed point, so the deterministic simulation never leaves it."

  2. Re-simulates each with x0 perturbed by a small random kick (3 independent
     trials per seed, different random directions) and re-classifies the
     resulting trajectory using the SAME support/flatness logic used
     throughout the rest of the pipeline. Reports the corrected category.

Standalone / non-invasive: reads existing params only, writes nothing back,
does not alter results_prefix_bug/ or any existing script's behavior.
"""

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from support_sequence_analysis import hardened_weights, spec, p2m, x0, parse_seed_zstd
from graph_rules_check import analyze_graph_rules, measured_fixedpoint_support
from oscillon.ctln.dynamics import simulate

BURN_IN = 200
REF_PERIOD = 2.0 * np.pi
dt = 0.1
N_STEPS = int(BURN_IN + 8 * (REF_PERIOD / dt))
TAIL_FRAC = 0.3
FLAT_THRESH = 1e-3
PERTURB_SCALE = 1e-4
N_TRIALS = 3

RESULTS_DIR = Path("results")  # adjust to match your current results dir

STUCK_SEEDS = [
    (10,1.0),(12,3.0),(15,3.0),(19,2.0),(24,1.0),(25,1.0),(25,2.0),(27,3.0),
    (2,1.0),(30,1.0),(34,2.0),(37,1.0),(37,3.0),(38,3.0),(47,3.0),(49,3.0)
]


def classify(xs, tail_frac=TAIL_FRAC, flat_thresh=FLAT_THRESH):
    """Same support/flatness logic used throughout the rest of the pipeline."""
    tail = xs[-int(tail_frac * xs.shape[0]):]
    tail_mean = tail.mean(axis=0)
    scale = np.abs(xs).max()
    support = tuple(i for i in range(len(tail_mean)) if tail_mean[i] > 0.01 * scale)

    tail_range = tail.max(axis=0) - tail.min(axis=0)
    rel_range = tail_range / max(scale, 1e-12)
    is_flat = bool(rel_range.max() < flat_thresh)

    if len(support) == 5:
        category = "full_support_oscillation" if not is_flat else "full_support_STILL_STUCK"
    elif is_flat:
        category = "true_fixedpoint_collapse"
    else:
        category = "subcycle_collapse"
    return category, len(support), float(rel_range.max())


def find_param_file(seed, z_std):
    candidates = list(RESULTS_DIR.glob(f"params_seed{seed}_z{z_std}.npy"))
    if not candidates:
        raise FileNotFoundError(f"no params file for seed {seed} z={z_std} in {RESULTS_DIR}")
    return candidates[0]


if __name__ == "__main__":
    print("=" * 78)
    print("STEP 1: theory prediction for the full-support fixed point")
    print("=" * 78)
    theory_results = {}
    for seed, z_std in STUCK_SEEDS:
        pf = find_param_file(seed, z_std)
        gr = analyze_graph_rules(pf)
        theory_results[(seed, z_std)] = gr
        print(f"seed {seed:>3} z={z_std:.1f} | full_support_has_fixed_point="
              f"{gr['full_support_has_fixed_point']!s:5} | full_support_stable="
              f"{gr['full_support_stable']!s:5} | n_stable_fps={gr['n_stable_fixed_points']}")

    n_consistent = sum(
        1 for r in theory_results.values()
        if r["full_support_has_fixed_point"] and r["full_support_stable"] is False
    )
    print(f"\n{n_consistent}/{len(STUCK_SEEDS)} cases match the hypothesis exactly "
          f"(a real, theory-confirmed UNSTABLE full-support fixed point that x0 "
          f"happens to sit on).")

    print("\n" + "=" * 78)
    print("STEP 2: re-simulate with perturbed x0, re-classify")
    print("=" * 78)

    rng = np.random.default_rng(0)
    n = len(np.asarray(x0))

    for seed, z_std in STUCK_SEEDS:
        pf = find_param_file(seed, z_std)
        params = jnp.asarray(np.load(pf))
        W_hard, theta_hard = hardened_weights(spec, params)

        print(f"\nseed {seed:>3} z={z_std:.1f}:")
        outcomes = []
        for trial in range(N_TRIALS):
            direction = rng.normal(size=n)
            direction /= np.linalg.norm(direction)
            x0_perturbed = jnp.asarray(np.asarray(x0) + PERTURB_SCALE * direction)

            xs = np.asarray(
                simulate(W_hard, theta_hard, x0_perturbed, dt=dt, n_steps=N_STEPS)
            )[BURN_IN:]
            category, supp_size, max_rel = classify(xs)
            outcomes.append(category)
            print(f"  trial {trial}: support_size={supp_size}  max_rel_range={max_rel:.3e}"
                  f"  -> {category}")

        agree = len(set(outcomes)) == 1
        print(f"  trials agree: {agree}  ->  corrected category: "
              f"{outcomes[0] if agree else 'DISAGREEMENT -- ' + str(outcomes)}")
