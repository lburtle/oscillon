"""
check_task_fidelity_v2.py

Same question as check_task_fidelity.py (after hardening, is the network still
usable for the learned task?), with three fixes:

    1. Non-flat reduced-support cases are re-simulated at a 3x window before being
    called sub-cycles, since verify_mismatches_extended_v2.py showed slowly
    converging fixed point (6 of them in the post-fix sweep) fail the
    trained-window flatness test.
    2. Full-support cases that are frozen at x0 (bit-exact flat hardened
    trajectory, an x0-on-unstable-fixed-point artifact, see
    reverify_stuck_seeds.py) are split out, since their hard_ms reflects a
    constant trajectory, not oscillation.
    3. NaN-safe (nanmedian) with NaN counts reported.

PARAMS AND RECORDS MUST COME FROM THE SAME SWEEP: the readout fix changes ES
fitness, so params in results/ differ from results_before_bug_fix/.
"""

import json
from collections import defaultdict
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from diagnose_boundedness import hard_len1_is_genuine_fixedpoint
from graph_rules_check import measured_fixedpoint_support
from oscillon.dynamics import simulate
from support_sequence_analysis import hardened_weights, spec, x0, parse_seed_zstd

RESULTS_DIR = Path("/scratch/xfd3tf/oscillon/examples/results")
RECORDS_PATH = RESULTS_DIR / "uniform_pop512.json"

dt = 0.1
BURN_IN = 200
TRAINED_N_STEPS = int(BURN_IN + 8 * (2.0 * np.pi / dt))
EXT_MULT = 3
FLAT_THRESH = 1e-3
TAIL_FRAC = 0.3


def hard_xs(params, n_steps):
    W_hard, theta_hard = hardened_weights(spec, params)
    return np.asarray(simulate(W_hard, theta_hard, x0, dt=dt, n_steps=n_steps))[BURN_IN:]

def max_rel_range(xs):
    tail = xs[-int(TAIL_FRAC * xs.shape[0]):]
    rng = tail.max(axis=0) - tail.min(axis=0)
    return float(np.max(rng / max(np.abs(xs).max(), 1e-12)))

with open(RECORDS_PATH) as f:
    sweep_records = json.load(f)
by_key = {(r["seed"], r["z_std"]): r for r in sweep_records}

by_category = defaultdict(list)

for pf in sorted(RESULTS_DIR.glob("params_seed*_z*.npy")):
    seed, z_std = parse_seed_zstd(pf)
    rec = by_key.get((seed, z_std))
    if rec is None:
        print(f"WARNING: no record for seed {seed} z={z_std} -- skipping")
        continue
    
    params = jnp.asarray(np.load(pf))
    support = measured_fixedpoint_support(pf)

    if len(support) == 5:
        frozen = max_rel_range(hard_xs(params, TRAINED_N_STEPS)) < FLAT_THRESH
        category = "oscillation_frozen_at_x0" if frozen else "full_support_oscillation"
    else:
        fp_check = hard_len1_is_genuine_fixedpoint(pf)
        if max(fp_check["tail_range_rel_to_scale"]) < FLAT_THRESH:
            category = "true_fixedpoint_collapse"
        else:
            xs_ext = hard_xs(params, TRAINED_N_STEPS * EXT_MULT)
            category = ("true_fixedpoint_collapse"
                        if max_rel_range(xs_ext) < FLAT_THRESH else "subcycle_collapse")

    by_category[category].append({
        "seed": seed, "z_std": z_std,
        "soft_mse": rec["soft_mse"],
        "hard_mse": rec["hard_mse"],
        "hard_dyn_divergence": rec["hard_dyn_divergence"],
    })

def summarize(label, rows):
    soft = np.array([r["soft_mse"] for r in rows], dtype=float)
    hard = np.array([r["hard_mse"] for r in rows], dtype=float)
    div = np.array([r["hard_dyn_divergence"] for r in rows], dtype=float)
    n_nan = int(np.isnan(hard).sum())
    ms, mh, md = np.nanmedian(soft), np.nanmedian(hard), np.nanmedian(div)
    ratio = mh / ms if ms > 0 else float("inf")
    print(f"{label:>34} | {len(rows):>3} | {n_nan:>5} | {ms:>9.4f} | {mh:>9.4f} | "
              f"{md:>9.4f} | {ratio:>8.2f}")


print(f"{'category':>34} | {'n':>3} | {'n_nan':>5} | {'soft_mse':>9} | "
      f"{'hard_mse':>9} | {'diverg.':>9} | {'h/s':>8}   (medians)")

for cat in ["full_support_oscillation", "oscillation_frozen_at_x0",
            "true_fixedpoint_collapse", "subcycle_collapse"]:
    if cat in by_category:
        summarize(cat, by_category[cat])

# the full-support oscillation bucket as counted in the headline (genuine + frozen-at-x0)
if "full_support_oscillation" in by_category:
    summarize("ALL full-support (headline bucket)",
              by_category["full_support_oscillation"]
              + by_category.get("oscillation_frozen_at_x0", []))

print("\ncategory counts:", {k: len(v) for k, v in by_category.items()})

