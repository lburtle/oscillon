"""
build_continuous_trace_figure_v2.py

Same continuous-amplitude visualization as build_continuous_trace_figure.py,
but fixes the example-selection methodology:

  - Uses your ACTUAL validated pipeline functions (measured_fixedpoint_support
    from graph_rules_check.py, hard_len1_is_genuine_fixedpoint from
    diagnose_boundedness.py) instead of a hand-rolled classify(), so category
    assignment is guaranteed consistent with everything else in the audit.

  - Rather than taking the first match per category, scans ALL candidates and
    picks the one that best represents its category's cluster:
      * true_fixedpoint_collapse: SMALLEST max(tail_range_rel_to_scale)
        (deepest in the flat cluster -- cleanest possible fixed-point example)
      * subcycle_collapse: LARGEST max(tail_range_rel_to_scale)
        (deepest in the oscillating cluster -- should show clearly visible
        amplitude banding, unlike the marginal case picked last time)
      * full_support_oscillation: first non-stuck case (this category was
        already uniformly ~0.95-1.0 per the earlier audit, so no ranking
        needed here)

Standalone / non-invasive: reads existing params only, writes only the two
figure files.
"""

from pathlib import Path

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from support_sequence_analysis import hardened_weights, spec, p2m, x0, parse_seed_zstd
from graph_rules_check import measured_fixedpoint_support
from diagnose_boundedness import hard_len1_is_genuine_fixedpoint
from oscillon.ctln.dynamics import simulate

BURN_IN = 200
REF_PERIOD = 2.0 * np.pi
dt = 0.1
N_STEPS = int(BURN_IN + 8 * (REF_PERIOD / dt))

RESULTS_DIR = Path("results")  # adjust to match your current results dir

STUCK_SEEDS = {(10,1.0),(12,3.0),(15,3.0),(19,2.0),(24,1.0),(25,1.0),(25,2.0),(27,3.0),
       (2,1.0),(30,1.0),(34,2.0),(37,1.0),(37,3.0),(38,3.0),(47,3.0),(49,3.0)}


def hard_trajectory(params):
    W_hard, theta_hard = hardened_weights(spec, params)
    xs = np.asarray(simulate(W_hard, theta_hard, x0, dt=dt, n_steps=N_STEPS))[BURN_IN:]
    return xs


def soft_trajectory(params):
    W_soft, theta_soft = p2m(params)  # adjust if p2m's signature differs
    xs = np.asarray(simulate(W_soft, theta_soft, x0, dt=dt, n_steps=N_STEPS))[BURN_IN:]
    return xs


if __name__ == "__main__":
    param_files = sorted(RESULTS_DIR.glob("params_seed*_z*.npy"))

    osc_candidate = None
    fp_candidates = []       # (max_rel, seed, z_std, params)
    subcycle_candidates = []  # (max_rel, seed, z_std, params)

    for pf in param_files:
        seed, z_std = parse_seed_zstd(pf)
        params = jnp.asarray(np.load(pf))
        support = measured_fixedpoint_support(pf)

        if len(support) == 5:
            if osc_candidate is None and (seed, z_std) not in STUCK_SEEDS:
                osc_candidate = (seed, z_std, params)
            continue

        fp_check = hard_len1_is_genuine_fixedpoint(pf)
        max_rel = max(fp_check["tail_range_rel_to_scale"])
        is_flat = max_rel < 1e-3

        if is_flat:
            fp_candidates.append((max_rel, seed, z_std, params))
        else:
            subcycle_candidates.append((max_rel, seed, z_std, params))

    if osc_candidate is None or not fp_candidates or not subcycle_candidates:
        raise RuntimeError("couldn't find a candidate for one or more categories -- "
                            f"osc={osc_candidate is not None} "
                            f"fp={len(fp_candidates)} subcycle={len(subcycle_candidates)}")

    fp_candidates.sort(key=lambda t: t[0])          # smallest max_rel first -- cleanest flat case
    subcycle_candidates.sort(key=lambda t: -t[0])    # largest max_rel first -- clearest oscillation

    examples = {
        "full_support_oscillation": osc_candidate,
        "true_fixedpoint_collapse": fp_candidates[0][1:],
        "subcycle_collapse": subcycle_candidates[0][1:],
    }

    print("Selected examples:")
    print(f"  full_support_oscillation: seed {osc_candidate[0]} z={osc_candidate[1]}")
    print(f"  true_fixedpoint_collapse: seed {fp_candidates[0][1]} z={fp_candidates[0][2]} "
          f"(max_rel={fp_candidates[0][0]:.2e})")
    print(f"  subcycle_collapse: seed {subcycle_candidates[0][1]} z={subcycle_candidates[0][2]} "
          f"(max_rel={subcycle_candidates[0][0]:.3f})")

    col_titles = {
        "full_support_oscillation": "Sustained oscillation",
        "true_fixedpoint_collapse": "Fixed-point collapse",
        "subcycle_collapse": "Sub-cycle collapse",
    }
    col_order = ["full_support_oscillation", "true_fixedpoint_collapse", "subcycle_collapse"]

    fig, axes = plt.subplots(2, 3, figsize=(9, 3.6), sharex=True)

    for col, cat in enumerate(col_order):
        seed, z_std, params = examples[cat]
        xs_soft = soft_trajectory(params)
        xs_hard = hard_trajectory(params)

        for row, (xs, row_label) in enumerate([(xs_soft, "Trained"), (xs_hard, "Hardened")]):
            mat = np.clip(xs.T, 0, None)  # threshold-linear: negative part not informative
            vmax = mat.max() if mat.max() > 0 else 1.0

            ax = axes[row, col]
            ax.imshow(mat, aspect="auto", cmap="Greys", vmin=0, vmax=vmax,
                      interpolation="nearest")
            ax.set_yticks(range(mat.shape[0]))
            ax.set_yticklabels([str(i) for i in range(mat.shape[0])], fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{row_label}\nneuron", fontsize=8)
            if row == 0:
                ax.set_title(f"{col_titles[cat]}\n(seed {seed}, z={z_std})", fontsize=8)
            if row == 1:
                ax.set_xlabel("time step", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig("continuous_trace_figure_v2.pdf", bbox_inches="tight")
    fig.savefig("continuous_trace_figure_v2.png", bbox_inches="tight", dpi=200)
    print("\nwrote continuous_trace_figure_v2.pdf and continuous_trace_figure_v2.png")
