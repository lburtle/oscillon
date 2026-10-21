"""
build_continuous_trace_figure.py

Same 2x3 layout as build_support_raster_figure.py (rows = soft/hardened,
columns = the 3 outcome categories), but shows the raw continuous trace
amplitude as a grayscale intensity raster instead of a hard on/off threshold.
This is meant to reveal periodic amplitude variation that stays entirely
above active_support's activation threshold (e.g. the sub-cycle case, whose
binary raster looked like a static fixed point even though it's genuinely
oscillating in magnitude) -- see the discussion that led here.

Each panel is normalized to its OWN max (vmin=0, vmax=that panel's max),
since soft and hardened amplitudes can differ substantially in scale and a
single global scale would wash out the smaller one.

Standalone / non-invasive: reads existing params only, writes only the two
figure files. Uses the SAME example-selection logic as the binary version
(same seeds), so this is a direct point of comparison, not a different set
of examples.
"""

from pathlib import Path

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from support_sequence_analysis import hardened_weights, spec, p2m, x0, parse_seed_zstd
from oscillon.ctln.dynamics import simulate

BURN_IN = 200
REF_PERIOD = 2.0 * np.pi
dt = 0.1
N_STEPS = int(BURN_IN + 8 * (REF_PERIOD / dt))
TAIL_FRAC = 0.3
FLAT_THRESH = 1e-3

RESULTS_DIR = Path("results_before_bug_fix")  # adjust to match your current results dir

STUCK_SEEDS = {
    (10, 1.0), (10, 3.0), (13, 2.0), (14, 1.0), (1, 1.0), (1, 3.0),
    (24, 3.0), (25, 2.0), (25, 3.0), (27, 1.0), (37, 1.0), (37, 2.0),
    (39, 2.0), (40, 2.0), (41, 1.0),
}


def classify(xs, tail_frac=TAIL_FRAC, flat_thresh=FLAT_THRESH):
    tail = xs[-int(tail_frac * xs.shape[0]):]
    tail_mean = tail.mean(axis=0)
    scale = np.abs(xs).max()
    support = tuple(i for i in range(len(tail_mean)) if tail_mean[i] > 0.01 * scale)

    tail_range = tail.max(axis=0) - tail.min(axis=0)
    rel_range = tail_range / max(scale, 1e-12)
    is_flat = bool(rel_range.max() < flat_thresh)

    if len(support) == 5:
        category = "full_support_oscillation" if not is_flat else "STUCK"
    elif is_flat:
        category = "true_fixedpoint_collapse"
    else:
        category = "subcycle_collapse"
    return category


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

    examples = {}
    for pf in param_files:
        seed, z_std = parse_seed_zstd(pf)
        if "full_support_oscillation" not in examples and (seed, z_std) in STUCK_SEEDS:
            continue

        params = jnp.asarray(np.load(pf))
        xs_hard = hard_trajectory(params)
        category = classify(xs_hard)

        if category in ("full_support_oscillation", "true_fixedpoint_collapse",
                         "subcycle_collapse") and category not in examples:
            examples[category] = (seed, z_std, params)

        if len(examples) == 3:
            break

    missing = {"full_support_oscillation", "true_fixedpoint_collapse", "subcycle_collapse"} - examples.keys()
    if missing:
        raise RuntimeError(f"couldn't find an example for: {missing}")

    col_titles = {
        "full_support_oscillation": "Sustained oscillation\n(preserved, 60.0%)",
        "true_fixedpoint_collapse": "Fixed-point collapse\n(theory-matched, 30.7%)",
        "subcycle_collapse": "Sub-cycle collapse\n(9.3%)",
    }
    col_order = ["full_support_oscillation", "true_fixedpoint_collapse", "subcycle_collapse"]

    fig, axes = plt.subplots(2, 3, figsize=(9, 3.6), sharex=True)

    for col, cat in enumerate(col_order):
        seed, z_std, params = examples[cat]
        xs_soft = soft_trajectory(params)
        xs_hard = hard_trajectory(params)

        for row, (xs, row_label) in enumerate([(xs_soft, "Soft"), (xs_hard, "Hardened")]):
            mat = xs.T  # (n_neurons, T), raw continuous amplitude
            mat = np.clip(mat, 0, None)  # threshold-linear: negative part is clipped, not informative
            vmax = mat.max() if mat.max() > 0 else 1.0

            ax = axes[row, col]
            im = ax.imshow(mat, aspect="auto", cmap="Greys", vmin=0, vmax=vmax,
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

    fig.suptitle("Continuous activity: soft vs. hardened dynamics", fontsize=10, y=1.04)
    fig.tight_layout()
    fig.savefig("continuous_trace_figure.pdf", bbox_inches="tight")
    fig.savefig("continuous_trace_figure.png", bbox_inches="tight", dpi=200)
    print("wrote continuous_trace_figure.pdf and continuous_trace_figure.png")
    for cat in col_order:
        seed, z_std, _ = examples[cat]
        print(f"  {cat}: seed {seed} z={z_std}")
