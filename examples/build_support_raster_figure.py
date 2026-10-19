"""
build_support_raster_figure.py

Builds the Cosyne abstract figure: active-support raster (soft vs. hardened)
for one representative example per outcome category --

  1. Sustained oscillation preserved (the 60% case)
  2. Genuine fixed-point collapse, theory-matched (the 30.7% case)
  3. Sub-cycle collapse (the 9.3% case)

Selects examples dynamically by scanning params_seed*_z*.npy and applying the
same classification logic validated throughout this session's audit (support
size + flatness check), so it doesn't rely on hardcoded seed numbers that
might not match your actual current results directory. Explicitly excludes
the 15 seeds known to sit exactly on an unstable full-support equilibrium
(see reverify_stuck_seeds.py) when picking the oscillation example, since
those are flat by simulation-artifact, not representative.

Output: support_raster_figure.pdf (vector, for LaTeX embedding) and a .png
preview at the same resolution.

Standalone / non-invasive: reads existing params only, writes only the two
figure files.
"""

from pathlib import Path

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from support_sequence_analysis import (
    hardened_weights, spec, p2m, x0, parse_seed_zstd, active_support,
)
from oscillon.dynamics import simulate

BURN_IN = 200
REF_PERIOD = 2.0 * np.pi
dt = 0.1
N_STEPS = int(BURN_IN + 8 * (REF_PERIOD / dt))
TAIL_FRAC = 0.3
FLAT_THRESH = 1e-3
REL_THRESH = 0.02  # active_support threshold, matching the validated setting

RESULTS_DIR = Path("results")  # adjust to match your current results dir

# seeds confirmed sitting on an unstable full-support equilibrium at this x0 --
# excluded when picking the oscillation example (see reverify_stuck_seeds.py)
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
    W_soft, theta_soft = p2m(params)
    xs = np.asarray(simulate(W_soft, theta_soft, x0, dt=dt, n_steps=N_STEPS))[BURN_IN:]
    return xs


def support_matrix(xs, rel_thresh=REL_THRESH):
    """active_support returns a list of tuples (bool per neuron per timestep);
    stack into a (T, n) boolean array, then transpose to (n, T) for the raster."""
    sigma = active_support(xs, rel_thresh=rel_thresh)
    return np.asarray(sigma, dtype=bool).T  # (n_neurons, T)


if __name__ == "__main__":
    param_files = sorted(RESULTS_DIR.glob("params_seed*_z*.npy"))

    examples = {}  # category -> (seed, z_std, params)
    for pf in param_files:
        seed, z_std = parse_seed_zstd(pf)
        if "full_support_oscillation" not in examples and (seed, z_std) in STUCK_SEEDS:
            continue  # skip known-stuck seeds entirely when looking for the osc. example

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
        raise RuntimeError(f"couldn't find an example for: {missing} -- "
                            f"scanned {len(param_files)} files, check RESULTS_DIR")

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

        sigma_soft = support_matrix(xs_soft)
        sigma_hard = support_matrix(xs_hard)

        for row, (sigma, row_label) in enumerate([(sigma_soft, "Soft"), (sigma_hard, "Hardened")]):
            ax = axes[row, col]
            ax.imshow(sigma, aspect="auto", cmap="Greys", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_yticks(range(sigma.shape[0]))
            ax.set_yticklabels([str(i) for i in range(sigma.shape[0])], fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{row_label}\nneuron", fontsize=8)
            if row == 0:
                ax.set_title(f"{col_titles[cat]}\n(seed {seed}, z={z_std})", fontsize=8)
            if row == 1:
                ax.set_xlabel("time step", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle("Active-support raster: soft vs. hardened dynamics", fontsize=10, y=1.04)
    fig.tight_layout()
    fig.savefig("support_raster_figure.pdf", bbox_inches="tight")
    fig.savefig("support_raster_figure.png", bbox_inches="tight", dpi=200)
    print("wrote support_raster_figure.pdf and support_raster_figure.png")
    for cat in col_order:
        seed, z_std, _ = examples[cat]
        print(f"  {cat}: seed {seed} z={z_std}")
