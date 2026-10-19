"""
hardening_weight_sweep.py

Robustness of the audit to the two canonical weight values. The hardened graph (gate > 0.5) is fixed; edge weights
are set to -1+eps (edge present) and -1-delta (absent) for a grid of (eps, delta) that satisfy the legal-CTLN
condition 0 < eps < delta/(delta+1). The drive theta is kept as trained (one shared scalar). For each setting:
outcome split (five-node oscillation / fixed point / subset oscillation; frozen-at-x0 cases rescued from small
perturbations as elsewhere) and the tie-aware directed-cycle test (ties within 2% of the period) with its
graph-specific chance level.

Standalone / non-invasive. Run from the folder with results/, next to the other scripts.
"""
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from check_order_ties import compatible, group_phases, null_fraction_groups
from check_order_vs_hamiltonian import edge_matrix
from compare_order_cutoffs import BURN_IN, RESCUE_SCALE, RESCUE_TRIES, WINDOW, analyze, dt, is_flat
from oscillon.dynamics import simulate
from support_sequence_analysis import hardened_weights, spec, x0

RESULTS_DIR = Path("results")
GRID = [(0.05, 0.25), (0.10, 0.25), (0.05, 0.5), (0.10, 0.5), (0.20, 0.5),
        (0.05, 1.0), (0.10, 1.0), (0.20, 1.0), (0.05, 2.0), (0.10, 2.0), (0.20, 2.0)]
FRAC, TOL = 0.05, 0.02


def run(W, th, x0_in=x0):
    return np.asarray(simulate(W, th, x0_in, dt=dt, n_steps=BURN_IN + WINDOW))[BURN_IN:]


if __name__ == "__main__":
    files = sorted(RESULTS_DIR.glob("params_seed*_z*.npy"))
    base = []
    for pf in files:
        params = jnp.asarray(np.load(pf))
        W0, th = hardened_weights(spec, params)
        W0 = np.asarray(W0)
        n = W0.shape[0]
        present = (W0 > -1.2) & ~np.eye(n, dtype=bool)      # eps=0.1 -> -0.9 (present), delta=0.5 -> -1.5 (absent)
        base.append((present, th))
    print("=" * 78)
    for eps, delta in GRID:
        legal = eps < delta / (delta + 1)
        rng = np.random.default_rng(0)
        counts = {"five-node": 0, "fixed point": 0, "subset": 0}
        n_cyc = hit = rescued = 0
        exp = 0.0
        for present, th in base:
            n = present.shape[0]
            W = np.where(present, -1.0 + eps, -1.0 - delta)
            np.fill_diagonal(W, 0.0)
            W = jnp.asarray(W)
            xs = run(W, th)
            if is_flat(xs):
                ok = False
                for _ in range(RESCUE_TRIES):
                    xt = run(W, th, x0 + RESCUE_SCALE * jnp.asarray(rng.standard_normal(np.shape(x0))))
                    if not is_flat(xt):
                        xs, ok = xt, True
                        break
                if not ok:
                    counts["fixed point"] += 1
                    continue
                rescued += 1
            a = analyze(xs, FRAC)
            if a is None:
                continue
            phases = a[1]
            counts["five-node" if len(phases) == 5 else "subset"] += 1
            if len(phases) >= 3:
                E = edge_matrix(np.asarray(W))
                groups = group_phases(phases, TOL)
                hit += compatible(groups, E)
                exp += null_fraction_groups(sorted(phases), [len(g) for g in groups], E)
                n_cyc += 1
        print(f"eps={eps:.2f} delta={delta:.2f} (legal: {legal}): split {counts['five-node']}/{counts['fixed point']}/"
              f"{counts['subset']} (five-node/fixed point/subset; rescued {rescued});  "
              f"graph cycle {hit}/{n_cyc} (chance {exp:.1f})")
    print("=" * 78)
