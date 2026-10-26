"""
period_tuning.py

Reviewer question: can a LEGAL CTLN on the same graph, with eps and delta tuned, reproduce the trained period? (theta only
scales amplitude -- the dynamics are homogeneous in theta -- so eps and delta are the only legal parameters that can
move the period.) If yes, the 1.19 period ratio is a property of the canonical (0.10, 0.50) choice, not evidence that the
trained values add timing beyond the topology. If no, legal CTLNs on this graph cannot match the trained timing.

For each of the 107 oscillators (hardened graph, gate > 0.5; rescued from frozen-at-x0 as elsewhere), for every legal
(eps, delta) on a grid (0 < eps < delta/(delta+1)), the hardened network is simulated and its fundamental period measured
(peak-to-peak of the highest-variance neuron, as in readout_refit_period_matched.py), then compared with the trained
network's fundamental period. Reports: canonical within 5%; best grid point per network within 5% / 10%; the single
(eps, delta) pair that matches the most networks; and how often the best-matching network still oscillates.

Standalone / non-invasive. Run from the folder with results/, next to the other scripts.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from compare_order_cutoffs import RESCUE_SCALE, RESCUE_TRIES, dt, is_flat
from readout_refit_period_matched import BURN_LONG, N_LONG, period_est, sim
from support_sequence_analysis import hardened_weights, p2m, spec, x0

RESULTS_DIR = Path("results")
EPS_GRID = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
DELTA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
ROWS, COLS, _, _, W_CROSS = spec.compile_indices()
ROWS, COLS = jnp.asarray(ROWS), jnp.asarray(COLS)
N_EDGES = spec.n_edges


def build_W(present, eps, delta):
    w_off = np.where(present, -1.0 + eps, -1.0 - delta)
    return jnp.asarray(W_CROSS).at[ROWS, COLS].set(jnp.asarray(w_off))


def period_of(W, th, rng):
    w = sim(W, th, x0, N_LONG)[BURN_LONG:]
    if is_flat(w):
        for _ in range(RESCUE_TRIES):
            w2 = sim(W, th, x0 + RESCUE_SCALE * jnp.asarray(rng.standard_normal(np.shape(x0))), N_LONG)[BURN_LONG:]
            if not is_flat(w2):
                return period_est(w2)
        return None
    return period_est(w)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    legal = [(e, d) for d in DELTA_GRID for e in EPS_GRID if e < d / (d + 1)]
    nets = []
    for pf in sorted(RESULTS_DIR.glob("params_seed*_z*.npy")):
        params = jnp.asarray(np.load(pf))
        W, th = hardened_weights(spec, params)
        Ws, ths = p2m(params)
        T_tr = period_est(sim(Ws, ths, x0, N_LONG)[BURN_LONG:])
        T_h = period_of(W, th, rng)
        if T_h is None or T_tr is None:
            continue
        present = np.asarray(jax.nn.sigmoid(params[:N_EDGES])) > 0.5
        nets.append((pf.stem, present, th, T_tr, T_h))
    print("=" * 78)
    print(f"{len(nets)} oscillators; legal (eps, delta) grid points: {len(legal)}")
    ratios = np.array([t_h / t_tr for _, _, _, t_tr, t_h in nets])
    print(f"canonical (0.10, 0.50): median ratio {np.median(ratios):.2f}; within 5%: {int((abs(ratios - 1) < .05).sum())}/{len(nets)}")

    table = np.full((len(nets), len(legal)), np.nan)
    for i, (name, present, th, T_tr, T_h) in enumerate(nets):
        for j, (e, d) in enumerate(legal):
            T = period_of(build_W(present, e, d), th, rng)
            if T is not None:
                table[i, j] = T / T_tr
    dev = np.abs(table - 1)
    best = np.nanmin(np.where(np.isnan(dev), np.inf, dev), axis=1)
    print(f"best legal grid point per network: within 5% in {int((best < .05).sum())}/{len(nets)}, "
          f"within 10% in {int((best < .10).sum())}/{len(nets)}; median best |ratio - 1| {np.median(best):.3f}")
    counts = [(int(np.sum(dev[:, j] < .05)), legal[j]) for j in range(len(legal))]
    counts.sort(reverse=True)
    print("single (eps, delta) matching the most networks within 5%: "
          + "; ".join(f"{c}/{len(nets)} at {p}" for c, p in counts[:5]))
    med = [(float(np.nanmedian(table[:, j])), legal[j]) for j in range(len(legal))]
    print("median period ratio by (eps, delta): " + "; ".join(f"{p}: {m:.2f}" for m, p in med))
    print("=" * 78)
