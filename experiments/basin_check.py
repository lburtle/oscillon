"""
basin_check.py

Initial-condition robustness of the classification. Every hardened network is simulated from its default start
plus N_LOCAL local starts (x0 + 1e-2 N(0,1)) and N_BROAD broad starts (theta * U(0,1) per neuron). Per network:
  * outcome class at each start: fixed point / five-node oscillation / subset oscillation,
  * for oscillating starts, whether the tie-aware firing order (ties within 2% of the period) follows a directed
    cycle of the hardened graph.
Reports how many networks keep one outcome class across all starts, how many keep one cycle status, and lists
the ones that do not (with the fraction of starts in each class).

Standalone / non-invasive. Run from the folder with results/, next to the other scripts. Takes a few minutes.
"""
from collections import Counter
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from check_order_ties import compatible, group_phases
from check_order_vs_hamiltonian import edge_matrix
from compare_order_cutoffs import BURN_IN, WINDOW, analyze, dt, is_flat
from oscillon.ctln.dynamics import simulate
from support_sequence_analysis import hardened_weights, spec, x0

RESULTS_DIR = Path("results")
N_LOCAL, N_BROAD = 20, 20
FRAC, TOL = 0.05, 0.02


def classify(W, th, E, x_start):
    xs = np.asarray(simulate(W, th, x_start, dt=dt, n_steps=BURN_IN + WINDOW))[BURN_IN:]
    if is_flat(xs):
        return "fixed point", None
    a = analyze(xs, FRAC)
    if a is None:
        return "unresolved", None
    phases = a[1]
    cls = "five-node" if len(phases) == 5 else "subset"
    if len(phases) < 3:
        return cls, None
    return cls, bool(compatible(group_phases(phases, TOL), E))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows = []
    for pf in sorted(RESULTS_DIR.glob("params_seed*_z*.npy")):
        params = jnp.asarray(np.load(pf))
        W, th = hardened_weights(spec, params)
        E = edge_matrix(np.asarray(W))
        theta = float(np.asarray(th)[0])
        starts = [("default", x0)]
        starts += [("local", x0 + 1e-2 * jnp.asarray(rng.standard_normal(np.shape(x0)))) for _ in range(N_LOCAL)]
        starts += [("broad", theta * jnp.asarray(rng.uniform(0, 1, np.shape(x0)))) for _ in range(N_BROAD)]
        res = [classify(W, th, E, s) for _, s in starts]
        rows.append((pf.stem, res))
    n = len(rows)
    one_class = [r for r in rows if len({c for c, _ in r[1]}) == 1]
    print("=" * 78)
    print(f"{n} hardened networks, {1 + N_LOCAL + N_BROAD} starts each")
    print(f"single outcome class across all starts: {len(one_class)}/{n}")
    mixed = [r for r in rows if len({c for c, _ in r[1]}) > 1]
    for name, res in mixed:
        print(f"   mixed outcome classes: {name}: {dict(Counter(c for c, _ in res))}")
    osc_only = [r for r in rows if all(c in ("five-node", "subset") for c, _ in r[1])]
    cyc_known = [(nm, [cy for _, cy in res if cy is not None]) for nm, res in osc_only]
    consistent = [nm for nm, cys in cyc_known if cys and len(set(cys)) == 1]
    print(f"oscillating at every start: {len(osc_only)}; of these, cycle status identical across starts: {len(consistent)}")
    for nm, cys in cyc_known:
        if cys and len(set(cys)) > 1:
            print(f"   cycle status varies: {nm}: follows a cycle in {sum(cys)}/{len(cys)} starts")
    print("=" * 78)
