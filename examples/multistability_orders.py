"""
multistability_orders.py

Does the hardened graph alone determine the firing order, or can one graph have several attractors with
different orders? For each named circuit it simulates the hardened network from N_STARTS random starts and
tallies the outcomes (fixed point, or firing order with whether it is a directed cycle of the graph).

Because theta is a single shared scalar, the hardened dynamics are homogeneous: scaling theta only rescales
the start relative to theta. So circuits with the same hardened graph but different trained theta, started at
the same x0 = 0.1, are sampled at different points of the same state space. Two start families are used:
  * "local":  x0 = 0.1 + 1e-2 * N(0, 1)         (like the frozen-case rescue)
  * "broad":  x0 = theta * U(0, 1) per neuron    (covers the box up to the drive level)

Default circuits: the three-circuit class containing seeds 32 z1.0, 32 z2.0, 36 z1.0; seed 15 z3.0; and a
control from the 30-circuit class (seed 11 z2.0), which should show one attractor.

Standalone / non-invasive. Run from the folder with results/, next to the other scripts.
"""
import sys
from collections import Counter
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from check_order_vs_hamiltonian import edge_matrix, is_cycle
from compare_order_cutoffs import BURN_IN, WINDOW, analyze, cyclic_order, dt, is_flat
from oscillon.dynamics import simulate
from support_sequence_analysis import hardened_weights, spec, x0

RESULTS_DIR = Path("results")
N_STARTS = 100
DEFAULT = ["params_seed32_z1.0", "params_seed32_z2.0", "params_seed36_z1.0",
           "params_seed15_z3.0", "params_seed11_z2.0"]
FRAC = 0.05


def classify(W, th, E, x_start):
    xs = np.asarray(simulate(W, th, x_start, dt=dt, n_steps=BURN_IN + WINDOW))[BURN_IN:]
    if is_flat(xs):
        return "fixed point"
    a = analyze(xs, FRAC)
    if a is None:
        return "unresolved"
    _, phases = a
    active = sorted(phases)
    order = tuple(cyclic_order(phases, active))
    tag = "GRAPH CYCLE" if is_cycle(list(order), E) else "not a cycle"
    return f"{tag}: {order}"


if __name__ == "__main__":
    names = sys.argv[1:] or DEFAULT
    rng = np.random.default_rng(0)
    print("=" * 78)
    for name in names:
        pf = RESULTS_DIR / f"{name}.npy"
        params = jnp.asarray(np.load(pf))
        W, th, = hardened_weights(spec, params)
        E = edge_matrix(np.asarray(W))
        theta = float(np.asarray(th)[0])
        print(f"{name}: theta = {theta:.4f}; directed edges {int(E.sum())}; "
              f"default start x0=0.1 (= {0.1 / theta:.3f} * theta): {classify(W, th, E, x0)}")
        for label, draw in (("local", lambda: x0 + 1e-2 * jnp.asarray(rng.standard_normal(np.shape(x0)))),
                            ("broad", lambda: theta * jnp.asarray(rng.uniform(0, 1, np.shape(x0))))):
            tally = Counter(classify(W, th, E, draw()) for _ in range(N_STARTS))
            print(f"  {label} starts ({N_STARTS}):")
            for k, v in tally.most_common():
                print(f"     {v:3d}  {k}")
        print("-" * 78)
    print("=" * 78)
