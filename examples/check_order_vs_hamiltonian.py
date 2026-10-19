"""
check_order_vs_hamiltonian.py

Turns "full oscillations are consistent with theory but not positively
predicted" into a testable prediction. Parmelee et al. (2022b) show that
cycle-structured CTLN graphs produce sequential limit cycles whose firing order
follows the cycle. So for each HARDENED network that oscillates, test whether
its measured cyclic firing order is a directed cycle through the active
neurons in the HARDENED GRAPH (edge j->i  <=>  W[i,j] is the weak value),
in the forward or the reversed orientation (reported separately so the
direction convention is visible rather than assumed).

Null: for the same graph and the same active set, the fraction of all cyclic
orders that would satisfy the same edge test. The summed null is the number of
hits expected if firing order were unrelated to the graph.

Reuses the measurement helpers from compare_period_and_order.py (same window,
thresholds and peak-phase definition). Standalone / non-invasive.
"""

from itertools import permutations
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from compare_period_and_order import analyze, cyclic_order, is_flat, simulate_xs
from support_sequence_analysis import hardened_weights, spec, parse_seed_zstd

RESULTS_DIR = Path("results")


def edge_matrix(W):
    n = W.shape[0]
    off = W[~np.eye(n, dtype=bool)]
    mid = 0.5 * (off.max() + off.min())
    E = np.zeros((n, n), dtype=bool)     # E[j, i] = edge j -> i
    for i in range(n):
        for j in range(n):
            if i != j and W[i, j] > mid:
                E[j, i] = True
    return E


def is_cycle(order, E, reverse=False):
    k = len(order)
    for a in range(k):
        u, v = order[a], order[(a + 1) % k]
        if reverse:
            u, v = v, u
        if not E[u, v]:
            return False
    return True


def null_fraction(active, E):
    active = list(active)
    first, rest = active[0], active[1:]
    total = fwd = rev = either = 0
    for perm in permutations(rest):
        order = [first] + list(perm)
        total += 1
        f, r = is_cycle(order, E), is_cycle(order, E, reverse=True)
        fwd += f
        rev += r
        either += (f or r)
    return fwd / total, rev / total, either / total


if __name__ == "__main__":
    n = hit_f = hit_r = hit_any = 0
    exp_f = exp_r = exp_any = 0.0
    for pf in sorted(RESULTS_DIR.glob("params_seed*_z*.npy")):
        params = jnp.asarray(np.load(pf))
        xs_h = simulate_xs(params, "hard")
        if is_flat(xs_h):
            continue
        a = analyze(xs_h)
        if a is None:
            continue
        _, phases = a
        active = sorted(phases)
        if len(active) < 3:
            continue
        W = np.asarray(hardened_weights(spec, params)[0])
        E = edge_matrix(W)
        order = list(cyclic_order(phases, active))
        f, r = is_cycle(order, E), is_cycle(order, E, reverse=True)
        pf_, pr_, pa_ = null_fraction(active, E)
        n += 1
        hit_f += f
        hit_r += r
        hit_any += (f or r)
        exp_f += pf_
        exp_r += pr_
        exp_any += pa_

    print("=" * 66)
    print(f"oscillating hardened networks with >=3 active neurons: n={n}")
    print(f"firing order is a graph cycle, forward:  {hit_f}/{n}   (expected by chance {exp_f:.1f})")
    print(f"firing order is a graph cycle, reversed: {hit_r}/{n}   (expected by chance {exp_r:.1f})")
    print(f"either orientation:                      {hit_any}/{n}   (expected by chance {exp_any:.1f})")
    print("=" * 66)
