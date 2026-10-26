"""
check_order_ties.py

Tie-aware version of the directed-cycle test. Neurons that fire together (phases within TOL of the period) have
no defined relative order, so the order inside such a group is decided by floating-point noise (this is what made
two exactly equivalent circuits, seeds 32 and 36, look different). Here:

  * neurons whose phases are within TOL (circularly, chain-linked) form a group,
  * the observed firing structure is the cyclic sequence of groups,
  * the test is: is there an ordering of the neurons inside the groups that makes the full sequence a directed
    cycle of the hardened graph (forward; or in the reversed orientation as the control)?
  * chance: assign the same active neurons to the same group-size structure uniformly at random (all k! ways)
    and count how often the same test succeeds, summed over networks (graph-specific, as before).

TOL = 0 reproduces the original test (only exactly equal phases tie). Reported for several TOL and both
activity cutoffs, on the same 107 oscillators (frozen-at-x0 cases rescued as in compare_order_cutoffs.py).

Standalone / non-invasive. Run from the folder with results/, next to the other scripts.
Environment: RESCUE_SEED (default 0).
"""
import os
from itertools import permutations, product
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from check_order_vs_hamiltonian import edge_matrix, is_cycle
from count_distinct_graphs import canonical
from compare_order_cutoffs import CUTOFFS, RESCUE_SCALE, RESCUE_TRIES, analyze, is_flat, simulate_xs
from support_sequence_analysis import hardened_weights, spec, x0

RESULTS_DIR = Path("results")
TOLS = (0.0, 0.01, 0.02, 0.05)


def group_phases(phases, tol):
    """Cyclic sequence of groups of neurons with (chain-linked) phase gaps <= tol."""
    items = sorted(phases.items(), key=lambda kv: kv[1])
    groups, last = [[items[0][0]]], items[0][1]
    for neuron, ph in items[1:]:
        if ph - last <= tol:
            groups[-1].append(neuron)
        else:
            groups.append([neuron])
        last = ph
    if len(groups) > 1 and items[0][1] + 1.0 - items[-1][1] <= tol:
        groups[0] = groups[-1] + groups[0]
        groups.pop()
    return groups


def compatible(groups, E, reverse=False):
    """Does some ordering inside the groups make the cyclic sequence a directed cycle of the graph?"""
    for choice in product(*[list(permutations(g)) for g in groups]):
        order = [n for g in choice for n in g]
        if is_cycle(order, E, reverse=reverse):
            return True
    return False


def null_fraction_groups(active, sizes, E, reverse=False):
    """Fraction of all assignments of the active neurons to the group-size structure that pass the test."""
    total = ok = 0
    for perm in permutations(active):
        groups, i = [], 0
        for s in sizes:
            groups.append(list(perm[i:i + s]))
            i += s
        total += 1
        ok += compatible(groups, E, reverse)
    return ok / total


if __name__ == "__main__":
    rescue_seed = int(os.environ.get("RESCUE_SEED", 0))
    rng = np.random.default_rng(rescue_seed)
    hard = []
    for pf in sorted(RESULTS_DIR.glob("params_seed*_z*.npy")):
        params = jnp.asarray(np.load(pf))
        xs_s, xs_h = simulate_xs(params, "soft"), simulate_xs(params, "hard")
        if is_flat(xs_s):
            continue
        if is_flat(xs_h):
            ok = False
            for _ in range(RESCUE_TRIES):
                xs_try = simulate_xs(params, "hard",
                                     x0 + RESCUE_SCALE * jnp.asarray(rng.standard_normal(np.shape(x0))))
                if not is_flat(xs_try):
                    xs_h, ok = xs_try, True
                    break
            if not ok:
                continue
        hard.append((pf.stem, params, xs_h))
    print("=" * 78)
    print(f"oscillating hardened networks: {len(hard)} (rescue seed {rescue_seed})")

    for frac in CUTOFFS:
        for tol in TOLS:
            n = hit_f = hit_r = tied = 0
            exp_f = exp_r = 0.0
            misses = []
            cls_all, cls_miss = set(), set()
            for name, params, xs_h in hard:
                a = analyze(xs_h, frac)
                if a is None:
                    continue
                _, phases = a
                if len(phases) < 3:
                    continue
                E = edge_matrix(np.asarray(hardened_weights(spec, params)[0]))
                groups = group_phases(phases, tol)
                sizes = [len(g) for g in groups]
                active = sorted(phases)
                f, r = compatible(groups, E), compatible(groups, E, reverse=True)
                cls = canonical(E)
                cls_all.add(cls)
                if not f:
                    cls_miss.add(cls)
                n += 1
                tied += max(sizes) > 1
                hit_f += f
                hit_r += r
                exp_f += null_fraction_groups(active, sizes, E)
                exp_r += null_fraction_groups(active, sizes, E, reverse=True)
                if not f:
                    misses.append((name, sizes))
            print("-" * 78)
            print(f"cutoff {frac:.0%}, tie tolerance {tol:.2f} of the period: {n} networks, {tied} with a tied group")
            print(f"   forward  {hit_f}/{n}  (expected by chance {exp_f:.1f})    "
                  f"reversed {hit_r}/{n}  (expected {exp_r:.1f})")
            print(f"   distinct graphs: {len(cls_all)}; graphs where every circuit follows a cycle: {len(cls_all) - len(cls_miss)}")
            print(f"   non-hits ({len(misses)}): {misses}")
    print("=" * 78)
