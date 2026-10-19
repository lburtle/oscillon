"""
compare_order_ties.py

Tie-aware version of "firing order preserved between trained and hardened networks" (the 78/107 statistic).
Neurons whose phases are within TOL of the period are tied (no defined relative order). For each network, using
the neurons active (>= cutoff of the largest std) in BOTH runs, the order is preserved if some ordering inside
the tied groups makes the trained and hardened cyclic orders equal. Chance: assign the shared neurons to the
hardened group structure uniformly at random (all k! ways) against the observed trained arrangement, summed.

TOL = 0 reproduces the original test (only exactly equal phases tie). Same 107 oscillators as the other scripts
(frozen-at-x0 hardened networks rescued from small perturbations, RESCUE_SEED env, default 0).
"""
import os
from itertools import permutations, product
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from check_order_ties import group_phases
from compare_order_cutoffs import CUTOFFS, RESCUE_SCALE, RESCUE_TRIES, analyze, is_flat, simulate_xs
from support_sequence_analysis import x0

RESULTS_DIR = Path("results")
TOLS = (0.0, 0.01, 0.02, 0.05)


def arrangements(groups):
    """All canonical cyclic orders consistent with the cyclic sequence of tied groups."""
    out = set()
    for choice in product(*[list(permutations(g)) for g in groups]):
        order = [n for g in choice for n in g]
        k = len(order)
        out.add(min(tuple(order[i:] + order[:i]) for i in range(k)))
    return out


def chance_preserved(soft_arr, sizes_h, shared):
    total = ok = 0
    for perm in permutations(shared):
        groups, i = [], 0
        for s in sizes_h:
            groups.append(list(perm[i:i + s]))
            i += s
        total += 1
        ok += bool(soft_arr & arrangements(groups))
    return ok / total


if __name__ == "__main__":
    rng = np.random.default_rng(int(os.environ.get("RESCUE_SEED", 0)))
    pairs = []
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
        pairs.append((pf.stem, xs_s, xs_h))
    print("=" * 78)
    print(f"oscillating pairs: {len(pairs)}")
    for frac in CUTOFFS:
        for tol in TOLS:
            n = kept = 0
            expected = 0.0
            for name, xs_s, xs_h in pairs:
                a_s, a_h = analyze(xs_s, frac), analyze(xs_h, frac)
                if a_s is None or a_h is None:
                    continue
                ph_s, ph_h = a_s[1], a_h[1]
                shared = sorted(set(ph_s) & set(ph_h))
                if len(shared) < 3:
                    continue
                gs = group_phases({k: ph_s[k] for k in shared}, tol)
                gh = group_phases({k: ph_h[k] for k in shared}, tol)
                soft_arr = arrangements(gs)
                n += 1
                kept += bool(soft_arr & arrangements(gh))
                expected += chance_preserved(soft_arr, [len(g) for g in gh], shared)
            print(f"cutoff {frac:.0%}, tie tolerance {tol:.2f}: order preserved {kept}/{n}   (expected by chance {expected:.1f})")
        print("-" * 78)
    print("=" * 78)
