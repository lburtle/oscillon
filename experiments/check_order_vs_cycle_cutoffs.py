"""
check_order_vs_cycle_cutoffs.py

Directed-cycle test (firing order vs a directed cycle of the hardened graph) at both activity cutoffs,
on the same oscillator population as compare_order_cutoffs.py:
  * hardened networks flat at x0 are re-simulated from small random perturbations (same rng, same order,
    same scale as compare_order_cutoffs.py, so the same 107 oscillators and the same perturbed starts),
  * the cutoff is a parameter, not a baked-in constant,
  * the edge test, cycle test and null are the originals from check_order_vs_hamiltonian.py.

For each cutoff it reports forward / reversed / either hits with the number expected by chance, and the
distribution of active-set sizes k (the null is stricter for larger k: roughly 1/(k-1)! of orders are cycles
only if the graph is complete, so read the printed expectation rather than assuming).

Standalone / non-invasive. Run from the folder with results/, next to compare_order_cutoffs.py and
check_order_vs_hamiltonian.py.
"""
from collections import Counter
from itertools import permutations
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from check_order_vs_hamiltonian import edge_matrix, is_cycle, null_fraction
from compare_order_cutoffs import (CUTOFFS, RESCUE_SCALE, RESCUE_TRIES, analyze, cyclic_order, is_flat,
                                   simulate_xs)
from support_sequence_analysis import hardened_weights, spec, x0

RESULTS_DIR = Path("results")

if __name__ == "__main__":
    import os
    rescue_seed = int(os.environ.get("RESCUE_SEED", 0))   # 0 reproduces compare_order_cutoffs.py
    rng = np.random.default_rng(rescue_seed)
    hard = []   # (params, xs_h, name, rescued) for every oscillating hardened network
    for pf in sorted(RESULTS_DIR.glob("params_seed*_z*.npy")):
        params = jnp.asarray(np.load(pf))
        was_rescued = False
        xs_s, xs_h = simulate_xs(params, "soft"), simulate_xs(params, "hard")
        if is_flat(xs_s):
            continue
        if is_flat(xs_h):
            rescued = False
            for _ in range(RESCUE_TRIES):
                x0p = x0 + RESCUE_SCALE * jnp.asarray(rng.standard_normal(np.shape(x0)))
                xs_try = simulate_xs(params, "hard", x0p)
                if not is_flat(xs_try):
                    xs_h, rescued = xs_try, True
                    break
            if not rescued:
                continue
            was_rescued = True
        hard.append((params, xs_h, pf.stem, was_rescued))
    print("=" * 70)
    print(f"oscillating hardened networks: {len(hard)} (rescue seed {rescue_seed}; "
          f"{sum(h[3] for h in hard)} rescued)")

    for frac in CUTOFFS:
        n = hit_f = hit_r = hit_any = skipped = 0
        exp_f = exp_r = exp_any = 0.0
        sizes = Counter()
        misses = []
        for params, xs_h, name, was_rescued in hard:
            a = analyze(xs_h, frac)
            if a is None:
                skipped += 1
                continue
            _, phases = a
            active = sorted(phases)
            if len(active) < 3:
                skipped += 1
                continue
            E = edge_matrix(np.asarray(hardened_weights(spec, params)[0]))
            order = list(cyclic_order(phases, active))
            f, r = is_cycle(order, E), is_cycle(order, E, reverse=True)
            pf_, pr_, pa_ = null_fraction(active, E)
            if not (f or r):
                cyc_active = any(is_cycle([active[0]] + list(p), E) for p in permutations(active[1:]))
                cyc_all5 = any(is_cycle([0] + list(p), E) for p in permutations(range(1, E.shape[0])))
                misses.append((name, f"k={len(active)}", "rescued" if was_rescued else "x0",
                               f"cycle through active set: {cyc_active}", f"5-cycle in graph: {cyc_all5}",
                               f"firing order: {order}"))
            n += 1
            sizes[len(active)] += 1
            hit_f += f
            hit_r += r
            hit_any += (f or r)
            exp_f += pf_
            exp_r += pr_
            exp_any += pa_
        print("-" * 70)
        print(f"cutoff {frac:.0%} of largest std   (n with >=3 active neurons: {n}; skipped: {skipped})")
        print(f"  active-set sizes (k: cases): {dict(sorted(sizes.items()))}")
        print(f"  firing order is a graph cycle, forward:  {hit_f}/{n}   (expected by chance {exp_f:.1f})")
        print(f"  firing order is a graph cycle, reversed: {hit_r}/{n}   (expected by chance {exp_r:.1f})")
        print(f"  either orientation:                      {hit_any}/{n}   (expected by chance {exp_any:.1f})")
        print(f"  non-hits ({len(misses)}): (file, active-set size, start, cycle exists through active set, "
              f"5-cycle in graph, firing order)")
        for m in misses:
            print(f"     {m}")
    print("=" * 70)
