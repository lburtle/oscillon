"""
analyze_hardened_oscillators.py

One pass over all 150 networks that combines the period / firing-order
comparison (compare_period_and_order.py) with the graph-cycle test
(check_order_vs_hamiltonian.py), and fixes two things:

  1. Networks whose hardened trajectory is frozen at the fixed x0 are
     re-simulated from a small random perturbation (1e-4, seed 0, as in
     reverify_stuck_seeds.py). If the perturbed run oscillates, the network is
     included as an oscillator; if it stays flat, it is a genuine fixed point.
  2. The hardened oscillators whose firing order is NOT a directed cycle of the
     hardened graph are listed, with the active set, the measured order, whether
     the graph has any directed cycle through all 5 neurons, and whether the
     order passes when only clearly active neurons (std >= 20% of max) are used.

Standalone / non-invasive. Needs compare_period_and_order.py and
check_order_vs_hamiltonian.py in the same folder.
"""

from itertools import permutations
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from compare_period_and_order import (
    BURN_IN, WINDOW, PERIOD_TOL, analyze, cyclic_order, dt, is_flat,
)
from check_order_vs_hamiltonian import edge_matrix, is_cycle, null_fraction
from oscillon.dynamics import simulate
from support_sequence_analysis import hardened_weights, spec, p2m, x0, parse_seed_zstd

RESULTS_DIR = Path("results")
PERTURB_SCALE = 1e-4
STRONG_STD_FRAC = 0.20


def sim(params, which, x0_in):
    W, th = p2m(params) if which == "soft" else hardened_weights(spec, params)
    return np.asarray(simulate(W, th, x0_in, dt=dt, n_steps=BURN_IN + WINDOW))[BURN_IN:]


def hard_oscillation(params):
    """-> (xs, perturbed_flag) for an oscillating hardened run, or None if a fixed point."""
    xs = sim(params, "hard", x0)
    if not is_flat(xs):
        return xs, False
    rng = np.random.default_rng(0)
    x0p = jnp.asarray(np.asarray(x0) + PERTURB_SCALE * rng.standard_normal(np.asarray(x0).shape))
    xs = sim(params, "hard", x0p)
    return (xs, True) if not is_flat(xs) else None


def has_full_cycle(E):
    n = E.shape[0]
    return any(is_cycle([0] + list(p), E) for p in permutations(range(1, n)))


if __name__ == "__main__":
    n_total = n_static = n_osc = n_perturbed = 0
    ratios, same_period = [], 0
    same_order = comparable_order = 0
    hit_f = hit_r = 0
    exp_f = exp_r = 0.0
    failures = []

    for pf in sorted(RESULTS_DIR.glob("params_seed*_z*.npy")):
        n_total += 1
        seed, z = parse_seed_zstd(pf)
        params = jnp.asarray(np.load(pf))
        xs_s = sim(params, "soft", x0)
        res = None if is_flat(xs_s) else hard_oscillation(params)
        if res is None:
            n_static += 1
            continue
        xs_h, perturbed = res
        a_s, a_h = analyze(xs_s), analyze(xs_h)
        if a_s is None or a_h is None:
            n_static += 1
            continue
        n_osc += 1
        n_perturbed += perturbed
        (per_s, ph_s), (per_h, ph_h) = a_s, a_h

        ratios.append(per_h / per_s)
        same_period += abs(per_h / per_s - 1.0) <= PERIOD_TOL
        shared = set(ph_s) & set(ph_h)
        if len(shared) >= 3:
            comparable_order += 1
            same_order += cyclic_order(ph_s, shared) == cyclic_order(ph_h, shared)

        active = sorted(ph_h)
        if len(active) < 3:
            continue
        E = edge_matrix(np.asarray(hardened_weights(spec, params)[0]))
        order = list(cyclic_order(ph_h, active))
        f, r = is_cycle(order, E), is_cycle(order, E, reverse=True)
        pf_, pr_, _ = null_fraction(active, E)
        hit_f += f
        hit_r += r
        exp_f += pf_
        exp_r += pr_
        if not f:
            std = xs_h.std(axis=0)
            strong = [i for i in active if std[i] >= STRONG_STD_FRAC * std.max()]
            robust = (len(strong) >= 3 and
                      is_cycle(list(cyclic_order(ph_h, strong)), E))
            failures.append((seed, z, perturbed, active, order, has_full_cycle(E), robust))

    r = np.array(ratios)
    print("=" * 70)
    print(f"n={n_total}: fixed points/static: {n_static}, oscillating after hardening: {n_osc} "
          f"(of which {n_perturbed} needed perturbed x0)")
    print(f"period within {PERIOD_TOL:.0%} of trained: {same_period}/{len(r)}  "
          f"(median ratio {np.median(r):.3f})")
    print(f"cyclic firing order preserved: {same_order}/{comparable_order}")
    print(f"order is a directed graph cycle, forward:  {hit_f}/{n_osc}  (chance {exp_f:.1f})")
    print(f"order is a directed graph cycle, reversed: {hit_r}/{n_osc}  (chance {exp_r:.1f})")
    print("=" * 70)
    print(f"\nforward-cycle failures ({len(failures)}):")
    print("seed  z    perturbed  active        order            graph_has_5cycle  passes_with_strong_only")
    for seed, z, pert, active, order, ham, robust in failures:
        print(f"{seed:>4} {z:>4}  {str(pert):>9}  {str(active):<13} {str(order):<16} "
              f"{str(ham):>16}  {str(robust):>22}")
