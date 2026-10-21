"""
compare_order_cutoffs.py

Matched comparison of the "active neuron" cutoff (5% vs 20% of the largest std) on the SAME oscillators.

Differences from compare_period_and_order.py:
  * Hardened networks that are flat at x0 (the 16 sitting on an unstable full-support equilibrium) are
    re-simulated from small random perturbations of x0 (RESCUE_TRIES tries, fixed rng) before being called
    static, so the population should be the audit's 107 oscillators rather than 91.
  * Both cutoffs are evaluated on the identical pair of trajectories per network.
  * Reports, per cutoff: period statistics, how many neurons are shared between trained and hardened,
    same-cyclic-order counts for >=3 shared and for all five, the number of cases whose neuron sets differ,
    and the expected number correct by chance (1/(k-1)! for k shared neurons).

Standalone / non-invasive: reads existing params only, writes nothing. Run from the folder with results/.
"""
from collections import Counter
from math import factorial
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from oscillon.ctln.dynamics import simulate
from support_sequence_analysis import hardened_weights, spec, p2m, x0, parse_seed_zstd

RESULTS_DIR = Path("results")
dt = 0.1
BURN_IN = 1000
WINDOW = 6000
FLAT_THRESH = 1e-3
PERIOD_TOL = 0.05
CUTOFFS = (0.05, 0.20)
RESCUE_TRIES = 3
RESCUE_SCALE = 1e-2


def simulate_xs(params, which, x0_in=x0):
    W, th = p2m(params) if which == "soft" else hardened_weights(spec, params)
    return np.asarray(simulate(W, th, x0_in, dt=dt, n_steps=BURN_IN + WINDOW))[BURN_IN:]


def is_flat(xs):
    rng = xs.max(axis=0) - xs.min(axis=0)
    return float(np.max(rng) / max(np.abs(xs).max(), 1e-12)) < FLAT_THRESH


def peak_times(x):
    mid = x.min() + 0.5 * (x.max() - x.min())
    return np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] > mid))[0] + 1


def analyze(xs, frac):
    """-> (period_in_time_units, {neuron: circular phase in [0,1)}) or None."""
    std = xs.std(axis=0)
    active = [i for i in range(xs.shape[1]) if std[i] >= frac * std.max()]
    ref = int(np.argmax(std))
    pk = {i: peak_times(xs[:, i]) for i in active}
    if len(pk[ref]) < 4:
        return None
    period_steps = float(np.median(np.diff(pk[ref])))
    phases = {}
    for i in active:
        if len(pk[i]) < 3:
            continue
        ang = np.angle(np.mean(np.exp(2j * np.pi * pk[i] / period_steps)))
        phases[i] = (ang / (2 * np.pi)) % 1.0
    if ref not in phases:
        return None
    ref_phase = phases[ref]
    return period_steps * dt, {i: (p - ref_phase) % 1.0 for i, p in phases.items()}


def cyclic_order(phases, neurons):
    order = sorted(neurons, key=lambda n: phases[n])
    k = len(order)
    return min(tuple(order[i:] + order[:i]) for i in range(k)) if k else ()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pairs, n_total, n_static, n_rescued = [], 0, 0, 0
    for pf in sorted(RESULTS_DIR.glob("params_seed*_z*.npy")):
        n_total += 1
        seed, z = parse_seed_zstd(pf)
        params = jnp.asarray(np.load(pf))
        xs_s, xs_h = simulate_xs(params, "soft"), simulate_xs(params, "hard")
        if is_flat(xs_s):
            n_static += 1
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
                n_static += 1
                continue
            n_rescued += 1
        pairs.append((seed, z, xs_s, xs_h))

    print("=" * 70)
    print(f"n={n_total}: static after rescue: {n_static}; rescued from frozen-at-x0: {n_rescued}; "
          f"oscillating pairs: {len(pairs)}")
    for frac in CUTOFFS:
        ratios, same_per, n_unres = [], 0, 0
        shared_hist, same_ge3, comparable_ge3, same_all5, comparable_all5 = Counter(), 0, 0, 0, 0
        set_differs, chance = 0, 0.0
        for seed, z, xs_s, xs_h in pairs:
            a_s, a_h = analyze(xs_s, frac), analyze(xs_h, frac)
            if a_s is None or a_h is None:
                n_unres += 1
                continue
            (per_s, ph_s), (per_h, ph_h) = a_s, a_h
            ratios.append(per_h / per_s)
            same_per += abs(per_h / per_s - 1.0) <= PERIOD_TOL
            shared = set(ph_s) & set(ph_h)
            shared_hist[len(shared)] += 1
            set_differs += set(ph_s) != set(ph_h)
            if len(shared) >= 3:
                comparable_ge3 += 1
                chance += 1.0 / factorial(len(shared) - 1)
                same_ge3 += cyclic_order(ph_s, shared) == cyclic_order(ph_h, shared)
            if len(shared) == 5:
                comparable_all5 += 1
                same_all5 += cyclic_order(ph_s, shared) == cyclic_order(ph_h, shared)
        r = np.array(ratios)
        print("-" * 70)
        print(f"cutoff {frac:.0%} of largest std   (unresolved: {n_unres}, measured: {len(r)})")
        if len(r):
            print(f"  period ratio hard/soft: median {np.median(r):.3f}, IQR "
                  f"[{np.percentile(r, 25):.3f}, {np.percentile(r, 75):.3f}]; within {PERIOD_TOL:.0%}: {same_per}/{len(r)}")
        print(f"  shared active neurons (count: cases): {dict(sorted(shared_hist.items()))}")
        print(f"  neuron set differs between trained and hardened: {set_differs}/{len(r)}")
        print(f"  same cyclic order, >=3 shared: {same_ge3}/{comparable_ge3}   (expected by chance: {chance:.1f})")
        print(f"  same cyclic order, all 5 shared: {same_all5}/{comparable_all5}")
    print("=" * 70)
