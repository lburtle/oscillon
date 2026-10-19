"""
compare_period_and_order.py

Compares trained (soft) and hardened oscillations with continuous measures
instead of support sequences (which cannot resolve networks whose neurons all
stay active), directly addressing "compare periods, phase-aligned trajectories,
activation sequences" from the NeurReps review, and testing the period-shift
explanation for the near-chance hardened readout error.

For every network where BOTH the soft and hardened trajectories oscillate
(tail range / scale >= FLAT_THRESH) it reports:
  - period ratio hard/soft (peak spacing, long window)
  - cyclic firing order of the neurons active in both (peak phase relative to
    the strongest neuron), compared up to rotation

Hardened networks that are flat (fixed points) or frozen at x0 are counted
separately as "static", not as oscillations.

Standalone / non-invasive: reads existing params only, writes nothing.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from oscillon.dynamics import simulate
from support_sequence_analysis import hardened_weights, spec, p2m, x0, parse_seed_zstd

RESULTS_DIR = Path("results")
dt = 0.1
BURN_IN = 1000
WINDOW = 6000
FLAT_THRESH = 1e-3
ACTIVE_STD_FRAC = 0.05   # neuron counts as oscillating if its std >= this * max std
PERIOD_TOL = 0.05        # "same period" if within 5%
STRONG_STD_FRAC = 0.20   # switch out with ACTIVE_STD_FRAC


def simulate_xs(params, which):
    W, th = p2m(params) if which == "soft" else hardened_weights(spec, params)
    return np.asarray(simulate(W, th, x0, dt=dt, n_steps=BURN_IN + WINDOW))[BURN_IN:]


def is_flat(xs):
    rng = xs.max(axis=0) - xs.min(axis=0)
    return float(np.max(rng) / max(np.abs(xs).max(), 1e-12)) < FLAT_THRESH


def peak_times(x):
    mid = x.min() + 0.5 * (x.max() - x.min())
    idx = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] > mid))[0] + 1
    return idx


def analyze(xs):
    """-> (period_in_time_units, {neuron: circular phase in [0,1)}) or None."""
    std = xs.std(axis=0)
    active = [i for i in range(xs.shape[1]) if std[i] >= STRONG_STD_FRAC * std.max()]
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
    ref_phase = phases[ref]
    phases = {i: (p - ref_phase) % 1.0 for i, p in phases.items()}
    return period_steps * dt, phases


def cyclic_order(phases, neurons):
    order = [n for n in sorted(neurons, key=lambda n: phases[n])]
    k = len(order)
    return min(tuple(order[i:] + order[:i]) for i in range(k)) if k else ()


if __name__ == "__main__":
    n_total = n_static = n_unres = 0
    ratios, same_period, same_order, comparable_order = [], 0, 0, 0
    for pf in sorted(RESULTS_DIR.glob("params_seed*_z*.npy")):
        n_total += 1
        params = jnp.asarray(np.load(pf))
        xs_s, xs_h = simulate_xs(params, "soft"), simulate_xs(params, "hard")
        if is_flat(xs_s) or is_flat(xs_h):
            n_static += 1
            continue
        a_s, a_h = analyze(xs_s), analyze(xs_h)
        if a_s is None or a_h is None:
            n_unres += 1
            continue
        (per_s, ph_s), (per_h, ph_h) = a_s, a_h
        ratios.append(per_h / per_s)
        if abs(per_h / per_s - 1.0) <= PERIOD_TOL:
            same_period += 1
        shared = set(ph_s) & set(ph_h)
        if len(shared) >= 3:
            comparable_order += 1
            if cyclic_order(ph_s, shared) == cyclic_order(ph_h, shared):
                same_order += 1

    r = np.array(ratios)
    print("=" * 62)
    print(f"n={n_total}: static (soft or hard flat): {n_static}, unresolved: {n_unres}, "
          f"both oscillating and measured: {len(r)}")
    if len(r):
        print(f"period ratio hard/soft: median {np.median(r):.3f}, "
              f"IQR [{np.percentile(r, 25):.3f}, {np.percentile(r, 75):.3f}]")
        print(f"period within {PERIOD_TOL:.0%}: {same_period}/{len(r)}")
        print(f"same cyclic firing order (>=3 shared neurons): {same_order}/{comparable_order}")
    print("=" * 62)
