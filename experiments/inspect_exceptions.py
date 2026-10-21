"""
inspect_exceptions.py

Diagnoses the firing-order exceptions (seeds 32 z=1.0, 32 z=2.0, 15 z=3.0) next
to one passing control (seed 0 z=2.0). For each network it prints:

  - the hardened graph (edge j->i iff W[i,j] is the weak value), the
    bidirectional pairs (these can co-fire and have nearly tied phases), and
    every directed cycle through all 5 neurons
  - per neuron: std as a fraction of the max, peaks per cycle (>1 means a
    multi-peak waveform, where circular-mean phase is unreliable), and phase
    concentration R (1 = every peak at the same phase, near 0 = unreliable)
  - the measured cyclic firing order, which consecutive steps are NOT graph
    edges, and whether the order is a cycle once near-tied neurons are allowed
    to swap

Standalone / non-invasive. Needs compare_period_and_order.py,
check_order_vs_hamiltonian.py and analyze_hardened_oscillators.py in the folder.
"""

from itertools import permutations
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from analyze_hardened_oscillators import hard_oscillation
from check_order_vs_hamiltonian import edge_matrix, is_cycle
from compare_period_and_order import analyze, cyclic_order, peak_times
from support_sequence_analysis import hardened_weights, spec

RESULTS_DIR = Path("results")
CASES = [(32, 1.0), (32, 2.0), (15, 3.0), (0, 2.0)]   # last one is a passing control
TIE_TOL = 0.04                                         # phases closer than this count as tied


def all_five_cycles(E):
    n = E.shape[0]
    seen = set()
    for p in permutations(range(1, n)):
        order = [0] + list(p)
        if is_cycle(order, E):
            seen.add(tuple(order))
    return sorted(seen)


def per_neuron(xs):
    std = xs.std(axis=0)
    ref = int(np.argmax(std))
    pk = {i: peak_times(xs[:, i]) for i in range(xs.shape[1])}
    period = float(np.median(np.diff(pk[ref])))
    out = {}
    for i in range(xs.shape[1]):
        if len(pk[i]) < 2:
            out[i] = (std[i] / std.max(), 0.0, float("nan"), float("nan"))
            continue
        z = np.mean(np.exp(2j * np.pi * pk[i] / period))
        out[i] = (std[i] / std.max(), len(pk[i]) * period / len(xs), abs(z),
                  (np.angle(z) / (2 * np.pi)) % 1.0)
    return out, period


def tied_swap_cycle(phases, E):
    """Is there a cycle if neurons with nearly tied phases may swap places?"""
    neurons = sorted(phases, key=lambda n: phases[n])
    k = len(neurons)
    groups, cur = [], [neurons[0]]
    for a, b in zip(neurons, neurons[1:]):
        if (phases[b] - phases[a]) % 1.0 <= TIE_TOL:
            cur.append(b)
        else:
            groups.append(cur)
            cur = [b]
    groups.append(cur)
    # group order is fixed by phase; only within-group swaps are allowed
    def expand(gs):
        if not gs:
            yield []
            return
        for perm in permutations(gs[0]):
            for rest in expand(gs[1:]):
                yield list(perm) + rest
    return any(is_cycle(o, E) for o in expand(groups))


if __name__ == "__main__":
    for seed, z in CASES:
        pf = RESULTS_DIR / f"params_seed{seed}_z{z}.npy"
        params = jnp.asarray(np.load(pf))
        W = np.asarray(hardened_weights(spec, params)[0])
        E = edge_matrix(W)
        res = hard_oscillation(params)
        print("\n" + "=" * 72)
        print(f"seed {seed} z={z}   (hardened trajectory perturbed x0: {res[1] if res else 'n/a'})")
        print("edges j->i:", [(j, i) for j in range(5) for i in range(5) if E[j, i]])
        bidir = [(i, j) for i in range(5) for j in range(i + 1, 5) if E[i, j] and E[j, i]]
        print("bidirectional pairs (can co-fire):", bidir)
        cycles = all_five_cycles(E)
        print(f"directed 5-cycles in graph ({len(cycles)}):", cycles)
        if res is None:
            print("hardened network is a fixed point; skipping")
            continue
        xs = res[0]
        stats, period = per_neuron(xs)
        print(f"period (time units) {period * 0.1:.2f}")
        print("neuron | std/max | peaks/cycle | phase conc. R | phase")
        for i, (s, ppc, R, ph) in stats.items():
            print(f"{i:>6} | {s:>7.3f} | {ppc:>11.2f} | {R:>13.3f} | {ph:>5.3f}")
        _, phases = analyze(xs)
        order = list(cyclic_order(phases, sorted(phases)))
        bad = [(order[a], order[(a + 1) % len(order)]) for a in range(len(order))
               if not E[order[a], order[(a + 1) % len(order)]]]
        print("measured cyclic order:", order)
        print("steps that are not graph edges:", bad if bad else "none (order is a graph cycle)")
        print("cycle if near-tied neurons may swap:", tied_swap_cycle(phases, E))
