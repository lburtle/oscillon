"""
count_distinct_graphs.py

How many distinct hardened graphs are there among the 150 circuits?
  * exact: identical edge matrices
  * up to relabeling: identical after the best permutation of the 5 neurons (120 permutations)
For each class: size, number of directed edges, whether the graph has a directed 5-cycle, and members.
Also lists where the unusual cases sit (seeds 15 z3.0, 32 z1.0, 32 z2.0, 43 z3.0, 21 z2.0, 46 z3.0).

Standalone / non-invasive: reads existing params only. Run from the folder with results/.
"""
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from check_order_vs_hamiltonian import edge_matrix, is_cycle
from support_sequence_analysis import hardened_weights, spec

RESULTS_DIR = Path("results")
WATCH = {"params_seed15_z3.0", "params_seed32_z1.0", "params_seed32_z2.0",
         "params_seed43_z3.0", "params_seed21_z2.0", "params_seed46_z3.0"}


def canonical(E):
    n = E.shape[0]
    return min(E[np.ix_(p, p)].tobytes() for p in permutations(range(n)))


def has_5cycle(E):
    n = E.shape[0]
    return any(is_cycle([0] + list(p), E) for p in permutations(range(1, n)))


if __name__ == "__main__":
    exact, iso, info = defaultdict(list), defaultdict(list), {}
    for pf in sorted(RESULTS_DIR.glob("params_seed*_z*.npy")):
        params = jnp.asarray(np.load(pf))
        E = edge_matrix(np.asarray(hardened_weights(spec, params)[0]))
        exact[E.tobytes()].append(pf.stem)
        c = canonical(E)
        iso[c].append(pf.stem)
        info[c] = (int(E.sum()), has_5cycle(E))
    n = sum(len(v) for v in exact.values())
    print("=" * 70)
    print(f"{n} circuits; distinct hardened graphs: exact {len(exact)}, up to relabeling {len(iso)}")
    sizes = sorted((len(v) for v in iso.values()), reverse=True)
    print(f"class sizes up to relabeling (largest first): {sizes}")
    print(f"circuits in the 5 largest classes: {sum(sizes[:5])}/{n}")
    print("-" * 70)
    print("classes (size, directed edges, has 5-cycle):")
    for c, members in sorted(iso.items(), key=lambda kv: -len(kv[1])):
        edges, ham = info[c]
        tag = " <-- contains watched case(s): " + ", ".join(m for m in members if m in WATCH) \
            if any(m in WATCH for m in members) else ""
        print(f"  size {len(members):3d}  edges {edges:2d}  5-cycle {ham}{tag}")
    print("-" * 70)
    print("graphs with more than one distinct seed (identical hardened graph, up to relabeling):")
    for c, members in sorted(iso.items(), key=lambda kv: -len(kv[1])):
        seeds = sorted({m.split("_z")[0] for m in members})
        if len(seeds) > 1:
            print(f"  {len(members)} circuits, {len(seeds)} seeds: {', '.join(members[:12])}"
                  f"{' ...' if len(members) > 12 else ''}")
    print("=" * 70)
