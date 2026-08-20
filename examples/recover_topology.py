"""
Recompute adjacency / Hamiltonian rate / cycle size from saved edge-commitment
values, without rerunning any training.

u was saved as the band-normalized position of each weight, so u > 0.5 is
exactly the band-midpoint threshold -- the correct edge test for BOTH the
sigmoid and the direct parameterization.

Usage:  python recover_topology.py ablation_direct_u.npz [N]
"""

import itertools
import sys

import numpy as np

# --- rebuild the same spec used in the sweep so edge order matches ---
from oscillon.topology import BlockSoftSpec

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5

spec = BlockSoftSpec(
    sizes=(N,),
    cross_deltas=np.zeros((1, 1)),
    eps=0.10,
    delta=0.50,
    theta_init=0.5,
)
rows, cols, _, _, _ = spec.compile_indices()
rows, cols = np.asarray(rows), np.asarray(cols)
n_edges = spec.n_edges

u = np.load(path)["u"]
assert u.size % n_edges == 0, f"{u.size} not divisible by n_edges={n_edges}"
U = u.reshape(-1, n_edges)
print(f"runs recovered: {U.shape[0]}  (n_edges={n_edges})")


def adjacency(u_row):
    A = np.zeros((N, N), dtype=bool)
    A[rows, cols] = u_row > 0.5  # A[i, j] = edge j -> i
    np.fill_diagonal(A, False)
    return A


def is_hamiltonian(A):
    for perm in itertools.permutations(range(1, N)):
        order = (0,) + perm
        if all(A[order[(k + 1) % N], order[k]] for k in range(N)):
            return True
    return False


def nodes_in_cycle(A):
    """Node i lies on a directed cycle iff i reaches itself in >=1 step."""
    reach = A.copy()
    acc = A.copy()
    for _ in range(N - 1):
        acc = acc @ A
        reach |= acc
    return int(np.sum(np.diag(reach)))


ham, sizes, degrees = [], [], []
for r in U:
    A = adjacency(r)
    ham.append(is_hamiltonian(A))
    sizes.append(nodes_in_cycle(A))
    degrees.append(int(A.sum()))

ham = np.array(ham)
sizes = np.array(sizes)
degrees = np.array(degrees)

print(f"\nHamiltonian:  {ham.sum()}/{len(ham)}  ({100 * ham.mean():.0f}%)")
print(f"edges per graph:  median={np.median(degrees):.0f}  "
      f"IQR=[{np.percentile(degrees, 25):.0f}, {np.percentile(degrees, 75):.0f}]")

print("\nnodes lying on some directed cycle:")
for k in range(N + 1):
    c = int((sizes == k).sum())
    if c:
        print(f"  {k} nodes: {c:3d}  ({100 * c / len(sizes):.0f}%)")

print("\ncommitment:")
print(f"  frac in [0,0.1):   {np.mean(u < 0.1):.3f}")
print(f"  frac in [0.1,0.9): {np.mean((u >= 0.1) & (u <= 0.9)):.3f}")
print(f"  frac in (0.9,1]:   {np.mean(u > 0.9):.3f}")
print(f"  within 0.05 of .5: {np.mean(np.abs(u - 0.5) < 0.05):.3f}")
