"""
check_homogeneity.py

Two quick tests behind the seed 32 / seed 36 puzzle (same graph class, different dominant attractor):

 1. Is circuit B's hardened weight matrix a permutation of circuit A's? (exact check of the off-diagonal weights,
    not just the edge pattern)
 2. Is the simulator homogeneous in theta?  For CTLN dynamics dx/dt = -x + [Wx + theta]_+ , scaling
    x -> theta*y removes theta, so xs(theta1, theta1*y0)/theta1 must equal xs(theta2, theta2*y0)/theta2.
    The max deviation is printed. A clear deviation means simulate() has an absolute scale somewhere
    (clip, saturation, noise); its source is printed so you can look.

Standalone / non-invasive. Run from the folder with results/.
"""
import inspect
import sys
from itertools import permutations
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from check_order_vs_hamiltonian import edge_matrix
from oscillon.ctln.dynamics import simulate
from support_sequence_analysis import hardened_weights, spec

RESULTS_DIR = Path("results")
dt = 0.1
N_STEPS = 2000


def load(name):
    params = jnp.asarray(np.load(RESULTS_DIR / f"{name}.npy"))
    W, th = hardened_weights(spec, params)
    return np.asarray(W), np.asarray(th)


if __name__ == "__main__":
    a, b = (sys.argv[1:3] if len(sys.argv) >= 3 else ["params_seed32_z1.0", "params_seed36_z1.0"])
    WA, thA = load(a)
    WB, thB = load(b)
    print("=" * 72)
    print(f"{a}: theta vector {np.round(thA, 4).tolist()}")
    print(f"{b}: theta vector {np.round(thB, 4).tolist()}")
    n = WA.shape[0]
    found = [p for p in permutations(range(n)) if np.array_equal(WA[np.ix_(p, p)], WB)]
    print(f"\n1. exact permutation of the weight matrix: {'YES, ' + str(len(found)) + ' permutation(s); first ' + str(found[0]) if found else 'NO'}")
    if not found:
        EA, EB = edge_matrix(WA), edge_matrix(WB)
        iso = [p for p in permutations(range(n)) if np.array_equal(EA[np.ix_(p, p)], EB)]
        print(f"   edge patterns isomorphic: {bool(iso)}; distinct weight values A: {sorted(set(np.round(WA[~np.eye(n, dtype=bool)], 4)))}, "
              f"B: {sorted(set(np.round(WB[~np.eye(n, dtype=bool)], 4)))}")

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(5):
        y0 = rng.uniform(0, 1, n)
        xa = np.asarray(simulate(WA, jnp.asarray(thA), jnp.asarray(thA[0] * y0), dt=dt, n_steps=N_STEPS)) / thA[0]
        xb = np.asarray(simulate(WA, jnp.asarray(thB), jnp.asarray(thB[0] * y0), dt=dt, n_steps=N_STEPS)) / thB[0]
        worst = max(worst, float(np.abs(xa - xb).max()))
    print(f"\n2. homogeneity in theta (graph of {a}, theta {thA[0]:.3f} vs {thB[0]:.3f}, 5 random y0): "
          f"max |x/theta difference| = {worst:.2e}   (expect ~1e-6 or smaller if exactly homogeneous)")
    print("\nsource of simulate():")
    print(inspect.getsource(simulate))
    print("=" * 72)
