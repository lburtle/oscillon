"""
check_analysis_invariance.py

Circuits A and B (default: seed 32 z1.0 and seed 36 z1.0) have hardened weight matrices that are exact
permutations of each other, and the simulator is homogeneous in theta. So started from the corresponding
points, their trajectories must be identical up to relabeling, and any measurement must agree after relabeling.
This script checks the trajectories and then the measurement (analyze) neuron by neuron.

For each of N_TRIALS random starts it reports:
  * max |difference| between A's relabeled, theta-normalized trajectory and B's (over the analysis window),
  * the reference neuron analyze() picked in each (in B's labels), the period estimates, std per neuron,
  * peaks per neuron in the window (a neuron with ~2x the peaks of the others fires twice per cycle),
  * the firing orders (A mapped into B's labels vs B as measured).

Standalone / non-invasive. Run from the folder with results/, next to the other scripts.
"""
import sys
from itertools import permutations
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from compare_order_cutoffs import BURN_IN, WINDOW, analyze, cyclic_order, dt, peak_times
from oscillon.dynamics import simulate
from support_sequence_analysis import hardened_weights, spec

RESULTS_DIR = Path("results")
N_TRIALS = 6
FRAC = 0.05


def load(name):
    W, th = hardened_weights(spec, jnp.asarray(np.load(RESULTS_DIR / f"{name}.npy")))
    return np.asarray(W), float(np.asarray(th)[0])


def sim(W, theta, y0):
    return np.asarray(simulate(W, jnp.full((W.shape[0],), theta), jnp.asarray(theta * y0),
                               dt=dt, n_steps=BURN_IN + WINDOW))[BURN_IN:]


if __name__ == "__main__":
    a, b = (sys.argv[1:3] if len(sys.argv) >= 3 else ["params_seed32_z1.0", "params_seed36_z1.0"])
    WA, thA = load(a)
    WB, thB = load(b)
    n = WA.shape[0]
    perms = [p for p in permutations(range(n)) if np.array_equal(WA[np.ix_(p, p)], WB)]
    assert perms, "circuits are not permutations of each other"
    p = perms[0]                    # B neuron i  <->  A neuron p[i]
    inv = [p.index(k) for k in range(n)]   # A neuron k  ->  B neuron inv[k]
    print("=" * 78)
    print(f"A = {a} (theta {thA:.3f}), B = {b} (theta {thB:.3f}); B neuron i corresponds to A neuron p[i], p = {p}")
    rng = np.random.default_rng(1)
    for trial in range(N_TRIALS):
        yA = rng.uniform(0, 1, n)
        yB = yA[list(p)]
        xsA, xsB = sim(WA, thA, yA), sim(WB, thB, yB)
        xsA_in_B = xsA[:, list(p)] / thA      # column i is A's neuron p[i] = B's neuron i
        diff = float(np.abs(xsA_in_B - xsB / thB).max())
        print(f"\ntrial {trial}: max |A(relabeled, /theta) - B(/theta)| over the window = {diff:.2e}")
        for tag, xs, relabel in (("A", xsA, lambda k: inv[k]), ("B", xsB, lambda k: k)):
            res = analyze(xs, FRAC)
            std = xs.std(axis=0)
            ref_phys = int(np.argmax(std))
            if res is None:
                print(f"  {tag}: analyze() returned None")
                continue
            per, phases = res
            order_own = list(cyclic_order(phases, sorted(phases)))
            # order in B's labels, rotated to start at the smallest label for comparison
            mapped = [relabel(k) for k in order_own]
            r = mapped.index(min(mapped))
            mapped = mapped[r:] + mapped[:r]
            peaks_B = [0] * n
            for k in range(n):
                peaks_B[relabel(k)] = len(peak_times(xs[:, k]))
            std_B = [0.0] * n
            for k in range(n):
                std_B[relabel(k)] = float(std[k])
            print(f"  {tag}: ref neuron (B labels) {relabel(ref_phys)}, period {per:.3f}, order (B labels) {mapped}")
            print(f"     std by neuron (B labels): {np.round(std_B, 4).tolist()}")
            print(f"     peaks per neuron (B labels): {peaks_B}")
    print("=" * 78)
