"""
check_seed31_multistability.py

Is hardened (seed 31, z=3.0) multistable?

Background: from the original x0 the hardened network settles on a 4-node
sub-cycle, but the graph rules predict a stable fixed point for its support.
analyze_hardened_oscillators.py never perturbs it, because it is not flat from
x0. This script starts the hardened network from many initial conditions and
records where each one ends up.

  1. Lists every fixed point the graph rules predict for the hardened weights
     (existence: x_sigma > 0 and inactive inputs <= 0; stability: eigenvalues of
     -I + W_sigma_sigma), so you can see which stable FP theory has in mind.
  2. Runs N_STARTS simulations: tiny perturbations of x0 (1e-4 .. 1e-1) and
     random starts over the observed state range.
  3. Classifies each end state (last WINDOW steps of a LONG_FACTOR x longer run,
     because the earlier size-2 cases converged slowly) as flat (fixed point,
     with its support) or oscillating (with its active set).

Standalone / non-invasive. Put it in the same folder as
analyze_hardened_oscillators.py and run from the folder that contains results/.
Usage:  python check_seed31_multistability.py [seed] [z]     (default 31 3.0)
"""

import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from analyze_hardened_oscillators import sim  # noqa: F401  (imports the setup too)
from compare_period_and_order import BURN_IN, WINDOW, dt, is_flat
from oscillon.ctln.dynamics import simulate
from support_sequence_analysis import hardened_weights, spec, x0

RESULTS_DIR = Path("results")
N_PERTURB = 20          # small perturbations of x0
N_RANDOM = 40           # random starts over the state range
PERTURB_SCALES = (1e-4, 1e-3, 1e-2, 1e-1)
LONG_FACTOR = 3         # run LONG_FACTOR x (BURN_IN + WINDOW) steps
ACTIVE_TOL = 1e-3       # a neuron is "on" if its tail mean exceeds this fraction of the max
STRONG_STD_FRAC = 0.20  # oscillating: neuron is active if std >= this fraction of max std


def predicted_fixed_points(W, th, tol=1e-9):
    """All fixed points of dx/dt = -x + [Wx + th]_+ allowed by the graph rules."""
    n = W.shape[0]
    out = []
    for k in range(1, n + 1):
        for sigma in combinations(range(n), k):
            s = list(sigma)
            A = np.eye(k) - W[np.ix_(s, s)]
            try:
                x = np.linalg.solve(A, th[s])
            except np.linalg.LinAlgError:
                continue
            if not np.all(x > tol):
                continue
            full = np.zeros(n)
            full[s] = x
            inp = W @ full + th
            off = [i for i in range(n) if i not in s]
            if off and not np.all(inp[off] <= tol):
                continue
            eig = np.linalg.eigvals(-np.eye(k) + W[np.ix_(s, s)])
            out.append((sigma, x, bool(np.all(eig.real < 0)), float(eig.real.max())))
    return out


def classify(xs):
    """xs: (WINDOW, n) tail of a run -> short label."""
    if is_flat(xs):
        m = xs.mean(axis=0)
        on = tuple(int(i) for i in np.where(m > ACTIVE_TOL * m.max())[0])
        return f"FIXED POINT  support={on}"
    std = xs.std(axis=0)
    on = tuple(int(i) for i in np.where(std >= STRONG_STD_FRAC * std.max())[0])
    return f"OSCILLATION  active={on}"


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 31
    z = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0

    matches = sorted(RESULTS_DIR.glob(f"params_seed{seed}_z*.npy"))
    pf = next((p for p in matches if float(p.stem.split("_z")[-1]) == z), None)
    if pf is None:
        sys.exit(f"no params file for seed {seed}, z={z} in {RESULTS_DIR}/ "
                 f"(found: {[p.name for p in matches]})")
    params = jnp.asarray(np.load(pf))
    W_j, th_j = hardened_weights(spec, params)
    W, th = np.asarray(W_j, dtype=float), np.asarray(th_j, dtype=float)
    n = W.shape[0]
    print(f"{pf.name}: n={n}")

    print("\nFixed points allowed by the graph rules (hardened weights):")
    fps = predicted_fixed_points(W, th)
    if not fps:
        print("  none")
    for sigma, x, stable, maxre in fps:
        print(f"  support={sigma}  x={np.round(x[x > 0], 3)}  "
              f"{'STABLE' if stable else 'unstable'}  (max Re eig = {maxre:+.3f})")

    n_steps = LONG_FACTOR * (BURN_IN + WINDOW)
    x0_np = np.asarray(x0, dtype=float)

    # reference run from the unperturbed x0 (should reproduce the sub-cycle)
    ref = np.asarray(simulate(W_j, th_j, x0, dt=dt, n_steps=n_steps))[-WINDOW:]
    print(f"\nreference (original x0): {classify(ref)}")

    # amplitude range for random starts, taken from the reference trajectory
    top = 1.5 * float(max(ref.max(), x0_np.max(), 1e-6))
    rng = np.random.default_rng(0)

    starts = []
    for i in range(N_PERTURB):
        scale = PERTURB_SCALES[i % len(PERTURB_SCALES)]
        starts.append((f"x0 + {scale:g}*noise", x0_np + scale * rng.standard_normal(n)))
    for _ in range(N_RANDOM):
        starts.append(("random in [0, %.2f]^n" % top, rng.uniform(0.0, top, size=n)))

    tally = Counter()
    by_kind = {}
    for kind, s in starts:
        s = np.maximum(s, 0.0)  # CTLN activity is non-negative
        xs = np.asarray(simulate(W_j, th_j, jnp.asarray(s, dtype=x0.dtype),
                                 dt=dt, n_steps=n_steps))[-WINDOW:]
        label = classify(xs)
        tally[label] += 1
        by_kind.setdefault(kind.split(" ")[0], Counter())[label] += 1

    print(f"\nend states from {len(starts)} starts "
          f"(each run {n_steps} steps, classified on the last {WINDOW}):")
    for label, c in tally.most_common():
        print(f"  {c:>3}  {label}")
    print("\nby start type:")
    for kind, cnt in by_kind.items():
        print(f"  {kind}: " + "; ".join(f"{c}x {l}" for l, c in cnt.most_common()))

    labels = set(tally)
    print("\nverdict:")
    if len(labels) == 1:
        print("  one end state from every start: no multistability found from these starts.")
    else:
        print("  more than one end state: multistable. Report it as coexisting attractors, "
              "not a contradiction of the graph rules, if a STABLE predicted FP appears above.")
