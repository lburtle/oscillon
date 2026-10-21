"""
compare_identity_examples.py

Quantifies the caption claim "the surviving subnetwork is not the trained
network's most active one" on the post-fix sweep (results/), two ways:

  A. Per-neuron soft vs hardened tail-mean activity (relative to each
     trajectory's own scale) for the three figure examples
     (seed 0 z=2.0 oscillation, seed 4 z=3.0 fixed point, seed 13 z=3.0 sub-cycle).

  B. Threshold-free overlap statistic over ALL reduced-support cases: let k be
     the hardened support size; ask how many of the hardened-active neurons are
     among the soft network's top-k neurons by tail-mean activity. Reports the
     distribution of overlap/k for fixed points and sub-cycles.

Standalone / non-invasive: reads existing params only, writes nothing.
Params must be from the same sweep as the figure (results/).
"""

from collections import Counter
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from graph_rules_check import measured_fixedpoint_support
from oscillon.ctln.dynamics import simulate
from support_sequence_analysis import hardened_weights, spec, p2m, x0, parse_seed_zstd

RESULTS_DIR = Path("results")
BURN_IN = 200
dt = 0.1
N_STEPS = int(BURN_IN + 8 * (2.0 * np.pi / dt))
TAIL_FRAC = 0.3
EXAMPLES = [(0, 2.0), (4, 3.0), (13, 3.0)]


def tail_means(xs):
    tail = xs[-int(TAIL_FRAC * xs.shape[0]):]
    scale = max(np.abs(xs).max(), 1e-12)
    return tail.mean(axis=0) / scale


def soft_xs(params):
    W, th = p2m(params)
    return np.asarray(simulate(W, th, x0, dt=dt, n_steps=N_STEPS))[BURN_IN:]


def hard_xs(params):
    W, th = hardened_weights(spec, params)
    return np.asarray(simulate(W, th, x0, dt=dt, n_steps=N_STEPS))[BURN_IN:]


if __name__ == "__main__":
    print("=== A. figure examples: tail mean / scale per neuron ===")
    for seed, z in EXAMPLES:
        pf = RESULTS_DIR / f"params_seed{seed}_z{z}.npy"
        params = jnp.asarray(np.load(pf))
        ms, mh = tail_means(soft_xs(params)), tail_means(hard_xs(params))
        print(f"\nseed {seed} z={z}")
        print(f"{'neuron':>7} | {'soft':>7} | {'hard':>7}")
        for i in range(len(ms)):
            print(f"{i:>7} | {ms[i]:>7.3f} | {mh[i]:>7.3f}")
        print("  soft rank (most active first):", list(np.argsort(-ms)))
        print("  hard support (>1% of scale):  ", [i for i in range(len(mh)) if mh[i] > 0.01])

    print("\n=== B. top-k overlap over all reduced-support cases ===")
    overlap = {"fixed_point_or_slow": [], "other_reduced": []}
    for pf in sorted(RESULTS_DIR.glob("params_seed*_z*.npy")):
        seed, z = parse_seed_zstd(pf)
        support = measured_fixedpoint_support(pf)
        if len(support) == 5:
            continue
        params = jnp.asarray(np.load(pf))
        ms = tail_means(soft_xs(params))
        k = len(support)
        topk = set(np.argsort(-ms)[:k])
        frac = len(topk & set(support)) / k
        # coarse bucket by hardened support size (2-3 mostly fixed points, 4 sub-cycles)
        overlap["fixed_point_or_slow" if k <= 3 else "other_reduced"].append(frac)

    for name, vals in overlap.items():
        if not vals:
            continue
        vals = np.array(vals)
        print(f"\n{name}: n={len(vals)}")
        print("  overlap fraction counts:", dict(sorted(Counter(np.round(vals, 2)).items())))
        print(f"  fully inside soft top-k: {(vals == 1.0).sum()}/{len(vals)}; "
              f"no overlap: {(vals == 0.0).sum()}/{len(vals)}")
