"""
verify_same_attractor_long_window.py

Checks the "hardened and trained networks never share the same attractor (0/150)"
claim with long windows, because support_sequence_analysis.py flagged
"windows possibly too short: 137/150".

For each network: simulate soft and hardened for a long window after a long
burn-in, convert each trajectory to its cyclic sequence of active-support
chambers (per-neuron relative threshold, flicker shorter than MIN_RUN steps
dropped, consecutive duplicates merged), find the minimal repeating cycle, and
compare the two cycles up to rotation (phase). A case with no detectable
repeating cycle in either regime is reported as UNRESOLVED, not as "different".

Standalone / non-invasive: reads existing params only, writes nothing.
Needs: simulate, hardened_weights, p2m, spec, x0, parse_seed_zstd (same imports
as the other scripts). Params must come from the sweep being reported (results/).
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from oscillon.dynamics import simulate
from support_sequence_analysis import hardened_weights, spec, p2m, x0, parse_seed_zstd

RESULTS_DIR = Path("results")
dt = 0.1
BURN_IN = 1000
WINDOW = 6000           # steps analysed after burn-in (~95 reference periods)
REL_THRESH = 0.02       # per-neuron fraction of that neuron's own peak
GLOBAL_FLOOR = 1e-3     # neurons below this fraction of the global peak count as silent
MIN_RUN = 3             # drop chamber runs shorter than this (threshold flicker)


def support_word(xs):
    peak = xs.max(axis=0)
    gpeak = max(peak.max(), 1e-12)
    thr = np.where(peak > GLOBAL_FLOOR * gpeak, REL_THRESH * peak, np.inf)
    active = xs > thr[None, :]
    tuples = [tuple(row) for row in active]

    runs = []  # (support, duration)
    for s in tuples:
        if runs and runs[-1][0] == s:
            runs[-1][1] += 1
        else:
            runs.append([s, 1])
    runs = runs[1:-1]                       # drop partial first/last chambers
    runs = [r for r in runs if r[1] >= MIN_RUN]

    word = []
    for s, _ in runs:
        if not word or word[-1] != s:
            word.append(s)
    return word


def canonical_cycle(word):
    """Minimal repeating cycle of the word, rotation-canonical; None if none found."""
    n = len(word)
    for p in range(1, n // 2 + 1):
        if all(word[i] == word[i + p] for i in range(n - p)):
            cyc = word[:p]
            return min(tuple(cyc[i:] + cyc[:i]) for i in range(p))
    return None


def run(params, which):
    n_steps = BURN_IN + WINDOW
    if which == "soft":
        W, th = p2m(params)
    else:
        W, th = hardened_weights(spec, params)
    return np.asarray(simulate(W, th, x0, dt=dt, n_steps=n_steps))[BURN_IN:]


if __name__ == "__main__":
    same, different, unresolved = [], 0, []
    files = sorted(RESULTS_DIR.glob("params_seed*_z*.npy"))
    for pf in files:
        seed, z = parse_seed_zstd(pf)
        params = jnp.asarray(np.load(pf))
        cs = canonical_cycle(support_word(run(params, "soft")))
        ch = canonical_cycle(support_word(run(params, "hard")))
        if cs is None or ch is None:
            unresolved.append((seed, z, cs is None, ch is None))
        elif cs == ch:
            same.append((seed, z))
        else:
            different += 1

    print("=" * 60)
    print(f"LONG-WINDOW ATTRACTOR COMPARISON (n={len(files)}, window={WINDOW} steps)")
    print(f"  same cycle up to phase: {len(same)}/{len(files)}")
    print(f"  different cycle:        {different}/{len(files)}")
    print(f"  unresolved:             {len(unresolved)}/{len(files)}")
    print("=" * 60)
    if same:
        print("same:", same)
    if unresolved:
        print("unresolved (seed, z, soft_none, hard_none):", unresolved)
