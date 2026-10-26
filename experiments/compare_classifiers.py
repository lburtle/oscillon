"""
compare_classifiers.py

Finds the network(s) classified differently by the audit (check_task_fidelity_v2.py: measured fixed-point support size;
size 5 = full support -> 97) and by the activity-cutoff classifier used in the sweeps (>= 5% of the largest std on all
five neurons -> 98). Prints, for every network where the two disagree about "all five neurons active", the measured
support, the per-neuron std fraction and the activity count, so the disagreement can be inspected directly.

Standalone / non-invasive. Run from the folder with results/, next to the other scripts. It imports
measured_fixedpoint_support from graph_rules_check (as check_task_fidelity_v2.py does).
"""
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from compare_order_cutoffs import RESCUE_SCALE, RESCUE_TRIES, analyze, is_flat
from graph_rules_check import measured_fixedpoint_support
from readout_refit_period_matched import BURN_LONG, N_LONG, sim
from support_sequence_analysis import hardened_weights, spec, x0

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("=" * 78)
    n_audit5 = n_mine5 = 0
    for pf in sorted(Path("results").glob("params_seed*_z*.npy")):
        params = jnp.asarray(np.load(pf))
        W, th = hardened_weights(spec, params)
        support = list(measured_fixedpoint_support(pf))
        w = sim(W, th, x0, N_LONG)[BURN_LONG:]
        rescued = False
        if is_flat(w):
            for _ in range(RESCUE_TRIES):
                w2 = sim(W, th, x0 + RESCUE_SCALE * jnp.asarray(rng.standard_normal(np.shape(x0))), N_LONG)[BURN_LONG:]
                if not is_flat(w2):
                    w, rescued = w2, True
                    break
        a = None if is_flat(w) else analyze(w, 0.05)
        k = len(a[1]) if a else 0
        audit5, mine5 = len(support) == 5, k == 5
        n_audit5 += audit5
        n_mine5 += mine5
        if audit5 != mine5:
            std = w.std(0)
            print(f"{pf.stem}: measured support {support} (size {len(support)}); activity count at 5% cutoff = {k}; "
                  f"rescued={rescued}; std / max std per neuron = {np.round(std / std.max(), 3).tolist()}")
    print(f"full support by measured support: {n_audit5};  by 5% activity cutoff (oscillating only): {n_mine5}")
    print("=" * 78)
