"""
period_diagnostic.py

The refit script measures a median hardened period of 2.48 x 2pi, while the abstract reports a median hardened/trained
ratio of 1.19. This prints, for every one of the 107 oscillators, the period from two estimators for the hardened and
the trained network, so the disagreement can be located:
  pe : peak-to-peak of the highest-variance neuron (readout_refit_period_matched.period_est), in time units
  an : analyze() from compare_order_cutoffs (units as returned by that function)
and the ratios hardened/trained for each, plus peak counts per neuron to see whether a neuron fires more than once
per cycle. Standalone; run next to the other scripts.
"""
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from compare_order_cutoffs import RESCUE_SCALE, RESCUE_TRIES, analyze, dt, is_flat
from readout_refit_period_matched import BURN_LONG, N_LONG, REF_PERIOD, period_est, sim
from support_sequence_analysis import hardened_weights, p2m, spec, x0

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows = []
    for pf in sorted(Path("results").glob("params_seed*_z*.npy")):
        params = jnp.asarray(np.load(pf))
        W, th = hardened_weights(spec, params)
        Ws, ths = p2m(params)
        start = x0
        wh = sim(W, th, start, N_LONG)[BURN_LONG:]
        if is_flat(wh):
            for _ in range(RESCUE_TRIES):
                s2 = x0 + RESCUE_SCALE * jnp.asarray(rng.standard_normal(np.shape(x0)))
                w2 = sim(W, th, s2, N_LONG)[BURN_LONG:]
                if not is_flat(w2):
                    start, wh = s2, w2
                    break
            else:
                continue
        wt = sim(Ws, ths, x0, N_LONG)[BURN_LONG:]
        ah, at = analyze(wh, 0.05), analyze(wt, 0.05)
        if ah is None or at is None:
            continue
        rows.append((pf.stem, period_est(wh), period_est(wt), ah[0], at[0]))
    print("=" * 78)
    print(f"{len(rows)} networks")
    ok = [r for r in rows if r[1] and r[2]]
    print(f"pe hardened / 2pi: median {np.median([r[1] for r in ok]) / REF_PERIOD:.2f};  "
          f"pe trained / 2pi: median {np.median([r[2] for r in ok]) / REF_PERIOD:.2f};  "
          f"pe ratio hardened/trained: median {np.median([r[1] / r[2] for r in ok]):.2f}")
    print(f"an hardened: median {np.median([r[3] for r in rows]):.3f};  an trained: median {np.median([r[4] for r in rows]):.3f};  "
          f"an ratio hardened/trained: median {np.median([r[3] / r[4] for r in rows]):.2f}")
    print("-" * 78)
    for r in rows[:8] + [r for r in rows if r[0] == "params_seed24_z2.0"]:
        print(f"{r[0]:22s} pe_h {r[1]}  pe_t {r[2]}  an_h {r[3]:.3f}  an_t {r[4]:.3f}")
    print("=" * 78)
