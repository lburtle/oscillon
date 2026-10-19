"""
emergence_rate_check.py

Apples-to-apples reproduction of the original NeurReps paper's oscillation
emergence rate (97%), applied to the corrected 150-seed sweep, using the
EXACT cycle_emerged + measure_period criterion from n_cycle.py -- copied
verbatim here (not imported) since n_cycle.py has unguarded top-level code
that would execute as an import side effect.

Applied to the SOFT trajectory (W_soft, theta_soft from p2m), matching what
the original paper measured -- this is about the training/discovery step,
not the hardened-topology question from the rest of this session's audit.

Standalone / non-invasive: reads existing params_seed*_z*.npy only, writes
nothing, does not touch n_cycle.py or any other script.
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
import jax.numpy as jnp

from support_sequence_analysis import spec, p2m, x0, parse_seed_zstd
from oscillon.dynamics import simulate

BURN_IN = 200
REF_PERIOD = 2.0 * np.pi
dt = 0.1
N_STEPS = int(BURN_IN + 8 * (REF_PERIOD / dt))
MIN_AMP = 0.05

RESULTS_DIR = Path("results")  # adjust to match your current results dir


# --- copied verbatim from n_cycle.py, not imported ---
def measure_period(xs, dt, rel_thresh=0.5):
    """Peak-to-peak period of neuron 0, in time units. None if no clear cycle."""
    x = np.asarray(xs[:, 0])
    hi = rel_thresh * x.max()
    peaks = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > hi))[0]
    if len(peaks) < 2:
        return None
    return float(np.mean(np.diff(peaks))) * dt


def cycle_emerged(W_t, theta_t, x0, dt, n_steps, burn_in, min_amp=0.05):
    """Judge whether the TRAINED network oscillates. Independent of target period.
    Criteria: sustained activity (tail variance) AND a detectable period."""
    xs = np.asarray(simulate(W_t, theta_t, x0, dt=dt, n_steps=n_steps))[burn_in:]
    tail = xs[-int(0.4 * len(xs)):]
    amp = float(tail.std(0).max())  # is anything still moving?
    per = measure_period(xs, dt)  # is it periodic?
    return (amp > min_amp and per is not None), amp, per
# --- end copied section ---


if __name__ == "__main__":
    param_files = sorted(RESULTS_DIR.glob("params_seed*_z*.npy"))
    print(f"scanning {len(param_files)} param files in {RESULTS_DIR.resolve()}\n")

    by_z = defaultdict(lambda: {"emerged": 0, "total": 0})
    n_emerged = 0
    rows = []

    for pf in param_files:
        seed, z_std = parse_seed_zstd(pf)
        params = jnp.asarray(np.load(pf))
        W_soft, theta_soft = p2m(params)  # <-- adjust if p2m's signature/order differs

        emerged, amp, per = cycle_emerged(
            np.asarray(W_soft), np.asarray(theta_soft), np.asarray(x0),
            dt=dt, n_steps=N_STEPS, burn_in=BURN_IN, min_amp=MIN_AMP,
        )

        by_z[z_std]["total"] += 1
        if emerged:
            by_z[z_std]["emerged"] += 1
            n_emerged += 1

        rows.append((seed, z_std, emerged, amp, per))

    print(f"{'seed/z':>14} | {'emerged':>7} | {'amp':>8} | {'period':>8}")
    for seed, z_std, emerged, amp, per in rows:
        per_str = f"{per:.3f}" if per is not None else "None"
        print(f"seed {seed:>3} z={z_std:.1f} | {str(emerged):>7} | {amp:>8.4f} | {per_str:>8}")

    print("\n" + "=" * 60)
    print(f"Overall: {n_emerged}/{len(param_files)} = {100*n_emerged/len(param_files):.1f}% "
          f"emerged (cycle_emerged criterion, same as original paper)")
    print("Original paper's reported figure: 97%")
    print("\nBy z_std:")
    for z_std in sorted(by_z):
        d = by_z[z_std]
        print(f"  z={z_std}: {d['emerged']}/{d['total']} = {100*d['emerged']/d['total']:.1f}%")
    print("=" * 60)
