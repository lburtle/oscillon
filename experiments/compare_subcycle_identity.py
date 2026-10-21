"""
compare_subcycle_identity.py

For the same sub-cycle example build_continuous_trace_figure_v2.py selects
(largest max_rel among subcycle_collapse candidates, via your validated
measured_fixedpoint_support / hard_len1_is_genuine_fixedpoint functions),
this reports the ACTUAL neuron identities with meaningful activity in the
soft trajectory vs. the hardened trajectory -- using the same 0.01*scale
support threshold as measured_fixedpoint_support, applied to both regimes
for an apples-to-apples comparison -- and states explicitly whether it's the
same node subset or a genuinely different one.

Standalone / non-invasive: reads existing params only, writes nothing.
"""

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from support_sequence_analysis import hardened_weights, spec, p2m, x0, parse_seed_zstd
from graph_rules_check import measured_fixedpoint_support
from diagnose_boundedness import hard_len1_is_genuine_fixedpoint
from oscillon.ctln.dynamics import simulate

BURN_IN = 200
REF_PERIOD = 2.0 * np.pi
dt = 0.1
N_STEPS = int(BURN_IN + 8 * (REF_PERIOD / dt))
TAIL_FRAC = 0.3
SUPPORT_THRESH = 0.01  # same as measured_fixedpoint_support

RESULTS_DIR = Path("results_before_bug_fix")  # adjust to match your current results dir


def hard_trajectory(params):
    W_hard, theta_hard = hardened_weights(spec, params)
    xs = np.asarray(simulate(W_hard, theta_hard, x0, dt=dt, n_steps=N_STEPS))[BURN_IN:]
    return xs


def soft_trajectory(params):
    W_soft, theta_soft = p2m(params)  # adjust if p2m's signature differs
    xs = np.asarray(simulate(W_soft, theta_soft, x0, dt=dt, n_steps=N_STEPS))[BURN_IN:]
    return xs


def support_and_means(xs, tail_frac=TAIL_FRAC, thresh_frac=SUPPORT_THRESH):
    tail = xs[-int(tail_frac * xs.shape[0]):]
    tail_mean = tail.mean(axis=0)
    scale = np.abs(xs).max()
    support = tuple(i for i in range(len(tail_mean)) if tail_mean[i] > thresh_frac * scale)
    rel_means = tail_mean / max(scale, 1e-12)  # per-neuron tail mean, relative to scale
    return support, rel_means


if __name__ == "__main__":
    param_files = sorted(RESULTS_DIR.glob("params_seed*_z*.npy"))

    subcycle_candidates = []  # (max_rel, seed, z_std, params)
    for pf in param_files:
        seed, z_std = parse_seed_zstd(pf)
        params = jnp.asarray(np.load(pf))
        support = measured_fixedpoint_support(pf)
        if len(support) == 5:
            continue
        fp_check = hard_len1_is_genuine_fixedpoint(pf)
        max_rel = max(fp_check["tail_range_rel_to_scale"])
        is_flat = max_rel < 1e-3
        if not is_flat:
            subcycle_candidates.append((max_rel, seed, z_std, params))

    if not subcycle_candidates:
        raise RuntimeError("no subcycle candidates found")

    subcycle_candidates.sort(key=lambda t: -t[0])
    max_rel, seed, z_std, params = subcycle_candidates[0]

    print(f"Selected sub-cycle example: seed {seed} z={z_std} (max_rel={max_rel:.3f})\n")

    xs_soft = soft_trajectory(params)
    xs_hard = hard_trajectory(params)

    support_soft, means_soft = support_and_means(xs_soft)
    support_hard, means_hard = support_and_means(xs_hard)

    print("Per-neuron tail mean (relative to scale):")
    print(f"{'neuron':>8} | {'soft':>10} | {'hard':>10}")
    for i in range(len(means_soft)):
        print(f"{i:>8} | {means_soft[i]:>10.4f} | {means_hard[i]:>10.4f}")

    print(f"\nSoft support (>{SUPPORT_THRESH}*scale):  {support_soft}")
    print(f"Hard support (>{SUPPORT_THRESH}*scale):  {support_hard}")

    set_soft, set_hard = set(support_soft), set(support_hard)
    if set_soft == set_hard:
        print("\n-> SAME neuron subset in both soft and hardened dynamics.")
    else:
        only_soft = set_soft - set_hard
        only_hard = set_hard - set_soft
        shared = set_soft & set_hard
        print("\n-> DIFFERENT neuron subsets.")
        print(f"   shared: {sorted(shared)}")
        print(f"   only in soft: {sorted(only_soft)}")
        print(f"   only in hard: {sorted(only_hard)}")
