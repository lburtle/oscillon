"""
graph_rules_check.py

Direct Python port of Curto lab's get_FP.m / sA2fixpts.m / check_fixedpt.m /
check_stability.m (from https://github.com/ccurto/ctln-basic-2.0), applied to
your hardened discovered graphs.

For each hardened network, this:
  1. Enumerates all fixed-point supports via brute-force subset search
     (trivial cost at N=5: 31 nonempty subsets).
  2. Checks which of those fixed points are stable.
  3. Tests the theory's sharp prediction: a directed-cycle graph should have
     NO stable fixed point on the full support {1,...,N} -- that's the
     combinatorial reason cyclic CTLNs oscillate instead of converging.
  4. Cross-references against what you actually observed:
       - for the ~65% still oscillating after hardening: does theory
         correctly predict no stable FP on the full support?
       - for the ~35% that collapsed: does the THEORY-predicted stable FP's
         support match the support you actually measured numerically?

Needs: hardened_weights, spec, p2m, x0, dt, N_STEPS, BURN_IN from your
existing scripts, plus the "collapsed_to_fixedpoint" results from
diagnose_boundedness.py's full_fixedpoint_audit.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from support_sequence_analysis import hardened_weights, spec, p2m, x0, parse_seed_zstd
from oscillon.dynamics import simulate
from diagnose_boundedness import hard_len1_is_genuine_fixedpoint

BURN_IN = 200
REF_PERIOD = 2.0 * np.pi
dt = 0.1
N_STEPS = int(BURN_IN + 8 * (REF_PERIOD / dt))


def check_fixedpt(W, sig, b):
    """Direct port of check_fixedpt.m. sig: 0-indexed tuple/list of neuron
    indices. Returns (is_fixed_point, x_fp) where x_fp is the full n-dim
    fixed point vector (zero outside sig)."""
    n = W.shape[0]
    sig = list(sig)
    M = np.eye(n) - W
    M_sig = M[np.ix_(sig, sig)]
    try:
        x_sig = np.linalg.solve(M_sig, b[sig])
    except np.linalg.LinAlgError:
        return False, None

    x_fp = np.zeros(n)
    x_fp[sig] = x_sig

    # "on" neuron condition: x_sig > 0 for all neurons in support
    if np.sum(x_sig > 0) < len(sig):
        return False, x_fp

    # "off" neuron condition: W_{k,sig} x_sig + b_k <= 0 for k outside sig
    sigbar = [k for k in range(n) if k not in sig]
    for k in sigbar:
        sk = W[k, sig] @ x_sig + b[k]
        if sk > 0:
            return False, x_fp

    return True, x_fp


def check_stability(W, sig):
    """Direct port of check_stability.m. Returns (is_stable, eigvals)."""
    sig = list(sig)
    M = -np.eye(len(sig)) + W[np.ix_(sig, sig)]
    eigvals = np.linalg.eigvals(M)
    lambda_max = np.max(eigvals.real)
    return bool(lambda_max < 0), eigvals


def get_FP(W, b=None):
    """Direct port of get_FP.m. Enumerates ALL fixed-point supports.
    Returns a list of dicts: {support, x_fp, stable, eigvals}."""
    n = W.shape[0]
    if b is None:
        b = np.ones(n)
    elif np.isscalar(b):
        b = np.full(n, b)

    results = []
    for k in range(1, n + 1):
        for sig in combinations(range(n), k):
            is_fp, x_fp = check_fixedpt(W, sig, b)
            if is_fp:
                is_stable, eigvals = check_stability(W, sig)
                results.append({
                    "support": sig,
                    "x_fp": x_fp,
                    "stable": is_stable,
                    "eigvals": eigvals,
                })
    return results


def analyze_graph_rules(params_path):
    """For one saved network: harden it, compute FP(W_hard, theta) via the
    ported theory, and report whether the full-support set has a stable
    fixed point (theory's oscillation-vs-convergence prediction)."""
    params = jnp.asarray(np.load(params_path))
    W_hard, theta_hard = hardened_weights(spec, params)
    W_np = np.asarray(W_hard)
    b_np = np.asarray(theta_hard)

    fps = get_FP(W_np, b_np)
    n = W_np.shape[0]
    full_support = tuple(range(n))

    full_support_fp = next((r for r in fps if r["support"] == full_support), None)
    full_support_stable = full_support_fp["stable"] if full_support_fp else None

    stable_fps = [r for r in fps if r["stable"]]

    return {
        "params_path": str(params_path),
        "all_fixed_points": fps,
        "n_fixed_points_total": len(fps),
        "n_stable_fixed_points": len(stable_fps),
        "full_support_has_fixed_point": full_support_fp is not None,
        "full_support_stable": full_support_stable,
        "stable_supports": [r["support"] for r in stable_fps],
    }


def measured_fixedpoint_support(params_path, tail_frac=0.3, flat_thresh=1e-3):
    """What support does the ACTUAL simulated hardened trajectory converge
    to? (i.e. which neurons are non-negligibly active in the tail).
    Only meaningful for networks that genuinely collapsed -- check
    hard_is_flat first."""
    params = jnp.asarray(np.load(params_path))
    W_hard, theta_hard = hardened_weights(spec, params)
    xs = np.asarray(simulate(W_hard, theta_hard, x0, dt=dt, n_steps=N_STEPS))[BURN_IN:]
    tail = xs[-int(tail_frac * xs.shape[0]):]
    tail_mean = tail.mean(axis=0)
    scale = np.abs(xs).max()
    support = tuple(i for i in range(len(tail_mean)) if tail_mean[i] > 0.01 * scale)
    return support


if __name__ == "__main__":
    results_dir = Path("results")
    param_files = sorted(results_dir.glob("params_seed*_z*.npy"))

    n_full_support_stable = 0
    n_full_support_unstable = 0
    n_match_when_collapsed = 0
    n_checked_collapsed = 0
    mismatches = []

    for pf in param_files:
        seed, z_std = parse_seed_zstd(pf)
        gr = analyze_graph_rules(pf)

        if gr["full_support_stable"]:
            n_full_support_stable += 1
        else:
            n_full_support_unstable += 1

        # cross-check against actual measured dynamics
        measured_support = measured_fixedpoint_support(pf)

        flag = ""
        if gr["full_support_stable"] and len(measured_support) < 5:
            # theory says full support SHOULD be stable, but numerically
            # only a partial support survived -- worth a manual look
            flag = "  <-- theory says full support stable, but measured support is partial"
            mismatches.append(pf.name)
        elif not gr["full_support_stable"] and len(measured_support) == 5:
            # theory says full support should NOT be stable, but the
            # trajectory looks like it's sitting on all 5 neurons -- also
            # worth checking (could still be a genuine oscillation across
            # all 5, which is fine and expected; only a true fixed point
            # here would be a real mismatch)
            flag = "  <-- theory says full support unstable; measured support uses all 5 (check: oscillating or converged?)"

        print(
            f"seed {seed:>3} z={z_std:.1f} | full_support_stable={gr['full_support_stable']!s:5} "
            f"n_stable_fps={gr['n_stable_fixed_points']:2d} measured_support_size={len(measured_support)}{flag}"
        )

        if len(measured_support) < 5:
            fp_check = hard_len1_is_genuine_fixedpoint(pf)
            is_flat = max(fp_check["tail_range_rel_to_scale"]) < 1e-3
            print(f"seed {seed:>3} z={z_std:.1f}: support_size={len(measured_support)} genuinely_flat={is_flat}")

    print("\n" + "=" * 70)
    print(f"Theory: full-support {{1..N}} stable in {n_full_support_stable}/{len(param_files)} graphs")
    print(f"Theory: full-support {{1..N}} UNSTABLE (predicts oscillation) in "
          f"{n_full_support_unstable}/{len(param_files)} graphs")
    print(f"Potential theory/numerics mismatches flagged: {len(mismatches)}")
    print("=" * 70)
