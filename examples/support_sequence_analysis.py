"""
support_sequence_analysis.py

Phase-invariant comparison of soft vs. hardened network dynamics, via the
sequence of active supports (chambers) visited rather than raw trajectory
MSE. Resolves whether the measured "hardening divergence" reflects a real
difference in dynamical regime, or is a phase-alignment artifact.

Reads saved params from a prior sweep (results_prefix_bug/params_seed*.npy).
These are untouched by the hardened_ctln_check bug -- they're written via
np.save(res.params) immediately after training, before that buggy call ever
runs, so they're safe to reuse even from the pre-fix batch.
"""

from pathlib import Path
import re

import jax
import jax.numpy as jnp
import numpy as np

from oscillon.dynamics import simulate
from oscillon.topology import BlockSoftSpec, make_block_param_to_model

# --- must match the spec that produced the saved params (from n_cycle.py) ---
N = 5
dt = 0.1
theta_init = 0.5
eps, delta = 0.10, 0.50

spec = BlockSoftSpec(
    sizes=(N,),
    cross_deltas=np.zeros((1, 1)),
    eps=eps,
    delta=delta,
    theta_init=theta_init,
)
p2m = make_block_param_to_model(spec)
x0 = jnp.full((N,), 0.1)

BURN_IN = 200
REF_PERIOD = 2.0 * np.pi
DT = 0.1
N_STEPS = int(BURN_IN + 8 * (REF_PERIOD / DT)) # == n_steps_sweep from run_sweep, ~702 steps


def hardened_weights(spec, params):
    """Mirror of the hardening step in hardened_ctln_check: threshold gates,
    snap to band endpoints, keep theta as trained."""
    n = spec.n_total
    rows, cols, eps_e, delta_e, W_cross = spec.compile_indices()
    eps_j, delta_j = jnp.asarray(eps_e), jnp.asarray(delta_e)
    n_edges = spec.n_edges

    if spec.learn_theta:
        theta = params[n_edges:]
        if theta.shape[0] == 1:
            theta = jnp.full((n,), theta[0])
    else:
        theta = jnp.full((n,), spec.theta_init)

    z = params[:n_edges]
    gate = jax.nn.sigmoid(z)
    edge_present = gate > 0.5
    w_hard_off = jnp.where(edge_present, -1.0 + eps_j, -1.0 - delta_j)
    W_hard = jnp.asarray(W_cross).at[jnp.asarray(rows), jnp.asarray(cols)].set(w_hard_off)
    return W_hard, theta


def active_support(xs, rel_thresh=0.02):
    """Binary support per timestep. Threshold is relative to this trajectory's
    own amplitude scale, so it's comparable across networks with very different
    absolute amplitudes, and sensitive enough to catch genuine near-zero
    boundary-hugging as an actual chamber transition."""
    scale = np.abs(xs).max()
    thresh = rel_thresh * scale
    sigma = np.asarray(xs) > thresh
    return [tuple(row) for row in sigma]


def collapse_to_word(sigma_seq):
    """Collapse consecutive repeats into the ordered sequence of DISTINCT
    chambers visited. This is where phase/timing gets discarded on purpose --
    only 'what comes after what' survives."""
    word = []
    for s in sigma_seq:
        if not word or word[-1] != s:
            word.append(s)
    return word


def is_rotation(word_a, word_b):
    """Is word_b a cyclic rotation of word_a? Handles pure phase-offset cases."""
    if len(word_a) != len(word_b):
        return False
    if len(word_a) == 0:
        return True
    doubled = word_a + word_a
    n = len(word_b)
    return any(doubled[i:i + n] == word_b for i in range(len(word_a)))


def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return dp[m, n]


def cyclic_edit_distance(word_a, word_b):
    """Edit distance between word_a and the best rotation of word_b.
    Fine for short words -- chamber sequences here are typically single-
    digit to low tens of entries, so the O(len_b) rotations x Levenshtein
    cost is trivial."""
    if len(word_b) == 0:
        return len(word_a)
    doubled = word_b + word_b
    return min(
        levenshtein(word_a, doubled[i:i + len(word_b)])
        for i in range(len(word_b))
    )


def half_window_consistent(sigma_seq):
    """Sanity check: does the chamber word from the first half of the window
    match (up to rotation) the word from the second half? If not, N_STEPS is
    probably too short to have settled into the actual limit cycle, and any
    soft-vs-hard comparison built on this trajectory isn't trustworthy yet."""
    mid = len(sigma_seq) // 2
    w1 = collapse_to_word(sigma_seq[:mid])
    w2 = collapse_to_word(sigma_seq[mid:])
    return is_rotation(w1, w2), w1, w2

def tail_consistent(sigma_seq, n_periods_to_check=3, approx_period_steps=int(REF_PERIOD / DT)):
    """Check whether the last few periods of the trajectory are internally
    consistent, i.e. dynamics have actually settled by the end of the verified window,
    rather than comparing arbitrary first/second halves."""
    window = n_periods_to_check * approx_period_steps
    if len(sigma_seq) < 2 * window:
        window = len(sigma_seq) // 2  # fall back if trajectory is short
    tail = sigma_seq[-2 * window:]
    w1 = collapse_to_word(tail[:window])
    w2 = collapse_to_word(tail[window:])
    return is_rotation(w1, w2), w1, w2

def analyze_seed(params_path, dt=dt, n_steps=N_STEPS, burn_in=BURN_IN):
    params = jnp.asarray(np.load(params_path))

    W_soft, theta_soft = p2m(params)
    W_hard, theta_hard = hardened_weights(spec, params)

    xs_soft = np.asarray(simulate(W_soft, theta_soft, x0, dt=dt, n_steps=n_steps))[burn_in:]
    xs_hard = np.asarray(simulate(W_hard, theta_hard, x0, dt=dt, n_steps=n_steps))[burn_in:]

    sigma_soft = active_support(xs_soft)
    sigma_hard = active_support(xs_hard)

    soft_settled, _, _ = tail_consistent(sigma_soft)
    hard_settled, _, _ = tail_consistent(sigma_hard)

    word_soft = collapse_to_word(sigma_soft)
    word_hard = collapse_to_word(sigma_hard)

    same_attractor = is_rotation(word_soft, word_hard)
    dist = 0 if same_attractor else cyclic_edit_distance(word_soft, word_hard)

    return {
        "params_path": str(params_path),
        "soft_settled": soft_settled,
        "hard_settled": hard_settled,
        "word_soft_len": len(word_soft),
        "word_hard_len": len(word_hard),
        "same_attractor_up_to_phase": same_attractor,
        "cyclic_edit_distance": dist,
        "word_soft": word_soft,
        "word_hard": word_hard,
        "sigma_soft": sigma_soft,
        "sigma_hard": sigma_hard,
    }


def parse_seed_zstd(path):
    m = re.match(r"params_seed(\d+)_z([\d.]+)\.npy", Path(path).name)
    return (int(m.group(1)), float(m.group(2))) if m else (None, None)


if __name__ == "__main__":
    results_dir = Path("results")
    param_files = sorted(results_dir.glob("params_seed*_z*.npy"))
    print(f"found {len(param_files)} saved parameter files")

    summary = []
    n_unsettled = 0
    for pf in param_files:
        seed, z_std = parse_seed_zstd(pf)
        r = analyze_seed(pf)
        r["seed"], r["z_std"] = seed, z_std
        summary.append(r)

        flag = ""
        if not (r["soft_settled"] and r["hard_settled"]):
            flag = "  <-- WINDOW MAY BE TOO SHORT, check manually"
            n_unsettled += 1

        print(
            f"seed {seed:>3} z={z_std:.1f} | "
            f"soft_len={r['word_soft_len']:>3} hard_len={r['word_hard_len']:>3} | "
            f"same_attractor={r['same_attractor_up_to_phase']!s:5} "
            f"edit_dist={r['cyclic_edit_distance']}{flag}"
        )

    n = len(summary)
    n_same = sum(r["same_attractor_up_to_phase"] for r in summary)
    print("\n" + "=" * 60)
    print(f"SUPPORT-SEQUENCE COMPARISON  (n={n} networks)")
    print(f"  same attractor up to phase: {n_same}/{n} ({100*n_same/n:.0f}%)")
    print(f"  windows possibly too short: {n_unsettled}/{n}")
    print("=" * 60)
