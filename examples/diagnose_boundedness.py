"""
diagnose_boundedness.py

One-off check: are the length-1 support sequences from support_sequence_analysis.py
genuine bounded oscillations, or unbounded transients that haven't diverged yet
within the training/eval window?
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from oscillon.dynamics import simulate
from oscillon.topology import BlockSoftSpec, make_block_param_to_model
from support_sequence_analysis import hardened_weights

# --- must match n_cycle.py / support_sequence_analysis.py exactly ---
N = 5
dt = 0.1
theta_init = 0.5
eps, delta = 0.10, 0.50
BURN_IN = 200
REF_PERIOD = 2.0 * np.pi
N_STEPS = int(BURN_IN + 8 * (REF_PERIOD / dt))  # ~702, the actual training/eval window

spec = BlockSoftSpec(
    sizes=(N,),
    cross_deltas=np.zeros((1, 1)),
    eps=eps,
    delta=delta,
    theta_init=theta_init,
)
p2m = make_block_param_to_model(spec)
x0 = jnp.full((N,), 0.1)


def diagnose_boundedness(params_path, n_steps_extended=5000):
    params = jnp.asarray(np.load(params_path))
    W_soft, theta_soft = p2m(params)

    xs = np.asarray(simulate(W_soft, theta_soft, x0, dt=dt, n_steps=n_steps_extended))

    train_window = xs[BURN_IN:N_STEPS]
    min_val_train = train_window.min()

    chunk = n_steps_extended // 5
    envelopes = [np.abs(xs[i * chunk:(i + 1) * chunk]).max() for i in range(5)]

    return {
        "params_path": str(params_path),
        "min_val_in_training_window": float(min_val_train),
        "envelope_by_fifth": envelopes,
        "still_growing": envelopes[-1] > 1.5 * envelopes[2],
    }

def diagnose_near_zero_pattern(params_path, n_steps=N_STEPS):
    """Which neuron(s) approach zero, and is it a brief periodic dip or a
    persistently suppressed node?"""
    params = jnp.asarray(np.load(params_path))
    W_soft, theta_soft = p2m(params)

    xs = np.asarray(simulate(W_soft, theta_soft, x0, dt=dt, n_steps=n_steps))
    tail = xs[BURN_IN:]

    near_zero_thresh = 0.01  # tune if needed relative to your amplitude scale
    per_neuron_min = tail.min(axis=0)
    per_neuron_frac_near_zero = (tail < near_zero_thresh).mean(axis=0)

    return {
        "params_path": str(params_path),
        "per_neuron_min": per_neuron_min.tolist(),
        "per_neuron_frac_time_near_zero": per_neuron_frac_near_zero.tolist(),
    }

def hard_len1_is_genuine_fixedpoint(params_path, tail_frac=0.3):
    """For hard_len==1 cases: is the hardened trajectory's tail genuinely
    flat (converged fixed point), or still changing at a scale the 0.02
    threshold happens to miss?"""
    params = jnp.asarray(np.load(params_path))
    W_hard, theta_hard = hardened_weights(spec, params)
    xs = np.array(simulate(W_hard, theta_hard, x0, dt=dt, n_steps=N_STEPS))[BURN_IN:]

    tail = xs[-int(tail_frac * xs.shape[0]):]
    tail_range = tail.max(axis=0) - tail.min(axis=0)
    overall_scale = np.abs(xs).max()

    return {
        "params_path": str(params_path),
        "tail_range_per_neuron": tail_range.tolist(),
        "tail_range_rel_to_scale": (tail_range / max(overall_scale, 1e-9)).tolist(),
    }

if __name__ == "__main__":
    results_dir = Path("results")
    all_files = sorted(results_dir.glob("params_seed*_z*.npy"))
    sample = all_files[::len(all_files) // 10][:10]

    for pf in sample:
        b = diagnose_boundedness(pf)
        z = diagnose_near_zero_pattern(pf)

        flag = "  <-- STILL GROWING" if b["still_growing"] else ""
        mins_str = ", ".join(f"{v:+.4f}" for v in z["per_neuron_min"])
        frac_str = ", ".join(f"{v:.3f}" for v in z["per_neuron_frac_time_near_zero"])

        print(f"{pf.name}{flag}")
        print(f"  min_val_train={b['min_val_in_training_window']:+.4f}  envelope={b['envelope_by_fifth']}")
        print(f"  per-neuron min:      [{mins_str}]")
        print(f"  frac time near zero: [{frac_str}]")
        print()

        r = hard_len1_is_genuine_fixedpoint(pf)
        print(f"{pf.name}")
        print(f"  tail_range_per_neuron:   {r['tail_range_per_neuron']}")
        print(f"  tail_range_rel_to_scale: {r['tail_range_rel_to_scale']}")

def soft_is_genuine_oscillator(params_path, tail_frac=0.3):
    """Same flatness check as hard_len1_is_genuine_fixedpoint, but on the
    SOFT trajectory -- confirms the soft network is a real oscillator,
    not just something word-length happened to call non-1."""
    params = jnp.asarray(np.load(params_path))
    W_soft, theta_soft = p2m(params)
    xs = np.asarray(simulate(W_soft, theta_soft, x0, dt=dt, n_steps=N_STEPS))[BURN_IN:]

    tail = xs[-int(tail_frac * xs.shape[0]):]
    tail_range = tail.max(axis=0) - tail.min(axis=0)
    overall_scale = np.abs(xs).max()
    is_flat = max(tail_range / max(overall_scale, 1e-9)) < 1e-3

    return not is_flat  # True == genuinely still oscillating

def wilson_ci(successes, n, z=1.96):
    """95% Wilson score interval for a binomial proportion."""
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return center - margin, center + margin

def full_fixedpoint_audit(results_dir="results_before_bug_fix"):
    """For every seed, check: (a) what hard_len support_sequence_analysis
    reported, (b) whether the hardened tail is genuinely flat. Produces the
    real headline rate: what fraction of trained networks lose their
    oscillation entirely upon hardening."""
    from support_sequence_analysis import analyze_seed, parse_seed_zstd

    param_files = sorted(Path(results_dir).glob("params_seed*_z*.npy"))
    n_collapsed = 0
    n_total = 0
    rows = []

    for pf in param_files:
        seq_result = analyze_seed(pf)
        fp_result = hard_len1_is_genuine_fixedpoint(pf)
        is_flat_hard = max(fp_result["tail_range_rel_to_scale"]) < 1e-3
        soft_oscillating = soft_is_genuine_oscillator(pf)
        seed, z_std = parse_seed_zstd(pf)

        rows.append({
            "params_path": str(pf),
            "seed": seed,
            "z_std": z_std,
            "soft_oscillating": soft_oscillating,
            "collapsed_to_fixedpoint": is_flat_hard,  # only meaningful if soft_oscillating
        })

    genuine = [r for r in rows if r["soft_oscillating"]]
    n_genuine = len(genuine)
    n_collapsed = sum(r["collapsed_to_fixedpoint"] for r in genuine)

    print(f"Genuine soft oscillators: {n_genuine}/{len(rows)}")
    print(f"Of those, collapsed to fixed point when hardened: {n_collapsed}/{n_genuine} "
          f"({100*n_collapsed/n_genuine:.0f}%)")

    print("\nBreakdown by z_std:")
    for z in sorted(set(r["z_std"] for r in genuine)):
        subset = [r for r in genuine if r["z_std"] == z]
        n_sub = len(subset)
        n_sub_collapsed = sum(r["collapsed_to_fixedpoint"] for r in subset)
        print(f"  z_std={z}: {n_sub_collapsed}/{n_sub} ({100*n_sub_collapsed/n_sub:.0f}%)")

    lo, hi = wilson_ci(n_collapsed, len(rows))
    print(f"{n_collapsed}/{len(rows)}% collapse rate, 95% CI: [{lo:.2f}, {hi:.2f}]")

    for thresh in [0.05, 0.1, 0.2, 0.3]:
        rows = []
        for pf in param_files:  # reuse your existing list
            fp_result = hard_len1_is_genuine_fixedpoint(pf)
            is_flat = max(fp_result["tail_range_rel_to_scale"]) < thresh
            rows.append(is_flat)
        n_flat = sum(rows)
        print(f"threshold={thresh}: {n_flat}/150 collapsed ({100*n_flat/150:.0f}%)")

    vals = []
    for pf in param_files:
        fp_result = hard_len1_is_genuine_fixedpoint(pf)
        vals.append(max(fp_result["tail_range_rel_to_scale"]))
    vals = np.array(vals)

    percentiles = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    for p, v in zip(percentiles, np.percentile(vals, percentiles)):
        print(f"  p{p:3d}: {v:.6f}")

    vals = np.array([max(hard_len1_is_genuine_fixedpoint(pf)["tail_range_rel_to_scale"])
                  for pf in param_files])
    collapsed = (vals < 0.03).sum()
    degraded = ((vals >= 0.03) & (vals < 0.9)).sum()
    preserved = (vals >= 0.9).sum()
    print(f"collapsed: {collapsed}, degraded: {degraded}, preserved: {preserved}")

    return rows

if __name__ == "__main__":
    full_fixedpoint_audit()
