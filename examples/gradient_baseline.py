import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

import jax
import jax.numpy as jnp

# imports: build spec, p2m builder, init, simulate, cycle_emerged, analyze_interior_fixed_point

from oscillon.topology import (
    BlockSoftSpec,
    block_gate_matrix,
    extract_block_adjacency,
    make_block_param_to_model,
    init_params_asymmetric
)

from oscillon.dynamics import simulate
from oscillon.graph import MotifSpec, NetworkSpec

from topology_analysis import analyze_interior_fixed_point

# imports: gradient_baseline function

from gradient import gradient_baseline


out = Path(__file__).parent / "results"; out.mkdir(exist_ok=True)

# ------------------
N = 5
dt, n_steps, burn_in, ridge = 0.1, 2000, 200, 1e-6
theta_init = 0.5

print("HYPERPARAMETERS")
print("=" * 50)
print(
    f"N={N} || dt={dt} || n_steps={n_steps} || burn_in={burn_in}, theta_init={theta_init}"
)

graph_params = []
graph_params.append(
    {
        "N": N,
        "dt": dt,
        "n_steps": n_steps,
        "burn_in": burn_in,
        "theta_init": theta_init,
    }
)
# ------------------

spec = BlockSoftSpec(
    sizes=(N,),
    cross_deltas=np.zeros((1, 1)),
    eps=0.10,
    delta=0.50,
    theta_init=theta_init,
)

def make_targets(n, t, period, amplitudes=None, phases=None, offset=0.0):
    """n sinusoids sharing one period; per-channel amplitude + phase.
    Shared period is required for single-motif linear decodability."""
    omega = 2.0 * np.pi / period
    if phases is None:
        phases = [2.0 * np.pi * k / n for k in range(n)]  # even n-phase spread
    if amplitudes is None:
        amplitudes = [1.0 + 0.5 * (k % 3) for k in range(n)]  # some amplitude variety
    cols = [
        a * jnp.sin(omega * t + ph) + offset
        for a, ph in zip(amplitudes, phases, strict=True)
    ]
    return jnp.stack(cols, axis=1)

def cycle_emerged(W_t, theta_t, x0, dt, n_steps, burn_in, min_amp=0.05):
    """Judge whether the TRAINED network oscillates. Independent of target period.
    Criteria: sustained activity (tail variance) AND a detectable period."""
    xs = np.asarray(simulate(W_t, theta_t, x0, dt=dt, n_steps=n_steps))[burn_in:]
    tail = xs[-int(0.4 * len(xs)) :]
    amp = float(tail.std(0).max())  # is anything still moving?
    per = measure_period(xs, dt)  # is it periodic?
    return (amp > min_amp and per is not None), amp, per

def measure_period(xs, dt, rel_thresh=0.5):
    """Peak-to-peak period of neuron 0, in time units. None if no clear cycle."""
    x = np.asarray(xs[:, 0])
    hi = rel_thresh * x.max()
    peaks = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > hi))[0]
    if len(peaks) < 2:
        return None
    return float(np.mean(np.diff(peaks))) * dt

cycle = NetworkSpec([MotifSpec.sparse(N)])
A_seed = cycle.to_torch_adjacency().numpy()

key = jax.random.PRNGKey(0)
key, sub = jax.random.split(key)

p2m = make_block_param_to_model(spec)
params0 = init_params_asymmetric(spec, sub, z_std=2.0)


x0 = jnp.full((N,), 0.1)
t = jnp.arange(n_steps) * dt
W0, th0 = p2m(params0)
xs0 = simulate(W0, th0, x0, dt=dt, n_steps=n_steps)[burn_in:]
period = measure_period(xs0, dt)

if period is None:
    period = 2.0 * np.pi
    print("WARN: warm start not oscillating; using fallback period", period)
print(
    f"N={N}  matched target period = {period:.3f} time units ({period / dt:.1f} steps)"
)

targets = make_targets(N, t, period)
tgt = targets[burn_in:]

results = []
for seed in range(70):
    key = jax.random.PRNGKey(1000 + seed)
    params0 = init_params_asymmetric(spec, key, z_std=2.0)

    gparams, ghist, xs, pred = gradient_baseline(p2m, params0, x0, targets,
                                       dt=dt, n_steps=n_steps,
                                       burn_in=burn_in, ridge=ridge)
    W, theta = p2m(gparams)
    W_np = np.asarray(W)
    diverged = not np.isfinite(W_np).all()

    if diverged:
        emerged, fp_stable, max_eig = False, False, float("nan")
        print(f"seed {seed} DIVERGED, max_eig set to {max_eig}")
    else:
        xs = simulate(W, theta, x0, dt=dt, n_steps=n_steps)[burn_in:]
        emerged, amp, per = cycle_emerged(W, theta, x0, dt, n_steps, burn_in)
        fp = analyze_interior_fixed_point(W, theta)
        fp_stable, max_eig = fp["stable"], fp["max_real_eig"]

        if pred is not None and seed % 5:

            # --- plots: vertical-offset stacking stays legible as N grows ---
            cmap = plt.cm.viridis(np.linspace(0, 1, N))
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 1.4 * N + 2), sharex=True)

            spacing_a = 1.2 * float(jnp.max(xs))
            for i in range(N):
                ax1.plot(xs[:, i] + i * spacing_a, color=cmap[i], lw=1.0)
                ax1.axvline(burn_in, ls="--", c="gray", lw=0.8)
                ax1.set_yticks([i * spacing_a for i in range(N)])
                ax1.set_yticklabels([f"n{i}" for i in range(N)])
                ax1.set_title(f"trained CTLN activations (N={N})")

            spacing_t = 1.2 * float(jnp.max(jnp.abs(targets)))
            for i in range(N):
                ax2.plot(pred[burn_in:, i] + i * spacing_t, color=cmap[i], lw=1.0)
                ax2.plot(tgt[:, i] + i * spacing_t, ls=":", c="k", lw=0.7)
                ax2.set_yticks([i * spacing_t for i in range(N)])
                ax2.set_yticklabels([f"ch{i}" for i in range(N)])
                ax2.set_title("readout (color) vs target (dotted)")

            plt.tight_layout()
            script_dir = Path(__file__).parent
            plot_dir = script_dir / "gradient_plots"
            plot_dir.mkdir(exist_ok=True)
            plt.savefig(plot_dir / f"scale_N{N}_{seed}.png", dpi=130)
            print(f"saved scale_N{N}_{seed}.png")

    rec = {"seed": seed, "emerged": bool(emerged),
           "final_loss": ghist[-1], "diverged": diverged,
           "fp_stable": bool(fp["stable"]),
           "max_real_eig": float(max_eig)}
    results.append(rec)
    with open(out / "grad_baseline.jsonl", "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    np.save(out / f"grad_params_seed{seed}.npy", np.asarray(gparams))

n = len(results)
print(f"cycle emerged: {sum(r['emerged'] for r in results)}/{n}")
print(f"stable FP:     {sum(r['fp_stable'] for r in results)}/{n}")
