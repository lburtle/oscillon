import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from oscillon.dynamics import simulate
from oscillon.graph import MotifSpec, NetworkSpec, plot_network_graph
from oscillon.model import train
from oscillon.readout import apply_readout, fit_readout
from oscillon.topology import (
    BlockSoftSpec,
    block_gate_matrix,
    extract_block_adjacency,
    init_params_asymmetric,
    make_block_param_to_model,
)

# ----------------------------------------------------------------------
N = 5  # <-- number of nodes; change this to scale
dt, n_steps, burn_in = 0.1, 2000, 200
theta_init = 0.5  # livelier warm start than the 3-cycle default (see notes)

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

# ----------------------------------------------------------------------

script_dir = Path(__file__).parent
output_dir = script_dir / "images"
output_dir.mkdir(exist_ok=True)

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
key = jax.random.PRNGKey(seed)


def measure_period(xs, dt, rel_thresh=0.5):
    """Peak-to-peak period of neuron 0, in time units. None if no clear cycle."""
    x = np.asarray(xs[:, 0])
    hi = rel_thresh * x.max()
    peaks = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > hi))[0]
    if len(peaks) < 2:
        return None
    return float(np.mean(np.diff(peaks))) * dt


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


# --- spec: single motif of size N ---
spec = BlockSoftSpec(
    sizes=(N,),
    cross_deltas=np.zeros((1, 1)),
    eps=0.10,
    delta=0.50,
    theta_init=theta_init,
)

# --- warm start from the discrete N-cycle ---
cycle = NetworkSpec([MotifSpec.sparse(N)])
A_seed = cycle.to_torch_adjacency().numpy()
key, sub = jax.random.split(key)

## Swap between these two for either seeded init (cyclic) or random asymm

# params0 = init_block_params_from_adjacency(spec, sub, A_seed)
params0 = init_params_asymmetric(spec, sub, z_std=2.0)

x0 = jnp.full((N,), 0.1)

# --- measure the warm-start cycle period, build target to match ---
t = jnp.arange(n_steps) * dt
p2m = make_block_param_to_model(spec)
W0, th0 = p2m(params0)
xs0 = simulate(W0, th0, x0, dt=dt, n_steps=n_steps)[burn_in:]
period = measure_period(xs0, dt)
if period is None:
    period = 2.0 * np.pi  # fallback if warm start didn't oscillate
    print("WARN: warm start not oscillating; using fallback period", period)
print(
    f"N={N}  matched target period = {period:.3f} time units ({period / dt:.1f} steps)"
)

targets = make_targets(N, t, period)

target_periods_in_window = 8  # how many full laps you want to see/fit
usable = target_periods_in_window * (period / dt)
n_steps = int(burn_in + usable)
print(f"N={N}  period={period / dt:.0f} steps  ->  n_steps={n_steps}")

# probe: rough period estimate with a generous fixed rollout
probe_steps = 4000
xs_probe = simulate(W0, th0, x0, dt=dt, n_steps=probe_steps)[burn_in:]
period = measure_period(xs_probe, dt)
if period is None:
    period = 2.0 * np.pi
n_steps = int(burn_in + 8 * (period / dt))  # real rollout sized to the cycle

# REBUILD t and targets at the new n_steps
t = jnp.arange(n_steps) * dt
targets = make_targets(N, t, period)

# --- train ---
result = train(
    spec,
    params0,
    x0,
    targets,
    key,
    p2m=p2m,
    dt=dt,
    n_steps=n_steps,
    burn_in=burn_in,
    warm=dict(n_iters=9000, pop=512, sigma=2, lr=0.05),
    cryst=dict(n_iters=5000, pop=512, sigma=1, lr=0.03),
)
print("final reward:", result.history[-1])

# --- evaluate on the trained network ---
xs = simulate(result.W, result.theta, x0, dt=dt, n_steps=n_steps)[burn_in:]
tgt = targets[burn_in:]

m = min(xs.shape[0], tgt.shape[0])
xs, tgt = xs[:m], tgt[:m]

R, b = fit_readout(xs, tgt, ridge=1e-6)
y = apply_readout(xs, R, b)
mse = float(jnp.mean((y - tgt) ** 2))
const = float(jnp.mean((tgt.mean(0) - tgt) ** 2))
print(f"readout MSE {mse:.4f}  vs constant-mean {const:.4f}")

res_dir = Path(__file__).parent / "results"
res_dir.mkdir(exist_ok=True)

emerged, amp, trained_per = cycle_emerged(
    result.W, result.theta, x0, dt=dt, n_steps=n_steps, burn_in=burn_in
)

with open(res_dir / "mse_runs.jsonl", "a") as f:
    f.write(json.dumps({
        "seed": seed,
        "init": "asymmetric",        # or "seeded" — never omit this again
        "N": N, "z_std": 2.0,
        "mse": mse, "const": const,
        "emerged": bool(emerged),
        "amp": float(amp),
        "trained_period": None if trained_per is None else float(trained_per),
        "target_period": float(period),
        "final_reward": float(result.history[-1]),
    }) + "\n")


## --- plots: vertical-offset stacking stays legible as N grows ---
#cmap = plt.cm.viridis(np.linspace(0, 1, N))
#fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 1.4 * N + 2), sharex=True)

#spacing_a = 1.2 * float(jnp.max(xs))
#for i in range(N):
#    ax1.plot(xs[:, i] + i * spacing_a, color=cmap[i], lw=1.0)
#ax1.axvline(burn_in, ls="--", c="gray", lw=0.8)
#ax1.set_yticks([i * spacing_a for i in range(N)])
#ax1.set_yticklabels([f"n{i}" for i in range(N)])
#ax1.set_title(f"trained CTLN activations (N={N})")

#spacing_t = 1.2 * float(jnp.max(jnp.abs(targets)))
#for i in range(N):
#    ax2.plot(y[burn_in:, i] + i * spacing_t, color=cmap[i], lw=1.0)
#    ax2.plot(tgt[:, i] + i * spacing_t, ls=":", c="k", lw=0.7)
#ax2.set_yticks([i * spacing_t for i in range(N)])
#ax2.set_yticklabels([f"ch{i}" for i in range(N)])
#ax2.set_title("readout (color) vs target (dotted)")

#plt.tight_layout()
#plt.savefig(output_dir / f"scale_N{N}.png", dpi=130)
#print(f"saved scale_N{N}.png")

# --- learned topology ---
A_learned = extract_block_adjacency(spec, result.params)
print("kept seed cycle:", np.array_equal(A_learned, A_seed))
#G = block_gate_matrix(spec, result.params)
#plot_network_graph(
#    G,
#    theta=np.asarray(result.theta),
#    save_path=str(output_dir / f"learned_graph_N{N}.png"),
#)

from phase_grid import plot_phase_grid
from phase_portrait import plot_phase_portrait

W0, th0 = p2m(params0)

#if N == 3:
    # 3-cycle, view neurons 0 and 1 directly, several initial conditions
  #  x0s = [
  #      np.array([0.1, 0.0, 0.0]),
  #      np.array([0.5, 0.5, 0.0]),
  #      np.array([1.0, 0.2, 0.6]),
  #  ]

 #   before_3 = plot_phase_portrait(
 #       W0,
 #       th0,
 #       dims=("pca", np.asarray(xs)),
 #       x0s=x0s,
 #       title="3-cycle phase portrait (before)",
 #       save_path="images/phase_portraits/3-cycle_phase_before.png",
 #   )

   #after_3 = plot_phase_portrait(
   #     result.W,
   #     result.theta,
  #      dims=("pca", np.asarray(xs)),
  #      x0s=x0s,
  #      title="3-cycle phase portrait (after)",
  #      save_path="images/phase_portraits/3-cycle_phase_after.png",
  #  )

 #   before_3_grid = plot_phase_grid(
 ##       W0,
 #       th0,
 #       x0s=x0s,
 #       title="3-cycle phase portrait (before)",
 #       save_path="images/phase_portraits/3-cycle_grid_before.png",
 #   )

#    after_3_grid = plot_phase_grid(
#        result.W,
#        result.theta,
#        x0s=x0s,
#        title="3-cycle phase portrait (after)",
#        save_path="images/phase_portraits/3-cycle_grid_after.png",
#    )

#else:
    # any N: project onto the PCA plane of the learned limit cycle
#    xs = simulate(result.W, result.theta, x0, dt=dt, n_steps=n_steps)[burn_in:]

#    before_n = plot_phase_portrait(
     #   W0,
    #    th0,
   #     dims=("pca", np.asarray(xs)),
  #      x0s=[x0],
 #       title=f"N={N} limit cycle (PCA projection)",
#        save_path=f"images/phase_portraits/{N}-cycle_phase_before.png",
 #   )

 #   after_n = plot_phase_portrait(
     #   result.W,
    #    result.theta,
   #     dims=("pca", np.asarray(xs)),
  #      x0s=[x0],
 #       title=f"N={N} limit cycle (PCA projection)",
#        save_path=f"images/phase_portraits/{N}-cycle_phase_after.png",
   # )

 #   before_n_grid = plot_phase_grid(
     #   W0,
    #    th0,
   #     x0s=[x0],
  #      title=f"N={N} limit cycle (slices)",
 #       save_path=f"images/phase_portraits/{N}-cycle_grid_before.png",
   # )

   # after_n_grid = plot_phase_grid(
    #    result.W,
   #     result.theta,
  #      x0s=[x0],
 #       title=f"N={N} limit cycle (slices)",
#        save_path=f"images/phase_portraits/{N}-cycle_grid_after.png",
#    )


# ======================================================================
#  DISCOVERY SWEEP: can ES find a limit cycle from empty asymmetric init?
# ======================================================================
from topology_analysis import analyze_interior_fixed_point

from oscillon.topology import extract_block_adjacency, init_params_asymmetric


def cycle_emerged(W_t, theta_t, x0, dt, n_steps, burn_in, min_amp=0.05):
    """Judge whether the TRAINED network oscillates. Independent of target period.
    Criteria: sustained activity (tail variance) AND a detectable period."""
    xs = np.asarray(simulate(W_t, theta_t, x0, dt=dt, n_steps=n_steps))[burn_in:]
    tail = xs[-int(0.4 * len(xs)) :]
    amp = float(tail.std(0).max())  # is anything still moving?
    per = measure_period(xs, dt)  # is it periodic?
    return (amp > min_amp and per is not None), amp, per


def discovered_cycle_graph(A_learned, N):
    """Does the learned adjacency contain a directed N-cycle (0->1->...->0
    or any rotation)? Checks each node has exactly the cyclic in/out structure."""
    canonical = MotifSpec.cyclic(N).adjacency
    # accept any rotation of the canonical cycle
    for shift in range(N):
        rolled = np.roll(np.roll(canonical, shift, axis=0), shift, axis=1)
        if np.array_equal(A_learned, rolled):
            return True
    return False


def contains_directed_cycle(A, N):
    """Does the adjacency contain a Hamiltonian directed cycle (visits all N,
    returns to start)? Looser than exact-match: allows extra edges."""
    import itertools

    # A[i,j] = edge j->i. Check if some cyclic permutation of nodes is all-connected.
    for perm in itertools.permutations(range(1, N)):
        order = (0,) + perm
        if all(A[order[(k + 1) % N], order[k]] for k in range(N)):
            return True
    return False


def edge_commitment(params, spec, mode="soft"):
    """
    Normalize learnable edge values to [0,1] and measure commitment.
    Returns (u_values, dict of summary stats).
    """
    _, _, eps_e, delta_e, _ = spec.compile_indices()
    p = np.asarray(params[: spec.n_edges])

    if mode == "soft":
        u = 1.0 / (1.0 + np.exp(-p))
    else:  # direct
        w = np.clip(p, -1.0 - delta_e, -1.0 + eps_e)
        u = (w + 1.0 + delta_e) / (eps_e + delta_e)

    stats = {
        "frac_ambiguous": float(np.mean((u > 0.4) & (u < 0.6))),
        "frac_committed": float(np.mean((u < 0.1) | (u > 0.9))),
        "mean_dist_mid": float(np.mean(np.abs(u - 0.5))),
    }
    return u, stats


from collections import defaultdict


def summarize_by_zstd(results):
    by_zstd = defaultdict(list)
    for r in results:
        by_zstd[r["z_std"]].append(r)

    print("\n" + "=" * 80)
    print(
        f"{'z_std':>6} | {'n':>3} | {'emerged':>8} | {'topology':>9} | "
        f"{'both':>6} | {'hamilton':>9} | {'fp_unstab':>9} | {'mean_eig':>9}"
    )
    print("-" * 80)

    summary = {}
    for z_std in sorted(by_zstd):
        rows = by_zstd[z_std]
        n = len(rows)
        n_emerged = sum(r["emerged"] for r in rows)
        n_topo = sum(r["cycle_topology"] for r in rows)
        n_both = sum(r["emerged"] and r["cycle_topology"] for r in rows)
        n_ham = sum(r["hamiltonian_cycle"] for r in rows)
        n_fp_unstable = sum(r["fp_unstable"] for r in rows)
        eigs = [r["max_real_eig"] for r in rows if not np.isnan(r["max_real_eig"])]
        mean_eig = float(np.mean(eigs)) if eigs else float("nan")

        summary[z_std] = {
            "n": n,
            "emerged_pct": 100 * n_emerged / n,
            "topology_pct": 100 * n_topo / n,
            "both_pct": 100 * n_both / n,
            "hamiltonian_pct": 100 * n_ham / n,
            "fp_unstable_pct": 100 * n_fp_unstable / n,
            "mean_max_real_eig": mean_eig,
        }
        print(
            f"{z_std:>6.1f} | {n:>3} | {n_emerged:>3}/{n:<3} {100 * n_emerged / n:>4.0f}%| "
            f"{n_topo:>3}/{n:<3} {100 * n_topo / n:>4.0f}%| "
            f"{100 * n_both / n:>5.0f}% | {n_ham:>3}/{n:<3} {100 * n_ham / n:>4.0f}%| "
            f"{n_fp_unstable:>3}/{n:<3} {100 * n_fp_unstable / n:>4.0f}%| "
            f"{mean_eig:>+9.3f}"
        )

    print("=" * 80)

    emerged_spread = max(s["emerged_pct"] for s in summary.values()) - min(
        s["emerged_pct"] for s in summary.values()
    )
    print(
        f"spread across z_std -> emerged: {emerged_spread:.0f}pp "
        f"(small spread supports invariance to init scale)"
    )

    return summary

def hardened_ctln_check(spec, params, p2m, x0, targets, *, dt, n_steps, burn_in, ridge):
    """
    Compare soft-network dynamics to the hardened uniform-CTLN dynamics.
    Returns (soft_mse, hard_mse, dyn_divergence) - the last is how much the
    hardened dynamics differ from the soft ones.
    """

    print("RUNNING HARDENED CTLN CHECK...")

    n = spec.n_total
    rows, cols, eps_e, delta_e, W_cross = spec.compile_indices()
    eps_j, delta_j = jnp.asarray(eps_e), jnp.asarray(delta_e)

    n_edges = spec.n_edges

    # theta (uniform: single scalar broadcast; per-node: the vector)
    if spec.learn_theta:
        theta = params[n_edges:]
        # for uniform theta spec this block is length 1; broadcast:
        if theta.shape[0] == 1:
            theta = jnp.full((n,), theta[0])
    else:
        theta = jnp.full((n,), spec.theta_init)

    # soft network
    z = params[:n_edges]
    W_soft, theta = p2m(params)
    xs_soft = simulate(W_soft, theta, x0, dt=dt, n_steps=n_steps)[burn_in:]

    # hardened uniform CTLN: snap each edge to the band endpoints
    gate = jax.nn.sigmoid(z)
    edge_present = gate > 0.5
    w_hard_off = jnp.where(edge_present, -1.0 + eps_j, -1.0 - delta_j)   # per-edge endpoints
    # edge -> -1+eps, non-edge -> -1-delta, zero diag
    W_hard = jnp.asarray(W_cross).at[jnp.asarray(rows), jnp.asarray(cols)].set(w_hard_off)
    xs_hard = simulate(W_hard, theta, x0, dt=dt, n_steps=n_steps)[burn_in:]

    # fit the same readout style to each, compare to target
    tgt = targets[burn_in:]
    R_s, b_s = fit_readout(xs_soft, tgt, ridge)
    R_h, b_h = fit_readout(xs_hard, tgt, ridge)
    soft_mse = float(jnp.mean((apply_readout(xs_soft, R_s, b_s) - tgt) ** 2))
    hard_mse = float(jnp.mean((apply_readout(xs_soft, R_h, b_h) - tgt) ** 2))

    # direct dynamics divergence (aligned length)
    m = min(xs_soft.shape[0], xs_hard.shape[0])
    dyn_divergence = float(jnp.mean((xs_soft[:m] - xs_hard[:m]) ** 2))

    return soft_mse, hard_mse, dyn_divergence

def run_sweep(n_seeds=20, z_std=2.0):
    # fixed target: build once at a reference period so every seed faces
    # the SAME problem (success = did a cycle emerge, not did it hit a period)
    ref_period = 2.0 * np.pi
    n_steps_sweep = int(burn_in + 8 * (ref_period / dt))
    t_sweep = jnp.arange(n_steps_sweep) * dt
    targets_sweep = make_targets(N, t_sweep, ref_period)

    warm_iters = 9000
    warm_pop = 512
    warm_sigma = 2
    warm_lr = 0.05

    cryst_iters = 5000
    cryst_pop = 512
    cryst_sigma = 1
    cryst_lr = 0.03

    hyperparameters = []
    hyperparameters.append(
        {
            "warm_iters": warm_iters,
            "warm_pop": warm_pop,
            "warm_sigma": warm_sigma,
            "warm_lr": warm_lr,
            "cryst_iters": cryst_iters,
            "cryst_pop": cryst_pop,
            "cryst_sigma": cryst_sigma,
            "cryst_lr": cryst_lr,
        }
    )

    out = Path(__file__).parent / "hyperparameters"
    out.mkdir(exist_ok=True)
    with open(out / "uniform_theta_hyperparams.json", "w") as f:
        json.dump(hyperparameters, f, indent=2, default=float)

    results = []
    all_u = []
    for seed in range(n_seeds):
        for z_std in [1.0, 2.0, 3.0]:
            key_s = jax.random.PRNGKey(1000 + seed)
            key_s, sub_s = jax.random.split(key_s)
            params0_s = init_params_asymmetric(spec, sub_s, z_std=z_std)

            res = train(
                spec,
                params0_s,
                x0,
                targets_sweep,
                key_s,
                p2m=p2m,
                dt=dt,
                n_steps=n_steps_sweep,
                burn_in=burn_in,
                warm=dict(n_iters=warm_iters, pop=warm_pop, sigma=2, lr=0.05),
                cryst=dict(n_iters=cryst_iters, pop=cryst_pop, sigma=1, lr=0.03),
            )

            # save model params
            np.save(Path(__file__).parent / "results" / f"params_seed{seed}_z{z_std}.npy", np.asarray(res.params))

            emerged, amp, per = cycle_emerged(
                res.W, res.theta, x0, dt, n_steps_sweep, burn_in
            )
            A_l = extract_block_adjacency(spec, res.params)
            is_cycle_graph = discovered_cycle_graph(A_l, N)
            hamiltonian = contains_directed_cycle(A_l, N)
            fp = analyze_interior_fixed_point(res.W, res.theta)

            u, stats = edge_commitment(res.params, spec, mode="soft")
            all_u.append(u)

            soft_mse, hard_mse, div = hardened_ctln_check(spec, res.params, p2m, x0, targets,
                                                          dt=dt, n_steps=n_steps,
                                                          burn_in=burn_in, ridge=1e-6)


            record = {
                "seed": seed,
                "z_std": z_std,
                "emerged": emerged,
                "amp": amp,
                "period": per,
                "cycle_topology": is_cycle_graph,
                "hamiltonian_cycle": hamiltonian,
                "fp_unstable": not fp["stable"],  # want True for a real cycle
                "max_real_eig": fp["max_real_eig"],
                "final_reward": res.history[-1],
                "hard_dyn_divergence": div,
                "hard_mse": hard_mse,
                "soft_mse": soft_mse,
                **stats,
            }

            results.append(record)

            with open(out / "ablation_soft.jsonl", "a") as f:
                f.write(json.dumps(record, default=float) + "\n")

            print(
                f"seed {seed:2d}: z_std={z_std:.2f} cycle={emerged!s:5} "
                f"topo={is_cycle_graph!s:5} fp_unstable={not fp['stable']!s:5} "
                f"hamiltonian={hamiltonian!s:5} "
                f"amp={amp:.3f} period={per} eig={fp['max_real_eig']:+.3f}"
            )

            n = len(results)
            n_emerged = sum(r["emerged"] for r in results)
            n_topo = sum(r["cycle_topology"] for r in results)
            n_both = sum(r["emerged"] and r["cycle_topology"] for r in results)
            n_hamiltonian = sum(r["hamiltonian_cycle"] for r in results)

    np.savez(out / "uniform_pop512_u.npz", u=np.concatenate(all_u))

    u_pooled = np.concatenate(all_u)
    np.save(Path(__file__).parent / "results" / "u_soft_pooled.npy", u_pooled)
    plot_commitment_histogram(u_pooled, save_path=str(Path(__file__).parent / "images" / "commitment_hist.png"))

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    with open(out / "uniform_pop512.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print(f"DISCOVERY SWEEP  (N={N}, {n_seeds} seeds, z_std={z_std})")
    print(f"  oscillation emerged:      {n_emerged}/{n}  ({100 * n_emerged / n:.0f}%)")
    print(f"  recovered cycle topology: {n_topo}/{n}  ({100 * n_topo / n:.0f}%)")
    print(f"  both (full discovery):    {n_both}/{n}  ({100 * n_both / n:.0f}%)")
    print(
        f"  hamiltonian cycle:        {n_hamiltonian}/{n}  ({100 * n_hamiltonian / n:.0f}%)"
    )
    print("=" * 60)
    return results, summarize_by_zstd(results), hyperparameters

    for z_std in sorted(by_zstd):
        rows = by_zstd[z_std]
        n = len(rows)
        osc = sum(r["emerged"] for r in rows)
        ham = sum(r["hamiltonian"] for r in rows)
        print(
            f"z_std={z_std:.1f}: oscillation {osc}/{n} ({100 * osc / n:.0f}%), "
            f"hamiltonian {ham}/{n} ({100 * ham / n:.0f}%)"
        )

def plot_commitment_histogram(u_soft, u_direct=None, save_path="commitment_hist.png"):
    """
    u_soft, u_direct: 1-D arrays of normalized edge values in [0,1],
    pooled across all runs (and edges) for each arm.
    """
    print("PLOTTING EDGE COMMITMENT HISTOGRAM...")
    fig, ax = plt.subplots(figsize=(7,4))
    bins = np.linspace(0, 1, 41)

    ax.hist(u_soft, bins=bins, density=True, alpha=0.6,
            color="tab:blue", label=f"soft (n={u_soft.size} edges)")
    if u_direct is not None:
        ax.hist(u_direct, bins=bins, density=True, alpha=0.6,
                color="tab:orange", label=f"direct (n={u_direct.size} edges)")

    ax.axvline(0.5, ls="--", c="gray", lw=1.0, label="threshold")
    ax.set_xlabel("normalized edge value $u = s(z)$")
    ax.set_ylabel("Edge commitment distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    print(f"HISTOGRAM SAVED AS: {save_path}")
    plt.close()


#if __name__ == "__main__" or True:  # set False to skip when running main script
#    sweep_results = run_sweep(n_seeds=50, z_std=2.0)
