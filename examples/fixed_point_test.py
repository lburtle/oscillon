import jax
import jax.numpy as jnp
import numpy as np

from oscillon.dynamics import simulate
from oscillon.readout import apply_readout, fit_readout
from oscillon.topology import (
    BlockSoftSpec,
    init_block_params_from_adjacency,
    make_block_param_to_model,
)
from oscillon.train.es import openai_es

N = 3
dt, n_steps, burn_in = 0.1, 1500, 500  # long burn_in: let transient die, judge the tail
key = jax.random.PRNGKey(0)

# --- constant target: a held static output vector ---
target_value = jnp.array([0.8, 0.1, 0.5])  # (N,) the value to settle-and-decode to
targets = jnp.broadcast_to(target_value, (n_steps, N))  # constant over time

# --- spec: a NON-cycle topology biases toward fixed points, not oscillation ---
# a cycle wants to oscillate; to get a fixed point, seed something acyclic.
spec = BlockSoftSpec(
    sizes=(N,), cross_deltas=np.zeros((1, 1)), eps=0.10, delta=0.50, theta_init=0.5
)

# seed an acyclic / feedforward-ish adjacency (no closed loop -> no traveling wave)
A_seed = np.zeros((N, N), dtype=bool)
A_seed[1, 0] = A_seed[2, 1] = True  # a chain 0->1->2, no wrap-around
cycle = None
key, sub = jax.random.split(key)
params0 = init_block_params_from_adjacency(spec, sub, A_seed)
x0 = jnp.full((N,), 0.1)

p2m = make_block_param_to_model(spec)


def make_fixed_point_fitness(
    p2m, x0, targets, *, dt, n_steps, burn_in, ridge, settle_weight, n_edges
):
    tgt = targets[burn_in:]

    def fitness(params):
        W, theta = p2m(params)
        xs = simulate(W, theta, x0, dt=dt, n_steps=n_steps)[burn_in:]
        R, b = fit_readout(xs, tgt, ridge)
        mse = jnp.mean((apply_readout(xs, R, b) - tgt) ** 2)
        settle = settle_weight * jnp.mean(
            jnp.var(xs, axis=0)
        )  # tail variance -> 0 at a fixed point
        reward = -(mse + settle)
        return jnp.where(jnp.isfinite(reward), reward, -1e6)

    return fitness


fit = make_fixed_point_fitness(
    p2m,
    x0,
    targets,
    dt=dt,
    n_steps=n_steps,
    burn_in=burn_in,
    ridge=1e-6,
    settle_weight=1.0,
    n_edges=spec.n_edges,
)
best_params, history = openai_es(
    fit, params0, key, n_iters=600, pop=128, sigma=0.1, lr=0.05
)
W_star, theta_star = p2m(best_params)

xs = simulate(W_star, theta_star, x0, dt=dt, n_steps=n_steps)
tail = xs[-200:]
x_star = np.asarray(tail.mean(0))
print("tail std per neuron:", np.asarray(tail.std(0)))  # -> near zero if converged
print("fixed point x*:", x_star)
# confirm it's actually a fixed point: dx/dt ~ 0 there
dxdt = -x_star + np.maximum(np.asarray(W_star) @ x_star + np.asarray(theta_star), 0)
print("||dx/dt|| at x*:", np.linalg.norm(dxdt))  # -> near zero


from pathlib import Path

import matplotlib.pyplot as plt
from phase_portrait import plot_phase_portrait
from phase_grid import plot_phase_grid

from oscillon.graph import plot_network_graph
from oscillon.topology import block_gate_matrix, extract_block_adjacency

out = Path(__file__).parent / "images"
out.mkdir(exist_ok=True)

# roll out the trained network once, reuse everywhere
xs_full = simulate(W_star, theta_star, x0, dt=dt, n_steps=n_steps)
xs = xs_full[burn_in:]
y = apply_readout(xs, *fit_readout(xs, targets[burn_in:], 1e-6))

# ---------------------------------------------------------------
# 1. ACTIVITY: state settling to the fixed point + decoded output
# ---------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for i in range(N):
    ax1.plot(xs_full[:, i], label=f"neuron {i}")
ax1.axvline(burn_in, ls="--", c="gray", lw=0.8)
ax1.axhline(0, c="k", lw=0.4)
ax1.set_title("state trajectory (settling to fixed point)")
ax1.legend(fontsize=8)

for i in range(N):
    ax2.plot(y[:, i], label=f"readout {i}")
    ax2.axhline(float(target_value[i]), ls=":", c="k", lw=0.7)  # constant targets
ax2.set_title("readout (solid) vs constant target (dotted)")
ax2.legend(fontsize=8)
plt.tight_layout()
plt.savefig(out / "fp_activity.png", dpi=130)
print("saved fp_activity.png")

# ---------------------------------------------------------------
# 2. GRAPH: learned topology with gate values + theta
# ---------------------------------------------------------------
G = block_gate_matrix(spec, best_params)
plot_network_graph(G, theta=np.asarray(theta_star), save_path=str(out / "fp_graph.png"))
A_learned = extract_block_adjacency(spec, best_params)
print("learned adjacency:\n", A_learned.astype(int))
print("kept seed (acyclic chain):", np.array_equal(A_learned, A_seed))

# ---------------------------------------------------------------
# 3. PHASE SPACE: before vs after, showing the fixed point form
# ---------------------------------------------------------------
# several initial conditions so basins/convergence are visible
rng = np.random.default_rng(0)
x0s = [rng.uniform(0, 1.2, size=N) for _ in range(6)] + [np.asarray(x0)]

xs_tmp = simulate(W_star, theta_star, x0, dt=dt, n_steps=n_steps)
x_star = np.asarray(xs_tmp[-200:].mean(0))

# BEFORE: warm-start dynamics
W0, th0 = p2m(params0)
plot_phase_portrait(
    W0,
    th0,
    dims=(1, 2),
    x0s=x0s,
    slice_value=float(x_star[0]),
    dt=dt,
    n_steps=n_steps,
    title="phase space BEFORE training (warm start)",
    save_path=str(out / "fp_phase_before.png"),
)

# AFTER: trained dynamics converging to x*
fig_after = plot_phase_portrait(
    W_star,
    theta_star,
    dims=(1, 2),
    x0s=x0s,
    slice_value=float(x_star[0]),
    dt=dt,
    n_steps=n_steps,
    title="phase space AFTER training (fixed point)",
    save_path=str(out / "fp_phase_after.png"),
)

print("saved phase portraits")


grid_before = plot_phase_grid(
    W0,
    th0,
    x0s=x0s,
    dt=dt,
    n_steps=n_steps,
    title="phase space BEFORE training (warm start)",
    save_path=str(out / "fp_grid_before.png")
)

# AFTER: trained dynamics converging to x*
fig_after = plot_phase_grid(
    W_star,
    theta_star,
    x0s=x0s,
    dt=dt,
    n_steps=n_steps,
    title="phase space AFTER training (fixed point)",
    save_path=str(out / "fp_grid_after.png")
)

print("saved phase grids")

def diagnose_fixed_point(W, theta, x_star, tol=1e-6):
    x_star = np.asarray(x_star)
    W, theta = np.asarray(W), np.asarray(theta)

    drive = W @ x_star + theta
    active = drive > tol  # neurons actually "on" (interior support)
    print("x*:", np.round(x_star, 4))
    print("drive (Wx*+theta):", np.round(drive, 4))
    print("active support:", active, f"({active.sum()}/{len(x_star)} active)")

    if active.sum() == 0:
        print("trivial fixed point (all zero) -- degenerate")
        return

    # Jacobian restricted to the active support only
    idx = np.where(active)[0]
    J_full = -np.eye(len(x_star)) + W
    J_active = J_full[np.ix_(idx, idx)]
    eigs = np.linalg.eigvals(J_active)
    print("eigenvalues on active submatrix:", np.round(eigs, 4))
    print("max real part:", eigs.real.max())
    print("-> stable" if eigs.real.max() < 0 else "-> UNSTABLE (saddle/source)")


diagnose_fixed_point(W_star, theta_star, x_star)

# ======================================================================
#  DISCOVERY SWEEP: can ES find a fixed point from random asymmetric init?
# ======================================================================

from oscillon.topology import extract_block_adjacency, init_params_asymmetric


def contains_directed_cycle(A, N):
    """Does the adjacency contain a Hamiltonian directed cycle? Used here as
    a NEGATIVE signal: a fixed point is more 'clean' if no closed loop formed."""
    import itertools

    for perm in itertools.permutations(range(1, N)):
        order = (0,) + perm
        if all(A[order[(k + 1) % N], order[k]] for k in range(N)):
            return True
    return False


def fixed_point_emerged(
    W_t,
    theta_t,
    x0,
    dt,
    n_steps,
    burn_in,
    tail_frac=0.2,
    var_thresh=1e-3,
    dxdt_thresh=1e-2,
):
    """Judge whether the TRAINED network settled to a fixed point.
    Criteria: low tail variance (state stopped moving) AND low residual
    dynamics (dx/dt ~ 0) at the tail mean. Independent of target value."""
    xs = np.asarray(simulate(W_t, theta_t, x0, dt=dt, n_steps=n_steps))[burn_in:]
    tail = xs[-int(tail_frac * len(xs)) :]
    tail_var = float(tail.var(0).max())
    x_star = tail.mean(0)
    dxdt = -x_star + np.maximum(np.asarray(W_t) @ x_star + np.asarray(theta_t), 0)
    dxdt_norm = float(np.linalg.norm(dxdt))
    settled = (tail_var < var_thresh) and (dxdt_norm < dxdt_thresh)
    return settled, tail_var, dxdt_norm, x_star


def is_stable_fixed_point(W, theta, x_star, tol=1e-6):
    W, theta, x_star = np.asarray(W), np.asarray(theta), np.asarray(x_star)
    drive = W @ x_star + theta
    active = drive > tol
    if active.sum() == 0:
        return False, np.array([])  # trivial/degenerate, treat as not stable
    idx = np.where(active)[0]
    J_active = (-np.eye(len(x_star)) + W)[np.ix_(idx, idx)]
    eigs = np.linalg.eigvals(J_active)
    return bool(eigs.real.max() < 0), eigs


from collections import defaultdict


def summarize_by_zstd(results):
    by_zstd = defaultdict(list)
    for r in results:
        by_zstd[r["z_std"]].append(r)

    print("\n" + "=" * 70)
    print(
        f"{'z_std':>6} | {'n':>3} | {'settled':>8} | {'acyclic':>8} | "
        f"{'both':>6} | {'fp_stable':>10} | {'mean_eig':>9}"
    )
    print("-" * 70)

    summary = {}
    for z_std in sorted(by_zstd):
        rows = by_zstd[z_std]
        n = len(rows)
        n_settled = sum(r["settled"] for r in rows)
        n_acyclic = sum(r["acyclic"] for r in rows)
        n_both = sum(r["settled"] and r["acyclic"] for r in rows)
        n_fp_stable = sum(r["fp_stable"] for r in rows)
        eigs = [r["max_real_eig"] for r in rows if not np.isnan(r["max_real_eig"])]
        mean_eig = float(np.mean(eigs)) if eigs else float("nan")
        n_dead = sum(r["network_died"] for r in rows)

        summary[z_std] = {
            "n": n,
            "settled_pct": 100 * n_settled / n,
            "acyclic_pct": 100 * n_acyclic / n,
            "both_pct": 100 * n_both / n,
            "fp_stable_pct": 100 * n_fp_stable / n,
            "mean_max_real_eig": mean_eig,
            "network_died_pct": 100 * n_dead / n
        }
        print(
            f"{z_std:>6.1f} | {n:>3} | {n_settled:>3}/{n:<3} {100 * n_settled / n:>4.0f}%| "
            f"{n_acyclic:>3}/{n:<3} {100 * n_acyclic / n:>4.0f}%| "
            f"{100 * n_both / n:>5.0f}% | {n_fp_stable:>3}/{n:<3} {100 * n_fp_stable / n:>4.0f}%| "
            f"{mean_eig:>+9.3f}"
            f"{n_dead:>3}/{n:<3} {100 * n_dead / n:>4.0f}%| "
        )

    print("=" * 70)

    # crude invariance check: spread of settled_pct and fp_stable_pct across z_std
    settled_spread = max(s["settled_pct"] for s in summary.values()) - min(
        s["settled_pct"] for s in summary.values()
    )
    stable_spread = max(s["fp_stable_pct"] for s in summary.values()) - min(
        s["fp_stable_pct"] for s in summary.values()
    )
    print(
        f"spread across z_std -> settled: {settled_spread:.0f}pp, "
        f"fp_stable: {stable_spread:.0f}pp"
    )
    print("(small spread supports invariance to init scale)")

    return summary


def run_sweep(n_seeds=20, z_stds=(1.0, 2.0, 3.0)):
    # fixed reference target so every seed faces the SAME problem
    target_ref = jnp.linspace(0.1, 0.9, N)
    targets_sweep = jnp.broadcast_to(target_ref, (n_steps, N))

    results = []
    for seed in range(n_seeds):
        for z_std in z_stds:
            key_s = jax.random.PRNGKey(2000 + seed)
            key_s, sub_s = jax.random.split(key_s)
            params0_s = init_params_asymmetric(spec, sub_s, z_std=z_std)

            fit_s = make_fixed_point_fitness(
                p2m,
                x0,
                targets_sweep,
                dt=dt,
                n_steps=n_steps,
                burn_in=burn_in,
                ridge=1e-6,
                settle_weight=1.0,
                n_edges=spec.n_edges,
            )

            best_params_s, history_s = openai_es(
                fit_s, params0_s, key_s, n_iters=600, pop=128, sigma=0.1, lr=0.05
            )
            W_s, theta_s = p2m(best_params_s)

            settled, tail_var, dxdt_norm, x_star = fixed_point_emerged(
                W_s, theta_s, x0, dt, n_steps, burn_in
            )

            A_l = extract_block_adjacency(spec, best_params_s)
            acyclic = not contains_directed_cycle(A_l, N)
            fp_stable, fp_eigs = is_stable_fixed_point(W_s, theta_s, x_star)

            if fp_eigs.size == 0:
                max_real = float("nan")
                fp_stable = True
                network_died = True
            else:
                max_real = float(fp_eigs.real.max())
                network_died = False
                fp_stable = bool((fp_eigs.real < 0).all())

            results.append(
                {
                    "seed": seed,
                    "z_std": z_std,
                    "settled": settled,
                    "tail_var": tail_var,
                    "dxdt_norm": dxdt_norm,
                    "acyclic": acyclic,
                    "fp_stable": fp_stable,
                    "max_real_eig": max_real,
                    "final_reward": history_s[-1],
                    "network_died": network_died,
                }
            )
            print(
                f"seed {seed:2d}: z_std={z_std:.2f} settled={settled!s:5} "
                f"acyclic={acyclic!s:5} fp_stable={fp_stable!s:5} "
                f"tail_var={tail_var:.4f} dxdt={dxdt_norm:.4f} "
                f"eig={max_real:+.3f}"
            )

    n = len(results)
    n_settled = sum(r["settled"] for r in results)
    n_acyclic = sum(r["acyclic"] for r in results)
    n_both = sum(r["settled"] and r["acyclic"] for r in results)
    n_fp_stable = sum(r["fp_stable"] for r in results)
    n_died = sum(r["network_died"] for r in results)

    summary_by_zstd = summarize_by_zstd(results)

    print("\n" + "=" * 60)
    print(
        f"FIXED-POINT DISCOVERY SWEEP  (N={N}, {n_seeds} seeds x {len(z_stds)} z_std)"
    )
    print(f"  settled to fixed point:   {n_settled}/{n}  ({100 * n_settled / n:.0f}%)")
    print(f"  acyclic topology:         {n_acyclic}/{n}  ({100 * n_acyclic / n:.0f}%)")
    print(f"  both (full discovery):    {n_both}/{n}  ({100 * n_both / n:.0f}%)")
    print(
        f"  fp analytically stable:   {n_fp_stable}/{n}  ({100 * n_fp_stable / n:.0f}%)"
    )
    print(f"  networks died:   {n_died}/{n}  ({100 * n_died / n:.0f}%)")
    print("=" * 60)
    return results


# Run the sweep
sweep_results = run_sweep(n_seeds=50, z_stds=(1.0, 2.0, 3.0))
