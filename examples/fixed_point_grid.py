import jax
import jax.numpy as jnp
import numpy as np

from oscillon.topology import BlockSoftSpec, init_block_params_from_adjacency, make_block_param_to_model
from oscillon.graph import MotifSpec, NetworkSpec
from oscillon.dynamics import simulate
from oscillon.readout import fit_readout, apply_readout
from oscillon.train.es import openai_es

N = 4
dt, n_steps, burn_in = 0.1, 1500, 500          # long burn_in: let transient die, judge the tail
key = jax.random.PRNGKey(0)

# --- constant target: a held static output vector ---
target_value = jnp.array([0.8, 0.3, 0.5, 0.1])   # (N,) the value to settle-and-decode to
targets = jnp.broadcast_to(target_value, (n_steps, N))   # constant over time

# --- spec: a NON-cycle topology biases toward fixed points, not oscillation ---
# a cycle wants to oscillate; to get a fixed point, seed something acyclic.
spec = BlockSoftSpec(sizes=(N,), cross_deltas=np.zeros((1, 1)),
                     eps=0.10, delta=0.50, theta_init=0.5)

# seed an acyclic / feedforward-ish adjacency (no closed loop -> no traveling wave)
A_seed = np.zeros((N, N), dtype=bool)
A_seed[1, 0] = A_seed[2, 1] = A_seed[3, 2] = True     # a chain 0->1->2->3, no wrap-around
cycle = None
key, sub = jax.random.split(key)
params0 = init_block_params_from_adjacency(spec, sub, A_seed)
x0 = jnp.full((N,), 0.1)

p2m = make_block_param_to_model(spec)


def make_fixed_point_fitness(p2m, x0, targets, *, dt, n_steps, burn_in, ridge, settle_weight, n_edges):
    tgt = targets[burn_in:]
    def fitness(params):
        W, theta = p2m(params)
        xs = simulate(W, theta, x0, dt=dt, n_steps=n_steps)[burn_in:]
        R, b = fit_readout(xs, tgt, ridge)
        mse = jnp.mean((apply_readout(xs, R, b) - tgt) ** 2)
        settle = settle_weight * jnp.mean(jnp.var(xs, axis=0))   # tail variance -> 0 at a fixed point
        reward = -(mse + settle)
        return jnp.where(jnp.isfinite(reward), reward, -1e6)
    return fitness


fit = make_fixed_point_fitness(p2m, x0, targets, dt=dt, n_steps=n_steps,
                               burn_in=burn_in, ridge=1e-6, settle_weight=1.0,
                               n_edges=spec.n_edges)
best_params, history = openai_es(fit, params0, key, n_iters=600, pop=128, sigma=0.1, lr=0.05)
W_star, theta_star = p2m(best_params)

xs = simulate(W_star, theta_star, x0, dt=dt, n_steps=n_steps)
tail = xs[-200:]
x_star = np.asarray(tail.mean(0))
print("tail std per neuron:", np.asarray(tail.std(0)))     # -> near zero if converged
print("fixed point x*:", x_star)
# confirm it's actually a fixed point: dx/dt ~ 0 there
dxdt = -x_star + np.maximum(np.asarray(W_star) @ x_star + np.asarray(theta_star), 0)
print("||dx/dt|| at x*:", np.linalg.norm(dxdt))            # -> near zero

import matplotlib.pyplot as plt
from oscillon.topology import extract_block_adjacency, block_gate_matrix
from oscillon.graph import plot_network_graph

from phase_grid import plot_phase_grid
from phase_portrait import plot_phase_portrait

from pathlib import Path
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
plot_network_graph(G, theta=np.asarray(theta_star),
                   save_path=str(out / "fp_graph.png"))
A_learned = extract_block_adjacency(spec, best_params)
print("learned adjacency:\n", A_learned.astype(int))
print("kept seed (acyclic chain):", np.array_equal(A_learned, A_seed))

# ---------------------------------------------------------------
# 3. PHASE SPACE: before vs after, showing the fixed point form
# ---------------------------------------------------------------
# several initial conditions so basins/convergence are visible
rng = np.random.default_rng(0)
x0s = [rng.uniform(0, 1.2, size=N) for _ in range(6)] + [np.asarray(x0)]

# BEFORE: warm-start dynamics
W0, th0 = p2m(params0)
plot_phase_grid(W0, th0, x0s=x0s,
                    dt=dt, n_steps=n_steps,
                    title="phase space BEFORE training (warm start)",
                    save_path=str(out / "fp_grid_before.png"))

# AFTER: trained dynamics converging to x*
fig_after = plot_phase_grid(W_star, theta_star, x0s=x0s,
                                dt=dt, n_steps=n_steps,
                                title="phase space AFTER training (fixed point)",
                                save_path=str(out / "fp_grid_after.png"))
print("saved phase portraits")
