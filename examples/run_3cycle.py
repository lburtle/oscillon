import jax
import jax.numpy as jnp
import numpy as np

from oscillon.topology import BlockSoftSpec, init_block_params_from_adjacency
from oscillon.graph import MotifSpec, NetworkSpec
from oscillon.model import train
from oscillon.readout import apply_readout

from pathlib import Path

script_dir = Path(__file__).parent
output_dir = script_dir / "images"
output_dir.mkdir(exist_ok=True)
save_path=output_dir / "demo_plot.png"

key = jax.random.PRNGKey(0)

# --- build the spec ---
spec = BlockSoftSpec(
    sizes=(3,),
    cross_deltas=np.zeros((1, 1)),   # single block: no cross-coupling
    eps=0.10, delta=0.50,
)

# --- targets: three sine waves over the rollout ---
n_steps, dt, burn_in = 2000, 0.1, 200
t = jnp.arange(n_steps) * dt
targets = jnp.stack([jnp.sin( 0.3 * (t + p)) for p in (0.0, 8 * 100.0, 16 * 100.0)], axis=1)  # (T, 3)

# --- warm start from the discrete 3-cycle ---
cycle = NetworkSpec([MotifSpec.cyclic(3)])
A = cycle.to_torch_adjacency().numpy()
key, sub = jax.random.split(key)
params0 = init_block_params_from_adjacency(spec, sub, A)

x0 = jnp.full((3,), 0.1)

# --- run ---
result = train(
            spec, params0, x0, targets, key,
            warm=dict(n_iters=900, pop=128, sigma=0.1, lr=0.05),  # manual override to specify training params
            cryst=dict(n_iters=500, pop=128, sigma=0.05, lr=0.03),
        )
print("final reward:", result.history[-1])

import matplotlib.pyplot as plt
from oscillon.dynamics import simulate

from oscillon.topology import make_block_param_to_model

p2m = make_block_param_to_model(spec)
W, theta = p2m(params0)
xs = simulate(result.W, result.theta, x0, dt=dt, n_steps=n_steps)
y = apply_readout(xs, result.R, result.b)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

for i in range(3):
    ax1.plot(xs[:, i], label=f"neuron {i}")
ax1.axvline(burn_in, ls="--", c="gray", lw=0.8)
ax1.set_title("trained gCTLN activations")
ax1.legend()

for i in range(3):
    ax2.plot(y[burn_in:, i], label=f"readout {i}")
    ax2.plot(targets[burn_in:, i], ls=":", c="k", lw=0.8)
ax2.set_title("readout (solid) vs target sines (dotted)")
ax2.legend()

print("Saving output figure")
plt.tight_layout()
plt.savefig(save_path)
print(f"Figure saved to {save_path}")

xs = simulate(result.W, result.theta, x0, dt=dt, n_steps=n_steps)[burn_in:]
y = apply_readout(xs, result.R, result.b)
tgt = targets[burn_in:]
print("readout MSE:", float(jnp.mean((y - tgt)**2)))
print("constant-mean MSE:", float(jnp.mean((tgt.mean(0) - tgt)**2)))
print("R norm:", float(jnp.linalg.norm(result.R)), "b:", np.asarray(result.b))

import numpy as np
x = np.asarray(xs[burn_in:, 0])
peaks = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > 0.5))[0]
if len(peaks) > 1:
    period_steps = np.mean(np.diff(peaks))
    print("cycle period (steps):", period_steps, " -> time:", period_steps * dt)
print("target period (time):", 2 * np.pi, " target period (steps):", 2 * np.pi / dt)

from oscillon.topology import extract_block_adjacency

A_learned = extract_block_adjacency(spec, result.params)  # (3, 3) bool
print("Learned adjacency:\n", A_learned.astype(int))


from oscillon.graph import plot_network_graph
from oscillon.topology import block_gate_matrix

G = block_gate_matrix(spec, result.params)
plot_network_graph(G, theta=np.asarray(result.theta), save_path="images/learned_graph.png")
