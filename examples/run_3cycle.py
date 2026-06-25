import jax
import jax.numpy as jnp
import numpy as np

from oscillon.topology import BlockSoftSpec, init_block_params_from_adjacency
from oscillon.graph import MotifSpec, NetworkSpec
from oscillon.model import train

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
targets = jnp.stack([jnp.sin(t + p) for p in (0.0, 2.094, 4.189)], axis=1)  # (T, 3)

# --- warm start from the discrete 3-cycle ---
cycle = NetworkSpec([MotifSpec.cyclic(3)])
A = cycle.to_torch_adjacency().numpy()
key, sub = jax.random.split(key)
params0 = init_block_params_from_adjacency(spec, sub, A)

x0 = jnp.full((3,), 0.1)

# --- run ---
result = train(spec, params0, x0, targets, key)
print("final reward:", result.history[-1])

import matplotlib.pyplot as plt
from oscillon.dynamics import simulate

from oscillon.topology import make_block_param_to_model

p2m = make_block_param_to_model(spec)
W, theta = p2m(params0)
xs = simulate(W, theta, jnp.full((3,), 0.1), dt=0.1, n_steps=2000)

plt.plot(xs[:, 0])
plt.savefig(save_path)