"""Evolve a gCTLN topology to track a target signal via OpenAI-ES.

Minimal demo of oscillon.search: wrap a BlockSoftSpec in a closed-form-
readout fitness function and optimize its topology with the evolutionary
strategy, printing the best reward per training phase.

Run with:
    python examples/evolve_topology.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from oscillon.ctln.topology import BlockSoftSpec, init_params_asymmetric
from oscillon.search import train


def main() -> None:
    n = 4
    n_steps = 500
    spec = BlockSoftSpec(sizes=(n,), cross_deltas=np.zeros((1, 1)))

    key = jax.random.key(0)
    key_params, key_x0 = jax.random.split(key)
    params0 = init_params_asymmetric(spec, key_params)
    x0 = jax.random.uniform(key_x0, (n,))

    # Target: a single slow sine wave for the readout to track.
    t = jnp.arange(n_steps) * 0.1
    targets = jnp.sin(0.2 * t)[:, None]

    result = train(
        spec,
        params0,
        x0,
        targets,
        key,
        dt=0.1,
        n_steps=n_steps,
        burn_in=50,
        warm=dict(n_iters=50, pop=64, sigma=0.1, lr=0.05),
        cryst=dict(n_iters=25, pop=64, sigma=0.5, lr=0.03),
    )

    print(f"best reward after warm-up:        {result.history[49]:+.5f}")
    print(f"best reward after crystallization: {result.history[-1]:+.5f}")


if __name__ == "__main__":
    main()
