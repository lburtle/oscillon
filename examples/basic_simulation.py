"""Simulate a small random CTLN and plot its trajectory.

Minimal end-to-end example: build a soft-topology spec, sample random
parameters, turn them into a weight matrix + drive, and roll out the
threshold-linear ODE.

Run with:
    python examples/basic_simulation.py
"""

from __future__ import annotations

import jax
import matplotlib.pyplot as plt

from oscillon import SoftNetworkSpec, init_params, make_param_to_model, simulate


def main() -> None:
    n = 5
    spec = SoftNetworkSpec(n=n, eps=0.10, delta=0.50)

    key = jax.random.key(0)
    key_params, key_x0 = jax.random.split(key)
    params = init_params(spec, key_params)

    param_to_model = make_param_to_model(spec)
    W, theta = param_to_model(params)

    x0 = jax.random.uniform(key_x0, (n,))
    xs = simulate(W, theta, x0, dt=0.1, n_steps=2000)

    fig, ax = plt.subplots()
    ax.plot(xs)
    ax.set_xlabel("step")
    ax.set_ylabel("activity")
    ax.set_title(f"CTLN trajectory (n={n})")
    fig.savefig("basic_simulation.png", dpi=150, bbox_inches="tight")
    print("wrote basic_simulation.png")


if __name__ == "__main__":
    main()
