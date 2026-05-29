# Demo: discover a soft topology whose dynamics trace a target 2D limit cycle

# Run (after 'pip install -e .'): python examples/softtopology_demo.property
# Or standalone:                  PYTHONPATH=src python examples/softtopology_demo.property

import jax
import jax.numpy as jnp

from oscillon.topology import(
    SoftNetworkSpec, make_param_to_model, init_params, extract_adjacency,
)
from oscillon.dynamics import simulate
from oscillon.train.es import openai_es


def main():
    spec = SoftNetworkSpec(n=6, eps=0.10, delta=0.50, learn_theta=True)
    key = jax.random.PRNGKey(0)

    to_model = make_param_to_model(spec)
    dt, n_steps = 0.1, 400
    t = jnp.arange(n_steps) * dt
    omega = 0.4

    # Target: 2D limit cycle for first two neurons to trace
    target = jnp.stack([0.5 + 0.35 * jnp.sin(omega * t),
                        0.5 + 0.35 * jnp.cos(omega * t)], axis=1) # (T, 2)
    x0 = 0.1 * jnp.ones((spec.n,))

    def fitness(params):
        W, theta = to_model(params)
        xs = simulate(W, theta, x0, dt=dt, n_steps=n_steps)
        pred = xs[:, :2]
        return -jnp.mean((pred - target) ** 2) # reward = -loss

    params0 = init_params(spec, key, z_mean=0.0, z_std=1.0)
    print(f"n = {spec.n}, n_params = {spec.n_params} "
        f"(z logits: {spec.n_edges}, theta drives: {spec.n if spec.learn_theta else 0})")
    print(f"init reward: {float(fitness(params0)):+.5f}")

    params, _ = openai_es(fitness, params0, key, n_iters=60, pop=64, sigma=0.15, lr=0.05, verbose_every=15)

    print(f"Final reward: {float(fitness(params)):+.5f}")
    A = extract_adjacency(spec, params[:spec.n_edges])

    # Ratio of correct materialized edges
    print(f"discovered edges (gate > 0.5): {int(A.sum())} / {spec.n_edges}")

if __name__ == "__main__":
    main()