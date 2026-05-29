# Utilizes OpenAI-ES, a blackbox evolution strategy

# fitness_fn(params,) -> scalar reward which is problem agnostic

from __future__ import annotations

import jax
import jax.numpy as jnp

def openai_es(fitness_fn, params0, key, *,
                n_iters=200, pop=128, sigma=0.1, lr=0.05,
                antithetic=True, verbose_every=0):
    """
    fitness_fn: (params,) -> scalar reward. Must be a pure JAX function so it can be jitted and vmapped over a population

    Update: theta <- theta + lr * (1 / (pop * sigma)) * sum_i r_i * eps_i
            r_i is the standardized rewards (zero-mean, unit-variance)

    Returns:
    params - optimizer parameter vector
    history - list of best per-gen rewards
    """
    params = params0
    batched_fit = jax.jit(jax.vmap(fitness_fn))
    half = pop // 2
    history = []

    for iter in range (n_iters):
        key, sub = jax.random.split(key)
        if antithetic: # mirrored noise pairs
            eps = jax.random.normal(sub, (half, params.size))
            eps = jnp.concatenate([eps, -eps], axis=0)
        else:
            eps = jax.random.normal(sub, (pop, params.size))

        pop_params = params[None, :] + sigma * eps # (pop, n_params)
        rewards = batched_fit(pop_params)          # (pop,)

        r = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        grad = (eps.T @ r) / (pop * sigma)         # (n_params,)
        params = params + lr * grad

        best = float(rewards.max())
        history.append(best)
        if verbose_every and (iter % verbose_every == 0 or iter == n_iters - 1):
            print(f" iter {iter:4d} || best reward {best:+.5f} "
                  f"|| mean {float(rewards.mean()):+.5f}")

    return params, history