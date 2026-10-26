# Utilizes OpenAI-ES, a blackbox evolution strategy

# fitness_fn(params,) -> scalar reward which is problem agnostic

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def openai_es(
    fitness_fn: Callable[[jax.Array], jax.Array],
    params0: jax.Array,
    key: jax.Array,
    *,
    n_iters: int = 200,
    pop: int = 128,
    sigma: float = 0.1,
    lr: float = 0.05,
    antithetic: bool = True,
    verbose_every: int = 0,
) -> tuple[jax.Array, list[float]]:
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
    best_params = params0
    best_fit = -float("inf")

    for it in range(n_iters):
        key, sub = jax.random.split(key)
        if antithetic:  # mirrored noise pairs
            eps = jax.random.normal(sub, (half, params.size))
            eps = jnp.concatenate([eps, -eps], axis=0)
        else:
            eps = jax.random.normal(sub, (pop, params.size))

        pop_params = params[None, :] + sigma * eps  # (pop, n_params)
        rewards = batched_fit(pop_params)  # (pop,)

        # Track best individual ever seen (elitism — free, uses rewards already)
        best_idx = int(jnp.argmax(rewards))
        if float(rewards[best_idx]) > best_fit:
            best_fit = float(rewards[best_idx])
            best_params = pop_params[best_idx]

        r = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        grad = (eps.T @ r) / (pop * sigma)  # (n_params,)
        params = params + lr * grad

        best = float(rewards.max())
        history.append(best)
        if verbose_every and (it % verbose_every == 0 or it == n_iters - 1):
            logger.info(
                "iter %4d | best reward %+.5f | mean %+.5f",
                it,
                best,
                float(rewards.mean()),
            )

    return best_params, history
