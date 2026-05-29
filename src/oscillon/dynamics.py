# gCTLN dynamics
# d/dx_i = -x + relu(Wx + theta)
#        = -x + max(Wx + theta, 0)
#
# Foward-Euler rollout of the threshold-linear ODE

from __future__ import annotations
from functools import partial

import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnames=("n_steps",))
def simulate(W, theta, x0, dt=0.1, n_steps=2000):
    """
    Params:
    W - (n, n) weight matrix
    theta - (n,) per-neuron drive
    x0 - (n,) initial state
    dt - Euler step size
    n_steps - num of steps (static to set scan length)

    Returns:
    xs - (n_steps, n) trajectory of states
    """
    def step(x, _):
        x = x + dt * (-x + jax.nn.relu(W @ x + theta))
        return x, x
    _, xs = jax.lax.scan(step, x0, xs=None, length=n_steps)
    return xs