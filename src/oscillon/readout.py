import jax
import jax.numpy as jnp
import numpy as np

def fit_readout(xs: jax.Array, targets: jax.Array, ridge: float) -> tuple[jax.Array, jax.Array]:
    """(T,n), (T,m) -> R (m,n), b (m,). Must be jittable + vmappable
    (closed form, no Python branching), because it runs inside fitness per candidate."""
    T = xs.shape[0]
    Xa = jnp.concatenate([xs, jnp.ones((T, 1))], axis=1)  # (T, n+1)

    n = xs.shape[1]
    A = Xa.T @ Xa + ridge * jnp.eye(n + 1)   # (n+1, n+1)
    B = Xa.T @ targets                       # (n+1, m)
    theta = jnp.linalg.solve(A, B)
    R = theta[:n].T                          # (m, n)
    b = theta[n]                             # (m,)

    return R, b

def apply_readout(xs: jax.Array, R: jax.Array, b: jax.Array) -> jax.Array:
    """(T,n) -> (T,m)."""
    y = xs @ R.T + b   # (T, n) @ (n, m) -> (T, m)
    
    return y