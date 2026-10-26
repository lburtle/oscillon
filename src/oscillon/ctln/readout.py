import jax
import jax.numpy as jnp
import numpy as np

def fit_readout_deprecated(xs: jax.Array, targets: jax.Array, ridge: float) -> tuple[jax.Array, jax.Array]:
    """(T,n), (T,m) -> R (m,n), b (m,). Must be jittable + vmappable
    (closed form, no Python branching), because it runs inside fitness per candidate."""
    T = xs.shape[0]
    Xa = jnp.concatenate([xs, jnp.ones((T, 1))], axis=1)  # (T, n+1)

    n = xs.shape[1]
    d = ridge * jnp.ones(n + 1)
    d = d.at[-1].set(0.0)
    A = Xa.T @ Xa + jnp.diag(d)   # (n+1, n+1)
    B = Xa.T @ targets                       # (n+1, m)
    theta = jnp.linalg.solve(A, B)
    R = theta[:n].T                          # (m, n)
    b = theta[n]                             # (m,)

    return R, b

def fit_readout_dep_2(xs: jax.Array, targets: jax.Array, ridge: float) -> tuple[jax.Array, jax.Array]:
    """(T,n), (T,m) -> R (m,n), b (m,). Must be jittable + vmappable
    (closed form, no Python branching), because it runs inside fitness per candidate."""
    T = xs.shape[0]
    Xa = jnp.concatenate([xs, jnp.ones((T, 1))], axis=1)  # (T, n+1)

    n = xs.shape[1]
    # Scale ridge relative to the design matrix's own diagonal magnitude so
    # regularization stays effective in float32 regardless of xs's variance.
    # A fixed absolute ridge becomes numerically invisible once Xa.T@Xa's
    # entries grow past float32's precision relative to it -- tie it to the
    # matrix's own scale instead, with a floor of 1.0 so degenerate (near-
    # zero-variance) trajectories still get at least the original ridge.
    diag_mean = jnp.mean(jnp.diag(Xa.T @ Xa))
    ridge_eff = ridge * jnp.maximum(diag_mean, 1.0)

    d = ridge_eff * jnp.ones(n + 1)
    d = d.at[-1].set(0.0)
    A = Xa.T @ Xa + jnp.diag(d)   # (n+1, n+1)
    B = Xa.T @ targets                       # (n+1, m)
    theta = jnp.linalg.solve(A, B)
    R = theta[:n].T                          # (m, n)
    b = theta[n]                             # (m,)

    return R, b

def fit_readout(xs, targets, ridge):
    T, n = xs.shape
    x_mean, y_mean = xs.mean(axis=0), targets.mean(axis=0)
    Xc, Yc = xs - x_mean, targets - y_mean
    G = Xc.T @ Xc
    ridge_eff = ridge * jnp.maximum(jnp.mean(jnp.diag(G)), 1.0)
    theta = jnp.linalg.solve(G + ridge_eff * jnp.eye(n), Xc.T @ Yc)  # (n, m)
    R = theta.T
    b = y_mean - R @ x_mean
    return R, b

def apply_readout(xs: jax.Array, R: jax.Array, b: jax.Array) -> jax.Array:
    """(T,n) -> (T,m)."""
    y = xs @ R.T + b   # (T, n) @ (n, m) -> (T, m)

    return y
