import jax
import jax.numpy as jnp
import optax
from oscillon.dynamics import simulate
from oscillon.readout import fit_readout, apply_readout

def gradient_baseline(
    p2m, params0, x0, targets, *,
    dt, n_steps, burn_in, ridge,
    n_iters=3000, lr=1e-2,
):
    """Naive graident descent (Adam) on the same objective ES optimizes.
    Shows collapse to a fixed point rather than a limit cycle.
    Returns (final_params, loss_history)."""
    tgt = targets[burn_in:]

    def loss(params):
        W, theta = p2m(params)
        xs = simulate(W, theta, x0, dt=dt, n_steps=n_steps)[burn_in:]
        R, b = fit_readout(xs, tgt, ridge)
        pred = apply_readout(xs, R, b)

        return jnp.mean((pred - tgt) ** 2)

    opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lr),
        )
    state = opt.init(params0)

    @jax.jit
    def step(params, state):
        l, g = jax.value_and_grad(loss)(params)
        updates, state = opt.update(g, state)
        params = optax.apply_updates(params, updates)

        return params, state, l

    params = params0
    history = []

    for i in range(n_iters):
        params, state, l = step(params, state)
        history.append(float(l))
        if i % 200 == 0:
            print(f"grad iter {i}: loss={float(l):.4f}")

    W, theta = p2m(params)
    xs = simulate(W, theta, x0, dt=dt, n_steps=n_steps)[burn_in:]
    if jnp.isfinite(xs).all():
        R, b = fit_readout(xs, tgt, ridge)
        pred = apply_readout(xs, R, b)
    else:
        R, b, pred = None, None, None      # diverged run

    return params, history, xs, pred
