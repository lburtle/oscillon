"""
diagnose_gradient_start.py

Checks the hypothesis that the gradient baseline sits at chance because it starts
(or ends up) in a regime where the post-burn-in state is constant. In that regime
the refit readout predicts the target mean whatever the constant state is, so the
loss equals the target variance and its gradient w.r.t. the network params is ~0.

Standalone / non-invasive: does not touch gradient.py. Call it with the same
arguments you pass to gradient_baseline, once per (seed) you want to inspect:

    from diagnose_gradient_start import diagnose_gradient_start
    d = diagnose_gradient_start(p2m, params0, x0, targets,
                                dt=dt, n_steps=n_steps, burn_in=burn_in, ridge=ridge)
    print(d)

Reading the result:
  flat=True, grad_norm ~ 0, loss ~ target_var  -> start is on the flat plateau; Adam
                                                  gets zero gradient and cannot move.
  flat=False, grad_norm > 0, loss ~ target_var -> not flat; look at the loss curve
                                                  (lr / steps / clipping).
  grad_finite=False                            -> NaN/inf gradient (the old failure).
"""

import jax
import jax.numpy as jnp
import numpy as np

from oscillon.ctln.dynamics import simulate
from oscillon.ctln.readout import apply_readout, fit_readout


def diagnose_gradient_start(p2m, params0, x0, targets, *, dt, n_steps, burn_in, ridge,
                            flat_rel_tol=1e-4):
    tgt = targets[burn_in:]

    def loss(params):
        W, theta = p2m(params)
        xs = simulate(W, theta, x0, dt=dt, n_steps=n_steps)[burn_in:]
        R, b = fit_readout(xs, tgt, ridge)
        pred = apply_readout(xs, R, b)
        return jnp.mean((pred - tgt) ** 2)

    l, g = jax.value_and_grad(loss)(params0)
    g_flat = jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(g)])

    W, theta = p2m(params0)
    xs = np.asarray(simulate(W, theta, x0, dt=dt, n_steps=n_steps)[burn_in:])
    std = xs.std(axis=0)
    scale = max(float(np.abs(xs).max()), 1e-12)

    return {
        "loss": float(l),
        "target_var": float(jnp.var(tgt, axis=0).mean()),   # loss of a constant-mean predictor
        "grad_norm": float(jnp.linalg.norm(g_flat)),
        "grad_finite": bool(jnp.isfinite(g_flat).all()),
        "state_std_per_neuron": std.tolist(),
        "state_scale": scale,
        "flat": bool(std.max() < flat_rel_tol * scale),
    }
