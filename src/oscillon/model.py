# pyrefly: ignore [missing-import]
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from oscillon.train.es import openai_es

from .dynamics import simulate
from .readout import apply_readout, fit_readout
from .topology import (
    BlockSoftSpec,
    clipped_saturation_penalty,
    gate_saturation_penalty,
    make_block_param_to_model,
)


def make_fitness(
    spec,
    param_to_model,
    x0,
    targets,
    *,
    dt,
    n_steps,
    burn_in,
    ridge,
    sat_weight,
    n_edges,
    ablation=None,
):
    """
    Manufacture the pure fitness(params) -> reward closure es.py wants.
    Readout is solved closed-form per candidate; ES vector is topology only.
    """
    tgt = targets[burn_in:]
    _, _, eps_e, delta_e, _ = spec.compile_indices()
    eps_j, delta_j = jnp.asarray(eps_e), jnp.asarray(delta_e)

    def fitness(params: jax.Array) -> jax.Array:
        W, theta = param_to_model(params)
        xs = simulate(W, theta, x0, dt=dt, n_steps=n_steps)[burn_in:]
        R, b = fit_readout(xs, tgt, ridge)
        mse = jnp.mean((apply_readout(xs, R, b) - tgt) ** 2)
        pen = sat_weight * gate_saturation_penalty(params[:n_edges])
        reward = -(mse + pen)  # reward = -loss
        return jnp.where(
            jnp.isfinite(reward), reward, -1e6
        )  # floor blown-up candidates

    def ablation_fitness(params: jax.Array) -> jax.Array:
        W, theta = param_to_model(params)
        xs = simulate(W, theta, x0, dt=dt, n_steps=n_steps)[burn_in:]
        R, b = fit_readout(xs, tgt, ridge)
        mse = jnp.mean((apply_readout(xs, R, b) - tgt) ** 2)
        p = params[:n_edges]
        w_eff = jnp.clip(p, -1.0 - delta_j, -1.0 + eps_j)
        pen = sat_weight * clipped_saturation_penalty(w_eff, eps_j, delta_j)
        reward = -(mse + pen)
        return jnp.where(jnp.isfinite(reward), reward, -1e6)

    if ablation == True:
        return ablation_fitness
    else:
        return fitness


@dataclass
class TrainResult:
    params: jax.Array
    W: jax.Array
    theta: jax.array
    R: jax.Array
    b: jax.Array
    history: list[float]


def train(
    spec: BlockSoftSpec,
    params0,
    x0,
    targets,
    key,
    *,
    p2m=None,
    dt=0.1,
    n_steps=2000,
    burn_in=200,
    ridge=1e-6,
    sat_weight=0.05,
    warm=None,
    cryst=None,
    ablation=None,
) -> TrainResult:
    if p2m is None:
        p2m = make_block_param_to_model(spec)
    warm = warm or dict(n_iters=300, pop=128, sigma=0.1, lr=0.05)
    cryst = cryst or dict(n_iters=150, pop=128, sigma=0.5, lr=0.03)

    if ablation == True:
        print("Using Ablation fitness...")

    # Discover dynamics (no saturation pressure)
    f0 = make_fitness(
        spec,
        p2m,
        x0,
        targets,
        dt=dt,
        n_steps=n_steps,
        burn_in=burn_in,
        ridge=ridge,
        sat_weight=0.0,
        n_edges=spec.n_edges,
        ablation=ablation,
    )
    params, h0 = openai_es(f0, params0, key, **warm)

    # Crystallize topology
    key, sub = jax.random.split(key)
    f1 = make_fitness(
        spec,
        p2m,
        x0,
        targets,
        dt=dt,
        n_steps=n_steps,
        burn_in=burn_in,
        ridge=ridge,
        sat_weight=sat_weight,
        n_edges=spec.n_edges,
        ablation=ablation,
    )
    params, h1 = openai_es(f1, params, sub, **cryst)

    # Freeze the readout once on the winner
    W, theta = p2m(params)
    xs = simulate(W, theta, x0, dt=dt, n_steps=n_steps)[burn_in:]
    R, b = fit_readout(xs, targets[burn_in:], ridge)
    return TrainResult(params, W, theta, R, b, h0 + h1)
