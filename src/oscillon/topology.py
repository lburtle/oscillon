from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@dataclass
class SoftNetworkSpec:
    """Continuous-topology gCTLN with an all-inhibitory weight band."""

    n: int
    eps: float
    delta: float
    learn_theta: bool = True
    theta_init: float = 0.10

    @property
    def n_edges(self) -> int:
        """Number of off-diagonal structural params"""
        return self.n * (self.n - 1)

    @property
    def n_params(self) -> int:
        return self.n_edges + (self.n if self.learn_theta else 0)

    def offdiag_indices(self) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Row/col indices of the off-diag entries, row-major (numpy)"""
        return cast(
            "tuple[NDArray[np.intp], NDArray[np.intp]]",
            np.nonzero(~np.eye(self.n, dtype=bool)),
        )


@partial(jax.jit, static_argnames=("n",))
def build_W(
    z_flat: jax.Array,
    n: int,
    eps: float,
    delta: float,
) -> jax.Array:
    """Map off-diag logits -> full (n, n) inhib weight matrix"""
    gate = jax.nn.sigmoid(z_flat)
    w_off = -1.0 - delta + (eps + delta) * gate
    rows, cols = jnp.nonzero(~jnp.eye(n, dtype=bool), size=n * (n - 1))
    return jnp.zeros((n, n)).at[rows, cols].set(w_off)  # zeros diag


def make_param_to_model(
    spec: SoftNetworkSpec,
) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    """
    Build a jitted params -> (W, theta) func for this spec
    Closes over the (static) spec fields so it is safe to vmap over a population of param vectors
    """
    n, eps, delta = spec.n, spec.eps, spec.delta
    n_edges = spec.n_edges
    learn_theta = spec.learn_theta
    theta_const = spec.theta_init

    @jax.jit  # type: ignore[untyped-decorator, unused-ignore]
    def f(params: jax.Array) -> tuple[jax.Array, jax.Array]:
        z = params[:n_edges]
        theta = params[n_edges:] if learn_theta else jnp.full((n,), theta_const)
        return build_W(z, n, eps, delta), theta

    return cast("Callable[[jax.Array], tuple[jax.Array, jax.Array]]", f)


def init_params(
    spec: SoftNetworkSpec, key: jax.Array, z_mean: float = 0.0, z_std: float = 1.0
) -> jax.Array:
    """
    Random asymmetric init of the flat param vector
    z_ij are drawn independently (z_ij != z_ji to avoid accidental symmetry)
    z_mean < 0 biases toward a sparser graph (more non-edges) at init
    """
    key_z, _ = jax.random.split(key)
    z = z_mean + z_std * jax.random.normal(key_z, (spec.n_edges,))
    if spec.learn_theta:
        theta = jnp.full((spec.n,), spec.theta_init)
        return jnp.concatenate([z, theta])
    return z


def init_params_from_adjacency(
    spec: SoftNetworkSpec,
    key: jax.Array,
    adjacency: NDArray[np.bool_],
    edge_logit: float = 3.0,
    non_edge_logit: float = -3.0,
    noise_std: float = 0.3,
) -> jax.Array:
    """
    Warm start z near a known discrete adjacency
    Sets z_ij ~ N(edge_logit, noise_std^2) where adjacency[i, j] is True, and z_ij ~ N(non_edge_logit, noise_std^2) where it is False.
    The resulting sigmoid gates are concentrated near 1 on the true edges and near 0 elsewhere,
    so ES starts inside the basin of the corresponding CTLN and only has to refine, rather than discover topology from scratch.

    Params:
    adjacency - (n, n) bool array (A[i, j] => edge j->i)
    edge_logit, non_edge_logit - pre-noise logit values for edge/non-edge
    noise_std - Gaussian noise std added on top of logit values
    """
    rows, cols = spec.offdiag_indices()
    z_vals = np.where(adjacency[rows, cols], edge_logit, non_edge_logit)
    z = jnp.array(z_vals).astype(jnp.float32)
    z = z + noise_std * jax.random.normal(key, (spec.n_edges,))
    if spec.learn_theta:
        theta = jnp.full((spec.n,), spec.theta_init)
        return jnp.concatenate([z, theta])
    return z


def gate_matrix(spec: SoftNetworkSpec, z_flat: jax.Array) -> NDArray[np.float64]:
    """(n, n) matrix of sigmoid gate vals (0 diag)"""
    rows, cols = spec.offdiag_indices()
    G = np.zeros((spec.n, spec.n))
    G[rows, cols] = np.asarray(jax.nn.sigmoid(z_flat))
    return G


def extract_adjacency(
    spec: SoftNetworkSpec, z_flat: jax.Array, thresh: float = 0.5
) -> NDArray[np.bool_]:
    """
    Bool adj ( A[i, j] = True -> edge j->i )
    Thresh > 0.5 on sigmoid to establish connectivity
    Feed into graph.MotifSpec(n=spec.n, adjacency=A) to visualize topology (plot_network_graph)
    """
    return gate_matrix(spec, z_flat) > thresh
