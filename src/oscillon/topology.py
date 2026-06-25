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

@dataclass
class BlockSoftSpec:
    """
    Block-structured soft topology
    eps_e: epsilon value per band (e.g. 0.10 for block a, 0.30 for block b). _e refers to "per edge"
    delta_e: delta value per band
    """
    sizes: tuple[int, ...]
    cross_deltas: NDArray[np.float64]
    eps: float | tuple[float, ...] = 0.10
    delta: float | tuple[float, ...] = 0.50
    learn_theta: bool = True
    theta_init: float = 1.0

    def __post_init__(self) -> None:
        k = len(self.sizes)

        def _broadcast(v: object, name: str) -> tuple[float, ...]:
            if np.isscalar(v):
                return tuple(float(v) for _ in range(k))
            t = tuple(float(x) for x in v)  # type: ignore[union-attr]
            if len(t) != k:
                raise ValueError(f"{name} has length {len(t)}, expected {k}")
            return t

        self.eps = _broadcast(self.eps, "eps")
        self.delta = _broadcast(self.delta, "delta")
        assert self.cross_deltas.shape == (k, k)

    @property
    def k(self) -> int: return len(self.sizes)
    @property
    def n_total(self) -> int: return sum(self.sizes)
    @property
    def n_edges(self) -> int: return sum(n_block * (n_block - 1) for n_block in self.sizes)
    @property
    def n_params(self) -> int:
        return self.n_edges + (self.n_total if self.learn_theta else 0)

    def _starts(self) -> list[int]:
        out, s = [], 0
        for n_block in self.sizes:
            out.append(s); s += n_block
        return out
    
    def compile_indices(
        self,
    ) -> tuple[NDArray[np.intp], NDArray[np.intp],
               NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Static arrays for the within-block edges + the fixed cross-block W"""
        N, starts = self.n_total, self._starts()
        rows, cols, eps_e, delta_e = [], [], [], []
        for block, n_block in enumerate(self.sizes):
            s = starts[block]
            for i in range(n_block):
                for j in range(n_block):
                    if i != j:
                        rows.append(s + i); cols.append(s + j)
                        eps_e.append(self.eps[block]); delta_e.append(self.delta[block])
        W_cross = np.zeros((N, N))
        for block_i in range(self.k):
            for block_j in range(self.k):
                if block_i != block_j:
                    start_i, n_i = starts[block_i], self.sizes[block_i]
                    start_j, n_j = starts[block_j], self.sizes[block_j]
                    W_cross[start_i:start_i + n_i, start_j:start_j + n_j] = -1.0 - self.cross_deltas[block_i, block_j]
        return (np.array(rows, np.intp), np.array(cols, np.intp),
                np.array(eps_e), np.array(delta_e), W_cross)

def make_block_param_to_model(
    spec: BlockSoftSpec,
) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    rows, cols, eps_e, delta_e, W_cross = spec.compile_indices()
    rows_j, cols_j = jnp.asarray(rows), jnp.asarray(cols)
    eps_j, delta_j = jnp.asarray(eps_e), jnp.asarray(delta_e)
    W0 = jnp.asarray(W_cross)
    n_edges, N = spec.n_edges, spec.n_total
    learn_theta, theta_const = spec.learn_theta, spec.theta_init

    @jax.jit # type: ignore[untyped-decorator, unused-ignore]
    def f(params: jax.Array) -> tuple[jax.Array, jax.Array]:
        z = params[:n_edges]
        theta = params[n_edges:] if learn_theta else jnp.full((N,), theta_const)
        gate = jax.nn.sigmoid(z)
        w = -1.0 - delta_j + (eps_j + delta_j) * gate
        return W0.at[rows_j, cols_j].set(w), theta

    return cast("Callable[[jax.Array], tuple[jax.Array, jax.Array]]", f)

def init_block_params_from_adjacency(
    spec: BlockSoftSpec, key: jax.Array, adjacency: NDArray[np.bool_],
    edge_logit: float = 3.0, non_edge_logit: float = -3.0, noise_std: float = 0.3,
) -> jax.Array:
    """Warm start from a full (N,N) adjacency, e.g. NetworkSpec.to_torch_adjacency()"""
    rows, cols, *_ = spec.compile_indices()
    a = np.asarray(adjacency)[rows, cols]
    z = jnp.asarray(np.where(a, edge_logit, non_edge_logit).astype(np.float32))
    z = z + noise_std * jax.random.normal(key, (spec.n_edges,))
    if spec.learn_theta:
        return jnp.concatenate([z, jnp.full((spec.n_total,), spec.theta_init)])
    return z

def gate_saturation_penalty(z_flat: jax.Array) -> jax.Array:
    """
    g(1-g): maximal at g=0.5, zero when gates saturate to 0/1.
    Penalizing this is what makes extract_adjacency's 0.5 cut meaningful.
    """
    g = jax.nn.sigmoid(z_flat)
    return jnp.mean(g * (1.0 - g))

def gate_sparsity_penalty(z_flat: jax.Array) -> jax.Array:
    """Mean gate value: pushes towards fewer edges (a different goal than bimodality)"""
    return jnp.mean(jax.nn.sigmoid(z_flat))

@partial(jax.jit, static_argnames=("n",))
def build_W(
    z_flat: jax.Array,
    n: int,
    eps: float,
    delta: float,
) -> jax.Array:
    """
    Map off-diag logits -> full (n, n) inhib weight matrix
    z_flat: (n*(n-1),)
    eps, delta: >0 parameters controlling range of weights (for stability)
    If the weights stay in [-1-delta, -1+eps] then you will stay in the gCTLN regime.
    """
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