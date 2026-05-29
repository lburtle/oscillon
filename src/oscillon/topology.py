from __future__ import annotations
from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

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
    def n_params(self) -> init:
        return self.n_edges + (self.n if self.learn_theta else 0)

    def offdiag_indices(self):
        """Row/col indices of the off-diag entries, row-major (numpy)"""
        return np.nonzero(~np.eye(self.n, dtype=bool))


@partial(jax.jit, static_argnames=("n",))
def build_W(z_flat, n, eps, delta):
    """Map off-diag logits -> full (n, n) inhib weight matrix"""
    gate = jax.nn.sigmoid(z_flat)
    w_off = -1.0 - delta + (eps + delta) * gate
    rows, cols = jnp.nonzero(~jnp.eye(n, dtype=bool), size=n * (n - 1))
    return jnp.zeros((n, n)).at[rows, cols].set(w_off)   # zeros diag

def make_param_to_model(spec: SoftNetworkSpec):
    """
    Build a jitted params -> (W, theta) func for this spec
    Closes over the (static) spec fields so it is safe to vmap over a population of param vectors
    """
    n, eps, delta = spec.n, spec.eps, spec.delta
    n_edges = spec.n_edges
    learn_theta = spec.learn_theta
    theta_const = spec.theta_init

    @jax.jit
    def f(params):
        z = params[:n_edges]
        theta = params[n_edges:] if learn_theta else jnp.full((n,), theta_const)
        return build_W(z, n, eps, delta), theta

    return f


def init_params(spec: SoftNetworkSpec, key, z_mean=0.0, z_std=1.0):
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

def gate_matrix(spec: SoftNetworkSpec, z_flat):
    """ (n, n) matrix of sigmoid gate vals (0 diag) """
    rows, cols = spec.offdiag_indices()
    G = np.zeros((spec.n, spec.n))
    G[rows, cols] = np.asarray(jax.nn.sigmoid(z_flat))
    return G

def extract_adjacency(spec: SoftNetworkSpec, z_flat, thresh=0.5):
    """ 
    Bool adj ( A[i, j] = True -> edge j->i )
    Thresh > 0.5 on sigmoid to establish connectivity
    Feed into graph.MotifSpec(n=spec.n, adjacency=A) to visualize topology (plot_network_graph)
    """
    return gate_matrix(spec, z_flat) > thresh