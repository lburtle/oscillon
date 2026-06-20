"""oscillon — graph-coupled threshold-linear networks for limit-cycle dynamics."""

from oscillon.dynamics import simulate
from oscillon.topology import (
    SoftNetworkSpec,
    build_W,
    extract_adjacency,
    gate_matrix,
    init_params,
    make_param_to_model,
)
from oscillon.train.es import openai_es

__all__ = [
    "SoftNetworkSpec",
    "build_W",
    "extract_adjacency",
    "gate_matrix",
    "init_params",
    "make_param_to_model",
    "openai_es",
    "simulate",
]
