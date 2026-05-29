"""oscillon — graph-coupled threshold-linear networks for limit-cycle dynamics."""
 
from oscillon.dynamics import simulate
from oscillon.topology import (
    SoftNetworkSpec,
    build_W,
    make_param_to_model,
    init_params,
    gate_matrix,
    extract_adjacency,
)
from oscillon.train.es import openai_es
 
__all__ = [
    "simulate",
    "SoftNetworkSpec",
    "build_W",
    "make_param_to_model",
    "init_params",
    "gate_matrix",
    "extract_adjacency",
    "openai_es",
]
 