"""ctln — gCTLN model primitives: weight construction, simulation, dynamics.

This subpackage holds the building blocks of a single graph-coupled
threshold-linear network (gCTLN): turning a topology spec into a weight
matrix (`topology`), constructing discrete motif graphs (`graph`), rolling
out the threshold-linear ODE (`dynamics`), and fitting a linear readout on
top of the resulting trajectory (`readout`).
"""

from __future__ import annotations

from .dynamics import simulate
from .graph import MotifSpec, NetworkSpec, plot_network_graph
from .readout import apply_readout, fit_readout
from .topology import (
    BlockSoftSpec,
    SoftNetworkSpec,
    block_gate_matrix,
    build_W,
    clipped_block_param_to_model,
    clipped_saturation_penalty,
    extract_adjacency,
    extract_block_adjacency,
    extract_block_adjacency_direct,
    gate_matrix,
    gate_saturation_penalty,
    gate_sparsity_penalty,
    init_block_params_from_adjacency,
    init_params,
    init_params_asymmetric,
    init_params_asymmetric_direct,
    init_params_from_adjacency,
    make_block_param_to_model,
    make_fixed_topology_p2m,
    make_param_to_model,
)

__all__ = [
    "BlockSoftSpec",
    "MotifSpec",
    "NetworkSpec",
    "SoftNetworkSpec",
    "apply_readout",
    "block_gate_matrix",
    "build_W",
    "clipped_block_param_to_model",
    "clipped_saturation_penalty",
    "extract_adjacency",
    "extract_block_adjacency",
    "extract_block_adjacency_direct",
    "fit_readout",
    "gate_matrix",
    "gate_saturation_penalty",
    "gate_sparsity_penalty",
    "init_block_params_from_adjacency",
    "init_params",
    "init_params_asymmetric",
    "init_params_asymmetric_direct",
    "init_params_from_adjacency",
    "make_block_param_to_model",
    "make_fixed_topology_p2m",
    "make_param_to_model",
    "plot_network_graph",
    "simulate",
]
