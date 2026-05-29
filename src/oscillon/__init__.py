"""Copyright (c) 2026 Landon Burtle. All rights reserved.

oscillon: A Python machine learning implementation for gCTLN models.
"""

from __future__ import annotations

from ._version import version as __version__
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("oscillon")
except PackageNotFoundError:
    __version__ = "unknown"

from oscillon.dynamics import simulate
from oscillon.topology import (
    SoftNetworkSpec,
    build_W,
    make_param_to_model,
    init_params,
    gate_matrix,
    extract_adjacency,
)
from oscillon.train import openai_es
import jax

__all__ = [
    "__version__",
    "simulate",
    "SoftNetworkSpec",
    "build_W",
    "make_param_to_model",
    "init_params",
    "gate_matrix",
    "extract_adjacency",
    "openai_es",
    "jax",
]