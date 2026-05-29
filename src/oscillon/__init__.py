"""Copyright (c) 2026 Landon Burtle. All rights reserved.

oscillon: A Python machine learning implementation for gCTLN models.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._version import version as __version__

try:
    __version__ = version("oscillon")
except PackageNotFoundError:
    __version__ = "unknown"

from oscillon.dynamics import simulate
from oscillon.topology import (
    SoftNetworkSpec,
    build_W,
    extract_adjacency,
    gate_matrix,
    init_params,
    make_param_to_model,
)
from oscillon.train import openai_es

__all__ = [
    "SoftNetworkSpec",
    "__version__",
    "build_W",
    "extract_adjacency",
    "gate_matrix",
    "init_params",
    "make_param_to_model",
    "openai_es",
    "simulate",
]
