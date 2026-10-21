"""Copyright (c) 2026 Landon Burtle. All rights reserved.

oscillon: A Python machine learning implementation for CTLN models.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._version import version as __version__

try:
    __version__ = version("oscillon")
except PackageNotFoundError:
    __version__ = "unknown"

from oscillon.ctln import (
    SoftNetworkSpec,
    build_W,
    extract_adjacency,
    gate_matrix,
    init_params,
    make_param_to_model,
    simulate,
)
from oscillon.search import openai_es

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
