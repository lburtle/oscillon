"""search — evolutionary search over gCTLN topologies: the ES optimizer,
fitness construction, and training orchestration.

`es` holds the problem-agnostic OpenAI-ES optimizer. `train` builds the
closed-form-readout fitness functions for a given topology spec and drives
the warm-up / crystallization training loop on top of it.
"""

from __future__ import annotations

from .es import openai_es
from .train import TrainResult, make_fitness, train

__all__ = [
    "TrainResult",
    "make_fitness",
    "openai_es",
    "train",
]
