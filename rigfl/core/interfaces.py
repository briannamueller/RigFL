"""Base class for RigFL algorithms.

Each algorithm subclasses :class:`Algorithm` and defines ``init_globals``,
``local_train``, ``aggregate``, and ``predict``, which the default
``iterative`` runner calls each round. ``shared`` is the state the server
sends to clients each round, and each algorithm decides what it contains.
"""

from __future__ import annotations

from typing import Any

import torch

from rigfl.core.config import AlgorithmConfig
from rigfl.prediction import (
    PROB_SUM_ATOL,
    PredictionError,
    Predictions,
    check_predictions,
    check_probabilities,
)

__all__ = [
    "Algorithm",
    "Predictions", "PredictionError", "check_predictions",
    "check_probabilities", "PROB_SUM_ATOL",
]


class Algorithm:
    """Base class for RigFL algorithms.

    The iterative runner sets ``device`` and ``total_rounds`` before
    ``init_globals()``, then updates ``round_idx`` at the start of every
    communication round. Algorithms only read these attributes when needed.
    """

    device: torch.device
    round_idx: int
    total_rounds: int

    def __init__(self, config: AlgorithmConfig):
        self.config = config

    @classmethod
    def from_config(cls, config: AlgorithmConfig, **resources):
        """Construct an algorithm from its validated configuration.

        Algorithms with no external construction dependencies inherit this
        implementation. Algorithms that require resolved models or experiment
        metadata override it in their own module.
        """
        return cls(config)

    def init_globals(self) -> Any:
        raise NotImplementedError

    def local_train(self, client, shared) -> Any:
        raise NotImplementedError

    def aggregate(self, uploads: list, shared) -> Any:
        raise NotImplementedError

    def predict(self, client, x, shared) -> Predictions:
        raise NotImplementedError

    def communication_payload_bytes(self, payload, *, kind: str) -> int:
        """Logical bytes carried by one algorithm payload."""
        from rigfl.eval.resources import payload_bytes
        return payload_bytes(payload)
