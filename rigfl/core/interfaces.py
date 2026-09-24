"""Operation contracts implemented by RigFL algorithms.

The registered algorithms implement the structural ``IterativeAlgorithm``
protocol: inheritance is optional and ``shared`` is deliberately an
algorithm-defined object.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch

from rigfl.core.config import AlgorithmConfig
from rigfl.prediction import (
    PROB_SUM_ATOL,
    PredictionError,
    Predictions,
    as_predictions,
    check_probabilities,
)

__all__ = [
    "IterativeAlgorithm", "Algorithm",
    "Predictions", "PredictionError", "as_predictions",
    "check_probabilities", "PROB_SUM_ATOL",
]


class IterativeAlgorithm(Protocol):
    """Operations required by the default ``iterative`` runner."""

    device: torch.device
    round_idx: int
    total_rounds: int

    def init_globals(self) -> Any: ...

    def local_train(self, client, shared) -> Any: ...

    def aggregate(self, uploads: list, shared) -> Any: ...

    def predict(self, client, x, shared) -> Predictions: ...


class Algorithm:
    """Optional common base; runners rely on the protocols above.

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

    def predict(self, client, x, shared) -> Predictions:
        raise NotImplementedError

    def communication_payload_bytes(self, payload, *, kind: str) -> int:
        """Logical bytes carried by one algorithm payload."""
        from rigfl.eval.resources import payload_bytes
        return payload_bytes(payload)
