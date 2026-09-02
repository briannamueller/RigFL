"""Operation contracts implemented by RigFL algorithms.

The runner named in the algorithm registry determines which contract applies:

``IterativeAlgorithm``
    Repeats client training followed by server aggregation for numbered rounds.

``P2POneShotAlgorithm``
    Performs local preparation, one peer-to-peer communication event, then one
    complete local computation per client. Its local computations may contain
    internally selected optimization steps, but they are not federated rounds.

These are structural protocols: inheritance is optional. ``shared`` and the
one-shot payloads are deliberately algorithm-defined objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from rigfl.core.config import AlgorithmConfig
from rigfl.prediction import (PROB_SUM_ATOL, PredictionError, Predictions,
                              as_predictions, check_probabilities)

__all__ = [
    "OneShotContext", "LocalSelection",
    "IterativeAlgorithm", "P2POneShotAlgorithm", "Algorithm",
    "Predictions", "PredictionError", "as_predictions",
    "check_probabilities", "PROB_SUM_ATOL",
]


@dataclass
class OneShotContext:
    """Context for one client's non-iterative one-shot operations."""

    device: torch.device
    client_id: int
    client_state: dict
    validation_loader: Any = None


@dataclass(frozen=True)
class LocalSelection:
    """How a one-shot local computation selected the model it retained.

    ``selected_step`` intentionally does not prescribe epochs: an algorithm may
    internally select an epoch, iteration, tree count, or another local step.
    """

    selected_step: int
    metric: str
    validation_value: float


class IterativeAlgorithm(Protocol):
    """Operations required by the default ``iterative`` runner."""

    device: torch.device
    round_idx: int
    total_rounds: int

    def init_globals(self) -> Any: ...

    def local_train(self, client, shared) -> Any: ...

    def aggregate(self, uploads: list, shared) -> Any: ...

    def predict(self, client, x, shared) -> Predictions: ...


class P2POneShotAlgorithm(Protocol):
    """Operations required by the ``p2p_one_shot`` runner."""

    def prepare(self, model, train_loader, ctx: OneShotContext) -> Any: ...

    def one_shot_communication(self, outgoing: list[Any]) -> list[Any]: ...

    def local_computation(self, model, incoming, train_loader,
                          ctx: OneShotContext) -> LocalSelection: ...

    def predict(self, client, x, shared) -> Predictions: ...


class Algorithm:
    """Optional common base; runners rely on the protocols above.

    Prediction is the only operation common to both current runner contracts.
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
