"""RigFL core: algorithm contracts and their execution runners."""

from rigfl.core.adapters import Adapter, AdaptivePool, Identity, LearnedProjection
from rigfl.core.interfaces import (Algorithm, IterativeAlgorithm, LocalSelection,
                                   OneShotContext, P2POneShotAlgorithm)
from rigfl.prediction import Predictions, as_predictions
from rigfl.core.model import ClientModel, assemble_model
from rigfl.core.round import Client, iterative, p2p_one_shot

__all__ = [
    "Adapter",
    "AdaptivePool",
    "Identity",
    "LearnedProjection",
    "ClientModel",
    "assemble_model",
    "Algorithm",
    "IterativeAlgorithm",
    "P2POneShotAlgorithm",
    "OneShotContext",
    "LocalSelection",
    "Predictions",
    "as_predictions",
    "Client",
    "iterative",
    "p2p_one_shot",
]
