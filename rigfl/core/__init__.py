"""RigFL core: algorithm contracts and their execution runners."""

from rigfl.core.adapters import Adapter, AdaptivePool, Identity, LearnedProjection
from rigfl.core.interfaces import Algorithm, IterativeAlgorithm
from rigfl.core.model import ClientModel, assemble_model, assemble_native_model
from rigfl.core.round import Client, iterative
from rigfl.prediction import Predictions, as_predictions

__all__ = [
    "Adapter",
    "AdaptivePool",
    "Identity",
    "LearnedProjection",
    "ClientModel",
    "assemble_model",
    "assemble_native_model",
    "Algorithm",
    "IterativeAlgorithm",
    "Predictions",
    "as_predictions",
    "Client",
    "iterative",
]
