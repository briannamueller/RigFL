"""RigFL core: algorithm contracts and their execution runners."""

from rigfl.core.adapters import Adapter, AdaptivePool, Identity, LearnedProjection
from rigfl.core.interfaces import Algorithm
from rigfl.core.model import ClientModel, assemble_model
from rigfl.core.round import Client, append_evaluation, iterative
from rigfl.prediction import Predictions, check_predictions

__all__ = [
    "Adapter",
    "AdaptivePool",
    "Identity",
    "LearnedProjection",
    "ClientModel",
    "assemble_model",
    "Algorithm",
    "Predictions",
    "check_predictions",
    "Client",
    "append_evaluation",
    "iterative",
]
