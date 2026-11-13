"""Base classes for algorithm settings.

Each algorithm defines its settings, such as ``lr`` or ``mu``, in a
subclass of :class:`AlgorithmConfig`. Settings not defined in the subclass
are rejected. Algorithms that train client models subclass
:class:`ClientOptimizerConfig` instead.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AlgorithmConfig(BaseModel):
    """Validation shared by every algorithm configuration."""

    model_config = ConfigDict(extra="forbid")   # a typo'd/unknown field is an error, not silent


class ClientOptimizerConfig(AlgorithmConfig):
    """Optimizer settings for client-model training (see :func:`rigfl.core.optim.client_optimizer`)."""

    optimizer: Literal["sgd", "adam"] = Field(
        "sgd", description="Optimizer for client-model training.")
    momentum: float = Field(0.0, ge=0, description="SGD momentum; not used by Adam.")
    weight_decay: float = Field(0.0, ge=0, description="Weight decay for client-model training.")
