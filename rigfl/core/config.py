"""Base configuration class for algorithm hyperparameters.

Each federated algorithm declares its complete configuration as a subclass of
:class:`AlgorithmConfig` in its own module. Pydantic validates and coerces values
from YAML and CLI strings, rejects unknown fields, and serializes the resolved
configuration for reproducibility.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AlgorithmConfig(BaseModel):
    """Validation shared by every algorithm configuration."""

    model_config = ConfigDict(extra="forbid")   # a typo'd/unknown field is an error, not silent
