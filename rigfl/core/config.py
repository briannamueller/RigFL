"""Base class for algorithm settings.

Each algorithm defines its settings, such as ``lr`` or ``mu``, in a
subclass of :class:`AlgorithmConfig`. Settings not defined in the subclass
are rejected.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AlgorithmConfig(BaseModel):
    """Validation shared by every algorithm configuration."""

    model_config = ConfigDict(extra="forbid")   # a typo'd/unknown field is an error, not silent
