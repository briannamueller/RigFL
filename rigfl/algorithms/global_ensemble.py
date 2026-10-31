"""Global Ensemble -- a fixed soft-voting ensemble of local models."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from pydantic import Field, field_validator

from rigfl.algorithms.local import Local, LocalConfig
from rigfl.eval.metrics import canonical
from rigfl.prediction import Predictions


class GlobalEnsembleConfig(LocalConfig):
    """Global Ensemble uses the same local-training settings as Local."""

    member_metric: Optional[str] = Field(
        None,
        description=(
            "Validation metric that picks each client's member checkpoint; "
            "defaults to the early-stopping metric. Training still stops on the "
            "early-stopping metric, so setting this to the reporting metric makes "
            "the members the models Local reports."
        ),
    )

    @field_validator("member_metric")
    @classmethod
    def _known_metric(cls, value):
        return None if value is None else canonical(value)


class GlobalEnsemble(Local):
    """Train as Local, then uniformly ensemble every selected client model."""

    # the ensemble exists only once every client has finished training
    scores_every_epoch = False

    def __init__(self, config: GlobalEnsembleConfig):
        super().__init__(config)
        self.members: list[torch.nn.Module] = []

    def finalize_local_models(
        self, models: list[torch.nn.Module]
    ) -> list[torch.nn.Module]:
        self.members = list(models)
        return self.members

    @torch.no_grad()
    def predict(self, client, x, shared=None) -> Predictions:
        members = self.members if shared is None else shared
        if not members:
            raise RuntimeError("global ensemble has no trained member models")
        probabilities = sum(F.softmax(model(x), dim=1) for model in members)
        return Predictions.from_probabilities(probabilities / len(members))
