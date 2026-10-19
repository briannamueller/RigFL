"""Global Ensemble -- a fixed soft-voting ensemble of local models."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rigfl.algorithms.local import Local, LocalConfig
from rigfl.prediction import Predictions


class GlobalEnsembleConfig(LocalConfig):
    """Global Ensemble uses the same local-training settings as Local."""


class GlobalEnsemble(Local):
    """Train as Local, then uniformly ensemble every selected client model."""

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
