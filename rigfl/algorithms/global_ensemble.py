"""Global Ensemble -- soft-vote all client models without dynamic selection.

Clients train locally without sharing. At evaluation, the shared object contains
all client models and prediction averages their class probabilities.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from pydantic import Field

from rigfl.core.config import AlgorithmConfig
from rigfl.core.interfaces import Algorithm
from rigfl.prediction import Predictions


class GlobalEnsembleConfig(AlgorithmConfig):
    local_epochs: int = Field(1, ge=1)
    lr: float = Field(0.01, gt=0)


class GlobalEnsemble(Algorithm):
    def init_globals(self) -> list:
        return []

    def local_train(self, client, shared):
        model, loader = client.model, client.train_loader
        device = self.device
        model.to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.lr)
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                loss = F.cross_entropy(model(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model                      # upload a reference to the trained model

    def aggregate(self, uploads: list, shared) -> list:
        return list(uploads)              # shared = every client's model

    @torch.no_grad()
    def predict(self, client, x, shared: list) -> Predictions:
        # The ensemble's own average of member probabilities: already a
        # normalized distribution, and the one the vote is taken over. There are
        # no ensemble-level logits to report -- the members have logits, the
        # average of their softmaxes does not.
        probs = sum(F.softmax(m(x), dim=1) for m in shared) / len(shared)
        return Predictions.from_probabilities(probs)
