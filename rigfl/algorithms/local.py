"""Local -- each client trains independently without communication.

There is no shared state or server update, so both ``init_globals`` and
``aggregate`` return ``None`` explicitly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from pydantic import Field

from rigfl.core.config import AlgorithmConfig
from rigfl.core.interfaces import Algorithm
from rigfl.prediction import Predictions


class LocalConfig(AlgorithmConfig):
    local_epochs: int = Field(1, ge=1)
    lr: float = Field(0.01, gt=0)


class Local(Algorithm):
    def init_globals(self) -> None:
        return None

    def local_train(self, client, shared) -> None:
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
        return None

    def aggregate(self, uploads: list, shared) -> None:
        return None

    @torch.no_grad()
    def predict(self, client, x, shared) -> Predictions:
        return Predictions.from_logits(client.model(x))     # softmax of the head
