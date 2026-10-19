"""Local -- each client trains independently without communication."""

from __future__ import annotations

import torch
from pydantic import Field

from rigfl.core.config import AlgorithmConfig
from rigfl.core.interfaces import Algorithm
from rigfl.core.losses import classification_loss
from rigfl.prediction import Predictions


class LocalConfig(AlgorithmConfig):
    local_epochs: int = Field(1, ge=1, description="Client training epochs per round.")
    lr: float = Field(0.01, gt=0, description="Client optimizer learning rate.")


class Local(Algorithm):
    def init_globals(self) -> None:
        return None

    def make_optimizer(self, model):
        """Create the optimizer used throughout one client's independent training."""
        return torch.optim.SGD(model.parameters(), lr=self.config.lr)

    def train_one_round(self, client, optimizer) -> None:
        """Advance one client's model by the configured number of local epochs."""
        model, loader = client.model, client.train_loader
        model.to(self.device)
        model.train()
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = classification_loss(model(x), y, client.class_weights)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def finalize_local_models(self, models: list[torch.nn.Module]) -> None:
        """Local evaluation needs no shared server state."""
        return None

    # Retain the ordinary Algorithm operations for callers that explicitly use
    # the iterative runner. The registry uses the independent runner instead.
    def local_train(self, client, shared) -> None:
        self.train_one_round(client, self.make_optimizer(client.model))
        return None

    def aggregate(self, uploads: list, shared) -> None:
        return None

    @torch.no_grad()
    def predict(self, client, x, shared) -> Predictions:
        return Predictions.from_logits(client.model(x))
