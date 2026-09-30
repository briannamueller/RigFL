"""FedGH -- Federated Global Header (Yi et al., ACM MM 2023).

Clients keep their feature extractors but share a global classifier head. Each
round, the server broadcasts the head; each client installs it,
fine-tunes locally, and uploads per-class prototypes (mean representations);
the server then *trains* the head on those (prototype -> label) pairs and keeps
it for the next round. Predictions use each client's locally fine-tuned model.

Server header training: Algorithm 1 / Eq. 4.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field

from rigfl.algorithms.fedproto import local_prototypes
from rigfl.core.config import AlgorithmConfig
from rigfl.core.interfaces import Algorithm
from rigfl.prediction import Predictions

Prototypes = dict[int, torch.Tensor]


class FedGHConfig(AlgorithmConfig):
    local_epochs: int = Field(1, ge=1, description="Client training epochs per round.")
    lr: float = Field(0.01, gt=0, description="Client optimizer learning rate.")
    server_epochs: int = Field(1, ge=1, description="Server header-training epochs per round.")
    server_lr: float = Field(0.01, gt=0, description="Server optimizer learning rate.")


class FedGH(Algorithm):
    def __init__(self, config: FedGHConfig, shared_dim: int, num_classes: int):
        super().__init__(config)
        self.shared_dim = shared_dim
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, config, *, experiment, **resources):
        return cls(config, experiment.shared_dim, experiment.num_classes)

    def init_globals(self) -> nn.Linear:
        return nn.Linear(self.shared_dim, self.num_classes)   # the shared, server-trained head

    def local_train(self, client, global_head: nn.Linear) -> Prototypes:
        model, loader = client.model, client.train_loader
        device = self.device
        model.head.load_state_dict(global_head.state_dict())
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

        return local_prototypes(model, loader, device)

    def aggregate(self, uploads: list[Prototypes],
                  global_head: nn.Linear) -> nn.Linear:
        device = self.device
        pairs = [(proto, c) for protos in uploads for c, proto in protos.items()]
        global_head = global_head.to(device)
        global_head.train()
        optimizer = torch.optim.SGD(
            global_head.parameters(), lr=self.config.server_lr)

        for _ in range(self.config.server_epochs):
            for proto, c in pairs:
                logit = global_head(proto.unsqueeze(0).to(device))
                loss = F.cross_entropy(logit, torch.tensor([c], device=device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return global_head

    @torch.no_grad()
    def predict(self, client, x, global_head: nn.Linear) -> Predictions:
        return Predictions.from_logits(client.model(x))
