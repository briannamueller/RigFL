"""FedGH -- Federated Global Header (Yi et al., ACM MM 2023).

Clients keep their feature extractors but share a global classifier head. Each
round, the server broadcasts the head; each client installs it,
fine-tunes locally, and uploads per-class prototypes (mean representations);
the server then *trains* the head on those (prototype -> label) pairs and keeps
it for the next round. Predictions use the head.

Server-side header training follows Algorithm 1 / Eq. 4: the optimizer updates
the header from uploaded (representation, label) pairs. See DEVIATIONS.md.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pydantic import Field

from rigfl.core.config import AlgorithmConfig
from rigfl.core.interfaces import Algorithm
from rigfl.prediction import Predictions

Prototypes = dict[int, torch.Tensor]


class FedGHConfig(AlgorithmConfig):
    local_epochs: int = Field(1, ge=1)
    lr: float = Field(0.01, gt=0)
    server_epochs: int = Field(1, ge=1)
    server_lr: float = Field(0.01, gt=0)


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
        model.head.load_state_dict(global_head.state_dict())   # install the global head
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

        return self._local_prototypes(model, loader, device)

    @torch.no_grad()
    def _local_prototypes(self, model, loader, device) -> Prototypes:
        model.eval()
        total: dict[int, torch.Tensor] = {}
        count: dict[int, int] = defaultdict(int)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            for r, label in zip(model.rep(x), y):
                c = label.item()
                total[c] = r.clone() if c not in total else total[c] + r
                count[c] += 1
        return {c: total[c] / count[c] for c in total}

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
        model = client.model
        # Local backbone + the server-trained head: the model that actually
        # predicts, and so the one whose logits become the distribution.
        model.head.load_state_dict(global_head.state_dict())
        return Predictions.from_logits(model(x))
