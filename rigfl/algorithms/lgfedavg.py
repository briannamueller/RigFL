"""LG-FedAvg -- clients keep local backbones and share the classifier head.

Client heads are averaged by local sample count. Evaluation combines each local
backbone with the global head.

FedGH also shares a head but trains it on the server; LG-FedAvg averages client
heads. See DEVIATIONS.md for differences from the released LG-FedAvg setup.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field

from rigfl.core.config import AlgorithmConfig
from rigfl.core.interfaces import Algorithm
from rigfl.prediction import Predictions


class LGFedAvgConfig(AlgorithmConfig):
    local_epochs: int = Field(1, ge=1)
    lr: float = Field(0.1, gt=0)          # LG-FedAvg paper uses lr 0.1


class LGFedAvg(Algorithm):
    def __init__(self, config: LGFedAvgConfig, shared_dim: int, num_classes: int):
        super().__init__(config)
        self.shared_dim = shared_dim
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, config, *, experiment, **resources):
        return cls(config, experiment.shared_dim, experiment.num_classes)

    def init_globals(self) -> nn.Linear:
        return nn.Linear(self.shared_dim, self.num_classes)

    def local_train(self, client, global_head: nn.Linear):
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
        return (model.head.state_dict(), len(loader.dataset))   # head + sample count

    def aggregate(self, uploads, global_head: nn.Linear) -> nn.Linear:
        global_head = global_head.to(self.device)
        total = sum(n for _, n in uploads)
        avg = {k: sum(head[k].to(self.device) * (n / total) for head, n in uploads)
               for k in uploads[0][0]}
        global_head.load_state_dict(avg)
        return global_head

    @torch.no_grad()
    def predict(self, client, x, global_head: nn.Linear) -> Predictions:
        model = client.model
        # The inference model is the local backbone + the global head, so those
        # are the logits the distribution comes from.
        model.head.load_state_dict(global_head.state_dict())
        return Predictions.from_logits(model(x))
