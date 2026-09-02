"""FedTGP -- Trainable Global Prototypes (Zhang et al., AAAI 2024).

Clients upload class prototypes and pull local features toward global prototypes.
The server trains a persistent prototype generator with a margin-based
contrastive objective instead of directly averaging the uploads.

The adaptive contrastive margin follows paper Eq. 9: each round it is the
largest classwise nearest-other-class prototype distance, capped at
``margin_cap`` (the paper's ``τ`` = 100).
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pydantic import Field

from rigfl.core.config import AlgorithmConfig
from rigfl.core.interfaces import Algorithm
from rigfl.algorithms.fedproto import prototype_prediction
from rigfl.prediction import Predictions

Prototypes = dict[int, torch.Tensor]


class TrainableGlobalPrototypes(nn.Module):
    """Maps a class id to a prototype vector, via an embedding + small MLP."""

    def __init__(self, num_classes: int, feature_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_classes, feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, class_ids: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embed(class_ids))


class FedTGPConfig(AlgorithmConfig):
    local_epochs: int = Field(1, ge=1)
    lr: float = Field(0.01, gt=0)
    lamda: float = Field(0.1, ge=0)          # client proto-pull weight; paper Eq. 11: 0.1 (official code: 10)
    server_epochs: int = Field(1, ge=1)      # paper uses S=100 server epochs/round; set it in validation
    server_lr: float = Field(0.01, gt=0)     # paper ties server lr to the client lr
    margin_cap: float = Field(100.0, gt=0)   # tau: cap on the adaptive contrastive margin (Eq. 9)


class FedTGP(Algorithm):
    def __init__(self, config: FedTGPConfig, num_classes: int, feature_dim: int):
        super().__init__(config)
        self.num_classes = num_classes
        self.feature_dim = feature_dim

    @classmethod
    def from_config(cls, config, *, experiment, **resources):
        return cls(config, experiment.num_classes, experiment.shared_dim)

    def init_globals(self) -> dict:
        # shared carries the persistent server generator + the current prototypes
        return {"tgp": TrainableGlobalPrototypes(self.num_classes, self.feature_dim), "protos": None}

    # ---- client side: train + pull toward global protos, then upload protos ----
    def local_train(self, client, shared: dict) -> Prototypes:
        model, loader = client.model, client.train_loader
        device = self.device
        global_protos = shared["protos"]
        model.to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.lr)

        for _ in range(self.config.local_epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                rep = model.rep(x)
                loss = F.cross_entropy(model.head(rep), y)
                if global_protos is not None:
                    target = rep.detach().clone()
                    for i, label in enumerate(y):
                        p = global_protos.get(label.item())
                        if p is not None:
                            target[i] = p
                    loss = loss + self.config.lamda * F.mse_loss(rep, target)
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

    # ---- server side: TRAIN the generator on the uploaded prototypes ----
    def aggregate(self, uploads: list[Prototypes], shared: dict) -> dict:
        device = self.device
        tgp = shared["tgp"].to(device)
        tgp.train()
        pairs = [(p.to(device), c) for protos in uploads for c, p in protos.items()]
        margin = self._adaptive_margin(uploads, device)             # Eq. 9, recomputed each round
        optimizer = torch.optim.SGD(
            tgp.parameters(), lr=self.config.server_lr)
        all_ids = torch.arange(self.num_classes, device=device)

        for _ in range(self.config.server_epochs):
            for proto, c in pairs:
                gen = tgp(all_ids)                                   # (num_classes, feature_dim)
                dist = torch.cdist(proto.unsqueeze(0), gen).squeeze(0)  # (num_classes,)
                onehot = F.one_hot(torch.tensor(c, device=device), self.num_classes).float()
                logits = -(dist + margin * onehot)                  # margin on the true class
                loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([c], device=device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            protos = {c: tgp(torch.tensor(c, device=device)) for c in range(self.num_classes)}
        return {"tgp": tgp, "protos": protos}

    @torch.no_grad()
    def _adaptive_margin(self, uploads: list[Prototypes], device) -> float:
        """Adaptive contrastive margin (Eq. 9 / official ``servertgp.py``).

        The largest over classes of a class's *nearest-other-class* distance,
        among this round's per-class average prototypes, capped at ``margin_cap``
        (``tau``). Falls back to the cap when fewer than two classes are present.
        """
        per_class: dict[int, list] = defaultdict(list)
        for protos in uploads:
            for c, p in protos.items():
                per_class[c].append(p.to(device))
        classes = sorted(per_class)
        if len(classes) < 2:
            return self.config.margin_cap
        avg = torch.stack([torch.stack(per_class[c]).mean(0) for c in classes])  # (C, D)
        d = torch.cdist(avg, avg)
        d.fill_diagonal_(float("inf"))
        max_gap = d.min(dim=1).values.max().item()   # max over classes of nearest-other-class distance
        return min(max_gap, self.config.margin_cap)

    @torch.no_grad()
    def predict(self, client, x, shared: dict) -> Predictions:
        return prototype_prediction(
            client.model.rep(x), shared["protos"], self.num_classes)
