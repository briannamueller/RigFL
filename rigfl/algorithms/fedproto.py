"""FedProto -- Federated Prototype Learning (Tan et al., AAAI 2022).

Clients keep their own (possibly different) models and never share weights.
Instead, each client shares one *prototype* per class -- the average feature
vector of its examples in that class. The server averages these prototypes into
global prototypes and hands them back. Each client trains normally, plus a term
that pulls each sample's feature toward its class's global prototype. At test
time a sample is labelled by its nearest global prototype.

Shares in representation space (the ``d`` interface); the model's adapter is
what gives every client the same representation width.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from pydantic import Field

from rigfl.core.config import AlgorithmConfig
from rigfl.core.interfaces import Algorithm
from rigfl.prediction import Predictions

Prototypes = dict[int, torch.Tensor]


def prototype_prediction(rep: torch.Tensor, global_protos: Prototypes,
                         num_classes: int) -> Predictions:
    """Nearest-prototype labels, plus a distribution over the same distances.

    Shared by FedProto and FedTGP: both label a sample by the class whose global
    prototype is nearest in representation space, so both need the same answer to
    "how confident was that".

    * **Distance** is Euclidean (L2) -- ``torch.cdist`` with ``p=2``, the same
      distance the nearest-prototype rule has always used. Nothing here changes
      which prototype is nearest.
    * **Probabilities** are ``softmax(-d)`` over classes with a global prototype.
      Negative distance is the score used by the nearest-prototype rule, so its
      cross-entropy is the predictive validation loss for early stopping.
    * **Classes with no global prototype** get probability zero. They are classes
      no client reported an example of, and the nearest-prototype rule could
      never have predicted them either.
    * **Labels** are taken directly from ``argmin(d)``.

    ``num_classes`` is the width of the reported distribution and must be the
    evaluator's class count, not the number of prototypes -- a class no client
    has an example of still exists, and its column is zero.

    """
    classes = sorted(global_protos)
    protos = torch.stack([global_protos[c].to(rep.device) for c in classes])
    d = torch.cdist(rep, protos)                                  # [N, K], Euclidean
    labels = torch.tensor([classes[i] for i in d.argmin(dim=1).tolist()],
                          device=rep.device)
    scores = torch.softmax(-d, dim=1)                             # [N, K]
    probs = torch.zeros(rep.shape[0], num_classes, device=rep.device, dtype=scores.dtype)
    probs[:, torch.as_tensor(classes, device=rep.device)] = scores
    return Predictions.from_probabilities(probs, labels=labels)


def resolve_num_classes(declared, model, global_protos) -> int:
    """The width of the reported distribution.

    A prototype algorithm never needed a class count before -- its prediction was
    the id of the nearest prototype. Now it reports a distribution, which must be
    as wide as the evaluator's class count or every metric downstream is
    comparing different-shaped things. Taken from an explicit value when given,
    then from the model's own head, and only then from the prototypes -- which
    is a lower bound, since a class nobody reported has no prototype.
    """
    if declared is not None:
        return int(declared)
    out = getattr(getattr(model, "head", None), "out_features", None)
    if out:
        return int(out)
    return int(max(global_protos)) + 1


class FedProtoConfig(AlgorithmConfig):
    local_epochs: int = Field(1, ge=1)
    lr: float = Field(0.01, gt=0)
    lamda: float = Field(0.1, ge=0)          # proto-alignment weight; paper: 0.1 (CIFAR-10), 1.0 (MNIST)


class FedProto(Algorithm):
    def init_globals(self) -> None:
        return None                       # no global prototypes yet on round 0

    def local_train(self, client, global_protos: Prototypes | None) -> Prototypes:
        model, loader = client.model, client.train_loader
        device = self.device
        model.to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.lr)

        for _ in range(self.config.local_epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                rep = model.rep(x)
                loss = F.cross_entropy(model.head(rep), y)

                # pull each sample's representation toward its class prototype
                if global_protos is not None:
                    target = rep.detach().clone()
                    for i, label in enumerate(y):
                        proto = global_protos.get(label.item())
                        if proto is not None:
                            target[i] = proto
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

    def aggregate(self, uploads: list[Prototypes], shared) -> Prototypes:
        total: dict[int, torch.Tensor] = {}
        count: dict[int, int] = defaultdict(int)
        for protos in uploads:
            for c, proto in protos.items():
                total[c] = proto.clone() if c not in total else total[c] + proto
                count[c] += 1
        return {c: total[c] / count[c] for c in total}

    @torch.no_grad()
    def predict(self, client, x, global_protos: Prototypes) -> Predictions:
        model = client.model
        nc = resolve_num_classes(None, model, global_protos)
        return prototype_prediction(model.rep(x), global_protos, nc)
