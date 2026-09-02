"""FML -- Federated Mutual Learning (Shen et al., 2020).

Each client has its private (heterogeneous) model and a copy of a shared
homogeneous "meme" model. On local data the two mutually distill via logit KL
(both directions). The server FedAvg-averages the memes. Predictions use the
private model.

Each per-client meme copy lives in ``client.state``.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from pydantic import Field

from rigfl.core.config import AlgorithmConfig
from rigfl.core.interfaces import Algorithm
from rigfl.prediction import Predictions
from rigfl.core.model import ClientModel


class FMLConfig(AlgorithmConfig):
    local_epochs: int = Field(1, ge=1)
    lr: float = Field(0.01, gt=0)
    alpha: float = Field(0.5, ge=0, le=1)    # private-model CE vs. distillation weight (official 0.5)
    beta: float = Field(0.5, ge=0, le=1)     # meme CE vs. distillation weight (official 0.5)


class FML(Algorithm):
    def __init__(self, config: FMLConfig, meme_factory: Callable[[], ClientModel]):
        super().__init__(config)
        self.meme_factory = meme_factory

    @classmethod
    def from_config(cls, config, *, aux_model_factory, **resources):
        return cls(config, aux_model_factory)

    def init_globals(self) -> dict:
        return self.meme_factory().state_dict()

    def local_train(self, client, meme_params: dict) -> dict:
        model, loader = client.model, client.train_loader
        device = self.device
        meme = client.state.setdefault("meme", self.meme_factory()).to(device)
        meme.load_state_dict(meme_params)

        model.to(device)
        model.train()
        meme.train()
        opt = torch.optim.SGD(model.parameters(), lr=self.config.lr)
        opt_m = torch.optim.SGD(meme.parameters(), lr=self.config.lr)

        for _ in range(self.config.local_epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out, out_m = model(x), meme(x)
                # Each model's teacher is detached, matching the FML objective.
                loss = (self.config.alpha * F.cross_entropy(out, y)
                        + (1 - self.config.alpha) * F.kl_div(F.log_softmax(out, 1), F.softmax(out_m.detach(), 1), reduction="batchmean"))
                loss_m = (self.config.beta * F.cross_entropy(out_m, y)
                          + (1 - self.config.beta) * F.kl_div(F.log_softmax(out_m, 1), F.softmax(out.detach(), 1), reduction="batchmean"))
                opt.zero_grad()
                opt_m.zero_grad()
                loss.backward()          # graphs are disjoint now -> no retain_graph
                loss_m.backward()
                opt.step()
                opt_m.step()

        return meme.state_dict()

    def aggregate(self, uploads: list[dict], meme_params: dict) -> dict:
        keys = uploads[0].keys()
        return {k: sum(u[k] for u in uploads) / len(uploads) for k in keys}

    @torch.no_grad()
    def predict(self, client, x, meme_params: dict) -> Predictions:
        # Inference uses the private model, not the meme -- so its logits, not
        # the meme's, are what the reported distribution must come from.
        return Predictions.from_logits(client.model(x))
