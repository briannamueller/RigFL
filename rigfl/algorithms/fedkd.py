"""FedKD -- Communication-efficient FL via Knowledge Distillation
(Wu et al., Nature Communications 2022).

Each client has a private model and a copy of a shared mentee. The two mutually
distill logits and hidden features through a learned projection ``W_h``. Only
the SVD-compressed mentee is communicated; predictions use the private model.

Per-client mentee and projection state live in ``client.state``; the compression
schedule reads the experiment-wide ``round_idx`` maintained by RigFL.

See DEVIATIONS.md for changes required by the CNN implementation, including the
scope of SVD compression.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from pydantic import Field

from rigfl.core.config import AlgorithmConfig
from rigfl.core.interfaces import Algorithm
from rigfl.prediction import Predictions
from rigfl.core.model import ClientModel


class FedKDConfig(AlgorithmConfig):
    local_epochs: int = Field(1, ge=1)
    lr: float = Field(0.01, gt=0)
    t_start: float = Field(0.95, ge=0, le=1)   # SVD energy-keep schedule (start -> end)
    t_end: float = Field(0.98, ge=0, le=1)


class FedKD(Algorithm):
    def __init__(self, config: FedKDConfig,
                 mentee_factory: Callable[[], ClientModel], shared_dim: int):
        super().__init__(config)
        self.mentee_factory = mentee_factory   # builds a fresh homogeneous mentee
        self.shared_dim = shared_dim

    @classmethod
    def from_config(cls, config, *, aux_model_factory, experiment, **resources):
        return cls(config, aux_model_factory, experiment.shared_dim)

    def init_globals(self) -> dict:
        return self.mentee_factory().state_dict()   # shared object = mentee parameters

    def _energy(self) -> float:
        return (self.config.t_start
                + (self.round_idx / self.total_rounds)
                * (self.config.t_end - self.config.t_start))

    def local_train(self, client, mentee_params: dict) -> dict:
        model, loader = client.model, client.train_loader
        device = self.device
        state = client.state

        # this client's persistent mentee copy + hidden-alignment projection
        mentee = state.setdefault("mentee", self.mentee_factory()).to(device)
        mentee.load_state_dict(svd_recover(mentee_params))   # recover (no-op on raw round-0 params)
        w_h = state.setdefault("w_h", nn.Linear(self.shared_dim, self.shared_dim, bias=False)).to(device)

        model.to(device)
        model.train()
        mentee.train()
        opt = torch.optim.SGD(model.parameters(), lr=self.config.lr)
        opt_m = torch.optim.SGD(mentee.parameters(), lr=self.config.lr)
        opt_w = torch.optim.SGD(w_h.parameters(), lr=self.config.lr)

        for _ in range(self.config.local_epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                rep, rep_m = model.rep(x), mentee.rep(x)
                out, out_m = model.head(rep), mentee.head(rep_m)

                ce, ce_m = F.cross_entropy(out, y), F.cross_entropy(out_m, y)
                kl = F.kl_div(F.log_softmax(out, 1), F.softmax(out_m, 1), reduction="batchmean") / (ce + ce_m)
                kl_m = F.kl_div(F.log_softmax(out_m, 1), F.softmax(out, 1), reduction="batchmean") / (ce + ce_m)
                hidden = F.mse_loss(rep, w_h(rep_m)) / (ce + ce_m)

                # One combined objective, so the shared hidden-alignment term
                # (paper Eq. 5) is counted ONCE rather than once per model; KL stays
                # undetached (FedKD's choice) so gradients still cross both models.
                loss = ce + ce_m + kl + kl_m + hidden

                for o in (opt, opt_m, opt_w):
                    o.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10)
                nn.utils.clip_grad_norm_(mentee.parameters(), 10)
                for o in (opt, opt_m, opt_w):
                    o.step()

        # upload = the mentee, SVD-compressed at the current energy level
        return svd_compress(mentee.state_dict(), self._energy())

    def aggregate(self, uploads: list[dict], mentee_params: dict) -> dict:
        recovered = [svd_recover(u) for u in uploads]
        keys = recovered[0].keys()
        averaged = {k: sum(r[k] for r in recovered) / len(recovered) for k in keys}
        return svd_compress(averaged, self._energy())

    @torch.no_grad()
    def predict(self, client, x, mentee_params: dict) -> Predictions:
        # Inference uses the private model, not the mentee -- so its logits, not
        # the mentee's, are what the reported distribution must come from.
        return Predictions.from_logits(client.model(x))


# SVD-compress 2-D weights to the requested energy; other tensors pass through.
# See DEVIATIONS.md for the communication-cost limitation on CNNs.
def svd_compress(state_dict: dict, energy: float) -> dict:
    out = {}
    for name, p in state_dict.items():
        w = p.detach().cpu()
        if w.ndim == 2 and min(w.shape) > 1:
            U, S, V = torch.linalg.svd(w, full_matrices=False)
            k = _energy_rank(S, energy)
            out[name] = (U[:, :k], S[:k], V[:k, :])
        else:
            out[name] = w
    return out


def svd_recover(compressed: dict) -> dict:
    out = {}
    for name, v in compressed.items():
        if isinstance(v, tuple):
            U, S, V = v
            out[name] = U @ torch.diag(S) @ V
        else:
            out[name] = v
    return out


def _energy_rank(S: torch.Tensor, energy: float) -> int:
    squared = S.square()
    keep = torch.cumsum(squared, 0) < energy * squared.sum()
    return int(keep.sum().item()) + 1
