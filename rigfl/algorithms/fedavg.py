"""FedAvg -- homogeneous full-model federated averaging.

Every client starts a round from the same global ``ClientModel``, trains it with
local SGD, and uploads a detached state snapshot plus its local sample count.
The server averages floating-point state by sample count. Non-floating buffers
cannot be meaningfully averaged, so they are copied from the largest upload
(ties follow client/upload order).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from pydantic import Field

from rigfl.core.config import AlgorithmConfig
from rigfl.core.interfaces import Algorithm
from rigfl.prediction import Predictions


class FedAvgConfig(AlgorithmConfig):
    local_epochs: int = Field(1, ge=1)
    lr: float = Field(0.01, gt=0)


@dataclass(frozen=True)
class ModelUpload:
    """A complete locally trained model state and its aggregation weight."""

    state: dict[str, torch.Tensor]
    num_samples: int


def clone_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """A detached snapshot: later client training cannot mutate the upload."""
    return {name: value.detach().clone() for name, value in state.items()}


def validate_state_structure(local, global_state: Mapping[str, torch.Tensor], *,
                             algorithm: str, client_id=None) -> None:
    """Reject parameter/buffer layouts that full-model FL cannot synchronize."""
    local_state = local.state_dict() if hasattr(local, "state_dict") else local
    local_keys, global_keys = set(local_state), set(global_state)
    where = f" client {client_id}" if client_id is not None else ""
    prefix = f"{algorithm} requires homogeneous client models;{where}"
    if local_keys != global_keys:
        missing = sorted(global_keys - local_keys)
        extra = sorted(local_keys - global_keys)
        detail = []
        if missing:
            detail.append(f"missing keys {missing}")
        if extra:
            detail.append(f"extra keys {extra}")
        raise ValueError(f"{prefix} has an incompatible state structure ({'; '.join(detail)}).")
    for name, reference in global_state.items():
        value = local_state[name]
        if value.shape != reference.shape:
            raise ValueError(
                f"{prefix} state {name!r} has shape {tuple(value.shape)}, expected "
                f"{tuple(reference.shape)}."
            )
        if value.dtype != reference.dtype:
            raise ValueError(
                f"{prefix} state {name!r} has dtype {value.dtype}, expected "
                f"{reference.dtype}."
            )


def weighted_average_states(uploads: list[ModelUpload], *, device,
                            algorithm: str = "FedAvg") -> dict[str, torch.Tensor]:
    """Sample-count-weighted floating state plus deterministic integer buffers."""
    if not uploads:
        raise ValueError("FedAvg aggregation requires at least one client upload.")
    if any(upload.num_samples < 0 for upload in uploads):
        raise ValueError("FedAvg upload sample counts must be non-negative.")
    total = sum(upload.num_samples for upload in uploads)
    if total <= 0:
        raise ValueError("FedAvg aggregation requires at least one training sample.")

    reference = uploads[0].state
    for upload in uploads[1:]:
        validate_state_structure(upload.state, reference, algorithm=f"{algorithm} upload")

    # There is no arithmetic mean for integer/categorical module state. Selecting
    # the largest-weight contributor preserves dtype and a real client value.
    source = max(enumerate(uploads), key=lambda item: (item[1].num_samples, -item[0]))[1]
    averaged: dict[str, torch.Tensor] = {}
    for name, first in reference.items():
        if torch.is_floating_point(first) or torch.is_complex(first):
            if torch.is_complex(first):
                work_dtype = torch.complex128 if first.dtype == torch.complex128 else torch.complex64
            else:
                work_dtype = torch.float64 if first.dtype == torch.float64 else torch.float32
            value = torch.zeros(first.shape, dtype=work_dtype, device=device)
            for upload in uploads:
                value.add_(upload.state[name].to(device=device, dtype=work_dtype),
                           alpha=upload.num_samples / total)
            averaged[name] = value.to(dtype=first.dtype)
        else:
            averaged[name] = source.state[name].to(device=device).clone()
    return averaged


class FedAvg(Algorithm):
    """Traditional FedAvg over the complete homogeneous client model."""

    algorithm_name = "FedAvg"

    def __init__(self, config: FedAvgConfig, model_template):
        super().__init__(config)
        if model_template is None:
            raise ValueError("FedAvg requires an initial homogeneous client model template.")
        self.model_template = copy.deepcopy(model_template).cpu()

    @classmethod
    def from_config(cls, config, *, model_template=None, **resources):
        return cls(config, model_template)

    def init_globals(self):
        return copy.deepcopy(self.model_template)

    def _round_reference(self, model) -> dict[str, torch.Tensor] | None:
        return None

    def _training_loss(self, model, logits, labels, reference) -> torch.Tensor:
        return F.cross_entropy(logits, labels)

    def local_train(self, client, global_model) -> ModelUpload:
        model, loader = client.model, client.train_loader
        global_state = global_model.state_dict()
        validate_state_structure(model, global_state, algorithm=self.algorithm_name,
                                 client_id=client.client_id)
        model.load_state_dict(global_state)
        model.to(self.device)
        model.train()
        reference = self._round_reference(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.lr)
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = self._training_loss(model, model(x), y, reference)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return ModelUpload(clone_state_dict(model.state_dict()), len(loader.dataset))

    def aggregate(self, uploads: list[ModelUpload], global_model):
        averaged = weighted_average_states(
            uploads, device=self.device, algorithm=self.algorithm_name
        )
        validate_state_structure(averaged, global_model.state_dict(),
                                 algorithm=self.algorithm_name)
        global_model.to(self.device)
        global_model.load_state_dict(averaged)
        return global_model

    @torch.no_grad()
    def predict(self, client, x, global_model) -> Predictions:
        global_model.to(x.device)
        return Predictions.from_logits(global_model(x))
