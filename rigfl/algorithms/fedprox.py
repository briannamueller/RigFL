"""FedProx -- FedAvg with a proximal local objective."""

from __future__ import annotations

from typing import Mapping

import torch
from pydantic import Field

from rigfl.core.config import AlgorithmConfig
from rigfl.algorithms.fedavg import FedAvg


class FedProxConfig(AlgorithmConfig):
    local_epochs: int = Field(1, ge=1)
    lr: float = Field(0.01, gt=0)
    mu: float = Field(0.01, ge=0)


def proximal_penalty(model, reference: Mapping[str, torch.Tensor], mu: float) -> torch.Tensor:
    """``mu / 2 * ||w - w_global||^2`` over trainable parameters."""
    parameters = list(model.named_parameters())
    if not parameters:
        return torch.tensor(0.0)
    penalty = parameters[0][1].new_zeros(())
    for name, parameter in parameters:
        penalty = penalty + (parameter - reference[name]).square().sum()
    return 0.5 * mu * penalty


class FedProx(FedAvg):
    """Traditional FedProx with full-model synchronization and aggregation."""

    algorithm_name = "FedProx"

    def _round_reference(self, model) -> dict[str, torch.Tensor]:
        return {name: parameter.detach().clone()
                for name, parameter in model.named_parameters()}

    def _training_loss(self, model, logits, labels, reference) -> torch.Tensor:
        return (super()._training_loss(model, logits, labels, reference)
                + proximal_penalty(model, reference, self.config.mu))
