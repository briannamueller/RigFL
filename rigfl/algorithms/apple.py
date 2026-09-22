"""APPLE -- Adaptive Personalized Cross-Silo Federated Learning (IJCAI 2022).

Each client learns a private directed-relationship vector over a server-held
pool of client core models.  Only the client's own core model and its private
relationship vector are optimized locally; peer core models remain frozen.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from pydantic import Field
from torch import nn

from rigfl.algorithms.fedavg import (
    ModelUpload,
    clone_state_dict,
    validate_state_structure,
)
from rigfl.core.config import AlgorithmConfig
from rigfl.core.interfaces import Algorithm
from rigfl.prediction import Predictions

State = dict[str, torch.Tensor]


class APPLEConfig(AlgorithmConfig):
    local_epochs: int = Field(5, ge=1, description="Client training epochs per round.")
    core_lr: float = Field(0.01, gt=0, description="Learning rate for core models.")
    relationship_lr: float = Field(
        0.001, gt=0, description="Learning rate for directed-relationship vectors."
    )
    momentum: float = Field(0.9, ge=0, description="SGD momentum for local core training.")
    lr_decay: float = Field(
        1.0, gt=0, le=1, description="Per-round multiplier for both learning rates."
    )
    mu: float = Field(
        0.1, ge=0, description="Weight of the directed-relationship proximal term."
    )
    regularization_fraction: float = Field(
        0.1, gt=0, le=1,
        description="Fraction of rounds during which relationship regularization decays.",
    )
    scheduler: Literal["cosine", "exponential"] = Field(
        "exponential", description="Decay shape for relationship regularization."
    )


@dataclass(frozen=True)
class APPLEState:
    """The complete server-held core model pool."""

    cores: tuple[State, ...]


def _mixed_state(
    model,
    cores: tuple[Mapping[str, torch.Tensor], ...],
    relationships: torch.Tensor,
    client_id: int,
    *,
    differentiable: bool,
) -> State:
    """Build APPLE Eq. 4 while differentiating only through the local core."""
    own_parameters = dict(model.named_parameters())
    own_state = model.state_dict()
    mixed: State = {}
    for name in own_state:
        if name not in own_parameters:
            mixed[name] = (
                own_state[name]
                if differentiable else own_state[name].detach().clone()
            )
            continue
        value = None
        for source, source_state in enumerate(cores):
            parameter = (
                own_parameters[name]
                if source == client_id
                else source_state[name].detach().to(relationships.device)
            )
            term = relationships[source] * parameter
            value = term if value is None else value + term
        mixed[name] = value if differentiable else value.detach().clone()
    return mixed


class APPLE(Algorithm):
    """Learn one private collaboration vector and one core model per client."""

    algorithm_name = "APPLE"

    def __init__(self, config: APPLEConfig, initial_client_models,
                 client_sample_counts):
        super().__init__(config)
        if not initial_client_models:
            raise ValueError("APPLE requires independently initialized client core models.")
        if client_sample_counts is None or len(client_sample_counts) != len(initial_client_models):
            raise ValueError("APPLE requires one training sample count per client.")
        if any(count < 0 for count in client_sample_counts) or sum(client_sample_counts) <= 0:
            raise ValueError("APPLE requires non-negative counts and at least one sample.")
        self.initial_models = tuple(copy.deepcopy(model).cpu() for model in initial_client_models)
        reference = self.initial_models[0].state_dict()
        for client_id, model in enumerate(self.initial_models[1:], start=1):
            validate_state_structure(
                model, reference, algorithm=self.algorithm_name, client_id=client_id
            )
        total = float(sum(client_sample_counts))
        self.p0 = torch.tensor(
            [count / total for count in client_sample_counts], dtype=torch.float32
        )

    @classmethod
    def from_config(
        cls, config, *, initial_client_models=None, client_sample_counts=None,
        **resources,
    ):
        return cls(config, initial_client_models, client_sample_counts)

    def init_globals(self) -> APPLEState:
        return APPLEState(tuple(
            clone_state_dict(model.state_dict()) for model in self.initial_models
        ))

    def _regularization_strength(self) -> float:
        if self.config.mu == 0:
            return 0.0
        dynamic_rounds = int(self.total_rounds * self.config.regularization_fraction)
        if dynamic_rounds == 0 or self.round_idx >= dynamic_rounds:
            return 0.0
        progress = self.round_idx / dynamic_rounds
        if self.config.scheduler == "cosine":
            decay = (math.cos(math.pi * progress) + 1.0) / 2.0
        else:
            decay = 0.001 ** progress
        return self.config.mu * decay

    def local_train(self, client, shared: APPLEState) -> ModelUpload:
        client_id = client.client_id
        if client_id is None or not 0 <= client_id < len(shared.cores):
            raise ValueError("APPLE requires a valid client_id for every client.")
        model = client.model
        validate_state_structure(
            model, shared.cores[client_id],
            algorithm=self.algorithm_name, client_id=client_id,
        )
        model.load_state_dict(shared.cores[client_id])
        model.to(self.device)
        model.train()

        relationships = client.state.get("apple_relationships")
        if relationships is None:
            relationships = nn.Parameter(self.p0.to(self.device).clone())
            client.state["apple_relationships"] = relationships
        elif relationships.device != self.device:
            relationships = nn.Parameter(relationships.detach().to(self.device))
            client.state["apple_relationships"] = relationships

        round_decay = self.config.lr_decay ** self.round_idx
        optimizer = torch.optim.SGD(
            [
                {"params": model.parameters(), "lr": self.config.core_lr * round_decay},
                {"params": [relationships],
                 "lr": self.config.relationship_lr * round_decay},
            ],
            momentum=self.config.momentum,
        )
        p0 = self.p0.to(self.device)
        regularization = self._regularization_strength()
        cores = tuple({
            name: value.to(self.device) for name, value in state.items()
        } for state in shared.cores)
        for _ in range(self.config.local_epochs):
            for x, y in client.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                personalized = _mixed_state(
                    model, cores, relationships, client_id, differentiable=True
                )
                logits = torch.func.functional_call(model, personalized, (x,))
                loss = F.cross_entropy(logits, y)
                if regularization:
                    loss = loss + 0.5 * regularization * (
                        relationships - p0
                    ).square().sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        updated_core = clone_state_dict(model.state_dict())
        retained_cores = list(cores)
        retained_cores[client_id] = updated_core
        client.state["apple_personalized_state"] = _mixed_state(
            model, tuple(retained_cores), relationships, client_id,
            differentiable=False,
        )
        return ModelUpload(updated_core, len(client.train_loader.dataset))

    def aggregate(self, uploads: list[ModelUpload], shared: APPLEState) -> APPLEState:
        if len(uploads) != len(shared.cores):
            raise ValueError("APPLE requires one upload from every client each round.")
        cores = tuple(upload.state for upload in uploads)
        for client_id, core in enumerate(cores):
            validate_state_structure(
                core, shared.cores[0],
                algorithm=self.algorithm_name, client_id=client_id,
            )
        return APPLEState(cores)

    @torch.no_grad()
    def predict(self, client, x, shared: APPLEState) -> Predictions:
        personalized = client.state.get("apple_personalized_state")
        if personalized is None:
            raise RuntimeError("APPLE cannot predict before the client has trained.")
        state = {name: value.to(x.device) for name, value in personalized.items()}
        client.model.to(x.device)
        return Predictions.from_logits(
            torch.func.functional_call(client.model, state, (x,))
        )
