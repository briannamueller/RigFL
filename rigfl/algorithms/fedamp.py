"""FedAMP -- federated attentive message passing (AAAI 2021).

The server keeps one personalized model and one cloud (prox-center) model per
client.  Cloud models are convex combinations of the previous personalized
models, with larger weights assigned to nearby clients.  Clients optimize their
own empirical risk with a squared proximal penalty toward their cloud model.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pydantic import Field

from rigfl.algorithms.fedavg import (
    ModelUpload,
    clone_state_dict,
    validate_state_structure,
)
from rigfl.core.config import AlgorithmConfig
from rigfl.core.interfaces import Algorithm
from rigfl.eval.resources import payload_bytes
from rigfl.prediction import Predictions

State = dict[str, torch.Tensor]


class FedAMPConfig(AlgorithmConfig):
    local_epochs: int = Field(1, ge=1, description="Client training epochs per round.")
    lr: float = Field(0.01, gt=0, description="Client optimizer learning rate.")
    self_weight: float = Field(
        0.5, ge=0, le=1,
        description="Weight of the client's own model in its personalized cloud.",
    )
    sigma: float = Field(
        10.0, gt=0,
        description="Distance scale for normalized peer attention.",
    )
    beta: float = Field(
        10_000.0, gt=0,
        description="Denominator of the squared proximal penalty.",
    )


@dataclass(frozen=True)
class FedAMPState:
    """Server state; clouds appear only after the initial local phase."""

    models: tuple[State, ...]
    clouds: tuple[State, ...] | None


def _parameter_names(model) -> tuple[str, ...]:
    return tuple(name for name, _ in model.named_parameters())


def _squared_parameter_distance(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    parameter_names: tuple[str, ...],
) -> float:
    distance = 0.0
    for name in parameter_names:
        delta = left[name].detach().to(device="cpu", dtype=torch.float64) - right[
            name
        ].detach().to(device="cpu", dtype=torch.float64)
        distance += delta.square().sum().item()
    return distance


def attentive_clouds(
    models: tuple[State, ...],
    *,
    parameter_names: tuple[str, ...],
    self_weight: float,
    sigma: float,
) -> tuple[State, ...]:
    """Mix each model with a normalized distance-weighted average of its peers."""
    clouds = []
    for target, target_state in enumerate(models):
        peers = []
        logits = []
        for source, source_state in enumerate(models):
            if source == target:
                continue
            distance = _squared_parameter_distance(
                target_state, source_state, parameter_names
            )
            peers.append(source)
            logits.append(-distance / sigma)

        if not peers:
            clouds.append(clone_state_dict(target_state))
            continue
        peer_weights = (1.0 - self_weight) * torch.softmax(
            torch.tensor(logits, dtype=torch.float64), dim=0
        )

        # The paper combines model parameters. Module buffers are not part of
        # that vector, so retain the target client's buffers unchanged.
        cloud = clone_state_dict(target_state)
        for name in parameter_names:
            value = target_state[name].clone().mul_(self_weight)
            for weight, source in zip(peer_weights, peers):
                value.add_(
                    models[source][name].to(value.device), alpha=weight.item()
                )
            cloud[name] = value
        clouds.append(cloud)
    return tuple(clouds)


class FedAMP(Algorithm):
    """Personalized full-model collaboration through attentive prox-centers."""

    algorithm_name = "FedAMP"

    def __init__(self, config: FedAMPConfig, initial_client_models):
        super().__init__(config)
        if not initial_client_models:
            raise ValueError("FedAMP requires an initialized model for every client.")
        reference_model = copy.deepcopy(initial_client_models[0]).cpu()
        reference = reference_model.state_dict()
        for client_id, model in enumerate(initial_client_models[1:], start=1):
            validate_state_structure(
                model, reference, algorithm=self.algorithm_name, client_id=client_id
            )
        self.initial_models = tuple(
            copy.deepcopy(reference_model) for _ in initial_client_models
        )
        self.parameter_names = _parameter_names(reference_model)

    @classmethod
    def from_config(cls, config, *, initial_client_models=None, **resources):
        return cls(config, initial_client_models)

    def init_globals(self) -> FedAMPState:
        models = tuple(clone_state_dict(model.state_dict()) for model in self.initial_models)
        return FedAMPState(models, None)

    def local_train(self, client, shared: FedAMPState) -> ModelUpload:
        client_id = client.client_id
        if client_id is None or not 0 <= client_id < len(shared.models):
            raise ValueError("FedAMP requires a valid client_id for every client.")
        model = client.model
        local_state = shared.models[client_id]
        validate_state_structure(
            model, local_state, algorithm=self.algorithm_name, client_id=client_id
        )
        cloud_state = (
            None if shared.clouds is None else shared.clouds[client_id]
        )
        # First obtain data-informed client models before measuring similarity.
        starting_state = local_state if cloud_state is None else cloud_state
        model.load_state_dict(starting_state)
        model.to(self.device)
        model.train()
        reference = None if cloud_state is None else {
            name: cloud_state[name].detach().to(self.device)
            for name in self.parameter_names
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.lr)
        proximal_scale = 1.0 / (2.0 * self.config.beta)
        for _ in range(self.config.local_epochs):
            for x, y in client.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = F.cross_entropy(model(x), y)
                if reference is not None and proximal_scale:
                    penalty = sum(
                        (parameter - reference[name]).square().sum()
                        for name, parameter in model.named_parameters()
                    )
                    loss = loss + proximal_scale * penalty
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return ModelUpload(
            clone_state_dict(model.state_dict()), len(client.train_loader.dataset)
        )

    def aggregate(self, uploads: list[ModelUpload], shared: FedAMPState) -> FedAMPState:
        if len(uploads) != len(shared.models):
            raise ValueError("FedAMP requires one upload from every client each round.")
        models = tuple(upload.state for upload in uploads)
        for client_id, model_state in enumerate(models):
            validate_state_structure(
                model_state, shared.models[0],
                algorithm=self.algorithm_name, client_id=client_id,
            )
        clouds = attentive_clouds(
            models,
            parameter_names=self.parameter_names,
            self_weight=self.config.self_weight,
            sigma=self.config.sigma,
        )
        return FedAMPState(models, clouds)

    @torch.no_grad()
    def predict(self, client, x, shared: FedAMPState) -> Predictions:
        client.model.to(x.device)
        return Predictions.from_logits(client.model(x))

    def communication_payload_bytes(self, payload, *, kind: str) -> int:
        if kind == "server_to_client" and isinstance(payload, FedAMPState):
            state = payload.models[0] if payload.clouds is None else payload.clouds[0]
            return payload_bytes(state)
        return payload_bytes(payload)
