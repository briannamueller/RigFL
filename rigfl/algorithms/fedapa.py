"""FedAPA -- adaptive personalized aggregation (IJCAI 2025).

The server keeps one feature extractor and one directed aggregation vector per
client.  Classifier heads remain private; only ``backbone + adapter`` state is
communicated and used to construct the next personalized extractor.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pydantic import Field

from rigfl.algorithms.fedavg import clone_state_dict, validate_state_structure
from rigfl.core.config import AlgorithmConfig
from rigfl.core.interfaces import Algorithm
from rigfl.eval.resources import payload_bytes
from rigfl.prediction import Predictions

State = dict[str, torch.Tensor]
_REPRESENTATION_PREFIXES = ("backbone.", "adapter.")


class FedAPAConfig(AlgorithmConfig):
    local_epochs: int = Field(5, ge=1, description="Client training epochs per round.")
    lr: float = Field(0.01, gt=0, description="Client optimizer learning rate.")
    aggregation_lr: float = Field(
        0.01, gt=0,
        description="Server learning rate for client aggregation-weight vectors.",
    )
    mu: float = Field(
        0.5, gt=0, le=1,
        description="Self-weight assigned before each aggregation-vector normalization.",
    )


@dataclass(frozen=True)
class FedAPAState:
    """Server-held client extractors and row-wise aggregation vectors."""

    extractors: tuple[State, ...]
    weights: torch.Tensor


def extractor_state(model) -> State:
    """Snapshot the representation-producing ``backbone + adapter`` state."""
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name.startswith(_REPRESENTATION_PREFIXES)
    }


def load_extractor_state(model, state: Mapping[str, torch.Tensor]) -> None:
    """Load representation state without altering the private classifier head."""
    expected = extractor_state(model)
    validate_state_structure(state, expected, algorithm="FedAPA extractor")
    complete = model.state_dict()
    complete.update(state)
    model.load_state_dict(complete)


def personalized_extractor(
    extractors: tuple[State, ...], weights: torch.Tensor, target: int
) -> State:
    """FedAPA Eq. 5, retaining the target's real value for nonnumeric buffers."""
    result: State = {}
    for name, first in extractors[target].items():
        if torch.is_floating_point(first) or torch.is_complex(first):
            value = torch.zeros_like(first)
            for source, state in enumerate(extractors):
                value.add_(
                    state[name].to(value.device),
                    alpha=float(weights[target, source]),
                )
            result[name] = value
        else:
            result[name] = first.detach().clone()
    return result


def update_aggregation_weights(
    weights: torch.Tensor,
    previous_extractors: tuple[State, ...],
    uploaded_extractors: tuple[State, ...],
    *,
    parameter_names: tuple[str, ...],
    aggregation_lr: float,
    mu: float,
) -> torch.Tensor:
    """FedAPA Eqs. 6-7 followed by clipping, self-weighting, and normalization."""
    updated = weights.detach().clone().to(dtype=torch.float64, device="cpu")
    count = len(previous_extractors)
    for target in range(count):
        downloaded = personalized_extractor(previous_extractors, weights, target)
        delta = {
            name: uploaded_extractors[target][name].detach().cpu().to(torch.float64)
            - downloaded[name].detach().cpu().to(torch.float64)
            for name in parameter_names
        }
        for source in range(count):
            derivative_dot_delta = sum(
                (
                    previous_extractors[source][name].detach().cpu().to(torch.float64)
                    * delta[name]
                ).sum().item()
                for name in parameter_names
            )
            updated[target, source] -= aggregation_lr * derivative_dot_delta
        updated[target].clamp_(0.0, 1.0)
        updated[target, target] = mu
        updated[target] /= updated[target].sum()
    return updated.to(dtype=weights.dtype, device=weights.device)


class FedAPA(Algorithm):
    """Server-learned personalized aggregation of homogeneous extractors."""

    algorithm_name = "FedAPA"

    def __init__(self, config: FedAPAConfig, initial_client_models):
        super().__init__(config)
        if not initial_client_models:
            raise ValueError("FedAPA requires initial client models.")
        self.initial_models = tuple(
            copy.deepcopy(model).cpu() for model in initial_client_models
        )
        reference = extractor_state(self.initial_models[0])
        for client_id, model in enumerate(self.initial_models[1:], start=1):
            validate_state_structure(
                extractor_state(model), reference,
                algorithm=self.algorithm_name, client_id=client_id,
            )
        self.parameter_names = tuple(
            name for name, _ in self.initial_models[0].named_parameters()
            if name.startswith(_REPRESENTATION_PREFIXES)
        )

    @classmethod
    def from_config(cls, config, *, initial_client_models=None, **resources):
        return cls(config, initial_client_models)

    def init_globals(self) -> FedAPAState:
        extractors = tuple(extractor_state(model) for model in self.initial_models)
        weights = torch.eye(len(extractors), dtype=torch.float32)
        return FedAPAState(extractors, weights)

    def local_train(self, client, shared: FedAPAState) -> State:
        client_id = client.client_id
        if client_id is None or not 0 <= client_id < len(shared.extractors):
            raise ValueError("FedAPA requires a valid client_id for every client.")
        model = client.model
        downloaded = personalized_extractor(
            shared.extractors, shared.weights, client_id
        )
        validate_state_structure(
            extractor_state(model), downloaded,
            algorithm=self.algorithm_name, client_id=client_id,
        )
        load_extractor_state(model, downloaded)
        model.to(self.device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.lr)
        for _ in range(self.config.local_epochs):
            for x, y in client.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = F.cross_entropy(model(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return extractor_state(model)

    def aggregate(self, uploads: list[State], shared: FedAPAState) -> FedAPAState:
        if len(uploads) != len(shared.extractors):
            raise ValueError("FedAPA requires one upload from every client each round.")
        reference = shared.extractors[0]
        for client_id, upload in enumerate(uploads):
            validate_state_structure(
                upload, reference, algorithm=self.algorithm_name, client_id=client_id
            )
        uploaded = tuple(clone_state_dict(upload) for upload in uploads)
        weights = update_aggregation_weights(
            shared.weights,
            shared.extractors,
            uploaded,
            parameter_names=self.parameter_names,
            aggregation_lr=self.config.aggregation_lr,
            mu=self.config.mu,
        )
        return FedAPAState(uploaded, weights)

    @torch.no_grad()
    def predict(self, client, x, shared: FedAPAState) -> Predictions:
        client_id = client.client_id
        if client_id is None:
            raise ValueError("FedAPA requires a valid client_id for every client.")
        personalized = personalized_extractor(
            shared.extractors, shared.weights, client_id
        )
        client.model.to(x.device)
        state = {
            name: value.to(x.device)
            for name, value in client.model.state_dict().items()
        }
        state.update({name: value.to(x.device) for name, value in personalized.items()})
        return Predictions.from_logits(
            torch.func.functional_call(client.model, state, (x,))
        )

    def communication_payload_bytes(self, payload, *, kind: str) -> int:
        if kind == "server_to_client" and isinstance(payload, FedAPAState):
            return payload_bytes(payload.extractors[0])
        return payload_bytes(payload)
