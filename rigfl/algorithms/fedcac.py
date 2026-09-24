"""FedCAC -- cautiously aggressive parameter collaboration (CVPR 2023).

Clients identify parameter-wise critical regions from their local update.  The
server globally averages non-critical parameters and builds a separate critical
parameter model for every client from clients selected by mask similarity.
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
Mask = dict[str, torch.Tensor]


class FedCACConfig(AlgorithmConfig):
    local_epochs: int = Field(5, ge=1, description="Client training epochs per round.")
    lr: float = Field(0.1, gt=0, description="Client optimizer learning rate.")
    tau: float = Field(
        0.5, ge=0, le=1,
        description="Fraction of parameters selected as critical within each tensor.",
    )
    beta: int = Field(
        100, ge=1,
        description="Round at which critical-parameter collaboration ends.",
    )


@dataclass(frozen=True)
class FedCACUpload:
    """One locally trained model and its parameter-wise critical mask."""

    state: State
    mask: Mask


@dataclass(frozen=True)
class FedCACState:
    """Global and client-customized models sent by the paper's server step."""

    global_model: State
    customized_models: tuple[State, ...]
    masks: tuple[Mask, ...]


def critical_parameter_mask(
    initial: Mapping[str, torch.Tensor],
    trained: Mapping[str, torch.Tensor],
    *,
    parameter_names: tuple[str, ...],
    tau: float,
) -> Mask:
    """FedCAC Eqs. 5-6: top-``tau`` sensitivity within each parameter tensor."""
    mask: Mask = {}
    for name in parameter_names:
        sensitivity = ((trained[name] - initial[name]) * trained[name]).abs().flatten()
        count = int(tau * sensitivity.numel())
        selected = torch.zeros_like(sensitivity, dtype=torch.bool)
        if count:
            indices = torch.topk(sensitivity, count, sorted=False).indices
            selected[indices] = True
        mask[name] = selected.reshape_as(trained[name])
    return mask


def mask_distance(left: Mask, right: Mask) -> float:
    """FedCAC mask-location quantity ``||M_i-M_j||_1 / (2n)``."""
    differing = sum(
        torch.logical_xor(left[name], right[name]).sum().item() for name in left
    )
    total = sum(value.numel() for value in left.values())
    return differing / (2.0 * total) if total else 0.0


def collaborators(
    masks: tuple[Mask, ...], *, round_number: int, beta: int
) -> tuple[tuple[int, ...], ...]:
    """FedCAC Eqs. 7-8, using the paper's one-indexed round number."""
    count = len(masks)
    if count < 2:
        return tuple(() for _ in masks)
    if round_number > beta:
        # Section 3.4 states that critical parameters train independently after
        # beta, including the degenerate case where every pairwise value ties.
        return tuple(() for _ in masks)
    distances = {
        (i, j): mask_distance(masks[i], masks[j])
        for i in range(count) for j in range(count) if i != j
    }
    average = sum(distances.values()) / len(distances)
    maximum = max(distances.values())
    threshold = average + (round_number / beta) * (maximum - average)
    return tuple(
        tuple(j for j in range(count) if j != i and distances[i, j] >= threshold)
        for i in range(count)
    )


def _mean_states(states: list[State]) -> State:
    """Equal-average floating model state; retain real values for other buffers."""
    result: State = {}
    for name, first in states[0].items():
        if torch.is_floating_point(first) or torch.is_complex(first):
            value = torch.zeros_like(first)
            for state in states:
                value.add_(state[name].to(value.device), alpha=1.0 / len(states))
            result[name] = value
        else:
            result[name] = first.detach().clone()
    return result


def _personalized_state(global_model: State, customized: State, mask: Mask) -> State:
    result = clone_state_dict(global_model)
    for name, critical in mask.items():
        result[name] = torch.where(
            critical.to(global_model[name].device),
            customized[name].to(global_model[name].device),
            global_model[name],
        )
    return result


class FedCAC(Algorithm):
    """Parameter-wise collaboration over homogeneous personalized models."""

    algorithm_name = "FedCAC"

    def __init__(self, config: FedCACConfig, model_template, num_clients: int):
        super().__init__(config)
        if model_template is None:
            raise ValueError("FedCAC requires a homogeneous model template.")
        if num_clients < 1:
            raise ValueError("FedCAC requires at least one client.")
        self.model_template = copy.deepcopy(model_template).cpu()
        self.num_clients = num_clients
        self.parameter_names = tuple(
            name for name, _ in self.model_template.named_parameters()
        )

    @classmethod
    def from_config(cls, config, *, model_template=None, experiment=None, **resources):
        return cls(config, model_template, experiment.num_clients)

    def init_globals(self) -> FedCACState:
        if self.config.beta > self.total_rounds:
            raise ValueError(
                "FedCAC beta cannot exceed the experiment's total number of rounds."
            )
        initial = clone_state_dict(self.model_template.state_dict())
        empty_mask = {
            name: torch.zeros_like(initial[name], dtype=torch.bool)
            for name in self.parameter_names
        }
        return FedCACState(
            initial,
            tuple(clone_state_dict(initial) for _ in range(self.num_clients)),
            tuple({name: value.clone() for name, value in empty_mask.items()}
                  for _ in range(self.num_clients)),
        )

    def _state_for(self, client_id: int, shared: FedCACState) -> State:
        return _personalized_state(
            shared.global_model,
            shared.customized_models[client_id],
            shared.masks[client_id],
        )

    def local_train(self, client, shared: FedCACState) -> FedCACUpload:
        client_id = client.client_id
        if client_id is None or not 0 <= client_id < self.num_clients:
            raise ValueError("FedCAC requires a valid client_id for every client.")
        initial = self._state_for(client_id, shared)
        model = client.model
        validate_state_structure(
            model, initial, algorithm=self.algorithm_name, client_id=client_id
        )
        model.load_state_dict(initial)
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
        trained = clone_state_dict(model.state_dict())
        mask = critical_parameter_mask(
            initial, trained, parameter_names=self.parameter_names,
            tau=self.config.tau,
        )
        return FedCACUpload(trained, mask)

    def aggregate(self, uploads: list[FedCACUpload], shared: FedCACState) -> FedCACState:
        if len(uploads) != self.num_clients:
            raise ValueError("FedCAC requires one upload from every client each round.")
        for client_id, upload in enumerate(uploads):
            validate_state_structure(
                upload.state, shared.global_model,
                algorithm=self.algorithm_name, client_id=client_id,
            )
        states = [upload.state for upload in uploads]
        masks = tuple(upload.mask for upload in uploads)
        selected = collaborators(
            masks, round_number=self.round_idx + 1, beta=self.config.beta
        )
        global_model = _mean_states(states)
        customized = tuple(
            _mean_states([states[source] for source in (*peers, target)])
            for target, peers in enumerate(selected)
        )
        return FedCACState(global_model, customized, masks)

    @torch.no_grad()
    def predict(self, client, x, shared: FedCACState) -> Predictions:
        client_id = client.client_id
        if client_id is None:
            raise ValueError("FedCAC requires a valid client_id for every client.")
        state = {
            name: value.to(x.device)
            for name, value in self._state_for(client_id, shared).items()
        }
        client.model.to(x.device)
        return Predictions.from_logits(
            torch.func.functional_call(client.model, state, (x,))
        )

    def communication_payload_bytes(self, payload, *, kind: str) -> int:
        if kind == "server_to_client" and isinstance(payload, FedCACState):
            return payload_bytes(payload.global_model) + payload_bytes(
                payload.customized_models[0]
            )
        if kind == "client_to_server" and isinstance(payload, FedCACUpload):
            # Binary masks are logically transmitted as one bit per parameter.
            bits = sum(mask.numel() for mask in payload.mask.values())
            return payload_bytes(payload.state) + (bits + 7) // 8
        return payload_bytes(payload)
