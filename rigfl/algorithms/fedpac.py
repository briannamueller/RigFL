"""FedPAC -- feature alignment and personalized classifier collaboration.

Clients share a sample-weighted feature extractor, align local features to
class-wise global centroids, and upload local classifier heads.  The server
solves one non-negative simplex quadratic program per client to construct its
personalized classifier from all uploaded heads.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pydantic import Field
from torch import nn

from rigfl.algorithms.fedavg import (
    ModelUpload,
    clone_state_dict,
    validate_state_structure,
    weighted_average_states,
)
from rigfl.core.config import AlgorithmConfig
from rigfl.core.interfaces import Algorithm
from rigfl.eval.resources import payload_bytes
from rigfl.prediction import Predictions

State = dict[str, torch.Tensor]
Prototypes = dict[int, torch.Tensor]


class FedPACConfig(AlgorithmConfig):
    feature_epochs: int = Field(
        1, ge=1, description="Feature-extractor training epochs per round."
    )
    feature_lr: float = Field(
        0.01, gt=0, description="Feature-extractor optimizer learning rate."
    )
    head_lr: float = Field(
        0.1, gt=0, description="Classifier learning rate for its one local epoch."
    )
    lamda: float = Field(
        1.0, ge=0, description="Weight of global feature-centroid alignment."
    )
    momentum: float = Field(0.5, ge=0, description="Momentum for local SGD.")
    weight_decay: float = Field(
        0.0005, ge=0, description="Weight decay for local SGD."
    )
    qp_weight_threshold: float = Field(
        1e-3, ge=0,
        description="Classifier weights at or below this value are set to zero.",
    )
    qp_eigen_threshold: float = Field(
        0.01, gt=0,
        description="Eigenvalue threshold used when repairing a non-PSD QP matrix.",
    )
    qp_max_iterations: int = Field(
        5000, ge=1, description="Maximum projected-gradient iterations per classifier QP."
    )
    qp_tolerance: float = Field(
        1e-10, gt=0, description="Convergence tolerance for classifier QP solves."
    )


@dataclass(frozen=True)
class FedPACState:
    feature: State
    heads: tuple[State, ...]
    prototypes: Prototypes


@dataclass(frozen=True)
class FedPACUpload:
    feature: State
    head: State
    variance: float
    class_features: torch.Tensor
    prototypes: Prototypes
    class_counts: dict[int, int]
    num_samples: int


def _feature_state(model) -> State:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("head.")
    }


def _load_feature_state(model, state: Mapping[str, torch.Tensor]) -> None:
    current = model.state_dict()
    current.update(state)
    model.load_state_dict(current)


def _head_state(model) -> State:
    return clone_state_dict(model.head.state_dict())


def _project_simplex(vector: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto {x >= 0, sum(x) = 1}."""
    ordered, _ = torch.sort(vector, descending=True)
    cumulative = torch.cumsum(ordered, dim=0) - 1.0
    indices = torch.arange(
        1, vector.numel() + 1, dtype=vector.dtype, device=vector.device
    )
    positive = ordered - cumulative / indices > 0
    rho = torch.nonzero(positive, as_tuple=False)[-1, 0]
    theta = cumulative[rho] / indices[rho]
    return torch.clamp(vector - theta, min=0)


def _repair_qp_matrix(matrix: torch.Tensor, eigen_threshold: float) -> torch.Tensor:
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    if eigenvalues.min().item() < -1e-10:
        kept = torch.where(
            eigenvalues >= eigen_threshold, eigenvalues, torch.zeros_like(eigenvalues)
        )
        matrix = (eigenvectors * kept.unsqueeze(0)) @ eigenvectors.T
    return (matrix + matrix.T) / 2.0


def solve_classifier_weights(
    variances: list[float],
    class_features: list[torch.Tensor],
    *,
    weight_threshold: float,
    eigen_threshold: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[torch.Tensor, ...]:
    """Solve FedPAC Eq. 11 for every target client on the probability simplex."""
    if not variances or len(variances) != len(class_features):
        raise ValueError("FedPAC requires matching non-empty feature statistics.")
    summaries = torch.stack([
        value.detach().to(device="cpu", dtype=torch.float64)
        for value in class_features
    ])
    count = summaries.shape[0]
    variance = torch.tensor(variances, dtype=torch.float64)
    solutions = []
    for target in range(count):
        differences = summaries[target].unsqueeze(0) - summaries
        flattened = differences.flatten(1)
        distance = flattened @ flattened.T
        matrix = _repair_qp_matrix(
            torch.diag(variance) + distance, eigen_threshold
        )
        largest = torch.linalg.eigvalsh(matrix).max().item()
        weights = torch.full((count,), 1.0 / count, dtype=torch.float64)
        if largest > 0:
            step = 1.0 / (2.0 * largest)
            for _ in range(max_iterations):
                updated = _project_simplex(weights - step * (2.0 * matrix @ weights))
                if torch.max(torch.abs(updated - weights)).item() <= tolerance:
                    weights = updated
                    break
                weights = updated
        weights[weights <= weight_threshold] = 0
        total = weights.sum()
        if total <= 0 or not torch.isfinite(total):
            weights.zero_()
            weights[target] = 1.0
        else:
            weights /= total
        solutions.append(weights.to(dtype=torch.float32))
    return tuple(solutions)


def _combine_states(states: tuple[State, ...], weights: torch.Tensor,
                    target: int) -> State:
    combined = clone_state_dict(states[target])
    for name, reference in states[target].items():
        if torch.is_floating_point(reference) or torch.is_complex(reference):
            value = torch.zeros_like(reference)
            for weight, state in zip(weights, states):
                value.add_(state[name].to(value.device), alpha=float(weight))
            combined[name] = value
    return combined


class FedPAC(Algorithm):
    """A shared feature extractor with per-client QP-combined linear heads."""

    algorithm_name = "FedPAC"

    def __init__(self, config: FedPACConfig, initial_client_models,
                 num_classes: int):
        super().__init__(config)
        if not initial_client_models:
            raise ValueError("FedPAC requires an initial homogeneous client model pool.")
        self.initial_models = tuple(copy.deepcopy(model).cpu() for model in initial_client_models)
        reference = self.initial_models[0].state_dict()
        for client_id, model in enumerate(self.initial_models[1:], start=1):
            validate_state_structure(
                model, reference, algorithm=self.algorithm_name, client_id=client_id
            )
        if not all(isinstance(model.head, nn.Linear) for model in self.initial_models):
            raise ValueError("FedPAC requires one linear classifier head per client.")
        self.num_classes = num_classes

    @classmethod
    def from_config(
        cls, config, *, initial_client_models=None, experiment, **resources,
    ):
        return cls(config, initial_client_models, experiment.num_classes)

    def init_globals(self) -> FedPACState:
        # FedPAC starts all clients from one common model. The extra independently
        # initialized RigFL models establish the client count and compatibility.
        initial = self.initial_models[0]
        head = _head_state(initial)
        return FedPACState(
            _feature_state(initial),
            tuple(clone_state_dict(head) for _ in self.initial_models),
            {},
        )

    @torch.no_grad()
    def _feature_statistics(self, model, loader):
        model.eval()
        totals: dict[int, torch.Tensor] = {}
        squared: dict[int, torch.Tensor] = {}
        counts: dict[int, int] = defaultdict(int)
        total_samples = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            features = model.rep(x)
            total_samples += y.numel()
            for feature, label in zip(features, y):
                cls = int(label.item())
                totals[cls] = feature.clone() if cls not in totals else totals[cls] + feature
                norm = feature.square().sum()
                squared[cls] = norm.clone() if cls not in squared else squared[cls] + norm
                counts[cls] += 1
        if total_samples <= 0:
            raise ValueError("FedPAC requires at least one training sample per client.")
        feature_dim = next(iter(totals.values())).numel()
        class_features = torch.zeros(
            self.num_classes, feature_dim, device=self.device
        )
        centroids: Prototypes = {}
        variance = torch.zeros((), dtype=torch.float64, device=self.device)
        for cls, total in totals.items():
            centroid = total / counts[cls]
            probability = counts[cls] / total_samples
            centroids[cls] = centroid.detach().clone()
            class_features[cls] = probability * centroid
            variance += (
                probability * squared[cls].to(torch.float64) / counts[cls]
                - probability ** 2 * centroid.to(torch.float64).square().sum()
            )
        variance /= total_samples
        return float(variance.item()), class_features.detach(), centroids, dict(counts)

    def local_train(self, client, shared: FedPACState) -> FedPACUpload:
        client_id = client.client_id
        if client_id is None or not 0 <= client_id < len(shared.heads):
            raise ValueError("FedPAC requires a valid client_id for every client.")
        model = client.model
        validate_state_structure(
            _feature_state(model), shared.feature,
            algorithm=self.algorithm_name, client_id=client_id,
        )
        validate_state_structure(
            model.head, shared.heads[client_id],
            algorithm=self.algorithm_name, client_id=client_id,
        )
        _load_feature_state(model, shared.feature)
        model.head.load_state_dict(shared.heads[client_id])
        model.to(self.device)

        variance, class_features, local_centroids, _ = self._feature_statistics(
            model, client.train_loader
        )

        feature_parameters = [
            parameter for name, parameter in model.named_parameters()
            if not name.startswith("head.")
        ]
        for parameter in feature_parameters:
            parameter.requires_grad_(False)
        for parameter in model.head.parameters():
            parameter.requires_grad_(True)
        model.train()
        head_optimizer = torch.optim.SGD(
            model.head.parameters(), lr=self.config.head_lr,
            momentum=self.config.momentum, weight_decay=self.config.weight_decay,
        )
        for x, y in client.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            loss = F.cross_entropy(model(x), y)
            head_optimizer.zero_grad()
            loss.backward()
            head_optimizer.step()

        for parameter in feature_parameters:
            parameter.requires_grad_(True)
        for parameter in model.head.parameters():
            parameter.requires_grad_(False)
        feature_optimizer = torch.optim.SGD(
            feature_parameters, lr=self.config.feature_lr,
            momentum=self.config.momentum, weight_decay=self.config.weight_decay,
        )
        for _ in range(self.config.feature_epochs):
            for x, y in client.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                representation = model.rep(x)
                loss = F.cross_entropy(model.head(representation), y)
                if self.config.lamda:
                    targets = representation.detach().clone()
                    for index, label in enumerate(y):
                        cls = int(label.item())
                        prototype = shared.prototypes.get(cls, local_centroids[cls])
                        targets[index] = prototype.to(self.device)
                    loss = loss + self.config.lamda * F.mse_loss(
                        representation, targets
                    )
                feature_optimizer.zero_grad()
                loss.backward()
                feature_optimizer.step()
        for parameter in model.head.parameters():
            parameter.requires_grad_(True)

        _, _, prototypes, class_counts = self._feature_statistics(
            model, client.train_loader
        )
        return FedPACUpload(
            _feature_state(model), _head_state(model), variance,
            class_features, prototypes, class_counts,
            len(client.train_loader.dataset),
        )

    def aggregate(self, uploads: list[FedPACUpload], shared: FedPACState) -> FedPACState:
        if len(uploads) != len(shared.heads):
            raise ValueError("FedPAC requires one upload from every client each round.")
        feature = weighted_average_states(
            [ModelUpload(upload.feature, upload.num_samples) for upload in uploads],
            device=self.device, algorithm=self.algorithm_name,
        )

        prototype_totals: dict[int, torch.Tensor] = {}
        prototype_counts: dict[int, int] = defaultdict(int)
        for upload in uploads:
            for cls, prototype in upload.prototypes.items():
                count = upload.class_counts[cls]
                contribution = prototype.to(self.device) * count
                prototype_totals[cls] = (
                    contribution.clone() if cls not in prototype_totals
                    else prototype_totals[cls] + contribution
                )
                prototype_counts[cls] += count
        prototypes = {
            cls: total / prototype_counts[cls]
            for cls, total in prototype_totals.items()
        }

        weights = solve_classifier_weights(
            [upload.variance for upload in uploads],
            [upload.class_features for upload in uploads],
            weight_threshold=self.config.qp_weight_threshold,
            eigen_threshold=self.config.qp_eigen_threshold,
            max_iterations=self.config.qp_max_iterations,
            tolerance=self.config.qp_tolerance,
        )
        local_heads = tuple(upload.head for upload in uploads)
        heads = tuple(
            _combine_states(local_heads, client_weights, client_id)
            for client_id, client_weights in enumerate(weights)
        )
        return FedPACState(feature, heads, prototypes)

    @torch.no_grad()
    def predict(self, client, x, shared: FedPACState) -> Predictions:
        client_id = client.client_id
        _load_feature_state(client.model, shared.feature)
        client.model.head.load_state_dict(shared.heads[client_id])
        client.model.to(x.device)
        return Predictions.from_logits(client.model(x))

    def communication_payload_bytes(self, payload, *, kind: str) -> int:
        if kind == "server_to_client" and isinstance(payload, FedPACState):
            return (
                payload_bytes(payload.feature)
                + payload_bytes(payload.heads[0])
                + payload_bytes(payload.prototypes)
            )
        return payload_bytes(payload)
