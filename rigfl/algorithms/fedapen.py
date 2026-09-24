"""FedAPEN -- personalized ensemble with learned adaptability (KDD 2023).

Each client retains a private model, trains a complete homogeneous shared model,
and learns one private scalar that combines their probability distributions.
The scalar is learned on a deterministic subset carved only from training data.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from pydantic import Field
from torch import nn
from torch.utils.data import DataLoader, Subset

from rigfl.algorithms.fedavg import (
    ModelUpload,
    clone_state_dict,
    weighted_average_states,
)
from rigfl.core.config import AlgorithmConfig
from rigfl.core.interfaces import Algorithm
from rigfl.prediction import Predictions


class FedAPENConfig(AlgorithmConfig):
    local_epochs: int = Field(5, ge=1, description="Client ensemble-training epochs per round.")
    lr: float = Field(0.01, gt=0, description="Private- and shared-model learning rate.")
    adaptation_fraction: float = Field(
        0.05, gt=0, lt=1,
        description="Fraction of client training samples reserved to learn the ensemble weight.",
    )
    adaptation_epochs: int = Field(
        10, ge=1, description="Ensemble-weight training epochs per round."
    )
    adaptation_lr: float = Field(
        0.001, gt=0, description="Learning rate for the client ensemble weight."
    )
    initial_lambda: float = Field(
        0.5, ge=0, le=1,
        description="Initial weight assigned to the private model's probabilities.",
    )
    shared_model: str | None = Field(
        None, description="Architecture used for the complete homogeneous shared model."
    )


def split_adaptation_data(
    loader: DataLoader, *, fraction: float, seed: int
) -> tuple[DataLoader, DataLoader]:
    """Deterministically partition a training loader into training/adaptability data."""
    size = len(loader.dataset)
    if size < 2:
        raise ValueError(
            "FedAPEN requires at least two client training samples so its "
            "adaptability set is disjoint from model-training data."
        )
    adaptation_size = min(max(1, round(size * fraction)), size - 1)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(size, generator=generator).tolist()
    adaptation_indices = order[:adaptation_size]
    training_indices = order[adaptation_size:]
    batch_size = loader.batch_size or 1
    common = {
        "batch_size": batch_size,
        "collate_fn": loader.collate_fn,
        "num_workers": loader.num_workers,
        "pin_memory": loader.pin_memory,
    }
    training = DataLoader(
        Subset(loader.dataset, training_indices),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
        **common,
    )
    adaptation = DataLoader(
        Subset(loader.dataset, adaptation_indices),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 2),
        **common,
    )
    return training, adaptation


def _probabilities(logits: torch.Tensor) -> torch.Tensor:
    probabilities = Predictions.from_logits(logits).probabilities
    assert probabilities is not None
    return probabilities


def ensemble_probabilities(
    private_logits: torch.Tensor,
    shared_logits: torch.Tensor,
    private_weight: torch.Tensor,
) -> torch.Tensor:
    """FedAPEN Eq. 6."""
    return (
        private_weight * _probabilities(private_logits)
        + (1.0 - private_weight) * _probabilities(shared_logits)
    )


def _distribution_cross_entropy(
    probabilities: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    return F.nll_loss(probabilities.clamp_min(1e-12).log(), labels)


def _kl(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = first.clamp_min(1e-12)
    second = second.clamp_min(1e-12)
    return (first * (first.log() - second.log())).sum(dim=1).mean()


def adapt_ensemble_weight(
    private_model,
    shared_model,
    loader: DataLoader,
    private_weight: nn.Parameter,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    """FedAPEN Eq. 8 while both model parameter sets remain fixed."""
    private_model.eval()
    shared_model.eval()
    optimizer = torch.optim.SGD([private_weight], lr=lr)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                private_logits = private_model(x)
                shared_logits = shared_model(x)
            probabilities = ensemble_probabilities(
                private_logits, shared_logits, private_weight
            )
            loss = _distribution_cross_entropy(probabilities, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                private_weight.clamp_(0.0, 1.0)


class FedAPEN(Algorithm):
    """Learned private/shared probability ensembles with a shared full model."""

    algorithm_name = "FedAPEN"

    def __init__(
        self,
        config: FedAPENConfig,
        shared_model_factory: Callable[[], nn.Module],
        *,
        seed: int,
    ):
        super().__init__(config)
        if shared_model_factory is None:
            raise ValueError("FedAPEN requires a complete shared-model factory.")
        self.shared_model_factory = shared_model_factory
        self.seed = seed

    @classmethod
    def from_config(
        cls, config, *, shared_model_factory=None, experiment=None, **resources
    ):
        return cls(config, shared_model_factory, seed=experiment.seed)

    def init_globals(self):
        return self.shared_model_factory()

    def _client_components(self, client, global_shared):
        shared_model = client.state.get("fedapen_shared_model")
        if shared_model is None:
            shared_model = self.shared_model_factory()
            client.state["fedapen_shared_model"] = shared_model
        shared_model.to(self.device)
        shared_model.load_state_dict(global_shared.state_dict())

        private_weight = client.state.get("fedapen_lambda")
        if private_weight is None:
            private_weight = nn.Parameter(torch.tensor(self.config.initial_lambda))
        if private_weight.device != self.device:
            private_weight = nn.Parameter(private_weight.detach().to(self.device))
        client.state["fedapen_lambda"] = private_weight

        loaders = client.state.get("fedapen_loaders")
        if loaders is None:
            loaders = split_adaptation_data(
                client.train_loader,
                fraction=self.config.adaptation_fraction,
                seed=self.seed + 1_000_003 * int(client.client_id or 0),
            )
            client.state["fedapen_loaders"] = loaders
        return shared_model, private_weight, loaders

    def local_train(self, client, global_shared) -> ModelUpload:
        private_model = client.model.to(self.device)
        shared_model, private_weight, loaders = self._client_components(
            client, global_shared
        )
        training_loader, adaptation_loader = loaders

        adapt_ensemble_weight(
            private_model,
            shared_model,
            adaptation_loader,
            private_weight,
            epochs=self.config.adaptation_epochs,
            lr=self.config.adaptation_lr,
            device=self.device,
        )
        fixed_weight = private_weight.detach()
        private_optimizer = torch.optim.SGD(private_model.parameters(), lr=self.config.lr)
        shared_optimizer = torch.optim.SGD(shared_model.parameters(), lr=self.config.lr)

        for _ in range(self.config.local_epochs):
            for x, y in training_loader:
                x, y = x.to(self.device), y.to(self.device)

                private_model.train()
                shared_model.eval()
                with torch.no_grad():
                    shared_probabilities = _probabilities(shared_model(x))
                private_logits = private_model(x)
                private_probabilities = _probabilities(private_logits)
                private_ensemble = (
                    fixed_weight * private_probabilities
                    + (1.0 - fixed_weight) * shared_probabilities
                )
                private_loss = (
                    F.cross_entropy(private_logits, y)
                    + _kl(private_probabilities, shared_probabilities)
                    + _distribution_cross_entropy(private_ensemble, y)
                )
                private_optimizer.zero_grad()
                private_loss.backward()
                private_optimizer.step()

                private_model.eval()
                shared_model.train()
                with torch.no_grad():
                    private_probabilities = _probabilities(private_model(x))
                shared_logits = shared_model(x)
                shared_probabilities = _probabilities(shared_logits)
                shared_ensemble = (
                    fixed_weight * private_probabilities
                    + (1.0 - fixed_weight) * shared_probabilities
                )
                shared_loss = (
                    F.cross_entropy(shared_logits, y)
                    + _kl(shared_probabilities, private_probabilities)
                    + _distribution_cross_entropy(shared_ensemble, y)
                )
                shared_optimizer.zero_grad()
                shared_loss.backward()
                shared_optimizer.step()

        return ModelUpload(
            clone_state_dict(shared_model.state_dict()), len(training_loader.dataset)
        )

    def aggregate(self, uploads: list[ModelUpload], global_shared):
        equal_uploads = [ModelUpload(upload.state, 1) for upload in uploads]
        averaged = weighted_average_states(
            equal_uploads, device=self.device, algorithm=self.algorithm_name
        )
        global_shared.to(self.device)
        global_shared.load_state_dict(averaged)
        return global_shared

    @torch.no_grad()
    def predict(self, client, x, global_shared) -> Predictions:
        private_weight = client.state.get("fedapen_lambda")
        if private_weight is None:
            raise RuntimeError("FedAPEN cannot predict before the client has trained.")
        client.model.to(x.device)
        global_shared.to(x.device)
        probabilities = ensemble_probabilities(
            client.model(x), global_shared(x), private_weight.to(x.device)
        )
        return Predictions.from_probabilities(probabilities)
