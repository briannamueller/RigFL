"""FedAPEN adaptability splitting and ensemble behavior."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from rigfl.algorithms.fedapen import (
    FedAPEN,
    FedAPENConfig,
    adapt_ensemble_weight,
    ensemble_probabilities,
    split_adaptation_data,
)
from rigfl.core import Client, ClientModel, Identity, iterative
from rigfl.experiment.registry import algorithm_spec, build_algorithm
from tests.helpers import resolved_experiment


def _model(value: float = 0.0) -> ClientModel:
    backbone = nn.Linear(2, 2, bias=False)
    backbone.out_dim = 2
    model = ClientModel(backbone, Identity(2), nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        model.backbone.weight.fill_(value)
        model.head.weight.copy_(torch.eye(2))
    return model


def _loader(size: int = 20) -> DataLoader:
    x = torch.stack([
        torch.tensor([float(index % 2), float((index + 1) % 2)])
        for index in range(size)
    ])
    y = torch.arange(size) % 2
    return DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)


def test_adaptability_split_is_deterministic_disjoint_and_training_only():
    training, adaptation = split_adaptation_data(
        _loader(), fraction=0.2, seed=17
    )

    assert isinstance(training.dataset, Subset)
    assert isinstance(adaptation.dataset, Subset)
    assert set(training.dataset.indices).isdisjoint(adaptation.dataset.indices)
    assert set(training.dataset.indices) | set(adaptation.dataset.indices) == set(range(20))
    assert len(adaptation.dataset) == 4


def test_adaptation_changes_only_the_bounded_ensemble_weight():
    private = _model(1.0)
    shared = _model(-1.0)
    private_before = copy.deepcopy(private.state_dict())
    shared_before = copy.deepcopy(shared.state_dict())
    weight = nn.Parameter(torch.tensor(0.5))

    adapt_ensemble_weight(
        private, shared, _loader(8), weight,
        epochs=2, lr=0.1, device=torch.device("cpu"),
    )

    assert 0.0 <= weight.item() <= 1.0
    assert all(torch.equal(value, private.state_dict()[name])
               for name, value in private_before.items())
    assert all(torch.equal(value, shared.state_dict()[name])
               for name, value in shared_before.items())


def test_probability_ensemble_uses_lambda_for_private_model():
    private_logits = torch.tensor([[4.0, 0.0]])
    shared_logits = torch.tensor([[0.0, 4.0]])

    probabilities = ensemble_probabilities(
        private_logits, shared_logits, torch.tensor(0.75)
    )

    private = torch.softmax(private_logits, dim=1)
    shared = torch.softmax(shared_logits, dim=1)
    assert torch.allclose(probabilities, 0.75 * private + 0.25 * shared)


def test_fedapen_is_heterogeneous_and_runs_through_standard_factory():
    assert algorithm_spec("fedapen").supports_model_heterogeneity is True
    models = [_model(0.1), _model(0.2)]
    experiment = resolved_experiment(
        data_backend="flower",
        model="tabular_linear",
        model_family=None,
        input_kind="numeric",
        input_spec={"input_kind": "numeric", "shape": [2]},
        shared_dim=2,
        num_clients=2,
        num_classes=2,
        seed=3,
    )
    algorithm = build_algorithm(
        "fedapen", experiment,
        FedAPENConfig(local_epochs=1, adaptation_epochs=1),
        initial_client_models=models,
    )
    clients = [
        Client(copy.deepcopy(model), _loader(), _loader(), _loader())
        for model in models
    ]

    result = iterative(
        algorithm, clients, num_rounds=1, device=torch.device("cpu"),
        num_classes=2, verbose=False,
    )

    assert result["evaluation_history"]["evaluation_rounds"] == [0]


def test_fedapen_rejects_a_client_without_disjoint_training_samples():
    algorithm = FedAPEN(
        FedAPENConfig(local_epochs=1, adaptation_epochs=1), _model, seed=0
    )
    algorithm.device = torch.device("cpu")
    client = Client(_model(), _loader(1), client_id=0)

    with pytest.raises(ValueError, match="at least two"):
        algorithm.local_train(client, algorithm.init_globals())
