"""FedAPEN adaptability splitting and ensemble behavior."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.algorithms.fedapen import (
    adapt_ensemble_weight,
    ensemble_probabilities,
    split_adaptation_data,
)
from rigfl.core import ClientModel, Identity


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


def test_adaptability_split_is_a_disjoint_partition_of_training_data():
    training, adaptation = split_adaptation_data(
        _loader(), fraction=0.2, seed=17
    )

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
