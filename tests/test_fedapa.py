"""FedAPA feature-extractor aggregation behavior."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.algorithms.fedapa import (
    FedAPA,
    FedAPAConfig,
    extractor_state,
    personalized_extractor,
    update_aggregation_weights,
)
from rigfl.core import Client, ClientModel, Identity, iterative
from rigfl.experiment.registry import algorithm_spec, build_algorithm
from tests.helpers import resolved_experiment


def _model(backbone_value: float = 0.0, head_value: float = 0.0) -> ClientModel:
    backbone = nn.Linear(2, 2, bias=False)
    backbone.out_dim = 2
    model = ClientModel(backbone, Identity(2), nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        model.backbone.weight.fill_(backbone_value)
        model.head.weight.fill_(head_value)
    return model


def _loader() -> DataLoader:
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([0, 1])
    return DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)


def test_personalized_extractor_uses_one_client_weight_vector():
    extractors = (extractor_state(_model(1.0)), extractor_state(_model(3.0)))
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]])

    mixed = personalized_extractor(extractors, weights, target=0)

    assert torch.allclose(mixed["backbone.weight"], torch.full((2, 2), 2.5))


def test_weight_update_applies_eq7_then_clip_self_weight_and_normalize():
    previous = (extractor_state(_model(1.0)), extractor_state(_model(2.0)))
    uploaded = (extractor_state(_model(0.5)), extractor_state(_model(2.5)))
    weights = torch.eye(2)

    updated = update_aggregation_weights(
        weights, previous, uploaded,
        parameter_names=("backbone.weight",),
        aggregation_lr=0.1, mu=0.5,
    )

    assert torch.all(updated >= 0)
    assert torch.all(updated <= 1)
    assert torch.allclose(updated.sum(dim=1), torch.ones(2))
    # The self-weight is assigned before normalization, exactly as Algorithm 1.
    assert updated[0, 0].item() == pytest.approx(0.5 / 0.9)


def test_local_training_upload_excludes_private_head():
    models = [_model(1.0, 7.0), _model(1.0, 9.0)]
    algorithm = FedAPA(FedAPAConfig(local_epochs=1), models)
    algorithm.device = torch.device("cpu")
    algorithm.total_rounds = 1
    algorithm.round_idx = 0
    shared = algorithm.init_globals()
    client = Client(copy.deepcopy(models[0]), _loader(), client_id=0)

    upload = algorithm.local_train(client, shared)

    assert upload
    assert all(not name.startswith("head.") for name in upload)
    assert torch.all(client.model.head.weight != 0)


def test_fedapa_is_registered_as_homogeneous_and_runs_one_round():
    assert algorithm_spec("fedapa").supports_model_heterogeneity is False
    models = [_model(), _model()]
    algorithm = build_algorithm(
        "fedapa",
        resolved_experiment(num_clients=2, num_classes=2),
        FedAPAConfig(local_epochs=1),
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
