"""FedAMP, APPLE, and FedPAC collaboration-weight behavior."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.algorithms.apple import APPLE, APPLEConfig, _mixed_state
from rigfl.algorithms.fedamp import (
    FedAMP,
    FedAMPConfig,
    FedAMPState,
    attentive_clouds,
)
from rigfl.algorithms.fedpac import solve_classifier_weights
from rigfl.core import Client, ClientModel, Identity

DEVICE = torch.device("cpu")


def _model() -> ClientModel:
    backbone = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    backbone.out_dim = 3
    return ClientModel(backbone, Identity(3), nn.Linear(3, 2))


def _loader(offset: int = 0) -> DataLoader:
    x = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    y = (torch.arange(4) + offset) % 2
    return DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)


def test_fedamp_clouds_use_fixed_self_weight_and_normalized_peer_attention():
    models = (
        {"weight": torch.tensor([0.0])},
        {"weight": torch.tensor([2.0])},
        {"weight": torch.tensor([4.0])},
    )
    clouds = attentive_clouds(
        models, parameter_names=("weight",), self_weight=0.25, sigma=2.0
    )
    peer_weights = 0.75 * torch.softmax(torch.tensor([-2.0, -8.0]), dim=0)

    assert clouds[0]["weight"].item() == pytest.approx(
        peer_weights[0].item() * 2.0 + peer_weights[1].item() * 4.0
    )
    assert clouds[1]["weight"].item() == pytest.approx(2.0)


def test_fedamp_initializes_all_clients_from_one_model_before_attention():
    first, second = _model(), _model()
    with torch.no_grad():
        for parameter in second.parameters():
            parameter.add_(10.0)
    algorithm = FedAMP(FedAMPConfig(), [first, second])

    shared = algorithm.init_globals()

    assert shared.clouds is None
    for name, value in shared.models[0].items():
        assert torch.equal(shared.models[1][name], value)


def test_fedamp_first_local_phase_ignores_the_proximal_penalty():
    model = _model()
    strong_penalty = FedAMP(FedAMPConfig(beta=1.0), [model])
    weak_penalty = FedAMP(FedAMPConfig(beta=1e6), [model])
    for algorithm in (strong_penalty, weak_penalty):
        algorithm.device = DEVICE

    first_upload = strong_penalty.local_train(
        Client(copy.deepcopy(model), _loader(), client_id=0),
        strong_penalty.init_globals(),
    )
    second_upload = weak_penalty.local_train(
        Client(copy.deepcopy(model), _loader(), client_id=0),
        weak_penalty.init_globals(),
    )

    for name, value in first_upload.state.items():
        assert torch.equal(second_upload.state[name], value)


def test_fedamp_later_local_phase_starts_from_the_cloud_model():
    model = _model()
    algorithm = FedAMP(FedAMPConfig(), [model])
    algorithm.device = DEVICE
    initial = algorithm.init_globals().models[0]
    cloud = {name: value + 2.0 for name, value in initial.items()}
    empty = DataLoader(TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long)))
    client = Client(copy.deepcopy(model), empty, client_id=0)

    upload = algorithm.local_train(
        client, FedAMPState(models=(initial,), clouds=(cloud,))
    )

    for name, value in upload.state.items():
        assert torch.equal(value, cloud[name])


def test_apple_initializes_relationships_from_client_sample_proportions():
    algorithm = APPLE(APPLEConfig(), [_model(), _model()], [1, 3])

    assert torch.equal(algorithm.p0, torch.tensor([0.25, 0.75]))


def test_apple_parameter_mixture_keeps_gradients_for_own_core_and_relationships():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    relationships = nn.Parameter(torch.tensor([0.25, 0.75]))
    cores = (
        {"weight": torch.tensor([[2.0]])},
        {"weight": torch.tensor([[6.0]])},
    )

    mixed = _mixed_state(
        model, cores, relationships, client_id=0, differentiable=True
    )
    mixed["weight"].sum().backward()

    assert mixed["weight"].item() == pytest.approx(5.0)
    assert model.weight.grad.item() == pytest.approx(0.25)
    assert torch.allclose(relationships.grad, torch.tensor([2.0, 6.0]))


def test_fedpac_qp_weights_are_nonnegative_and_renormalized_after_thresholding():
    summaries = [
        torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        torch.tensor([[1.1, 0.0], [0.0, 0.0]]),
        torch.tensor([[-4.0, 0.0], [0.0, 0.0]]),
    ]
    weights = solve_classifier_weights(
        [0.1, 0.1, 0.1], summaries,
        weight_threshold=0.05, eigen_threshold=0.01,
        max_iterations=5000, tolerance=1e-12,
    )

    for vector in weights:
        assert torch.all(vector >= 0)
        assert vector.sum().item() == pytest.approx(1.0)
    assert weights[0][2].item() == 0.0
