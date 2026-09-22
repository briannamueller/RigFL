"""FedAMP, APPLE, and FedPAC collaboration-weight behavior."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.algorithms.apple import APPLE, APPLEConfig, _mixed_state
from rigfl.algorithms.fedamp import FedAMP, FedAMPConfig, attentive_clouds
from rigfl.algorithms.fedpac import FedPAC, FedPACConfig, solve_classifier_weights
from rigfl.core import Client, ClientModel, Identity, iterative
from rigfl.experiment.registry import (
    ALL_ALGORITHMS,
    BASELINES,
    REGISTRY,
    algorithm_spec,
    build_algorithm,
    config_class,
    resolve_algorithm_models,
)
from tests.helpers import resolved_experiment

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


def _clients(models) -> list[Client]:
    return [
        Client(model, _loader(cid), _loader(cid), _loader(cid))
        for cid, model in enumerate(models)
    ]


def test_fedamp_clouds_follow_the_published_attention_equation():
    models = (
        {"weight": torch.tensor([0.0])},
        {"weight": torch.tensor([2.0])},
    )
    clouds = attentive_clouds(
        models, parameter_names=("weight",), alpha=0.5, sigma=2.0
    )
    peer_weight = 0.5 * torch.exp(torch.tensor(-2.0)).item() / 2.0

    assert clouds[0]["weight"].item() == pytest.approx(peer_weight * 2.0)
    assert clouds[1]["weight"].item() == pytest.approx(
        (1.0 - peer_weight) * 2.0
    )


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


@pytest.mark.parametrize("name", ["fedamp", "apple", "fedpac"])
def test_collaboration_algorithms_are_homogeneous_explicit_algorithms(name):
    assert name in REGISTRY and name in ALL_ALGORITHMS and name not in BASELINES
    assert algorithm_spec(name).supports_model_heterogeneity is False
    assert algorithm_spec(name).ignored_experiment_fields == ("model_family",)
    exp = resolved_experiment(
        model="fedavg_cnn", model_family="image_heterogeneous_3"
    )
    assert resolve_algorithm_models(name, exp).resolved_models == ["fedavg_cnn"]


@pytest.mark.parametrize("name", ["fedamp", "apple", "fedpac"])
def test_standard_factory_supplies_initial_model_pool_and_sample_counts(name):
    models = [_model(), _model()]
    algorithm = build_algorithm(
        name,
        resolved_experiment(num_classes=2),
        config_class(name)(),
        model_template=models[0],
        initial_client_models=models,
        client_sample_counts=[1, 3],
    )

    assert isinstance(algorithm, (FedAMP, APPLE, FedPAC))
    assert len(algorithm.initial_models) == 2
    if name == "apple":
        assert torch.equal(algorithm.p0, torch.tensor([0.25, 0.75]))


@pytest.mark.parametrize("name", ["fedamp", "apple", "fedpac"])
def test_collaboration_algorithms_complete_one_standard_iterative_round(name):
    initial_models = [_model(), _model()]
    if name == "fedamp":
        algorithm = FedAMP(
            FedAMPConfig(lamda=0.001), initial_models
        )
    elif name == "apple":
        algorithm = APPLE(
            APPLEConfig(local_epochs=1), initial_models, [4, 4]
        )
    else:
        algorithm = FedPAC(
            FedPACConfig(feature_epochs=1, qp_max_iterations=200),
            initial_models, 2,
        )
    clients = _clients([copy.deepcopy(model) for model in initial_models])

    result = iterative(
        algorithm, clients, num_rounds=1, device=DEVICE,
        num_classes=2, verbose=False,
    )

    assert result["evaluation_history"]["evaluation_rounds"] == [0]
    assert set(result["evaluation_history"]["clients"]) == {"0", "1"}
