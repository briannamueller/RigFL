"""FedCAC parameter selection and collaboration behavior."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.algorithms.fedcac import (
    FedCAC,
    FedCACConfig,
    FedCACUpload,
    collaborators,
    critical_parameter_mask,
)
from rigfl.core import Client, ClientModel, Identity, iterative
from rigfl.experiment.registry import algorithm_spec, build_algorithm
from tests.helpers import resolved_experiment


def _model() -> ClientModel:
    backbone = nn.Linear(2, 2, bias=False)
    backbone.out_dim = 2
    return ClientModel(backbone, Identity(2), nn.Linear(2, 2, bias=False))


def _batch_norm_model() -> ClientModel:
    backbone = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    backbone.out_dim = 2
    return ClientModel(backbone, Identity(2), nn.Linear(2, 2))


def _loader() -> DataLoader:
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([0, 1])
    return DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)


def test_critical_mask_selects_top_tau_within_each_parameter_tensor():
    initial = {"first": torch.zeros(4), "second": torch.zeros(2)}
    trained = {
        "first": torch.tensor([1.0, 3.0, 2.0, 4.0]),
        "second": torch.tensor([5.0, 1.0]),
    }

    mask = critical_parameter_mask(
        initial, trained, parameter_names=("first", "second"), tau=0.5
    )

    assert mask["first"].tolist() == [False, True, False, True]
    assert mask["second"].tolist() == [True, False]


def test_batch_norm_running_statistics_are_always_critical():
    algorithm = FedCAC(FedCACConfig(tau=0.0, beta=1), _batch_norm_model(), 1)
    algorithm.device = torch.device("cpu")
    algorithm.total_rounds = 1
    algorithm.round_idx = 0
    shared = algorithm.init_globals()
    client = Client(_batch_norm_model(), _loader(), client_id=0)

    upload = algorithm.local_train(client, shared)

    assert algorithm.always_critical_names == (
        "backbone.1.running_mean", "backbone.1.running_var"
    )
    assert upload.mask["backbone.1.running_mean"].all()
    assert upload.mask["backbone.1.running_var"].all()


def test_collaborator_threshold_follows_the_paper_schedule():
    masks = (
        {"w": torch.tensor([True, True, False, False])},
        {"w": torch.tensor([True, False, True, False])},
        {"w": torch.tensor([False, False, True, True])},
    )

    early = collaborators(masks, round_number=1, beta=4)
    after_beta = collaborators(masks, round_number=5, beta=4)

    assert early == ((2,), (), (0,))
    assert after_beta == ((), (), ())


def test_aggregation_uses_custom_models_only_at_target_critical_parameters():
    algorithm = FedCAC(FedCACConfig(tau=0.5, beta=1), _model(), 2)
    algorithm.device = torch.device("cpu")
    algorithm.total_rounds = 2
    algorithm.round_idx = 1
    shared = algorithm.init_globals()
    states = []
    masks = []
    for value, critical in ((1.0, 0), (3.0, 1)):
        state = {name: torch.full_like(tensor, value)
                 for name, tensor in shared.global_model.items()}
        mask = {name: torch.zeros_like(tensor, dtype=torch.bool)
                for name, tensor in state.items()
                if name in algorithm.parameter_names}
        first = next(iter(mask.values()))
        first.flatten()[critical] = True
        states.append(state)
        masks.append(mask)

    result = algorithm.aggregate([
        FedCACUpload(states[0], masks[0]),
        FedCACUpload(states[1], masks[1]),
    ], shared)

    first_name = algorithm.parameter_names[0]
    client_zero = algorithm._state_for(0, result)[first_name].flatten()
    assert client_zero[0].item() == pytest.approx(1.0)
    assert client_zero[1].item() == pytest.approx(2.0)


def test_fedcac_is_registered_as_homogeneous_and_runs_one_round():
    assert algorithm_spec("fedcac").supports_model_heterogeneity is False
    model = _model()
    algorithm = build_algorithm(
        "fedcac",
        resolved_experiment(num_clients=2, num_classes=2),
        FedCACConfig(local_epochs=1, beta=1),
        model_template=model,
    )
    clients = [
        Client(copy.deepcopy(model), _loader(), _loader(), _loader())
        for _ in range(2)
    ]

    result = iterative(
        algorithm, clients, num_rounds=1, device=torch.device("cpu"),
        num_classes=2, verbose=False,
    )

    assert result["evaluation_history"]["evaluation_rounds"] == [0]


def test_fedcac_counts_binary_mask_as_one_bit_per_parameter():
    algorithm = FedCAC(FedCACConfig(beta=1), _model(), 1)
    model = _model()
    state = model.state_dict()
    mask = {
        name: torch.zeros_like(parameter, dtype=torch.bool)
        for name, parameter in model.named_parameters()
    }
    upload = FedCACUpload(state, mask)

    model_bytes = sum(value.numel() * value.element_size() for value in state.values())
    mask_bytes = (sum(value.numel() for value in mask.values()) + 7) // 8
    assert algorithm.communication_payload_bytes(
        upload, kind="client_to_server"
    ) == model_bytes + mask_bytes
