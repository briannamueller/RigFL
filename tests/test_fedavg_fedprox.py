"""FedAvg/FedProx behavior and the end-to-end run of every built-in algorithm."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.algorithms.fedavg import (
    FedAvg,
    FedAvgConfig,
    ModelUpload,
    weighted_average_states,
)
from rigfl.algorithms.fedprox import FedProx, FedProxConfig, proximal_penalty
from rigfl.core import Client, ClientModel, Identity
from rigfl.experiment.artifacts import validate_run_record
from rigfl.experiment.identity import fingerprint
from rigfl.experiment.registry import (
    BUILT_IN_ALGORITHMS,
    config_class,
    ignored_experiment_fields,
    resolve_algorithm_models,
    run_identity,
)
from rigfl.experiment.run import ResolvedData
from rigfl.models.registry import MODEL_ARCHITECTURE_REGISTRY
from tests.helpers import resolved_experiment, run_resolved

DEVICE = torch.device("cpu")


class TinyBackbone(nn.Module):
    out_dim = 3

    def __init__(self, input_spec=None):
        super().__init__()
        self.linear = nn.Linear(4, self.out_dim)

    def forward(self, x):
        return torch.relu(self.linear(x.flatten(1)))


def _model(hidden: int = 3) -> ClientModel:
    backbone = nn.Sequential(nn.Linear(4, hidden), nn.ReLU())
    backbone.out_dim = hidden
    return ClientModel(backbone, Identity(hidden), nn.Linear(hidden, 2))


def _empty_loader() -> DataLoader:
    return DataLoader(TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long)))


def _client(model, loader=None, client_id=0) -> Client:
    return Client(
        model, _empty_loader() if loader is None else loader,
        client_id=client_id,
    )


def _on_cpu(algorithm):
    algorithm.device = DEVICE
    algorithm.round_idx = 0
    algorithm.total_rounds = 1
    return algorithm


def _fingerprint(name, exp, algorithm_config):
    return fingerprint(run_identity(name, exp, algorithm_config))


def test_fedavg_synchronizes_client_from_global_before_local_training():
    algorithm = _on_cpu(FedAvg(
        FedAvgConfig(local_epochs=1, lr=0.1), _model()))
    global_model = algorithm.init_globals()
    client_model = _model()
    with torch.no_grad():
        for parameter in global_model.parameters():
            parameter.fill_(0.25)
        for parameter in client_model.parameters():
            parameter.fill_(9.0)

    upload = algorithm.local_train(_client(client_model), global_model)

    for name, value in global_model.state_dict().items():
        assert torch.equal(client_model.state_dict()[name], value)
        assert torch.equal(upload.state[name], value)


def test_fedavg_uses_sample_weighted_average():
    first = {"weight": torch.tensor([1.0, 3.0])}
    second = {"weight": torch.tensor([5.0, 7.0])}
    averaged = weighted_average_states(
        [ModelUpload(first, 1), ModelUpload(second, 3)], device=DEVICE
    )

    assert torch.allclose(averaged["weight"], torch.tensor([4.0, 6.0]))


def test_fedprox_penalty_matches_the_published_objective():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    reference = {"weight": torch.zeros_like(model.weight)}
    # mu / 2 * (2^2 + 2^2) = 2 when mu = 0.5
    assert proximal_penalty(model, reference, mu=0.5).item() == pytest.approx(2.0)


def test_fedprox_mu_zero_matches_fedavg_and_positive_mu_changes_local_training():
    template = _model()
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    y = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=1, shuffle=False)

    avg = _on_cpu(FedAvg(
        FedAvgConfig(local_epochs=1, lr=0.1), template))
    zero = _on_cpu(FedProx(
        FedProxConfig(local_epochs=1, lr=0.1, mu=0.0), template))
    positive = _on_cpu(FedProx(
        FedProxConfig(local_epochs=1, lr=0.1, mu=2.0), template))
    avg_upload = avg.local_train(
        _client(copy.deepcopy(template), loader), avg.init_globals())
    zero_upload = zero.local_train(
        _client(copy.deepcopy(template), loader), zero.init_globals())
    prox_upload = positive.local_train(
        _client(copy.deepcopy(template), loader), positive.init_globals())

    assert all(torch.equal(avg_upload.state[k], zero_upload.state[k])
               for k in avg_upload.state)
    assert any(not torch.equal(avg_upload.state[k], prox_upload.state[k])
               for k in avg_upload.state if torch.is_floating_point(avg_upload.state[k]))


@pytest.mark.parametrize(
    "name", ["fedavg", "fedprox", "fedcac", "fedapa", "fedamp", "apple", "fedpac"]
)
def test_homogeneous_algorithms_reject_a_model_family(name):
    exp = resolved_experiment(model_arch="image_heterogeneous_3")

    with pytest.raises(ValueError, match="identical client architectures"):
        resolve_algorithm_models(name, exp)


def test_algorithm_settings_change_run_fingerprints():
    exp = resolved_experiment(model_arch="fedavg_cnn")
    original = _fingerprint("fedprox", exp, FedProxConfig().model_dump())
    assert original != _fingerprint("fedprox", exp, FedProxConfig(mu=0.2).model_dump())


def _artifact(tmp_path):
    for cid in range(2):
        directory = tmp_path / "clients" / f"client_{cid}"
        directory.mkdir(parents=True)
        for split, n in (("train", 6), ("validation", 4), ("test", 4)):
            x = torch.randn(n, 4)
            y = (torch.arange(n) + cid) % 2
            torch.save((x, y), directory / f"{split}.pt")
    from rigfl.data.config import FlowerDatasetSettings
    settings = FlowerDatasetSettings(
        source_dataset="test/source",
        partition={"scheme": "dirichlet", "num_clients": 2},
    )
    return SimpleNamespace(
        partition_id="tiny-partition",
        path=tmp_path,
        settings=settings,
        manifest={
            "task": "classification",
            "num_clients": 2,
            "input_spec": {"kind": "image", "shape": [4]},
            "target_spec": {"num_classes": 2},
        },
    )


# fedcac's collaboration schedule (beta) must fit inside the one-round run
_E2E_SETTINGS = {"fedcac": {"beta": 1}}


@pytest.mark.parametrize("name", sorted(BUILT_IN_ALGORITHMS))
def test_algorithms_run_end_to_end_through_experiment_infrastructure(
    name, monkeypatch, tmp_path
):
    artifact = _artifact(tmp_path)
    monkeypatch.setitem(
        MODEL_ARCHITECTURE_REGISTRY, "tiny_image", ("image", TinyBackbone))
    resolved = resolved_experiment(
        dataset="tiny", partition_id=artifact.partition_id,
        num_clients=2, num_classes=2, model_arch="tiny_image",
        resolved_models=["tiny_image"],
        input_spec={"input_kind": "image", "shape": [4]},
        rounds=1, shared_dim=3, batch=2, quiet=True,
    )
    data = ResolvedData(settings=artifact.settings, artifact=artifact)

    config = config_class(name)(**_E2E_SETTINGS.get(name, {}))
    record = run_resolved(name, resolved, config, DEVICE, data)

    assert record["algorithm"] == name
    assert record["config"]["experiment"]["model_arch"] == "tiny_image"
    for field in ignored_experiment_fields(name):
        assert record["config"]["experiment"][field] is None
    assert record["result"]["evaluation_history"]["evaluation_rounds"] == [0]
    if name != "local":
        assert record["resources"]["observed"]["communication_bytes"]["total"] > 0
    validate_run_record(record)
