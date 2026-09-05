"""Traditional homogeneous FedAvg/FedProx behavior and integration."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.core import (Client, ClientModel, Identity, iterative,
                        p2p_one_shot)
from rigfl.data.builder import build_clients
from rigfl.experiment.artifacts import validate_run_record
from rigfl.experiment.config import ExperimentConfig, run_fingerprint
from rigfl.experiment.launch import build_grid
from rigfl.experiment.registry import (
    ALL_ALGORITHMS,
    BASELINES,
    REGISTRY,
    algorithm_spec,
    config_class,
    resolve_algorithm_config,
)
from tests.helpers import resolved_experiment
from rigfl.experiment.run import run_one
from rigfl.algorithms.fedavg import (FedAvg, FedAvgConfig, ModelUpload,
                                    weighted_average_states)
from rigfl.algorithms.fedprox import FedProx, FedProxConfig, proximal_penalty
from rigfl.models.registry import MODEL_ARCHITECTURE_REGISTRY


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


def test_fedavg_uses_sample_weighted_floating_average_and_preserves_integer_state():
    first = {
        "weight": torch.tensor([1.0, 3.0]),
        "num_batches_tracked": torch.tensor(4, dtype=torch.long),
    }
    second = {
        "weight": torch.tensor([5.0, 7.0]),
        "num_batches_tracked": torch.tensor(11, dtype=torch.long),
    }
    averaged = weighted_average_states(
        [ModelUpload(first, 1), ModelUpload(second, 3)], device=DEVICE
    )

    assert torch.allclose(averaged["weight"], torch.tensor([4.0, 6.0]))
    assert averaged["num_batches_tracked"].dtype == torch.long
    assert averaged["num_batches_tracked"].item() == 11


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


def test_one_selected_architecture_constructs_separate_homogeneous_models():
    def source(cid, split):
        n = 6 if split == "train" else 3
        return torch.randn(n, 4), torch.arange(n) % 2, None

    clients = build_clients(
        source, num_clients=2, num_classes=2, backbones=[TinyBackbone],
        shared_dim=3, val_frac=0.25, batch=2,
        adapter=lambda native, shared: Identity(shared),
    )

    assert clients[0].model is not clients[1].model
    assert clients[0].model.backbone is not clients[1].model.backbone
    assert clients[0].model.adapter is not clients[1].model.adapter
    assert clients[0].model.head is not clients[1].model.head
    assert {k: v.shape for k, v in clients[0].model.state_dict().items()} == {
        k: v.shape for k, v in clients[1].model.state_dict().items()
    }


@pytest.mark.parametrize("name", ["fedavg", "fedprox"])
def test_homogeneous_algorithms_reject_multiple_or_implicit_architectures(name):
    cfg = config_class(name)()
    with pytest.raises(ValueError, match="requires exactly one model architecture"):
        resolve_algorithm_config(name, ExperimentConfig(), cfg)
    with pytest.raises(ValueError, match="requires exactly one model architecture"):
        resolve_algorithm_config(
            name,
            ExperimentConfig(
                model_architectures=["fedavg_cnn", "cifar_resnet18"]),
            cfg,
        )
    with pytest.raises(ValueError, match="resolves to 3 architectures"):
        resolve_algorithm_config(
            name,
            ExperimentConfig(
                model_architecture_family="image_heterogeneous_3"),
            cfg,
        )


@pytest.mark.parametrize("name", ["fedavg", "fedprox"])
def test_homogeneous_algorithms_reject_models_incompatible_with_the_input(name):
    exp = resolved_experiment(
        data_backend="biosilo", partition_scheme=None, input_kind="temporal",
        input_spec={"input_kind": "temporal", "n_ts": 3, "n_static": 2,
                    "seq_len": 8},
        model_architectures=["fedavg_cnn"],
    )
    with pytest.raises(ValueError, match="do not accept temporal inputs"):
        resolve_algorithm_config(name, exp, config_class(name)())


def test_fedavg_rejects_an_incompatible_client_structure_before_training():
    algorithm = _on_cpu(FedAvg(FedAvgConfig(), _model(hidden=3)))
    with pytest.raises(ValueError, match="requires homogeneous client models"):
        algorithm.local_train(
            _client(_model(hidden=5), client_id=1), algorithm.init_globals())


def test_algorithms_are_registered_configured_and_sweepable_without_joining_baselines():
    assert REGISTRY["fedavg"].algorithm is FedAvg
    assert REGISTRY["fedprox"].algorithm is FedProx
    assert {"fedavg", "fedprox"} <= set(ALL_ALGORITHMS)
    assert "fedavg" not in BASELINES and "fedprox" not in BASELINES
    assert config_class("fedavg")().model_dump() == {"local_epochs": 1, "lr": 0.01}
    assert config_class("fedprox")(mu=0.2).mu == 0.2
    with pytest.raises(Exception):
        config_class("fedprox")(mu=-0.1)

    assert algorithm_spec("fedavg").runner is iterative
    assert algorithm_spec("local").runner is iterative
    assert algorithm_spec("feddes").runner is p2p_one_shot

    grid = build_grid({
        "algorithms": ["fedavg", "fedprox"],
        "base": {"experiment": {"model_architectures": ["fedavg_cnn"]}},
        "sweep": {"algorithm.mu": [0.1, 0.2]},
    })
    assert sum(task["algorithm"] == "fedavg" for task in grid) == 1
    assert sum(task["algorithm"] == "fedprox" for task in grid) == 2


@pytest.mark.parametrize("name", ["fedavg", "fedprox"])
def test_algorithm_settings_change_run_fingerprints(name):
    exp = resolved_experiment(model_architectures=["fedavg_cnn"])
    Cfg = config_class(name)
    original = run_fingerprint(exp, Cfg().model_dump())
    assert original != run_fingerprint(exp, Cfg(lr=0.02).model_dump())
    assert original != run_fingerprint(exp, Cfg(local_epochs=2).model_dump())
    if name == "fedprox":
        assert original != run_fingerprint(exp, Cfg(mu=0.2).model_dump())


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


@pytest.mark.parametrize("name", ["fedavg", "fedprox"])
def test_algorithms_run_end_to_end_through_experiment_infrastructure(
    name, monkeypatch, tmp_path
):
    artifact = _artifact(tmp_path)
    exp = ExperimentConfig(
        dataset="tiny", model_architectures=["tiny_image"],
        rounds=1, shared_dim=3, batch=2, quiet=True,
    )
    monkeypatch.setitem(
        MODEL_ARCHITECTURE_REGISTRY, "tiny_image", ("image", TinyBackbone))
    from rigfl.experiment.run import ResolvedData
    resolved = resolved_experiment(
        dataset="tiny", partition_id=artifact.partition_id,
        num_clients=2, num_classes=2, model_architectures=["tiny_image"],
        input_spec={"input_kind": "image", "shape": [4]},
        rounds=1, shared_dim=3, batch=2, quiet=True,
    )
    monkeypatch.setattr(
        "rigfl.experiment.run.resolve_experiment_data",
        lambda value: (resolved, ResolvedData(
            settings=artifact.settings, artifact=artifact)),
    )

    record = run_one(name, exp, config_class(name)(), DEVICE)

    assert record["algorithm"] == name
    assert record["config"]["experiment"]["model_architectures"] == [
        "tiny_image"]
    assert record["result"]["evaluation_history"]["evaluation_rounds"] == [0]
    validate_run_record(record)
