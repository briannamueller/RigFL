"""One experiment-level architecture selection shared by every algorithm."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from rigfl.core import ClientModel
from rigfl.data.config import FlowerDatasetSettings
from rigfl.experiment.artifacts import validate_run_record
from rigfl.experiment.config import ExperimentConfig, run_fingerprint
from rigfl.experiment.launch import build_grid
from rigfl.experiment.registry import (build_algorithm, config_class,
                                       resolve_algorithm_config)
from rigfl.experiment.run import (ResolvedData, resolve_experiment_architectures,
                                  resolve_experiment_data, run_one)
from rigfl.models.registry import (
    MODEL_ARCHITECTURE_FAMILIES,
    MODEL_ARCHITECTURE_REGISTRY,
    instantiate_backbones,
    resolve_model_architectures,
)
from tests.helpers import resolved_experiment


def test_model_architectures_are_configured_independently_of_dataset_name():
    names = resolve_model_architectures(
        architecture_family=None,
        architectures=["fedavg_cnn", "cifar_resnet18"],
        input_kind="image",
    )
    factories = instantiate_backbones(
        names, input_spec={"input_kind": "image", "shape": (1, 28, 28)}
    )
    assert names == ["fedavg_cnn", "cifar_resnet18"]
    assert factories[0]() is not factories[0]()
    assert factories[0]()(torch.randn(2, 1, 28, 28)).shape == (2, 512)


def test_mnist_architecture_family_constructs_distinct_backbones():
    names = resolve_model_architectures(
        architecture_family="mnist_heterogeneous_3",
        architectures=None,
        input_kind="image",
    )
    factories = instantiate_backbones(
        names, input_spec={"kind": "image", "shape": (1, 28, 28)}
    )

    assert names == ["lenet5", "fedavg_mnist_cnn", "small_cnn"]
    assert [factory()(torch.randn(2, 1, 28, 28)).shape for factory in factories] == [
        (2, 84), (2, 512), (2, 128)
    ]


@pytest.mark.parametrize("name", ["lenet5", "fedavg_mnist_cnn"])
def test_mnist_architectures_reject_other_image_sizes(name):
    factory = instantiate_backbones(
        [name], input_spec={"kind": "image", "shape": (3, 32, 32)}
    )[0]

    with pytest.raises(ValueError, match="28x28"):
        factory()


def test_tabular_architecture_family_constructs_distinct_backbones():
    names = resolve_model_architectures(
        architecture_family="tabular_heterogeneous_3",
        architectures=None,
        input_kind="numeric",
    )
    factories = instantiate_backbones(
        names, input_spec={"kind": "numeric", "shape": (12,)}
    )

    assert names == ["tabular_linear", "tabular_mlp", "tabular_residual_mlp"]
    assert [factory()(torch.randn(2, 12)).shape for factory in factories] == [
        (2, 64),
        (2, 64),
        (2, 64),
    ]


def test_numeric_inputs_use_the_tabular_family_by_default():
    assert resolve_model_architectures(
        architecture_family=None,
        architectures=None,
        input_kind="numeric",
    ) == ["tabular_linear", "tabular_mlp", "tabular_residual_mlp"]


def test_model_families_only_contain_registered_architectures():
    assert all(
        name in MODEL_ARCHITECTURE_REGISTRY
        for family in MODEL_ARCHITECTURE_FAMILIES.values()
        for name in family
    )


def test_model_architecture_family_and_list_are_mutually_exclusive():
    with pytest.raises(
        Exception, match="model_architecture_family or model_architectures"
    ):
        ExperimentConfig(
            model_architecture_family="image_heterogeneous_3",
            model_architectures=["fedavg_cnn"],
        )


def test_family_and_explicit_architectures_have_one_resolved_identity():
    family = resolve_experiment_architectures(
        resolved_experiment(model_architecture_family="image_heterogeneous_3"),
        input_kind="image",
    )
    explicit = resolve_experiment_architectures(
        resolved_experiment(model_architectures=[
            "fedavg_cnn", "cifar_resnet18", "cifar_mobilenet_v2"
        ]),
        input_kind="image",
    )

    assert family.model_dump() == explicit.model_dump()
    assert family.model_architecture_family is None
    assert explicit.model_architecture_family is None
    assert family.model_architectures == [
        "fedavg_cnn", "cifar_resnet18", "cifar_mobilenet_v2"
    ]
    assert run_fingerprint(family, {}) == run_fingerprint(explicit, {})


def test_default_model_architectures_are_recorded_explicitly(monkeypatch):
    from rigfl.data.config import FlowerDatasetSettings

    settings = FlowerDatasetSettings(
        source_dataset="example/source",
        partition={
            "scheme": "dirichlet", "num_clients": 2, "alpha": 0.1,
            "val_frac": 0.2, "train_per_client": 10, "test_per_client": 5,
        },
    )
    artifact = SimpleNamespace(
        partition_id="generated-partition",
        settings=settings,
        manifest={
            "task": "classification",
            "num_clients": 2,
            "input_spec": {"kind": "image", "shape": [4]},
            "target_spec": {"num_classes": 2},
        },
    )
    monkeypatch.setattr(
        "rigfl.experiment.run.load_partition", lambda *args, **kwargs: artifact
    )
    monkeypatch.setattr(
        "rigfl.experiment.run.dataset_settings",
        lambda *args, **kwargs: artifact.settings,
    )

    resolved, loaded = resolve_experiment_data(
        ExperimentConfig(dataset="generated")
    )

    assert loaded.artifact is artifact
    assert resolved.model_architecture_family is None
    assert resolved.model_architectures == [
        "fedavg_cnn", "cifar_resnet18", "cifar_mobilenet_v2"
    ]


def test_registered_architecture_compatibility_is_validated(monkeypatch):
    monkeypatch.setitem(
        MODEL_ARCHITECTURE_REGISTRY, "custom_numeric", ("numeric", nn.Identity)
    )
    with pytest.raises(ValueError, match="do not accept image inputs"):
        resolve_model_architectures(
            architecture_family=None,
            architectures=["custom_numeric"],
            input_kind="image",
        )


def test_numeric_backbone_receives_the_resolved_input_spec(monkeypatch):
    class NumericBackbone(nn.Module):
        def __init__(self, input_spec):
            super().__init__()
            self.input_spec = input_spec
            self.out_dim = 4
            self.linear = nn.Linear(input_spec["shape"][0], self.out_dim)

        def forward(self, x):
            return self.linear(x)

    monkeypatch.setitem(
        MODEL_ARCHITECTURE_REGISTRY,
        "custom_numeric",
        ("numeric", NumericBackbone),
    )
    input_spec = {"input_kind": "numeric", "shape": (10,)}
    make_backbone = instantiate_backbones(
        ["custom_numeric"], input_spec=input_spec
    )[0]

    backbone = make_backbone()

    assert backbone.input_spec == input_spec
    assert backbone(torch.randn(2, 10)).shape == (2, 4)


def test_unresolved_dataset_does_not_assume_image_inputs(monkeypatch):
    class NumericBackbone(nn.Module):
        pass

    monkeypatch.setitem(
        MODEL_ARCHITECTURE_REGISTRY,
        "custom_numeric",
        ("numeric", NumericBackbone),
    )
    exp = ExperimentConfig(model_architectures=["custom_numeric"])

    assert resolve_algorithm_config(
        "feddes", exp, config_class("feddes")()
    ) == config_class("feddes")()

    grid = build_grid({
        "algorithms": ["local"],
        "base": {"experiment": {"model_architectures": ["custom_numeric"]}},
    })
    assert grid[0]["experiment"]["model_architectures"] == ["custom_numeric"]


def test_numeric_partition_runs_through_experiment_infrastructure(
    monkeypatch, tmp_path
):
    class NumericBackbone(nn.Module):
        def __init__(self, input_spec):
            super().__init__()
            self.out_dim = 4
            self.linear = nn.Linear(input_spec["shape"][0], self.out_dim)

        def forward(self, x):
            return torch.relu(self.linear(x))

    monkeypatch.setitem(
        MODEL_ARCHITECTURE_REGISTRY,
        "custom_numeric",
        ("numeric", NumericBackbone),
    )
    for cid in range(2):
        directory = tmp_path / "clients" / f"client_{cid}"
        directory.mkdir(parents=True)
        for split, size in (("train", 6), ("validation", 4), ("test", 4)):
            inputs = torch.randn(size, 10)
            targets = (torch.arange(size) + cid) % 2
            torch.save((inputs, targets), directory / f"{split}.pt")

    settings = FlowerDatasetSettings(
        source_dataset="test/source",
        partition={"scheme": "dirichlet", "num_clients": 2},
    )
    artifact = SimpleNamespace(
        dataset="numeric",
        partition_id="numeric-partition",
        path=tmp_path,
        settings=settings,
        manifest={
            "task": "classification",
            "num_clients": 2,
            "input_spec": {"kind": "numeric", "shape": [10]},
            "target_spec": {"num_classes": 2},
        },
    )
    exp = resolved_experiment(
        dataset="numeric",
        partition_id=artifact.partition_id,
        input_kind="numeric",
        input_spec={"input_kind": "numeric", "shape": [10]},
        num_classes=2,
        model_architectures=["custom_numeric"],
        rounds=1,
        shared_dim=3,
        batch=2,
    )

    record = run_one(
        "local",
        exp,
        config_class("local")(local_epochs=1),
        torch.device("cpu"),
        data=ResolvedData(settings=settings, artifact=artifact),
    )

    assert record["config"]["experiment"]["input_kind"] == "numeric"
    assert record["result"]["evaluation_history"]["evaluation_rounds"] == [0]
    validate_run_record(record)


def test_unknown_architecture_fails_during_algorithm_validation():
    exp = ExperimentConfig(model_architectures=["does_not_exist"])
    with pytest.raises(ValueError, match="Unknown model architecture"):
        resolve_algorithm_config("feddes", exp, config_class("feddes")())


def test_dataset_supplies_architecture_compatibility_context():
    exp = resolved_experiment(
        input_kind="numeric",
        input_spec={"input_kind": "numeric", "shape": [10]},
        model_architectures=["fedavg_cnn"],
    )
    with pytest.raises(ValueError, match="do not accept numeric inputs"):
        resolve_algorithm_config("feddes", exp, config_class("feddes")())


def test_feddes_has_no_separate_model_selection():
    Cfg = config_class("feddes")
    assert "models" not in Cfg.model_fields
    assert "model_family" not in Cfg.model_fields
    assert "local_epochs" not in Cfg.model_fields
    assert "lr" not in Cfg.model_fields
    assert {"base_epochs", "base_lr"} <= set(Cfg.model_fields)

    with pytest.raises(Exception, match="models"):
        Cfg(models=["fedavg_cnn"])
    with pytest.raises(Exception, match="model_family"):
        Cfg(model_family="image_heterogeneous_3")


def test_feddes_relevant_settings_still_change_its_fingerprint():
    exp = resolved_experiment()
    default = resolve_algorithm_config("feddes", exp, config_class("feddes")())
    changed_epochs = resolve_algorithm_config(
        "feddes", exp, config_class("feddes")(base_epochs=101)
    )
    changed_lr = resolve_algorithm_config(
        "feddes", exp, config_class("feddes")(base_lr=0.001)
    )

    original = run_fingerprint(exp, default.model_dump())
    assert original != run_fingerprint(exp, changed_epochs.model_dump())
    assert original != run_fingerprint(exp, changed_lr.model_dump())


def test_local_training_settings_remain_on_every_algorithm_that_uses_them():
    locally_trained = {
        "local", "global", "fedproto", "fedgh", "lgfedavg", "fml",
        "fedkd", "fedtgp", "fedavg", "fedprox",
    }
    for name in locally_trained:
        assert {"local_epochs", "lr"} <= set(config_class(name).model_fields)


def test_every_algorithm_config_inherits_directly_from_the_universal_base():
    from rigfl.core.config import AlgorithmConfig
    from rigfl.experiment.registry import ALL_ALGORITHMS

    for name in ALL_ALGORITHMS:
        assert config_class(name).__bases__ == (AlgorithmConfig,)


def test_feddes_builds_its_pool_from_the_experiment_architectures(monkeypatch):
    class TinyBackbone(nn.Module):
        out_dim = 3

        def __init__(self, input_spec=None):
            super().__init__()
            self.linear = nn.Linear(4, self.out_dim)

        def forward(self, x):
            return torch.relu(self.linear(x))

    monkeypatch.setitem(
        MODEL_ARCHITECTURE_REGISTRY, "custom", ("image", TinyBackbone)
    )
    exp = resolved_experiment(
        num_classes=3, shared_dim=5, model_architectures=["custom"],
        input_spec={"input_kind": "image", "shape": [4]},
    )
    algorithm = build_algorithm(
        "feddes", exp, config_class("feddes")(cache_dir=""),
        model_input_spec={"input_kind": "image", "shape": (4,)})

    assert algorithm.model_ids == ["custom"]
    assert len(algorithm.base_models) == 1
    assert isinstance(algorithm.base_models[0], ClientModel)
    assert isinstance(algorithm.base_models[0].backbone, TinyBackbone)
    assert algorithm.base_models[0](torch.randn(2, 4)).shape == (2, 3)
