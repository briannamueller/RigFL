"""One experiment-level architecture selection shared by every algorithm."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from rigfl.core import ClientModel
from rigfl.experiment.config import ExperimentConfig, run_fingerprint
from rigfl.experiment.launch import build_grid
from rigfl.experiment.registry import (build_algorithm, config_class,
                                       resolve_algorithm_config)
from rigfl.experiment.run import (resolve_experiment_architectures,
                                  resolve_experiment_data)
from rigfl.models.registry import (
    MODEL_ARCHITECTURE_REGISTRY,
    instantiate_backbones,
    resolve_model_architectures,
)


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
        ExperimentConfig(model_architecture_family="image_heterogeneous_3"),
        input_kind="image",
    )
    explicit = resolve_experiment_architectures(
        ExperimentConfig(model_architectures=[
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
    artifact = SimpleNamespace(
        partition_id="generated-partition",
        settings=SimpleNamespace(partition=SimpleNamespace(
            alpha=0.1, val_frac=0.2,
            train_per_client=10, test_per_client=5,
        )),
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

    resolved, loaded = resolve_experiment_data(
        ExperimentConfig(dataset="generated", scheme="generated")
    )

    assert loaded is artifact
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


def test_unknown_architecture_fails_during_algorithm_validation():
    exp = ExperimentConfig(model_architectures=["does_not_exist"])
    with pytest.raises(ValueError, match="Unknown model architecture"):
        resolve_algorithm_config("feddes", exp, config_class("feddes")())


def test_dataset_supplies_architecture_compatibility_context():
    exp = ExperimentConfig(
        scheme="natural", dataset="eICU", partition="p",
        model_architectures=["fedavg_cnn"],
    )
    with pytest.raises(ValueError, match="do not accept temporal inputs"):
        resolve_algorithm_config("feddes", exp, config_class("feddes")())


def test_incompatible_architecture_fails_before_submission():
    with pytest.raises(SystemExit, match="do not accept temporal inputs"):
        build_grid({
            "algorithms": ["feddes"],
            "base": {
                "experiment": {
                    "scheme": "natural", "dataset": "eICU", "partition": "p",
                    "model_architectures": ["fedavg_cnn"],
                },
            },
        })


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
    exp = ExperimentConfig()
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
    exp = ExperimentConfig(
        num_classes=3, shared_dim=5, model_architectures=["custom"])
    algorithm = build_algorithm(
        "feddes", exp, config_class("feddes")(cache_dir=""),
        model_input_spec={"input_kind": "image", "shape": (4,)})

    assert algorithm.model_ids == ["custom"]
    assert len(algorithm.base_models) == 1
    assert isinstance(algorithm.base_models[0], ClientModel)
    assert isinstance(algorithm.base_models[0].backbone, TinyBackbone)
    assert algorithm.base_models[0](torch.randn(2, 4)).shape == (2, 3)
