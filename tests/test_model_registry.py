"""One experiment-level architecture selection shared by every algorithm."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from rigfl.core import ClientModel, Identity
from rigfl.data.config import FlowerDatasetSettings
from rigfl.experiment.artifacts import validate_run_record
from rigfl.experiment.config import ExperimentConfig, run_fingerprint
from rigfl.experiment.launch import build_grid
from rigfl.experiment.registry import (algorithm_run_fingerprint, build_algorithm,
                                       config_class, resolve_algorithm_config,
                                       resolve_algorithm_models)
from rigfl.experiment.run import ResolvedData, resolve_experiment_data, run_one
from rigfl.models.registry import (
    MODEL_FAMILIES,
    MODEL_ARCHITECTURE_REGISTRY,
    instantiate_backbones,
    resolve_models,
)
from tests.helpers import resolved_experiment


def test_model_is_configured_independently_of_dataset_name():
    names = resolve_models(
        model="fedavg_cnn", model_family=None, input_kind="image"
    )
    factories = instantiate_backbones(
        names, input_spec={"input_kind": "image", "shape": (1, 28, 28)}
    )
    assert names == ["fedavg_cnn"]
    assert factories[0]() is not factories[0]()
    assert factories[0]()(torch.randn(2, 1, 28, 28)).shape == (2, 512)


def test_mnist_architecture_family_constructs_distinct_backbones():
    names = resolve_models(
        model="lenet5",
        model_family="mnist_heterogeneous_3",
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
    names = resolve_models(
        model="tabular_mlp",
        model_family="tabular_heterogeneous_3",
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


def test_omitting_a_family_uses_the_model():
    assert resolve_models(
        model="tabular_mlp",
        model_family=None,
        input_kind="numeric",
    ) == ["tabular_mlp"]


def test_token_sequence_model_constructs():
    names = resolve_models(
        model="phishing_byte_cnn",
        model_family=None,
        input_kind="token_sequence",
    )
    factory = instantiate_backbones(
        names,
        input_spec={"input_kind": "token_sequence", "shape": (256,)},
    )[0]

    assert names == ["phishing_byte_cnn"]
    assert factory()(torch.randint(0, 258, (2, 256))).shape == (2, 128)


def test_model_families_only_contain_registered_architectures():
    assert all(
        name in MODEL_ARCHITECTURE_REGISTRY
        for family in MODEL_FAMILIES.values()
        for name in family
    )


def test_family_order_is_preserved():
    names = resolve_models(
        model="fedavg_cnn",
        model_family="image_heterogeneous_3",
        input_kind="image",
    )
    assert names == [
        "fedavg_cnn", "cifar_resnet18", "cifar_mobilenet_v2"
    ]


def test_mixed_sweep_resolves_models_by_algorithm_capability():
    exp = resolved_experiment(
        model="fedavg_cnn", model_family="image_heterogeneous_3"
    )

    assert resolve_algorithm_models("fedavg", exp).resolved_models == [
        "fedavg_cnn"
    ]
    expected_family = [
        "fedavg_cnn", "cifar_resnet18", "cifar_mobilenet_v2"
    ]
    assert resolve_algorithm_models("fedproto", exp).resolved_models == expected_family
    assert resolve_algorithm_models("feddes", exp).resolved_models == expected_family

    grid = build_grid({
        "algorithms": ["fedavg", "fedproto", "feddes"],
        "base": {"experiment": {
            "model": "fedavg_cnn",
            "model_family": "image_heterogeneous_3",
        }},
    })
    assert [task["algorithm"] for task in grid] == [
        "fedavg", "fedproto", "feddes"
    ]


def test_model_family_sweep_does_not_duplicate_homogeneous_algorithms(monkeypatch):
    monkeypatch.setitem(
        MODEL_FAMILIES, "image_pair", ["fedavg_cnn", "cifar_resnet18"]
    )
    grid = build_grid({
        "algorithms": ["fedavg", "fedproto"],
        "base": {"experiment": {"model": "fedavg_cnn"}},
        "sweep": {"model_family": [
            "image_pair", "image_heterogeneous_3"
        ]},
    })

    assert sum(task["algorithm"] == "fedavg" for task in grid) == 1
    assert sum(task["algorithm"] == "fedproto" for task in grid) == 2


@pytest.mark.parametrize("algorithm", ["fml", "fedkd"])
def test_aux_model_defaults_to_first_family_member(algorithm):
    exp = resolved_experiment(
        model="cifar_resnet18", model_family="image_heterogeneous_3"
    )
    default = resolve_algorithm_config(algorithm, exp, config_class(algorithm)())
    explicit = resolve_algorithm_config(
        algorithm, exp, config_class(algorithm)(aux_model="fedavg_cnn")
    )
    changed = resolve_algorithm_config(
        algorithm, exp, config_class(algorithm)(aux_model="cifar_resnet18")
    )

    assert default.aux_model == "fedavg_cnn"
    default_fingerprint = algorithm_run_fingerprint(
        algorithm, resolve_algorithm_models(algorithm, exp), default.model_dump()
    )
    assert default_fingerprint == algorithm_run_fingerprint(
        algorithm, resolve_algorithm_models(algorithm, exp), explicit.model_dump()
    )
    assert default_fingerprint != algorithm_run_fingerprint(
        algorithm, resolve_algorithm_models(algorithm, exp), changed.model_dump()
    )


@pytest.mark.parametrize("algorithm", ["fml", "fedkd"])
def test_aux_model_defaults_to_model_without_a_family(algorithm):
    exp = resolved_experiment(model="cifar_resnet18", model_family=None)
    cfg = resolve_algorithm_config(algorithm, exp, config_class(algorithm)())

    assert cfg.aux_model == "cifar_resnet18"


def test_run_identity_uses_resolved_family_members(monkeypatch):
    exp = resolved_experiment(
        model="fedavg_cnn", model_family="image_heterogeneous_3"
    )
    original = resolve_algorithm_models("fedproto", exp)
    monkeypatch.setitem(
        MODEL_ARCHITECTURE_REGISTRY, "unrelated", ("image", nn.Identity)
    )
    unchanged = resolve_algorithm_models("fedproto", exp)
    monkeypatch.setitem(
        MODEL_FAMILIES,
        "same_members",
        list(MODEL_FAMILIES["image_heterogeneous_3"]),
    )
    alias = resolve_algorithm_models(
        "fedproto", exp.model_copy(update={"model_family": "same_members"})
    )
    monkeypatch.setitem(
        MODEL_FAMILIES,
        "reordered",
        list(reversed(MODEL_FAMILIES["image_heterogeneous_3"])),
    )
    reordered = resolve_algorithm_models(
        "fedproto", exp.model_copy(update={"model_family": "reordered"})
    )

    assert run_fingerprint(original, {}) == run_fingerprint(unchanged, {})
    assert run_fingerprint(original, {}) == run_fingerprint(alias, {})
    assert run_fingerprint(original, {}) != run_fingerprint(reordered, {})


def test_requested_models_are_recorded_explicitly(monkeypatch):
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
        ExperimentConfig(
            dataset="generated",
            model="fedavg_cnn",
            model_family="image_heterogeneous_3",
        )
    )

    assert loaded.artifact is artifact
    assert resolved.model == "fedavg_cnn"
    assert resolved.model_family == "image_heterogeneous_3"
    assert resolved.resolved_models == [
        "fedavg_cnn", "cifar_resnet18", "cifar_mobilenet_v2"
    ]


def test_registered_architecture_compatibility_is_validated(monkeypatch):
    monkeypatch.setitem(
        MODEL_ARCHITECTURE_REGISTRY, "custom_numeric", ("numeric", nn.Identity)
    )
    with pytest.raises(ValueError, match="does not accept image inputs"):
        resolve_models(
            model="custom_numeric",
            model_family=None,
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
    exp = ExperimentConfig(model="custom_numeric")

    assert resolve_algorithm_config(
        "feddes", exp, config_class("feddes")()
    ) == config_class("feddes")()

    grid = build_grid({
        "algorithms": ["local"],
        "base": {"experiment": {"model": "custom_numeric"}},
    })
    assert grid[0]["experiment"]["model"] == "custom_numeric"


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
        model="custom_numeric",
        resolved_models=["custom_numeric"],
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
    exp = ExperimentConfig(model="does_not_exist")
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_algorithm_config("feddes", exp, config_class("feddes")())


def test_dataset_supplies_architecture_compatibility_context():
    exp = resolved_experiment(
        input_kind="numeric",
        input_spec={"input_kind": "numeric", "shape": [10]},
        model="fedavg_cnn",
        resolved_models=["fedavg_cnn"],
    )
    with pytest.raises(ValueError, match="does not accept numeric inputs"):
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


def test_feddes_shared_dimension_does_not_change_its_fingerprint():
    exp = resolved_experiment(shared_dim=64)
    other = exp.model_copy(update={"shared_dim": 1024})
    cfg = config_class("feddes")().model_dump()

    assert algorithm_run_fingerprint("feddes", exp, cfg) == algorithm_run_fingerprint(
        "feddes", other, cfg
    )
    assert algorithm_run_fingerprint(
        "local", exp, config_class("local")().model_dump()
    ) != algorithm_run_fingerprint(
        "local", other, config_class("local")().model_dump()
    )


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
        num_classes=3, shared_dim=5, model="custom",
        resolved_models=["custom"],
        input_spec={"input_kind": "image", "shape": [4]},
    )
    algorithm = build_algorithm(
        "feddes", exp, config_class("feddes")(cache_dir=""),
        model_input_spec={"input_kind": "image", "shape": (4,)})

    assert algorithm.model_ids == ["custom"]
    assert len(algorithm.base_models) == 1
    assert isinstance(algorithm.base_models[0], ClientModel)
    assert isinstance(algorithm.base_models[0].backbone, TinyBackbone)
    assert isinstance(algorithm.base_models[0].adapter, Identity)
    assert algorithm.base_models[0].head.in_features == TinyBackbone.out_dim
    assert algorithm.base_models[0](torch.randn(2, 4)).shape == (2, 3)
