"""One experiment-level architecture selection shared by every algorithm."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from rigfl.experiment.identity import fingerprint
from rigfl.experiment.launch import expand
from rigfl.experiment.registry import (
    config_class,
    resolve_algorithm_config,
    resolve_algorithm_models,
    run_identity,
)
from rigfl.models.registry import (
    MODEL_ARCHITECTURE_REGISTRY,
    MODEL_FAMILIES,
    instantiate_backbones,
    resolve_models,
)
from tests.helpers import resolved_experiment


def _fingerprint(name, exp, algorithm_config):
    return fingerprint(run_identity(name, exp, algorithm_config))


def test_model_is_configured_independently_of_dataset_name():
    names = resolve_models("fedavg_cnn", input_kind="image")
    factories = instantiate_backbones(
        names, input_spec={"input_kind": "image", "shape": (1, 28, 28)}
    )
    assert names == ["fedavg_cnn"]
    assert factories[0]() is not factories[0]()
    assert factories[0]()(torch.randn(2, 1, 28, 28)).shape == (2, 512)


@pytest.mark.parametrize("model_arch, kind, x, expected_names, expected_dims", [
    ("mnist_heterogeneous_3", "image", torch.randn(2, 1, 28, 28),
     ["lenet5", "fedavg_mnist_cnn", "small_cnn"], [84, 512, 128]),
    ("tabular_heterogeneous_3", "numeric", torch.randn(2, 12),
     ["tabular_linear", "tabular_mlp", "tabular_residual_mlp"], [64, 64, 64]),
    ("phishing_byte_cnn", "token_sequence", torch.randint(0, 258, (2, 256)),
     ["phishing_byte_cnn"], [128]),
])
def test_architecture_family_constructs_distinct_backbones(
    model_arch, kind, x, expected_names, expected_dims
):
    names = resolve_models(model_arch, input_kind=kind)
    factories = instantiate_backbones(
        names, input_spec={"kind": kind, "shape": tuple(x.shape[1:])}
    )

    assert names == expected_names
    assert [factory()(x).shape for factory in factories] == [
        (2, dim) for dim in expected_dims
    ]


def test_model_arch_sweep_skips_families_for_homogeneous_algorithms():
    grid = expand({
        "sweep": {
            "algorithm": ["fedavg", "fedproto"],
            "model_arch": ["fedavg_cnn", "image_heterogeneous_3"],
        },
    })

    assert [
        (task["algorithm"], task["experiment"]["model_arch"]) for task in grid
    ] == [
        ("fedavg", "fedavg_cnn"),
        ("fedproto", "fedavg_cnn"),
        ("fedproto", "image_heterogeneous_3"),
    ]


@pytest.mark.parametrize("algorithm", ["fml", "pfedmoe", "fedapen"])
def test_aux_model_defaults_to_first_family_member(algorithm):
    field = "aux_model_arch"
    exp = resolved_experiment(model_arch="image_heterogeneous_3")
    default = resolve_algorithm_config(algorithm, exp, config_class(algorithm)())
    explicit = resolve_algorithm_config(
        algorithm, exp, config_class(algorithm)(**{field: "fedavg_cnn"})
    )
    changed = resolve_algorithm_config(
        algorithm, exp, config_class(algorithm)(**{field: "cifar_resnet18"})
    )

    assert getattr(default, field) == "fedavg_cnn"
    default_fingerprint = _fingerprint(
        algorithm, resolve_algorithm_models(algorithm, exp), default.model_dump()
    )
    assert default_fingerprint == _fingerprint(
        algorithm, resolve_algorithm_models(algorithm, exp), explicit.model_dump()
    )
    assert default_fingerprint != _fingerprint(
        algorithm, resolve_algorithm_models(algorithm, exp), changed.model_dump()
    )


def test_run_identity_uses_resolved_family_members(monkeypatch):
    exp = resolved_experiment(model_arch="image_heterogeneous_3")
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
        "fedproto", exp.model_copy(update={"model_arch": "same_members"})
    )
    monkeypatch.setitem(
        MODEL_FAMILIES,
        "reordered",
        list(reversed(MODEL_FAMILIES["image_heterogeneous_3"])),
    )
    reordered = resolve_algorithm_models(
        "fedproto", exp.model_copy(update={"model_arch": "reordered"})
    )

    def fp(exp):
        return _fingerprint("fedproto", exp, {})

    assert fp(original) == fp(unchanged)
    assert fp(original) == fp(alias)
    assert fp(original) != fp(reordered)

