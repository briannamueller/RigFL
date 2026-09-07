"""One registry for the model architectures used throughout an experiment."""

from __future__ import annotations

import torch.nn as nn

from rigfl.core import assemble_model
from rigfl.models.cifar import (
    CifarMobileNetV2,
    CifarResNet18,
    FedAvgCNN,
    SmallCNN,
)
from rigfl.models.mnist import FedAvgMNISTCNN, LeNet5
from rigfl.models.tabular import TabularLinear, TabularMLP, TabularResidualMLP


# Short YAML name -> input kind + feature-extractor class. The same resolved
# names construct the federated client models and FedDES's per-client pool.
MODEL_ARCHITECTURE_REGISTRY: dict[str, tuple[str, type[nn.Module]]] = {
    "lenet5": ("image", LeNet5),
    "fedavg_mnist_cnn": ("image", FedAvgMNISTCNN),
    "small_cnn": ("image", SmallCNN),
    "fedavg_cnn": ("image", FedAvgCNN),
    "cifar_resnet18": ("image", CifarResNet18),
    "cifar_mobilenet_v2": ("image", CifarMobileNetV2),
    "tabular_linear": ("numeric", TabularLinear),
    "tabular_mlp": ("numeric", TabularMLP),
    "tabular_residual_mlp": ("numeric", TabularResidualMLP),
}

MODEL_ARCHITECTURE_FAMILIES = {
    "mnist_heterogeneous_3": [
        "lenet5", "fedavg_mnist_cnn", "small_cnn"
    ],
    "image_heterogeneous_3": [
        "fedavg_cnn", "cifar_resnet18", "cifar_mobilenet_v2"
    ],
    "tabular_heterogeneous_3": [
        "tabular_linear", "tabular_mlp", "tabular_residual_mlp"
    ],
}


DEFAULT_ARCHITECTURE_FAMILY = {
    "image": "image_heterogeneous_3",
    "numeric": "tabular_heterogeneous_3",
}


def resolve_model_architectures(*, architecture_family: str | None,
                                architectures: list[str] | None,
                                input_kind: str | None) -> list[str]:
    """Resolve one family or explicit architecture list and validate it."""
    if architecture_family is not None and architectures is not None:
        raise ValueError(
            "Set model_architecture_family or model_architectures, not both.")
    if architecture_family is not None:
        if architecture_family not in MODEL_ARCHITECTURE_FAMILIES:
            known = ", ".join(sorted(MODEL_ARCHITECTURE_FAMILIES))
            raise ValueError(
                f"Unknown model_architecture_family {architecture_family!r}; "
                f"known: {known}.")
        names = list(MODEL_ARCHITECTURE_FAMILIES[architecture_family])
    elif architectures is not None:
        names = list(architectures)
    else:
        try:
            family = DEFAULT_ARCHITECTURE_FAMILY[input_kind]
            names = list(MODEL_ARCHITECTURE_FAMILIES[family])
        except KeyError as exc:
            raise ValueError(
                f"No default model architectures support {input_kind!r} inputs; "
                "set experiment.model_architectures"
            ) from exc
    if not names:
        raise ValueError(
            "model_architectures must contain at least one registered name.")
    unknown = [name for name in names if name not in MODEL_ARCHITECTURE_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown model architecture(s): {', '.join(unknown)}; known: "
            f"{', '.join(sorted(MODEL_ARCHITECTURE_REGISTRY))}."
        )
    incompatible = [] if input_kind is None else [
        name for name in names
        if MODEL_ARCHITECTURE_REGISTRY[name][0] != input_kind
    ]
    if incompatible:
        raise ValueError(
            f"Model architecture(s) {', '.join(incompatible)} do not accept "
            f"{input_kind} inputs."
        )
    return names


def instantiate_backbones(names: list[str], *, input_spec: dict):
    """Return factories so every client receives a fresh backbone instance."""
    factories = []
    for name in names:
        _, backbone_class = MODEL_ARCHITECTURE_REGISTRY[name]
        factories.append(
            lambda cls=backbone_class: cls(input_spec=input_spec)
        )
    return factories


def instantiate_models(names: list[str], *, input_spec: dict,
                       shared_dim: int, num_classes: int, adapter) -> list[nn.Module]:
    """Construct complete templates from the same recipe used by the clients."""
    models = []
    for make_backbone in instantiate_backbones(names, input_spec=input_spec):
        backbone = make_backbone()
        models.append(assemble_model(
            backbone, shared_dim=shared_dim, num_classes=num_classes,
            adapter=adapter))
    return models
