"""One registry for the model architectures used throughout an experiment."""

from __future__ import annotations

import torch.nn as nn

from rigfl.core import assemble_model, assemble_native_model
from rigfl.models.cifar import (
    CifarMobileNetV2,
    CifarResNet18,
    FedAvgCNN,
    SmallCNN,
)
from rigfl.models.mnist import FedAvgMNISTCNN, LeNet5
from rigfl.models.phishing import PhishingByteCNN
from rigfl.models.tabular import TabularLinear, TabularMLP, TabularResidualMLP
from rigfl.models.temporal import TemporalCNN, TemporalGRU, TemporalLSTM


# Short YAML name -> input kind + feature-extractor class. The resolved names
# select both federated client models and FedDES's native classifier pool.
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
    "phishing_byte_cnn": ("token_sequence", PhishingByteCNN),
    "temporal_gru": ("temporal", TemporalGRU),
    "temporal_cnn": ("temporal", TemporalCNN),
    "temporal_lstm": ("temporal", TemporalLSTM),
}

MODEL_FAMILIES = {
    "mnist_heterogeneous_3": [
        "lenet5", "fedavg_mnist_cnn", "small_cnn"
    ],
    "image_heterogeneous_3": [
        "fedavg_cnn", "cifar_resnet18", "cifar_mobilenet_v2"
    ],
    "tabular_heterogeneous_3": [
        "tabular_linear", "tabular_mlp", "tabular_residual_mlp"
    ],
    "phishing_byte_cnn": ["phishing_byte_cnn"],
    "temporal_heterogeneous_3": [
        "temporal_gru", "temporal_cnn", "temporal_lstm"
    ],
}


def validate_model(name: str, input_kind: str | None) -> str:
    """Validate one registered model against the dataset input type."""
    if name not in MODEL_ARCHITECTURE_REGISTRY:
        known = ", ".join(sorted(MODEL_ARCHITECTURE_REGISTRY))
        raise ValueError(f"Unknown model {name!r}; known: {known}.")
    kind = MODEL_ARCHITECTURE_REGISTRY[name][0]
    if input_kind is not None and kind != input_kind:
        raise ValueError(f"Model {name!r} does not accept {input_kind} inputs.")
    return name


def resolve_models(*, model: str, model_family: str | None,
                   input_kind: str | None, use_family: bool = True) -> list[str]:
    """Resolve the model selection used by one algorithm."""
    validate_model(model, input_kind)
    if model_family is None:
        return [model]
    if model_family not in MODEL_FAMILIES:
        known = ", ".join(sorted(MODEL_FAMILIES))
        raise ValueError(f"Unknown model_family {model_family!r}; known: {known}.")
    names = list(MODEL_FAMILIES[model_family])
    if not names:
        raise ValueError(f"Model family {model_family!r} is empty.")
    unknown = [name for name in names if name not in MODEL_ARCHITECTURE_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown model architecture(s): {', '.join(unknown)}; known: "
            f"{', '.join(sorted(MODEL_ARCHITECTURE_REGISTRY))}."
        )
    if not use_family:
        return [model]
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


def instantiate_native_models(names: list[str], *, input_spec: dict,
                              num_classes: int) -> list[nn.Module]:
    """Construct classifiers without a representation-alignment adapter."""
    return [
        assemble_native_model(make_backbone(), num_classes=num_classes)
        for make_backbone in instantiate_backbones(names, input_spec=input_spec)
    ]
