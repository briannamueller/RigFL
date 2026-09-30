"""One registry for the model architectures used throughout an experiment."""

from __future__ import annotations

import torch.nn as nn

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

# Short YAML name -> input kind + feature-extractor class.
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

# A model_arch value is either one architecture name or one family name.
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
    "temporal_heterogeneous_3": [
        "temporal_gru", "temporal_cnn", "temporal_lstm"
    ],
}


def is_model_family(model_arch: str) -> bool:
    """Whether a model_arch value names a model family."""
    return model_arch in MODEL_FAMILIES


def validate_model(name: str, input_kind: str | None) -> str:
    """Validate one registered model against the dataset input type."""
    if name not in MODEL_ARCHITECTURE_REGISTRY:
        known = ", ".join(sorted(MODEL_ARCHITECTURE_REGISTRY))
        raise ValueError(f"Unknown model architecture {name!r}; known: {known}.")
    kind = MODEL_ARCHITECTURE_REGISTRY[name][0]
    if input_kind is not None and kind != input_kind:
        raise ValueError(f"Model {name!r} does not accept {input_kind} inputs.")
    return name


def resolve_models(model_arch: str, *, input_kind: str | None) -> list[str]:
    """Resolve a model_arch value to the architectures assigned to clients."""
    if model_arch in MODEL_FAMILIES:
        names = list(MODEL_FAMILIES[model_arch])
        if not names:
            raise ValueError(f"Model family {model_arch!r} is empty.")
    elif model_arch in MODEL_ARCHITECTURE_REGISTRY:
        names = [model_arch]
    else:
        raise ValueError(
            f"Unknown model_arch {model_arch!r}; known architectures: "
            f"{', '.join(sorted(MODEL_ARCHITECTURE_REGISTRY))}; known families: "
            f"{', '.join(sorted(MODEL_FAMILIES))}."
        )
    for name in names:
        validate_model(name, input_kind)
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

