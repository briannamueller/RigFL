"""One registry for the model architectures used throughout an experiment."""

from __future__ import annotations

import torch.nn as nn

from rigfl.models.brats import ConvMixer256D8, MobileNetV2
from rigfl.models.cifar import (
    CifarCNN37_74_148,
    CifarCNN43_86_172,
    CifarCNN49_98_196,
    CifarCNN55_110_220,
    CifarCNN61_122_244,
    CifarCNN67_134_268,
    CifarCNN73_146_292,
    CifarCNN79_158_316,
    CifarCNN85_170_340,
    CifarCNN91_182_364,
    CifarMobileNetV2,
    CifarResNet18,
    CifarThreeLayerCNN,
    FedAvgCNN,
    FedGHCNN1,
    FedGHCNN2,
    FedGHCNN3,
    FedGHCNN4,
    FedGHCNN5,
    SmallCNN,
)
from rigfl.models.mnist import FedAvgMNISTCNN, LeNet5
from rigfl.models.phishing import PhishingByteCNN
from rigfl.models.tabular import (
    TabularLinear,
    TabularLogisticRegression,
    TabularMLP,
    TabularMLP64,
    TabularResidualMLP,
)
from rigfl.models.temporal import (
    TemporalAttention,
    TemporalCNN,
    TemporalGRU,
    TemporalLSTM,
    TemporalTPC,
)

# Short YAML name -> input kind + feature-extractor class.
MODEL_ARCHITECTURE_REGISTRY: dict[str, tuple[str, type[nn.Module]]] = {
    "lenet5": ("image", LeNet5),
    "fedavg_mnist_cnn": ("image", FedAvgMNISTCNN),
    "small_cnn": ("image", SmallCNN),
    "fedavg_cnn": ("image", FedAvgCNN),
    "cifar_resnet18": ("image", CifarResNet18),
    "cifar_mobilenet_v2": ("image", CifarMobileNetV2),
    "cifar_cnn_37_74_148": ("image", CifarCNN37_74_148),
    "cifar_cnn_43_86_172": ("image", CifarCNN43_86_172),
    "cifar_cnn_49_98_196": ("image", CifarCNN49_98_196),
    "cifar_cnn_55_110_220": ("image", CifarCNN55_110_220),
    "cifar_cnn_61_122_244": ("image", CifarCNN61_122_244),
    "cifar_cnn_64_128_256": ("image", CifarThreeLayerCNN),
    "cifar_cnn_67_134_268": ("image", CifarCNN67_134_268),
    "cifar_cnn_73_146_292": ("image", CifarCNN73_146_292),
    "cifar_cnn_79_158_316": ("image", CifarCNN79_158_316),
    "cifar_cnn_85_170_340": ("image", CifarCNN85_170_340),
    "cifar_cnn_91_182_364": ("image", CifarCNN91_182_364),
    "fedgh_cnn_1": ("image", FedGHCNN1),
    "fedgh_cnn_2": ("image", FedGHCNN2),
    "fedgh_cnn_3": ("image", FedGHCNN3),
    "fedgh_cnn_4": ("image", FedGHCNN4),
    "fedgh_cnn_5": ("image", FedGHCNN5),
    "mobilenet_v2": ("image", MobileNetV2),
    "convmixer_256d8": ("image", ConvMixer256D8),
    "tabular_linear": ("numeric", TabularLinear),
    "tabular_logreg": ("numeric", TabularLogisticRegression),
    "tabular_mlp": ("numeric", TabularMLP),
    "tabular_mlp_64": ("numeric", TabularMLP64),
    "tabular_residual_mlp": ("numeric", TabularResidualMLP),
    "phishing_byte_cnn": ("token_sequence", PhishingByteCNN),
    "temporal_gru": ("temporal", TemporalGRU),
    "temporal_cnn": ("temporal", TemporalCNN),
    "temporal_lstm": ("temporal", TemporalLSTM),
    "temporal_tpc_light": ("temporal", TemporalTPC),
    "temporal_attention": ("temporal", TemporalAttention),
}

# A model_arch value is either one architecture name or one family name.
MODEL_FAMILIES = {
    "mnist_heterogeneous_3": [
        "lenet5", "fedavg_mnist_cnn", "small_cnn"
    ],
    "image_heterogeneous_3": [
        "fedavg_cnn", "cifar_resnet18", "cifar_mobilenet_v2"
    ],
    "cifar_width_heterogeneous_10": [
        "cifar_cnn_37_74_148",
        "cifar_cnn_43_86_172",
        "cifar_cnn_49_98_196",
        "cifar_cnn_55_110_220",
        "cifar_cnn_61_122_244",
        "cifar_cnn_67_134_268",
        "cifar_cnn_73_146_292",
        "cifar_cnn_79_158_316",
        "cifar_cnn_85_170_340",
        "cifar_cnn_91_182_364",
    ],
    "fedgh_heterogeneous_5": [
        "fedgh_cnn_1",
        "fedgh_cnn_2",
        "fedgh_cnn_3",
        "fedgh_cnn_4",
        "fedgh_cnn_5",
    ],
    "brats_heterogeneous_2": [
        "convmixer_256d8", "mobilenet_v2"
    ],
    "tabular_heterogeneous_3": [
        "tabular_linear", "tabular_mlp", "tabular_residual_mlp"
    ],
    "scrnaseq_heterogeneous_2": [
        "tabular_logreg", "tabular_mlp_64"
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

