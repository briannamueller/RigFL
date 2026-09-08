"""Registered conversions from source datasets to model-ready tensors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from rigfl.data.transforms.phishing import (
    MAX_LENGTH,
    PADDING_INDEX,
    REQUIRED_COLUMNS as PHISHING_REQUIRED_COLUMNS,
    SOURCE_URL as PHISHING_SOURCE_URL,
    VOCAB_SIZE,
    encode_urls,
)
from rigfl.data.transforms.paysim import REQUIRED_COLUMNS, SOURCE_URL, prepare_paysim


@dataclass(frozen=True)
class DataTransform:
    version: int
    input_kind: Literal["auto", "image", "numeric", "token_sequence"]
    input_column: str | None = None
    target_column: str | None = None
    task: Literal["classification", "regression"] | None = None
    class_names: tuple[str, ...] | None = None
    required_columns: tuple[str, ...] = ()
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None
    image_mode: Literal["rgb", "grayscale"] | None = None
    prepare: Callable | None = None
    convert: Callable | None = None
    source: str | None = None
    sequence_length: int | None = None
    vocab_size: int | None = None
    padding_index: int | None = None

    def identity(self) -> dict:
        parameters = {
            "input_kind": self.input_kind,
            "input_column": self.input_column,
            "target_column": self.target_column,
            "task": self.task,
            "class_names": list(self.class_names) if self.class_names else None,
            "required_columns": list(self.required_columns),
            "mean": list(self.mean) if self.mean else None,
            "std": list(self.std) if self.std else None,
            "image_mode": self.image_mode,
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "padding_index": self.padding_index,
        }
        return {
            "version": self.version,
            "source": self.source,
            "parameters": parameters,
        }


def _image(
    input_column: str,
    target_column: str,
    *,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
    image_mode: Literal["rgb", "grayscale"] | None = None,
) -> DataTransform:
    return DataTransform(
        version=1,
        input_kind="image",
        input_column=input_column,
        target_column=target_column,
        task="classification",
        required_columns=(input_column, target_column),
        mean=mean,
        std=std,
        image_mode=image_mode,
    )


DATA_TRANSFORMS = {
    "auto": DataTransform(version=1, input_kind="auto"),
    "image": DataTransform(version=1, input_kind="image"),
    "numeric": DataTransform(version=1, input_kind="numeric"),
    "mnist": _image("image", "label", mean=(0.1307,), std=(0.3081,)),
    "fashion_mnist": _image("image", "label", mean=(0.2860,), std=(0.3530,)),
    "cifar10": _image(
        "img",
        "label",
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
    "cifar100": _image(
        "img",
        "fine_label",
        mean=(0.5071, 0.4865, 0.4409),
        std=(0.2673, 0.2564, 0.2762),
    ),
    "tiny_imagenet": _image(
        "image",
        "label",
        mean=(0.4802, 0.4481, 0.3975),
        std=(0.2764, 0.2689, 0.2816),
        image_mode="rgb",
    ),
    "femnist": _image("image", "character", image_mode="grayscale"),
    "paysim_fraud": DataTransform(
        version=1,
        input_kind="numeric",
        input_column="features",
        target_column="isFraud",
        task="classification",
        class_names=("not_fraud", "fraud"),
        required_columns=REQUIRED_COLUMNS,
        prepare=prepare_paysim,
        source=SOURCE_URL,
    ),
    "phishing_urls": DataTransform(
        version=1,
        input_kind="token_sequence",
        input_column="url",
        target_column="label",
        task="classification",
        class_names=("benign", "phishing"),
        required_columns=PHISHING_REQUIRED_COLUMNS,
        convert=encode_urls,
        source=PHISHING_SOURCE_URL,
        sequence_length=MAX_LENGTH,
        vocab_size=VOCAB_SIZE,
        padding_index=PADDING_INDEX,
    ),
}


def get_data_transform(name: str) -> DataTransform:
    try:
        return DATA_TRANSFORMS[name]
    except KeyError as exc:
        known = ", ".join(sorted(DATA_TRANSFORMS))
        raise ValueError(f"unknown data_transform {name!r}; known: {known}") from exc


def data_transform_identity(name: str) -> dict:
    return {"name": name, **get_data_transform(name).identity()}


__all__ = [
    "DATA_TRANSFORMS",
    "DataTransform",
    "data_transform_identity",
    "get_data_transform",
]
