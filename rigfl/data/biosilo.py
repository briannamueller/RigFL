"""Load generated BioSilo partitions for RigFL experiments."""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

from rigfl.data.builder import build_clients


def load_biosilo_partition(settings):
    """Load the exact BioSilo partition named by a dataset configuration."""
    try:
        import biosilo
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The BioSilo data backend requires the optional 'biosilo' package."
        ) from exc
    return biosilo.load(
        settings.source_dataset,
        root=settings.data_root,
        partition=settings.partition,
    )


def biosilo_input_spec(handle) -> tuple[str, dict]:
    """Translate BioSilo's input description to RigFL's model description."""
    fields = [
        {
            "name": field["name"],
            "shape": list(field["shape"]),
            "dtype": field["dtype"],
        }
        for field in handle.inputs
    ]
    shapes = [tuple(field["shape"]) for field in fields]

    if len(shapes) == 1 and len(shapes[0]) == 1:
        kind = "numeric"
    elif len(shapes) == 1 and len(shapes[0]) == 3:
        kind = "image"
    elif len(shapes) == 2 and len(shapes[0]) == 2 and len(shapes[1]) == 1:
        kind = "temporal"
    else:
        raise ValueError(
            "RigFL cannot infer a supported model input from BioSilo fields "
            f"{fields}. Supported forms are one numeric vector, one "
            "channels-by-height-by-width image, or a time-series and static "
            "feature pair."
        )

    spec = {"input_kind": kind, "fields": fields}
    if len(fields) == 1:
        spec["shape"] = fields[0]["shape"]
    return kind, spec


def build_biosilo_clients(
    handle,
    *,
    shared_dim: int,
    batch: int,
    seed: int,
    backbones: list[Callable[[], nn.Module]],
    validation_fraction: float,
    adapter=None,
    build_models: bool = True,
):
    """Build RigFL clients without copying a generated BioSilo partition."""
    return build_clients(
        handle.client,
        handle.num_clients,
        handle.num_classes,
        backbones,
        shared_dim,
        val_frac=validation_fraction,
        batch=batch,
        adapter=adapter,
        seed=seed,
        build_models=build_models,
    )
