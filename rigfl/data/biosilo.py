"""Generate and load BioSilo partitions for RigFL experiments."""

from __future__ import annotations

import shlex

from filelock import FileLock


def _biosilo():
    try:
        import biosilo
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The BioSilo data backend requires the optional 'biosilo' package. "
            "Install it with: "
            "pip install 'rigfl[biosilo] @ git+https://github.com/briannamueller/RigFL.git'"
        ) from exc
    return biosilo


def _root(settings, data_dir):
    return settings.data_root or data_dir


def generate_biosilo_partition(settings, *, data_dir="data"):
    """Generate or reuse the configured BioSilo partition."""
    biosilo = _biosilo()
    expected = biosilo.expected_partition(
        settings.source_dataset,
        root=_root(settings, data_dir),
        **settings.parameters,
    )
    expected.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(f"{expected}.lock"):
        existed = (expected / "manifest.json").is_file()
        path = biosilo.generate(
            settings.source_dataset,
            root=_root(settings, data_dir),
            **settings.parameters,
        )
    handle = biosilo.load(
        settings.source_dataset,
        root=_root(settings, data_dir),
        partition=path.name,
    )
    return handle, not existed


def load_biosilo_partition(
    settings,
    *,
    data_dir="data",
    dataset_name=None,
    dataset_config=None,
    partition_seed_override=None,
):
    """Load the BioSilo partition determined by the configured parameters."""
    biosilo = _biosilo()
    path = biosilo.expected_partition(
        settings.source_dataset,
        root=_root(settings, data_dir),
        **settings.parameters,
    )
    if not (path / "manifest.json").is_file():
        configured_name = dataset_name or settings.source_dataset
        command = [
            "rigfl",
            "data",
            "generate",
            "--dataset",
            configured_name,
        ]
        if dataset_config is not None:
            command.extend(["--config", str(dataset_config)])
        command.extend(["--data-dir", str(data_dir)])
        if partition_seed_override is not None:
            command.extend(["--partition-seed", str(partition_seed_override)])
        formatted_command = " ".join(shlex.quote(part) for part in command)
        raise FileNotFoundError(
            f"BioSilo partition for {settings.source_dataset!r} was not found at "
            f"{path}. Run: {formatted_command}"
        )
    return biosilo.load(
        settings.source_dataset,
        root=_root(settings, data_dir),
        partition=path.name,
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
    feature_groups = getattr(handle, "feature_groups", {})
    if feature_groups:
        spec["feature_groups"] = feature_groups
    return kind, spec

