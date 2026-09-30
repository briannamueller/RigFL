"""Named filters and CSV output for public result reporting."""

from __future__ import annotations

import csv
import io
import math
import re
from pathlib import Path

import yaml

from rigfl.experiment.artifacts import atomic_write_text
from rigfl.experiment.config import ResolvedExperimentConfig
from rigfl.experiment.identity import ENVIRONMENT_ONLY_EXPERIMENT_FIELDS
from rigfl.experiment.paths import flatten_mapping, model_paths, nested_get

_MISSING = object()
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class ReportingConfigError(ValueError):
    """A reporting configuration cannot be applied unambiguously."""


def load_reporting_config(path: str | Path) -> dict:
    """Load and validate the reporting configuration."""
    source = Path(path)
    try:
        loaded = yaml.safe_load(source.read_text()) or {}
    except (OSError, yaml.YAMLError) as error:
        raise ReportingConfigError(
            f"cannot read reporting configuration {source}: {error}"
        ) from error
    if not isinstance(loaded, dict):
        raise ReportingConfigError("reporting configuration must be a mapping")
    unknown = sorted(
        set(loaded)
        - {"version", "metric", "round_selection", "client_weighting", "filters"}
    )
    if unknown:
        raise ReportingConfigError(
            "unknown top-level reporting setting(s): " + ", ".join(unknown)
        )
    if loaded.get("version") != 1:
        raise ReportingConfigError("reporting configuration version must be 1")
    filters = loaded.get("filters", {})
    if not isinstance(filters, dict):
        raise ReportingConfigError("filters must be a mapping")
    loaded["filters"] = filters
    _validate_names(loaded["filters"], "filter")
    return loaded


def _validate_names(values: dict, kind: str) -> None:
    invalid = [str(name) for name in values if not _NAME.fullmatch(str(name))]
    if invalid:
        raise ReportingConfigError(
            f"{kind} names may contain letters, numbers, underscores, and hyphens; "
            f"invalid: {', '.join(invalid)}"
        )


def record_value(record: dict, path: str, default=_MISSING):
    """Resolve one flat public setting against the internal run record."""
    if path == "algorithm":
        return record.get("algorithm", default)
    experiment_value = nested_get(record, f"config.experiment.{path}", _MISSING)
    algorithm_value = nested_get(record, f"config.algorithm.{path}", _MISSING)
    if experiment_value is not _MISSING and algorithm_value is not _MISSING:
        raise ReportingConfigError(
            f"reporting setting {path!r} is ambiguous between the shared "
            "experiment protocol and the algorithm configuration"
        )
    if experiment_value is not _MISSING:
        return experiment_value
    if algorithm_value is not _MISSING:
        return algorithm_value
    return default


def apply_named_filter(records: list[dict], config: dict, name: str) -> tuple[list[dict], dict]:
    """Apply OR within a field and AND across fields."""
    filters = config.get("filters", {})
    if name not in filters:
        raise ReportingConfigError(
            f"unknown filter {name!r}; available filters: "
            + (", ".join(sorted(filters)) or "none")
        )
    rule = filters[name]
    if not isinstance(rule, dict) or not rule:
        raise ReportingConfigError(f"filter {name!r} must be a nonempty mapping")
    normalized = {}
    for path, values in rule.items():
        if not isinstance(path, str):
            raise ReportingConfigError(f"filter {name!r} contains a non-string path")
        allowed = values if isinstance(values, list) else [values]
        if not allowed:
            raise ReportingConfigError(f"filter {name!r} path {path!r} has no values")
        if not any(record_value(record, path, _MISSING) is not _MISSING for record in records):
            raise ReportingConfigError(
                f"filter {name!r} references unknown path {path!r}"
            )
        normalized[path] = allowed
    matches = [
        record
        for record in records
        if all(
            _null_experiment_setting(record, path)
            or record_value(record, path, _MISSING) in allowed
            for path, allowed in normalized.items()
        )
    ]
    if not matches:
        raise ReportingConfigError(f"filter {name!r} matched no completed results")
    return matches, normalized


def _null_experiment_setting(record: dict, path: str) -> bool:
    """Whether the run recorded this experiment setting, or a parent of it, as null.

    Null marks a setting the algorithm ignores, so it matches any filter value.
    """
    value = record.get("config", {}).get("experiment", {})
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
        if value is None:
            return True
    return False


def reporting_settings(config: dict) -> dict:
    """Return the metric, round selection, and client weighting used by ``rigfl report``."""
    resolved = {
        "metric": config.get("metric", "accuracy"),
        "round_selection": config.get("round_selection", "shared"),
        "client_weighting": config.get("client_weighting", "uniform"),
    }
    allowed = {
        "round_selection": {"shared", "client-specific"},
        "client_weighting": {"uniform", "sample_count"},
    }
    for field, choices in allowed.items():
        if resolved[field] not in choices:
            raise ReportingConfigError(
                f"{field}={resolved[field]!r} is invalid; "
                f"allowed values: {', '.join(sorted(choices))}"
            )
    return resolved


def report_configuration(record: dict) -> dict:
    """Return one seed- and environment-independent reporting configuration."""
    experiment = dict(record.get("config", {}).get("experiment", {}))
    for field in ("partition_id", "partition_seed", "split_seed", "training_seed"):
        experiment.pop(field, None)
    for field in ENVIRONMENT_ONLY_EXPERIMENT_FIELDS:
        experiment.pop(field, None)
    return {
        "algorithm": record.get("algorithm"),
        "experiment": experiment,
        "algorithm_config": record.get("config", {}).get("algorithm", {}),
    }


def configuration_count(records: list[dict]) -> int:
    """Count seed-independent complete resolved configurations."""
    from rigfl.experiment.config import fingerprint

    return len({fingerprint(report_configuration(record)) for record in records})


def setting_columns(record: dict) -> tuple[dict, dict]:
    """Flat experiment and algorithm settings of one configuration, as CSV columns."""
    frozen = report_configuration(record)
    nested = {
        path.split(".")[0] for path in model_paths(ResolvedExperimentConfig) if "." in path
    }
    # a null nested setting leaves its sub-columns empty
    experiment = {
        key: value
        for key, value in flatten_mapping(frozen["experiment"]).items()
        if not (value is None and key in nested)
    }
    # already top-level columns
    experiment.pop("dataset", None)
    experiment.pop("input_spec.input_kind", None)
    return experiment, flatten_mapping(frozen["algorithm_config"])


def csv_text(rows: list[dict]) -> str:
    """Render dictionaries as CSV text."""
    if not rows:
        raise ReportingConfigError("cannot save an empty CSV table")
    fieldnames = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})
    return stream.getvalue()


def write_csv(path: str | Path, rows: list[dict]) -> Path:
    """Atomically write a derived CSV, replacing an earlier derived copy."""
    return atomic_write_text(Path(path), csv_text(rows))


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and not math.isfinite(value):
        raise ReportingConfigError("CSV output cannot contain NaN or infinity")
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return ";".join(f"{key}={value[key]}" for key in sorted(value))
    return value
