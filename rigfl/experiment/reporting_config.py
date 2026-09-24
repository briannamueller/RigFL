"""Named filters, comparisons, and CSV output for public result reporting."""

from __future__ import annotations

import csv
import io
import math
import re
from pathlib import Path

import yaml

from rigfl.experiment.artifacts import atomic_write_text
from rigfl.experiment.paths import flatten_mapping, nested_get

_MISSING = object()
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class ReportingConfigError(ValueError):
    """A reporting configuration cannot be applied unambiguously."""


def load_reporting_config(path: str | Path) -> dict:
    """Load and validate the small public reporting configuration."""
    source = Path(path)
    try:
        loaded = yaml.safe_load(source.read_text()) or {}
    except (OSError, yaml.YAMLError) as error:
        raise ReportingConfigError(
            f"cannot read reporting configuration {source}: {error}"
        ) from error
    if not isinstance(loaded, dict):
        raise ReportingConfigError("reporting configuration must be a mapping")
    unknown = sorted(set(loaded) - {"version", "defaults", "filters", "comparisons"})
    if unknown:
        raise ReportingConfigError(
            "unknown top-level reporting setting(s): " + ", ".join(unknown)
        )
    if loaded.get("version") != 1:
        raise ReportingConfigError("reporting configuration version must be 1")
    for section in ("defaults", "filters", "comparisons"):
        value = loaded.get(section, {})
        if not isinstance(value, dict):
            raise ReportingConfigError(f"{section} must be a mapping")
        loaded[section] = value
    _validate_names(loaded["filters"], "filter")
    _validate_names(loaded["comparisons"], "comparison")
    return loaded


def _validate_names(values: dict, kind: str) -> None:
    invalid = [str(name) for name in values if not _NAME.fullmatch(str(name))]
    if invalid:
        raise ReportingConfigError(
            f"{kind} names may contain letters, numbers, underscores, and hyphens; "
            f"invalid: {', '.join(invalid)}"
        )


def record_value(record: dict, path: str, default=_MISSING):
    """Read one real path from the persisted run-record structure."""
    if path == "algorithm":
        return record.get("algorithm", default)
    if not path.startswith("config.experiment.") and not path.startswith(
        "config.algorithm."
    ):
        raise ReportingConfigError(
            f"unknown reporting path {path!r}; use 'algorithm', "
            "'config.experiment.*', or 'config.algorithm.*'"
        )
    return nested_get(record, path, default)


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
        if all(record_value(record, path, _MISSING) in allowed for path, allowed in normalized.items())
    ]
    if not matches:
        raise ReportingConfigError(f"filter {name!r} matched no completed results")
    return matches, normalized


def selection_defaults(config: dict) -> dict:
    """Return the validated reporting selection protocol."""
    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ReportingConfigError("defaults must be a mapping")
    unknown = sorted(set(defaults) - {"metric", "confidence_level", "selection"})
    if unknown:
        raise ReportingConfigError(
            "unknown reporting default(s): " + ", ".join(unknown)
        )
    confidence = defaults.get("confidence_level", 0.95)
    if confidence != 0.95:
        raise ReportingConfigError(
            "confidence_level currently supports only 0.95"
        )
    selection = defaults.get("selection", {})
    if not isinstance(selection, dict):
        raise ReportingConfigError("defaults.selection must be a mapping")
    unknown_selection = sorted(
        set(selection) - {"split", "view", "aggregation", "tie_break"}
    )
    if unknown_selection:
        raise ReportingConfigError(
            "unknown selection setting(s): " + ", ".join(unknown_selection)
        )
    resolved = {
        "metric": defaults.get("metric", "accuracy"),
        "split": selection.get("split", "validation"),
        "view": selection.get("view", "global"),
        "aggregation": selection.get("aggregation", "mean"),
        "tie_break": selection.get("tie_break", "earliest"),
    }
    allowed = {
        "split": {"validation"},
        "view": {"global", "per-client"},
        "aggregation": {"mean", "weighted_mean"},
        "tie_break": {"earliest", "latest"},
    }
    for field, choices in allowed.items():
        if resolved[field] not in choices:
            raise ReportingConfigError(
                f"defaults.selection.{field}={resolved[field]!r} is invalid; "
                f"allowed values: {', '.join(sorted(choices))}"
            )
    return resolved


def seed_summary(records: list[dict]) -> dict:
    """Describe fixed and varied replicate seed components."""
    fields = {
        "partition": "partition_seed",
        "split": "split_seed",
        "training": "seed",
    }
    values = {
        label: sorted(
            {
                record.get("config", {}).get("experiment", {}).get(field)
                for record in records
            },
            key=repr,
        )
        for label, field in fields.items()
    }
    varied = [label for label, items in values.items() if len(items) > 1]
    fixed = [label for label, items in values.items() if len(items) <= 1]
    return {"values": values, "varied": varied, "fixed": fixed}


def configuration_count(records: list[dict]) -> int:
    """Count seed-independent complete resolved configurations."""
    from rigfl.eval.comparison import frozen_configuration
    from rigfl.experiment.config import fingerprint

    return len({fingerprint(frozen_configuration(record)) for record in records})


def human_configuration(record: dict) -> dict:
    """Flatten the seed-independent configuration using persisted field names."""
    from rigfl.eval.comparison import frozen_configuration

    frozen = frozen_configuration(record)
    output = {"algorithm": frozen["algorithm"]}
    output.update(
        flatten_mapping(frozen.get("experiment", {}), "config.experiment")
    )
    output.update(
        flatten_mapping(frozen.get("algorithm_config", {}), "config.algorithm")
    )
    return output


def csv_text(rows: list[dict]) -> str:
    """Render dictionaries as a stable, standards-compliant CSV."""
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
