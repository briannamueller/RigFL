"""Atomic artifact writing and basic completed-run validation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from rigfl.experiment.identity import fingerprint

RECORD_KIND = "rigfl.run_result"
RECORD_SCHEMA_VERSION = 1
STATUS_COMPLETE = "complete"


class ResultValidationError(ValueError):
    """A file cannot be treated as a completed run."""

    def __init__(self, reason: str, path: Optional[Path] = None):
        self.reason = reason
        self.path = Path(path) if path is not None else None
        super().__init__(f"{self.path}: {reason}" if self.path else reason)

    def report(self) -> str:
        return (
            "Existing result cannot be treated as complete:\n"
            f"{self.path}\nReason: {self.reason}\n\n"
            "The file was preserved. Re-run with --force to replace it."
        )


def dumps(payload: Any, *, indent: int | None = 2) -> str:
    """Serialize standards-compliant JSON, refusing NaN and infinity.

    With an indent, lists and small mappings of scalars stay on one line, so
    label counts and per-round metrics read as rows.
    """
    try:
        if indent is None:
            return json.dumps(payload, allow_nan=False)
        return _format_json(payload, " " * indent)
    except (TypeError, ValueError) as exc:
        raise ResultValidationError(f"payload is not valid JSON ({exc})") from exc


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _format_json(value: Any, indent: str, level: int = 0) -> str:
    prefix, child = indent * level, indent * (level + 1)
    if isinstance(value, dict):
        if not value or (len(value) <= 3 and all(map(_is_scalar, value.values()))):
            return json.dumps(value, allow_nan=False)
        entries = [f"{child}{json.dumps(str(key))}: {_format_json(item, indent, level + 1)}"
                   for key, item in value.items()]
        return "{\n" + ",\n".join(entries) + f"\n{prefix}}}"
    if isinstance(value, (list, tuple)):
        if all(map(_is_scalar, value)):
            return json.dumps(value, allow_nan=False)
        entries = [f"{child}{_format_json(item, indent, level + 1)}" for item in value]
        return "[\n" + ",\n".join(entries) + f"\n{prefix}]"
    return json.dumps(value, allow_nan=False)


def loads(text: str, *, path: Optional[Path] = None) -> Any:
    def reject_constant(value: str):
        raise ValueError(f"non-standard numeric constant {value}")

    try:
        return json.loads(text, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise ResultValidationError(
            f"invalid JSON at line {exc.lineno} column {exc.colno} ({exc.msg})", path
        ) from exc
    except ValueError as exc:
        raise ResultValidationError(f"invalid JSON ({exc})", path) from exc


def read_json(path: Path) -> Any:
    path = Path(path)
    try:
        return loads(path.read_text(), path=path)
    except OSError as exc:
        raise ResultValidationError(f"cannot be read ({exc})", path) from exc


def atomic_write_text(path: Path, text: str) -> Path:
    """Write beside the destination, then atomically replace it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".partial"
    )
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


def atomic_write_json(path: Path, payload: Any, *, indent: int | None = 2) -> Path:
    return atomic_write_text(path, dumps(payload, indent=indent))


def make_run_record(
    *,
    algorithm: str,
    experiment: dict,
    algorithm_config: dict,
    result: dict,
    run_fingerprint: str,
    identity_input: dict,
    **extra,
) -> dict:
    record = {
        "kind": RECORD_KIND,
        "schema_version": RECORD_SCHEMA_VERSION,
        "status": STATUS_COMPLETE,
        "run_fingerprint": run_fingerprint,
        "algorithm": algorithm,
        "config": {"experiment": experiment, "algorithm": algorithm_config},
        "identity": {"fingerprint_input": identity_input},
        **extra,
        "result": result,
    }
    return record


def is_run_result(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("kind") == RECORD_KIND


def validate_run_record(
    record: Any,
    *,
    path: Optional[Path] = None,
    expected_algorithm: Optional[str] = None,
    expected_fingerprint: Optional[str] = None,
) -> dict:
    """Check that a record is a complete run with a consistent identity."""

    def fail(reason: str):
        raise ResultValidationError(reason, path)

    if not isinstance(record, dict):
        fail("document root is not an object")
    if record.get("kind") != RECORD_KIND:
        fail(f"kind is {record.get('kind')!r}, expected {RECORD_KIND!r}")
    if record.get("schema_version") != RECORD_SCHEMA_VERSION:
        fail(
            f"schema_version is {record.get('schema_version')!r}, "
            f"expected {RECORD_SCHEMA_VERSION}"
        )
    if record.get("status") != STATUS_COMPLETE:
        fail(f"status is {record.get('status')!r}, not 'complete'")

    algorithm = record.get("algorithm")
    if expected_algorithm is not None and algorithm != expected_algorithm:
        fail(f"holds algorithm {algorithm!r}, but {expected_algorithm!r} was requested")

    identity_input = (record.get("identity") or {}).get("fingerprint_input")
    if not isinstance(identity_input, dict):
        fail("identity.fingerprint_input is missing")
    computed_fingerprint = fingerprint(identity_input)
    if record.get("run_fingerprint") != computed_fingerprint:
        fail("saved run fingerprint does not match its saved fingerprint input")
    if (
        expected_fingerprint is not None
        and computed_fingerprint != expected_fingerprint
    ):
        fail("saved fingerprint does not match the requested experiment")
    if not isinstance(record.get("result"), dict):
        fail("result is missing")
    return record


def existing_result_decision(
    path: Path,
    *,
    expected_algorithm: str,
    expected_fingerprint: str,
    force: bool = False,
    require_flops: bool = False,
) -> tuple[bool, str]:
    path = Path(path)
    if not path.exists():
        return False, ""
    try:
        record = read_json(path)
        validate_run_record(
            record,
            path=path,
            expected_algorithm=expected_algorithm,
            expected_fingerprint=expected_fingerprint,
        )
    except ResultValidationError as exc:
        if force:
            return False, f"--force: existing result is unusable ({exc.reason}); rerunning"
        raise
    if force:
        return False, f"--force: rerunning over validated result: {path.name}"
    resources = record.get("resources")
    flop_settings = (
        resources.get("measurement", {}).get("flop_estimation", {})
        if isinstance(resources, dict)
        else {}
    )
    if require_flops and not flop_settings.get("enabled"):
        return False, f"rerun: {path.name} does not contain requested FLOP estimates"
    return True, f"skip (validated complete): {path.name}"
