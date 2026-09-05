"""Atomic artifact writing and basic completed-run validation."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

RECORD_KIND = "rigfl.run_result"
RECORD_SCHEMA_VERSION = 3
RESULT_SCHEMA_VERSION = 3
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
    """Serialize standards-compliant JSON, refusing NaN and infinity."""
    try:
        return json.dumps(payload, indent=indent, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ResultValidationError(f"payload is not valid JSON ({exc})") from exc


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


def atomic_write_text(
    path: Path, text: str, validate: Callable[[str], None] | None = None
) -> Path:
    """Write beside the destination, then atomically replace it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".partial"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if validate is not None:
            validate(tmp.read_text())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    indent: int | None = 2,
    validate: Callable[[Any], None] | None = None,
) -> Path:
    text = dumps(payload, indent=indent)

    def check(written: str) -> None:
        parsed = loads(written, path=path)
        if validate is not None:
            validate(parsed)

    return atomic_write_text(path, text, validate=check)


def make_run_record(
    *,
    algorithm: str,
    experiment: dict,
    algorithm_config: dict,
    result: dict,
    run_fingerprint: str,
    **extra,
) -> dict:
    record = {
        "kind": RECORD_KIND,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "status": STATUS_COMPLETE,
        "run_fingerprint": run_fingerprint,
        "algorithm": algorithm,
        "config": {"experiment": experiment, "algorithm": algorithm_config},
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
    """Validate completion, training identity, and history structure."""

    def fail(reason: str):
        raise ResultValidationError(reason, path)

    if not isinstance(record, dict):
        fail("document root is not an object")
    if record.get("kind") != RECORD_KIND:
        fail(f"kind is {record.get('kind')!r}, expected {RECORD_KIND!r}")
    if record.get("record_schema_version") != RECORD_SCHEMA_VERSION:
        fail(
            f"record_schema_version is {record.get('record_schema_version')!r}, "
            f"expected {RECORD_SCHEMA_VERSION}"
        )
    if record.get("status") != STATUS_COMPLETE:
        fail(f"status is {record.get('status')!r}, not 'complete'")

    algorithm = record.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm:
        fail("algorithm is missing")
    if expected_algorithm is not None and algorithm != expected_algorithm:
        fail(f"holds algorithm {algorithm!r}, but {expected_algorithm!r} was requested")

    config = record.get("config")
    if not isinstance(config, dict):
        fail("config is missing or is not an object")
    if not isinstance(config.get("experiment"), dict):
        fail("config.experiment is missing or is not an object")
    if not isinstance(config.get("algorithm"), dict):
        fail("config.algorithm is missing or is not an object")

    resolved_fields = {
        "data_backend", "partition_id", "partition_scheme", "num_clients",
        "num_classes", "validation_fraction", "input_kind", "input_spec",
    }
    missing = sorted(resolved_fields - set(config["experiment"]))
    if missing:
        fail(
            "config.experiment lacks resolved dataset fields: "
            + ", ".join(missing)
        )

    from rigfl.experiment.config import ResolvedExperimentConfig, run_fingerprint
    from rigfl.experiment.registry import config_class

    try:
        experiment = ResolvedExperimentConfig(**config["experiment"])
        algorithm_config = config_class(algorithm)(**config["algorithm"])
    except Exception as exc:
        fail(f"saved configuration does not validate: {exc}")

    computed_fingerprint = run_fingerprint(experiment, algorithm_config.model_dump())
    if record.get("run_fingerprint") != computed_fingerprint:
        fail("saved run fingerprint does not match the saved configuration")
    if (
        expected_fingerprint is not None
        and computed_fingerprint != expected_fingerprint
    ):
        fail("saved configuration does not match the requested experiment")

    result = record.get("result")
    if not isinstance(result, dict):
        fail("result is missing or is not an object")
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        fail(
            f"result.schema_version is {result.get('schema_version')!r}, "
            f"expected {RESULT_SCHEMA_VERSION}"
        )
    history = result.get("evaluation_history")
    stopping = result.get("early_stopping")
    views = result.get("selection_views_supported")
    if (
        not isinstance(views, list)
        or not views
        or len(views) != len(set(views))
        or any(view not in {"global", "per-client"} for view in views)
    ):
        fail("result.selection_views_supported is invalid")
    iterative_result = "global" in views
    if not iterative_result and views != ["per-client"]:
        fail("a non-iterative result must support exactly the per-client view")
    _validate_history(history, experiment, fail)
    if iterative_result:
        if result.get("selection_provenance") is not None:
            fail("an iterative result must not contain selection_provenance")
    else:
        _validate_local_selection(result.get("selection_provenance"), experiment, fail)
    _validate_early_stopping(stopping, experiment, history, fail,
                             iterative=iterative_result)
    return record


def _validate_history(history: Any, experiment, fail) -> None:
    if not isinstance(history, dict):
        fail("result.evaluation_history is missing or is not an object")
    rounds = history.get("evaluation_rounds")
    if not isinstance(rounds, list) or not rounds:
        fail("evaluation_rounds is missing or empty")
    if not all(isinstance(r, int) and not isinstance(r, bool) for r in rounds):
        fail("evaluation_rounds contains a non-integer")
    if any(right <= left for left, right in zip(rounds, rounds[1:])):
        fail("evaluation_rounds is not strictly increasing")
    if rounds[0] < 0 or rounds[-1] >= experiment.rounds:
        fail("evaluation_rounds falls outside the configured round range")

    clients = history.get("clients")
    if not isinstance(clients, dict) or not clients:
        fail("evaluation_history.clients is missing or empty")
    expected_clients = {str(i) for i in range(experiment.num_clients)}
    if set(clients) != expected_clients:
        fail("evaluation_history client ids do not match num_clients")

    n_rounds = len(rounds)
    metric_names = None
    for client_id, splits in clients.items():
        if not isinstance(splits, dict):
            fail(f"client {client_id} history is not an object")
        for split in ("validation", "test"):
            metrics = splits.get(split)
            if not isinstance(metrics, dict) or not metrics:
                fail(f"client {client_id} {split} metrics are missing")
            if metric_names is None:
                metric_names = set(metrics)
            elif set(metrics) != metric_names:
                fail("clients and splits do not record the same metric names")
            for name, values in metrics.items():
                if not isinstance(values, list) or len(values) != n_rounds:
                    fail(f"client {client_id} {split}.{name} is not round-aligned")
                if any(
                    value is not None
                    and (
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(float(value))
                    )
                    for value in values
                ):
                    fail(f"client {client_id} {split}.{name} contains an invalid value")

    counts = history.get("client_sample_counts")
    if not isinstance(counts, dict):
        fail("client_sample_counts is missing")
    for split in ("validation", "test"):
        per_client = counts.get(split)
        if not isinstance(per_client, dict) or set(per_client) != set(clients):
            fail(f"client_sample_counts.{split} does not cover every client")
        for client_id, values in per_client.items():
            if not isinstance(values, list) or len(values) != n_rounds:
                fail(f"client_sample_counts.{split}.{client_id} is not round-aligned")


def _validate_local_selection(saved: Any, experiment, fail) -> None:
    """Validate provenance for models selected inside one-shot local computation."""
    if not isinstance(saved, dict):
        fail("result.selection_provenance is missing or is not an object")
    if saved.get("view") != "per-client":
        fail("selection_provenance.view must be 'per-client'")
    if saved.get("stage") != "local_computation":
        fail("selection_provenance.stage must be 'local_computation'")
    metric = saved.get("metric")
    if not isinstance(metric, str) or not metric:
        fail("selection_provenance.metric is missing")
    clients = saved.get("clients")
    expected = {str(i) for i in range(experiment.num_clients)}
    if not isinstance(clients, dict) or set(clients) != expected:
        fail("selection_provenance.clients does not cover every client")
    for client_id, selected in clients.items():
        if not isinstance(selected, dict):
            fail(f"selection provenance for client {client_id} is not an object")
        step = selected.get("selected_step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            fail(f"selection provenance for client {client_id} has an invalid selected_step")
        value = selected.get("validation_value")
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(float(value))):
            fail(f"selection provenance for client {client_id} has an invalid validation_value")


def _validate_early_stopping(saved: Any, experiment, history: dict, fail, *,
                             iterative: bool) -> None:
    if not isinstance(saved, dict):
        fail("result.early_stopping is missing or is not an object")
    configured = experiment.early_stopping
    if not iterative:
        if configured.enabled:
            fail("experiment early stopping cannot be enabled for a one-shot result")
        if saved.get("enabled") is not False:
            fail("one-shot early-stopping record must be disabled")
        if saved.get("termination_reason") != "not_applicable":
            fail("one-shot early-stopping termination reason must be 'not_applicable'")
        if saved.get("stopped_at_round") is not None:
            fail("one-shot result must not record a stopped_at_round")
        if history["evaluation_rounds"] != [0]:
            fail("one-shot result must contain exactly one final evaluation point")
        return
    if bool(saved.get("enabled")) != bool(configured.enabled):
        fail("early-stopping record disagrees with the experiment configuration")
    if saved.get("termination_reason") not in (
        "completed_all_rounds",
        "early_stopping",
    ):
        fail("early-stopping termination reason is invalid")
    stopped_at = saved.get("stopped_at_round")
    if (
        not isinstance(stopped_at, int)
        or isinstance(stopped_at, bool)
        or not 0 <= stopped_at < experiment.rounds
    ):
        fail("early-stopping stopped_at_round is invalid")
    expected_rounds = [
        rnd for rnd in range(stopped_at + 1)
        if rnd % experiment.eval_gap == 0 or rnd == stopped_at
    ]
    if history["evaluation_rounds"] != expected_rounds:
        fail("evaluation_rounds is incomplete for the recorded stopping point")
    if configured.enabled:
        for field in (
            "metric",
            "direction",
            "split",
            "aggregation",
            "patience",
            "min_delta",
        ):
            if saved.get(field) != getattr(configured, field):
                fail(f"early-stopping {field} disagrees with the configuration")


def existing_result_decision(
    path: Path,
    *,
    expected_algorithm: str,
    expected_fingerprint: str,
    force: bool = False,
) -> tuple[bool, str]:
    path = Path(path)
    if not path.exists():
        return False, ""
    try:
        validate_run_record(
            read_json(path),
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
    return True, f"skip (validated complete): {path.name}"


def write_run_record(
    path: Path,
    record: dict,
    *,
    expected_algorithm: str,
    expected_fingerprint: str,
) -> Path:
    def check(parsed: Any) -> None:
        validate_run_record(
            parsed,
            path=path,
            expected_algorithm=expected_algorithm,
            expected_fingerprint=expected_fingerprint,
        )

    return atomic_write_json(path, record, validate=check)
