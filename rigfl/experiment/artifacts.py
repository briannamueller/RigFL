"""Atomic artifact writing and basic completed-run validation."""

from __future__ import annotations

import json
import math
import os
import tempfile
from itertools import pairwise
from pathlib import Path
from typing import Any, Callable, Optional

RECORD_KIND = "rigfl.run_result"
RECORD_SCHEMA_VERSION = 4
READABLE_RECORD_SCHEMA_VERSIONS = {3, 4}
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
    record_version = 4 if "resources" in extra else 3
    record = {
        "kind": RECORD_KIND,
        "record_schema_version": record_version,
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
    record_version = record.get("record_schema_version")
    if record_version not in READABLE_RECORD_SCHEMA_VERSIONS:
        fail(
            f"record_schema_version is {record_version!r}, expected one of "
            f"{sorted(READABLE_RECORD_SCHEMA_VERSIONS)}"
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

    from rigfl.experiment.config import ResolvedExperimentConfig
    from rigfl.experiment.registry import algorithm_run_fingerprint, config_class

    try:
        experiment = ResolvedExperimentConfig(**config["experiment"])
        algorithm_config = config_class(algorithm)(**config["algorithm"])
    except Exception as exc:
        fail(f"saved configuration does not validate: {exc}")

    computed_fingerprint = algorithm_run_fingerprint(
        algorithm, experiment, algorithm_config.model_dump()
    )
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
    if record_version >= 4:
        _validate_resources(record.get("resources"), record.get("wall_seconds"),
                            experiment, fail)
    return record


def _valid_number(value, *, integer: bool = False) -> bool:
    kind = int if integer else (int, float)
    return (isinstance(value, kind) and not isinstance(value, bool)
            and value >= 0 and math.isfinite(float(value)))


def _same_number(left, right) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)


def _validate_hardware_signature(value: Any, fail) -> None:
    if not isinstance(value, dict):
        fail("resources timing hardware is missing")
    required = {
        "device_type", "torch_version", "python_version", "platform",
        "cpu_name", "cpu_identity_source", "device_name",
    }
    if any(not isinstance(value.get(name), str) for name in required):
        fail("resources timing hardware is invalid")


def _validate_flop_settings(value: Any, fail) -> None:
    if not isinstance(value, dict):
        fail("resources FLOP metadata is missing")
    enabled = value.get("enabled")
    if not isinstance(enabled, bool) or not isinstance(
            value.get("torch_version"), str):
        fail("resources FLOP metadata is invalid")
    method = value.get("method")
    convention = value.get("convention")
    if enabled:
        if not isinstance(method, str) or not isinstance(convention, str):
            fail("enabled FLOP estimation metadata is incomplete")
    elif method is not None or convention is not None:
        fail("disabled FLOP estimation metadata is inconsistent")


def _validate_resources(saved: Any, wall_seconds, experiment, fail) -> None:
    if not isinstance(saved, dict) or saved.get("schema_version") != 1:
        fail("resources is missing or uses an unsupported schema")
    measurement = saved.get("measurement")
    observed = saved.get("observed")
    if not isinstance(measurement, dict) or not isinstance(observed, dict):
        fail("resources measurement or observed values are missing")

    communication_settings = measurement.get("communication")
    if not isinstance(communication_settings, dict):
        fail("resources communication metadata is missing")
    if not isinstance(communication_settings.get("basis"), str):
        fail("resources communication basis is invalid")
    if not isinstance(
            communication_settings.get("includes_protocol_metadata"), bool):
        fail("resources communication metadata is invalid")

    timing_settings = measurement.get("timing")
    if not isinstance(timing_settings, dict):
        fail("resources timing metadata is missing")
    if not isinstance(timing_settings.get("clock"), str):
        fail("resources timing clock is invalid")
    if not isinstance(timing_settings.get("accelerator_synchronized"), bool):
        fail("resources timing synchronization metadata is invalid")
    hardware = timing_settings.get("hardware")
    _validate_hardware_signature(hardware, fail)

    communication = observed.get("communication_bytes")
    expected = {"client_to_server", "server_to_client", "peer_to_peer", "total"}
    if not isinstance(communication, dict) or set(communication) != expected:
        fail("resources communication totals are invalid")
    if any(not _valid_number(value, integer=True)
           for value in communication.values()):
        fail("resources communication contains an invalid byte count")
    if communication["total"] != sum(
            communication[name] for name in expected - {"total"}):
        fail("resources communication total is inconsistent")

    flop_settings = measurement.get("flop_estimation")
    flops = observed.get("flops")
    _validate_flop_settings(flop_settings, fail)
    resource_total_names = {"algorithm_operations", "evaluation", "total"}
    if not isinstance(flops, dict) or set(flops) != resource_total_names:
        fail("resources FLOP totals are missing")
    if bool(flop_settings.get("enabled")) != bool(experiment.estimate_flops):
        fail("resources FLOP setting disagrees with the experiment configuration")
    for name in ("algorithm_operations", "evaluation", "total"):
        value = flops.get(name)
        if value is not None and not _valid_number(value, integer=True):
            fail(f"resources flops.{name} is invalid")
    if experiment.estimate_flops:
        if any(flops.get(name) is None for name in
               ("algorithm_operations", "evaluation", "total")):
            fail("enabled FLOP estimation has a missing observed total")
        if flops["total"] != flops["algorithm_operations"] + flops["evaluation"]:
            fail("resources FLOP total is inconsistent")
    elif any(flops.get(name) is not None for name in
             ("algorithm_operations", "evaluation", "total")):
        fail("disabled FLOP estimation has observed totals")

    timing = observed.get("wall_seconds")
    if not isinstance(timing, dict) or set(timing) != resource_total_names:
        fail("resources wall time is missing")
    for name in ("algorithm_operations", "evaluation", "total"):
        if not _valid_number(timing.get(name)):
            fail(f"resources wall_seconds.{name} is invalid")
    if not _valid_number(wall_seconds):
        fail("wall_seconds is invalid")
    if round(timing["total"], 1) != wall_seconds:
        fail("wall_seconds disagrees with resources observed total")

    attributed = saved.get("attributed_training")
    if (not isinstance(attributed, dict)
            or set(attributed) != {
                "flops", "wall_seconds", "wall_seconds_comparable"
            }):
        fail("resources attributed-training totals are missing")
    for name in ("flops", "wall_seconds"):
        value = attributed.get(name)
        if value is not None and not _valid_number(
                value, integer=name == "flops"):
            fail(f"resources attributed_training.{name} is invalid")
    if not isinstance(attributed.get("wall_seconds_comparable"), bool):
        fail("resources attributed-training timing compatibility is missing")
    if (attributed["wall_seconds_comparable"]
            != (attributed.get("wall_seconds") is not None)):
        fail("resources attributed-training wall time is inconsistent")
    if not experiment.estimate_flops and attributed.get("flops") is not None:
        fail("disabled FLOP estimation has an attributed total")

    reused = saved.get("reused")
    if not isinstance(reused, list):
        fail("resources reused measurements are missing")
    for item in reused:
        access = item.get("current_access") if isinstance(item, dict) else None
        if (not isinstance(item, dict) or not isinstance(item.get("name"), str)
                or not isinstance(access, dict)
                or not _valid_number(access.get("wall_seconds"))):
            fail("resources reused measurement is invalid")
        if not _valid_number(item.get("wall_seconds")):
            fail("resources reused wall time is invalid")
        reused_flops = item.get("flops")
        if reused_flops is not None and not _valid_number(
                reused_flops, integer=True):
            fail("resources reused FLOPs are invalid")
        reused_measurement = item.get("measurement")
        if not isinstance(reused_measurement, dict):
            fail("resources reused measurement metadata is missing")
        _validate_hardware_signature(
            reused_measurement.get("hardware"), fail)
        _validate_flop_settings(
            reused_measurement.get("flop_estimation"), fail)
        access_flops = access.get("flops")
        if access_flops is not None and not _valid_number(
                access_flops, integer=True):
            fail("resources reused access FLOPs are invalid")
    missing_reused = saved.get("missing_reused_measurements")
    if (not isinstance(missing_reused, list)
            or any(not isinstance(name, str) for name in missing_reused)):
        fail("resources missing-reuse record is invalid")
    cache_reuse = saved.get("cache_reuse")
    if (not isinstance(cache_reuse, dict)
            or any(not isinstance(stage, str) for stage in cache_reuse)
            or any(value not in {"none", "partial", "complete", "disabled"}
                   for value in cache_reuse.values())):
        fail("resources cache-reuse record is invalid")

    operations = saved.get("operations")
    if not isinstance(operations, dict) or not operations:
        fail("resources operations are missing")
    for operation, values in operations.items():
        if (not isinstance(operation, str) or not isinstance(values, dict)
                or values.get("category") not in {"algorithm", "evaluation"}
                or not _valid_number(values.get("calls"), integer=True)
                or values["calls"] < 1
                or not _valid_number(values.get("wall_seconds"))):
            fail(f"resources operation {operation!r} is invalid")
        operation_flops = values.get("flops")
        if experiment.estimate_flops:
            if not _valid_number(operation_flops, integer=True):
                fail(f"resources operation {operation!r} FLOPs are invalid")
        elif operation_flops is not None:
            fail(f"resources operation {operation!r} has unexpected FLOPs")
    operation_seconds = {
        category: sum(values["wall_seconds"] for values in operations.values()
                      if values["category"] == category)
        for category in ("algorithm", "evaluation")
    }
    if not _same_number(
            operation_seconds["algorithm"], timing["algorithm_operations"]):
        fail("resources algorithm-operation wall time is inconsistent")
    if not _same_number(operation_seconds["evaluation"], timing["evaluation"]):
        fail("resources evaluation wall time is inconsistent")
    if experiment.estimate_flops:
        operation_flops = {
            category: sum(values["flops"] for values in operations.values()
                          if values["category"] == category)
            for category in ("algorithm", "evaluation")
        }
        if operation_flops["algorithm"] != flops["algorithm_operations"]:
            fail("resources algorithm-operation FLOPs are inconsistent")
        if operation_flops["evaluation"] != flops["evaluation"]:
            fail("resources evaluation FLOPs are inconsistent")
    clients = saved.get("clients")
    if not isinstance(clients, dict):
        fail("resources clients are missing")
    expected_client_ids = {str(client_id)
                           for client_id in range(experiment.num_clients)}
    if set(clients) != expected_client_ids:
        fail("resources clients do not match the configured client ids")
    categories = {"algorithm_operations", "evaluation"}
    for client_id, client in clients.items():
        if not isinstance(client, dict):
            fail(f"resources client {client_id} is invalid")
        for field in ("wall_seconds", "flops"):
            values = client.get(field)
            if not isinstance(values, dict) or set(values) != categories:
                fail(f"resources client {client_id} {field} is invalid")
            for value in values.values():
                if field == "wall_seconds" and not _valid_number(value):
                    fail(f"resources client {client_id} {field} is invalid")
                if field == "flops":
                    if (experiment.estimate_flops
                            and not _valid_number(value, integer=True)):
                        fail(f"resources client {client_id} {field} is invalid")
                    if not experiment.estimate_flops and value is not None:
                        fail(f"resources client {client_id} {field} is invalid")
        if (not _valid_number(client.get("sent_bytes"), integer=True)
                or not _valid_number(client.get("received_bytes"), integer=True)):
            fail(f"resources client {client_id} communication is invalid")
    sent = sum(client["sent_bytes"] for client in clients.values())
    received = sum(client["received_bytes"] for client in clients.values())
    if sent != communication["client_to_server"] + communication["peer_to_peer"]:
        fail("resources per-client sent bytes are inconsistent")
    if (received
            != communication["server_to_client"] + communication["peer_to_peer"]):
        fail("resources per-client received bytes are inconsistent")
    checkpoints = saved.get("checkpoints")
    if (not isinstance(checkpoints, list) or not checkpoints
            or any(not isinstance(checkpoint, dict)
                   for checkpoint in checkpoints)):
        fail("resources checkpoints are missing")
    rounds = [checkpoint.get("round") for checkpoint in checkpoints]
    if rounds != list(range(len(rounds))) and rounds != [0]:
        fail("resources checkpoints are not consecutive rounds")
    for checkpoint in checkpoints:
        required = {
            "round", "algorithm_wall_seconds", "algorithm_flops",
            "communication_bytes",
        }
        if set(checkpoint) != required:
            fail("resources checkpoint fields are invalid")
        if (not _valid_number(checkpoint.get("algorithm_wall_seconds"))
                or not _valid_number(
                    checkpoint.get("communication_bytes"), integer=True)):
            fail("resources checkpoint totals are invalid")
        checkpoint_flops = checkpoint.get("algorithm_flops")
        if experiment.estimate_flops:
            if not _valid_number(checkpoint_flops, integer=True):
                fail("resources checkpoint FLOPs are invalid")
        elif checkpoint_flops is not None:
            fail("resources checkpoint has unexpected FLOPs")
    if not _same_number(
            checkpoints[-1]["algorithm_wall_seconds"],
            timing["algorithm_operations"]):
        fail("resources final checkpoint wall time is inconsistent")
    if checkpoints[-1]["communication_bytes"] != communication["total"]:
        fail("resources final checkpoint communication is inconsistent")
    if (experiment.estimate_flops
            and checkpoints[-1]["algorithm_flops"]
            != flops["algorithm_operations"]):
        fail("resources final checkpoint FLOPs are inconsistent")
    for previous, current in pairwise(checkpoints):
        if (current["algorithm_wall_seconds"]
                < previous["algorithm_wall_seconds"]
                or current["communication_bytes"]
                < previous["communication_bytes"]):
            fail("resources checkpoint totals are not cumulative")
        if (experiment.estimate_flops
                and current["algorithm_flops"] < previous["algorithm_flops"]):
            fail("resources checkpoint FLOPs are not cumulative")

    reused_seconds = sum(item["wall_seconds"] for item in reused)
    access_seconds = sum(item["current_access"]["wall_seconds"] for item in reused)
    same_hardware = (
        not missing_reused
        and (not reused or (
            hardware.get("cpu_identity_source") != "generic_fallback"
            and all(item["measurement"]["hardware"] == hardware
                    for item in reused)
        ))
    )
    if attributed["wall_seconds_comparable"] != same_hardware:
        fail("resources attributed-training hardware compatibility is inconsistent")
    expected_wall = (
        max(0.0, timing["algorithm_operations"] - access_seconds)
        + reused_seconds
        if same_hardware else None
    )
    if (expected_wall is None) != (attributed["wall_seconds"] is None):
        fail("resources attributed-training wall time is inconsistent")
    if (expected_wall is not None
            and not _same_number(expected_wall, attributed["wall_seconds"])):
        fail("resources attributed-training wall time is inconsistent")

    expected_flops = None
    compatible_flops = (
        experiment.estimate_flops
        and not missing_reused
        and all(item["measurement"]["flop_estimation"] == flop_settings
                for item in reused)
        and all(item["flops"] is not None for item in reused)
        and all(item["current_access"]["flops"] is not None for item in reused)
    )
    if compatible_flops:
        expected_flops = (
            max(0, flops["algorithm_operations"] - sum(
                item["current_access"]["flops"] for item in reused))
            + sum(item["flops"] for item in reused)
        )
    if attributed["flops"] != expected_flops:
        fail("resources attributed-training FLOPs are inconsistent")


def _validate_history(history: Any, experiment, fail) -> None:
    if not isinstance(history, dict):
        fail("result.evaluation_history is missing or is not an object")
    rounds = history.get("evaluation_rounds")
    if not isinstance(rounds, list) or not rounds:
        fail("evaluation_rounds is missing or empty")
    if not all(isinstance(r, int) and not isinstance(r, bool) for r in rounds):
        fail("evaluation_rounds contains a non-integer")
    if any(right <= left for left, right in pairwise(rounds)):
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
