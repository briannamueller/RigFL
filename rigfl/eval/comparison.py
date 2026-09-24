"""Matched client and replicate calculations used by Local-relative reporting."""

from __future__ import annotations

import statistics

from rigfl.eval.metrics import canonical, direction_of
from rigfl.eval.report import (
    pooled_within_replicate_sd,
    replicate_statistics,
    selection_for,
)
from rigfl.experiment.config import (
    result_data_configuration,
    result_data_configuration_id,
)


class ConfigurationComparisonError(ValueError):
    """Raised when two sets of results cannot be compared as paired runs."""


def replicate_condition(record: dict) -> dict:
    """Return the data and training seeds that identify one replicate."""
    experiment = record.get("config", {}).get("experiment", {})
    fields = {
        "partition_seed": "partition_seed",
        "split_seed": "split_seed",
        "experiment_seed": "seed",
    }
    missing = [source for source in fields.values() if source not in experiment]
    if missing:
        raise ConfigurationComparisonError(
            "comparison requires partition_seed, split_seed, and seed in every "
            f"result; missing: {', '.join(missing)}"
        )
    return {name: experiment[source] for name, source in fields.items()}


_DATA_FIELDS = (
    "data_backend",
    "partition_scheme",
    "num_clients",
    "num_classes",
    "validation_fraction",
    "input_kind",
    "input_spec",
)


def comparison_context(record: dict) -> tuple[str, str]:
    """Return the dataset and seed-independent data-configuration identity."""
    try:
        configuration = result_data_configuration(record)
    except ValueError as error:
        raise ConfigurationComparisonError(str(error)) from error
    return configuration["dataset"], result_data_configuration_id(record)


def _record_key(record: dict) -> tuple:
    context = comparison_context(record)
    replicate = replicate_condition(record)
    return (
        *context,
        replicate["partition_seed"],
        replicate["split_seed"],
        replicate["experiment_seed"],
    )


def _index(records: list[dict], label: str) -> dict[tuple, dict]:
    indexed = {}
    for record in records:
        key = _record_key(record)
        if key in indexed:
            raise ConfigurationComparisonError(
                f"{label} contains more than one result for {key}"
            )
        indexed[key] = record
    if not indexed:
        raise ConfigurationComparisonError(f"{label} contains no run results")
    return indexed


def _check_data(left: dict, right: dict, key: tuple) -> None:
    left_exp = left.get("config", {}).get("experiment", {})
    right_exp = right.get("config", {}).get("experiment", {})
    different = [
        field for field in _DATA_FIELDS if left_exp.get(field) != right_exp.get(field)
    ]
    if different:
        raise ConfigurationComparisonError(
            f"paired runs for {key} use different data settings: "
            + ", ".join(different)
        )
    if left_exp.get("partition_id") != right_exp.get("partition_id"):
        raise ConfigurationComparisonError(
            f"paired runs for {key} use different generated partitions"
        )

def _client_values(selected: dict, metric: str, split: str) -> dict[str, float]:
    values = selected.get(split, {}).get(metric, [])
    client_ids = selected.get("client_ids") or [str(i) for i in range(len(values))]
    return {
        str(client_id): value
        for client_id, value in zip(client_ids, values)
        if value is not None
    }


def _client_weights(selected: dict, split: str) -> dict[str, float | None]:
    client_ids = selected.get("client_ids") or []
    weights = selected.get("sample_counts", {}).get(split, [])
    return {str(client_id): weight for client_id, weight in zip(client_ids, weights)}


def paired_client_differences(
    left_records: list[dict],
    right_records: list[dict],
    metric: str,
    *,
    left_label: str = "left",
    right_label: str = "right",
    view: str = "global",
    aggregation: str = "mean",
    tie_break: str = "earliest",
    evaluation_split: str = "test",
    align_selection_views: bool = False,
) -> dict:
    """Return validation-selected differences matched by run and client."""
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation split must be validation or test")
    left_index = _index(left_records, left_label)
    right_index = _index(right_records, right_label)
    if set(left_index) != set(right_index):
        missing_right = sorted(set(left_index) - set(right_index), key=repr)
        missing_left = sorted(set(right_index) - set(left_index), key=repr)
        details = []
        if missing_right:
            details.append(f"missing from {right_label}: {missing_right}")
        if missing_left:
            details.append(f"missing from {left_label}: {missing_left}")
        raise ConfigurationComparisonError(
            "comparison requires identical data configurations and replicate "
            "conditions ("
            + "; ".join(details)
            + ")"
        )

    name = canonical(metric)
    sign = 1.0 if direction_of(name) == "maximize" else -1.0
    pairs = []
    actual_views = {left_label: set(), right_label: set()}
    for key in sorted(left_index, key=repr):
        left = left_index[key]
        right = right_index[key]
        _check_data(left, right, key)
        experiment = left["config"]["experiment"]
        replicate = replicate_condition(left)
        selected_left = selection_for(
            left,
            name,
            view=view,
            aggregation=aggregation,
            tie_break=tie_break,
            include_test=evaluation_split == "test",
        )
        right_view = selected_left["selection_view"] if align_selection_views else view
        selected_right = selection_for(
            right,
            name,
            view=right_view,
            aggregation=aggregation,
            tie_break=tie_break,
            include_test=evaluation_split == "test",
        )
        actual_views[left_label].add(selected_left["selection_view"])
        actual_views[right_label].add(selected_right["selection_view"])
        if (
            align_selection_views
            and selected_left["selection_view"] != selected_right["selection_view"]
        ):
            raise ConfigurationComparisonError(
                "paired configurations do not support the same selection view"
            )
        left_values = _client_values(selected_left, name, evaluation_split)
        right_values = _client_values(selected_right, name, evaluation_split)
        if set(left_values) != set(right_values):
            raise ConfigurationComparisonError(
                f"client IDs differ for dataset={key[0]}, "
                f"replicate={replicate}"
            )
        client_ids = set(left_values)
        expected_client_count = left["config"]["experiment"].get("num_clients")
        if expected_client_count is not None and len(client_ids) != expected_client_count:
            raise ConfigurationComparisonError(
                f"dataset={key[0]}, replicate={replicate} contains "
                f"{len(client_ids)} clients; expected {expected_client_count}"
            )
        left_weights = _client_weights(selected_left, evaluation_split)
        right_weights = _client_weights(selected_right, evaluation_split)
        if left_weights != right_weights:
            raise ConfigurationComparisonError(
                f"sample counts differ for dataset={key[0]}, "
                f"replicate={replicate}"
            )
        for client_id in sorted(left_values):
            left_value = left_values[client_id]
            right_value = right_values[client_id]
            pairs.append(
                {
                    "dataset": key[0],
                    "data_configuration_hash": key[1],
                    "partition_id": experiment["partition_id"],
                    "replicate_condition": replicate,
                    "client_id": client_id,
                    "left_value": left_value,
                    "right_value": right_value,
                    "gain": sign * (left_value - right_value),
                    "sample_count": left_weights.get(client_id),
                }
            )
    if not pairs:
        raise ConfigurationComparisonError("comparison found no paired test values")
    for label, values in actual_views.items():
        if len(values) != 1:
            raise ConfigurationComparisonError(
                f"{label} uses different selection views across paired runs"
            )
    return {
        "metric": name,
        "direction": direction_of(name),
        "requested_selection_view": view,
        "evaluation_split": evaluation_split,
        "selection_views": {
            label: next(iter(values)) for label, values in actual_views.items()
        },
        "pairs": pairs,
    }


def _pair_replicate_key(pair: dict) -> tuple:
    replicate = pair["replicate_condition"]
    return (
        pair["dataset"],
        pair["data_configuration_hash"],
        replicate["partition_seed"],
        replicate["split_seed"],
        replicate["experiment_seed"],
    )


def pairs_by_replicate(pairs: list[dict]) -> dict[tuple, list[dict]]:
    """Group matched client differences by their complete replicate condition."""
    by_run: dict[tuple, list[dict]] = {}
    for pair in pairs:
        key = _pair_replicate_key(pair)
        by_run.setdefault(key, []).append(pair)
    return by_run


def matched_difference_summary(
    pairs: list[dict],
    aggregation: str = "mean",
    *,
    include_uncertainty: bool = True,
) -> tuple[dict, dict]:
    """Summarize client-paired differences after reducing within each replicate."""
    by_run = pairs_by_replicate(pairs)
    ordered = [by_run[key] for key in sorted(by_run, key=repr)]
    run_gains = [_run_gain(run_pairs, aggregation) for run_pairs in ordered]
    stats = replicate_statistics(
        run_gains, confidence_interval=include_uncertainty
    )
    effect = {
        "estimate": stats["mean"],
        "sd": stats["sd"],
        "ci_half_width": stats["ci_half_width"],
        "ci_low": stats["ci_low"],
        "ci_high": stats["ci_high"],
        "n": stats["n"],
        "df": stats["df"],
        "pooled_within_replicate_client_sd": pooled_within_replicate_sd(
            [[pair["gain"] for pair in run_pairs] for run_pairs in ordered]
        ),
    }
    available = include_uncertainty and stats["n"] > 1
    uncertainty = {
        "method": "replicate_level_t_interval",
        "confidence_level": 0.95,
        "available": available,
        "replicate_count": stats["n"],
        "df": stats["df"],
        "reason": (
            None
            if available
            else (
                "fewer than two replicate estimates"
                if stats["n"] < 2
                else "experiment seeds are reused across run conditions"
            )
        ),
    }
    return effect, uncertainty


def _run_gain(pairs: list[dict], aggregation: str) -> float:
    if aggregation == "mean":
        return statistics.mean(pair["gain"] for pair in pairs)
    weights = [pair.get("sample_count") for pair in pairs]
    if any(weight is None for weight in weights):
        raise ConfigurationComparisonError(
            "weighted comparison requires a sample count for every client"
        )
    total = sum(weights)
    if total <= 0:
        raise ConfigurationComparisonError(
            "weighted comparison requires positive total sample count"
        )
    return sum(pair["gain"] * weight for pair, weight in zip(pairs, weights)) / total
