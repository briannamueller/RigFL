"""Paired comparison of frozen experiment configurations."""

from __future__ import annotations

import itertools
import json
import math
import random
import statistics

from rigfl.eval.metrics import canonical, direction_of
from rigfl.eval.report import independent_replicates, mean_ci, selection_for
from rigfl.experiment.config import (
    algorithm_identity,
    fingerprint,
    result_data_configuration,
    result_data_configuration_id,
)

BOOTSTRAP_REPLICATES = 2000
RANDOMIZATION_REPLICATES = 10000
RANDOM_SEED = 0
MIN_RANDOMIZATION_RUNS = 6


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


_ENVIRONMENT_FIELDS = {
    "data_dir",
    "dataset_config",
    "device",
    "out_dir",
    "quiet",
    "wandb",
    "wandb_project",
}


def frozen_configuration(record: dict) -> dict:
    """Return the seed- and environment-independent configuration of one run."""
    experiment = dict(record.get("config", {}).get("experiment", {}))
    for field in ("partition_id", "partition_seed", "split_seed", "seed"):
        experiment.pop(field, None)
    for field in _ENVIRONMENT_FIELDS:
        experiment.pop(field, None)
    return {
        "algorithm": record.get("algorithm"),
        "experiment": experiment,
        "algorithm_config": algorithm_identity(
            record.get("config", {}).get("algorithm", {}),
            algorithm=record.get("algorithm"),
        ),
    }


def _configuration_summary(records: list[dict], label: str) -> dict:
    by_context: dict[tuple[str, str], set[str]] = {}
    identities: dict[str, dict] = {}
    for record in records:
        context = comparison_context(record)
        identity = frozen_configuration(record)
        encoded = json.dumps(identity, sort_keys=True)
        by_context.setdefault(context, set()).add(encoded)
        identities[encoded] = identity
    varying = [context for context, values in by_context.items() if len(values) != 1]
    if varying:
        raise ConfigurationComparisonError(
            f"{label} is not frozen within data configuration(s): {varying}"
        )
    configs = [
        identities[next(iter(by_context[context]))] for context in sorted(by_context)
    ]
    return {
        "label": label,
        "configuration_hash": fingerprint({"configurations": configs}),
        "configurations": configs,
        "source_files": sorted(
            record["_source_file"] for record in records if record.get("_source_file")
        ),
    }


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


def _point_metrics(
    pairs: list[dict], threshold: float, tail_fraction: float, aggregation: str
) -> dict:
    gains = [pair["gain"] for pair in pairs]
    by_run: dict[tuple, list[dict]] = {}
    for pair in pairs:
        key = _pair_replicate_key(pair)
        by_run.setdefault(key, []).append(pair)
    run_gains = [
        _run_gain(by_run[key], aggregation) for key in sorted(by_run, key=repr)
    ]
    harm = [-gain for gain in gains if gain < -threshold]
    benefit_count = sum(gain > threshold for gain in gains)
    count = len(gains)
    tail_count = max(1, math.ceil(tail_fraction * count))
    return {
        "mean_gain": statistics.mean(run_gains),
        "median_gain": statistics.median(run_gains),
        "benefit_rate": benefit_count / count,
        "harm_rate": len(harm) / count,
        "harm_magnitude": statistics.mean(harm) if harm else None,
        "harm_burden": sum(harm) / count,
        "worst_tail_gain": statistics.mean(sorted(gains)[:tail_count]),
    }


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _estimate(value, bootstrap: list[float | None], available: bool) -> dict:
    usable = [item for item in bootstrap if item is not None]
    return {
        "estimate": value,
        "ci_low": _percentile(usable, 0.025) if available else None,
        "ci_high": _percentile(usable, 0.975) if available else None,
        "ci_replicates": len(usable) if available else 0,
    }


def _hierarchical_bootstrap(
    pairs: list[dict], threshold: float, tail_fraction: float, aggregation: str
) -> tuple[dict, dict]:
    by_replicate: dict[tuple, list[dict]] = {}
    for pair in pairs:
        by_replicate.setdefault(_pair_replicate_key(pair), []).append(pair)
    replicates = sorted(by_replicate, key=repr)
    available = len(replicates) > 1 and all(
        len(by_replicate[replicate]) > 1 for replicate in replicates
    )
    sampled = {
        key: [] for key in _point_metrics(pairs, threshold, tail_fraction, aggregation)
    }
    if available:
        rng = random.Random(RANDOM_SEED)
        for _ in range(BOOTSTRAP_REPLICATES):
            replicate_sample = [rng.choice(replicates) for _ in replicates]
            resampled = []
            for replicate_index, replicate in enumerate(replicate_sample):
                source_pairs = by_replicate[replicate]
                client_sample = [
                    rng.choice(source_pairs) for _ in range(len(source_pairs))
                ]
                for client_index, sampled_pair in enumerate(client_sample):
                    pair = dict(sampled_pair)
                    pair["replicate_condition"] = {
                        "partition_seed": replicate_index,
                        "split_seed": replicate_index,
                        "experiment_seed": replicate_index,
                    }
                    pair["client_id"] = str(client_index)
                    resampled.append(pair)
            values = _point_metrics(resampled, threshold, tail_fraction, aggregation)
            for key, value in values.items():
                sampled[key].append(value)
    return sampled, {
        "method": "hierarchical_cluster_percentile_bootstrap",
        "clusters": ["replicate_condition", "client_within_replicate"],
        "confidence_level": 0.95,
        "replicates": BOOTSTRAP_REPLICATES,
        "random_seed": RANDOM_SEED,
        "available": available,
    }


def summarize_paired_effects(
    pairs: list[dict],
    practical_threshold: float,
    tail_fraction: float,
    aggregation: str = "mean",
    *,
    include_uncertainty: bool = True,
) -> tuple[dict, dict]:
    """Summarize paired gains across replicate conditions and clients."""
    point = _point_metrics(pairs, practical_threshold, tail_fraction, aggregation)
    if include_uncertainty:
        bootstrap, uncertainty = _hierarchical_bootstrap(
            pairs, practical_threshold, tail_fraction, aggregation
        )
    else:
        bootstrap = {key: [] for key in point}
        uncertainty = {
            "method": "hierarchical_cluster_percentile_bootstrap",
            "clusters": ["replicate_condition", "client_within_replicate"],
            "confidence_level": 0.95,
            "replicates": 0,
            "random_seed": RANDOM_SEED,
            "available": False,
            "reason": "experiment seeds are reused across run conditions",
        }
    effects = {
        key: _estimate(value, bootstrap[key], uncertainty["available"])
        for key, value in point.items()
    }
    return effects, uncertainty


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


def _randomization_test(pairs: list[dict], aggregation: str) -> dict:
    by_run: dict[tuple, list[dict]] = {}
    for pair in pairs:
        key = _pair_replicate_key(pair)
        by_run.setdefault(key, []).append(pair)
    values = [
        _run_gain(by_run[key], aggregation) for key in sorted(by_run, key=repr)
    ]
    if len(values) < MIN_RANDOMIZATION_RUNS:
        return {
            "available": False,
            "reason": (f"at least {MIN_RANDOMIZATION_RUNS} paired runs are required"),
            "run_count": len(values),
        }
    observed = abs(statistics.mean(values))
    if len(values) <= 20:
        null = [
            abs(statistics.mean(sign * value for sign, value in zip(signs, values)))
            for signs in itertools.product((-1, 1), repeat=len(values))
        ]
        p_value = sum(value >= observed - 1e-15 for value in null) / len(null)
        method = "exact_paired_sign_flip"
    else:
        rng = random.Random(RANDOM_SEED)
        exceed = 0
        for _ in range(RANDOMIZATION_REPLICATES):
            value = abs(statistics.mean(rng.choice((-1, 1)) * item for item in values))
            exceed += value >= observed - 1e-15
        p_value = (exceed + 1) / (RANDOMIZATION_REPLICATES + 1)
        method = "monte_carlo_paired_sign_flip"
    return {
        "available": True,
        "method": method,
        "experimental_unit": "replicate_condition",
        "run_count": len(values),
        "p_value": p_value,
        "adjusted_p_value": None,
    }


def _resource_value(record: dict, name: str):
    resources = record.get("resources", {})
    if name == "communication_bytes":
        return resources.get("observed", {}).get("communication_bytes", {}).get("total")
    return resources.get("attributed_training", {}).get(name)


def _resource_differences(left: dict, right: dict) -> dict:
    output = {}
    for name in ("communication_bytes", "flops", "wall_seconds"):
        left_values, right_values, differences = [], [], []
        hardware_signatures = set()
        for key in sorted(left, key=repr):
            left_value = _resource_value(left[key], name)
            right_value = _resource_value(right[key], name)
            if left_value is None or right_value is None:
                break
            if name == "wall_seconds":
                left_resource = left[key].get("resources", {})
                right_resource = right[key].get("resources", {})
                if not left_resource.get("attributed_training", {}).get(
                    "wall_seconds_comparable"
                ) or not right_resource.get("attributed_training", {}).get(
                    "wall_seconds_comparable"
                ):
                    break
                left_hardware = (
                    left_resource.get("measurement", {})
                    .get("timing", {})
                    .get("hardware")
                )
                right_hardware = (
                    right_resource.get("measurement", {})
                    .get("timing", {})
                    .get("hardware")
                )
                if left_hardware != right_hardware:
                    break
                hardware_signatures.add(json.dumps(left_hardware, sort_keys=True))
            left_values.append(left_value)
            right_values.append(right_value)
            differences.append(left_value - right_value)
        else:
            if name == "wall_seconds" and len(hardware_signatures) != 1:
                output[name] = {"available": False}
                continue
            mean_difference, ci = mean_ci(differences)
            output[name] = {
                "available": True,
                "left_mean": statistics.mean(left_values),
                "right_mean": statistics.mean(right_values),
                "difference": mean_difference,
                "difference_ci": ci,
                "run_count": len(differences),
            }
            continue
        output[name] = {"available": False}
    return output


def compare_configurations(
    left_records: list[dict],
    right_records: list[dict],
    metric: str,
    *,
    left_label: str,
    right_label: str,
    view: str = "global",
    aggregation: str = "mean",
    tie_break: str = "earliest",
    practical_threshold: float = 0.0,
    tail_fraction: float = 0.10,
    evaluation_split: str = "test",
) -> dict:
    """Compare two named, frozen configurations on matched results."""
    if not math.isfinite(practical_threshold) or practical_threshold < 0:
        raise ValueError("practical threshold must be finite and nonnegative")
    if not 0 < tail_fraction <= 1:
        raise ValueError("tail fraction must be greater than 0 and at most 1")
    if not independent_replicates(left_records) or not independent_replicates(
        right_records
    ):
        raise ConfigurationComparisonError(
            "final statistical comparison requires a distinct experiment_seed "
            "for every replicate condition; analyze crossed seed sweeps with "
            "python -m rigfl.experiment.variance"
        )
    contexts = {
        comparison_context(record) for record in [*left_records, *right_records]
    }
    if len(contexts) != 1:
        raise ConfigurationComparisonError(
            "each statistical contrast must cover one dataset and data "
            "configuration; report separate contrasts for separate data conditions"
        )
    left_config = _configuration_summary(left_records, left_label)
    right_config = _configuration_summary(right_records, right_label)
    paired = paired_client_differences(
        left_records,
        right_records,
        metric,
        left_label=left_label,
        right_label=right_label,
        view=view,
        aggregation=aggregation,
        tie_break=tie_break,
        evaluation_split=evaluation_split,
    )
    pairs = paired.pop("pairs")
    estimates, uncertainty = summarize_paired_effects(
        pairs, practical_threshold, tail_fraction, aggregation
    )
    interval = estimates["mean_gain"]
    decision = "inconclusive"
    if interval["ci_low"] is not None:
        if interval["ci_low"] > practical_threshold:
            decision = "left_better"
        elif interval["ci_high"] < -practical_threshold:
            decision = "right_better"
        elif (
            interval["ci_low"] >= -practical_threshold
            and interval["ci_high"] <= practical_threshold
        ):
            decision = "practically_equivalent"
    left_index = _index(left_records, left_label)
    right_index = _index(right_records, right_label)
    run_level = []
    for key in sorted(left_index, key=repr):
        record = left_index[key]
        replicate = replicate_condition(record)
        run_gains = [
            pair
            for pair in pairs
            if _pair_replicate_key(pair) == key
        ]
        run_level.append(
            {
                "dataset": key[0],
                "partition_id": record["config"]["experiment"]["partition_id"],
                "replicate_condition": replicate,
                "gain": _run_gain(run_gains, aggregation),
            }
        )
    context = next(iter(contexts))
    data_configuration = result_data_configuration(left_records[0])
    return {
        "schema_version": 1,
        "kind": "rigfl.statistical_comparison",
        "data_condition": {
            "dataset": context[0],
            "configuration_hash": context[1],
            "configuration": data_configuration,
            "partition_ids": sorted(
                {
                    record["config"]["experiment"]["partition_id"]
                    for record in [*left_records, *right_records]
                }
            ),
        },
        "left": left_config,
        "right": right_config,
        "protocol": {
            **paired,
            "selection_split": "validation",
            "evaluation_split": evaluation_split,
            "client_aggregation": aggregation,
            "round_tie_break": tie_break,
            "practical_threshold": practical_threshold,
            "tail_fraction": tail_fraction,
        },
        "coverage": {
            "pair_count": len(pairs),
            "run_count": len(left_index),
            "replicate_count": len({_pair_replicate_key(pair) for pair in pairs}),
            "client_count": len({pair["client_id"] for pair in pairs}),
        },
        "effects": estimates,
        "uncertainty": uncertainty,
        "practical_conclusion": decision,
        "randomization_test": _randomization_test(pairs, aggregation),
        "resources": _resource_differences(left_index, right_index),
        "run_level_differences": run_level,
        "paired_differences": pairs,
    }


def apply_holm(comparisons: list[dict]) -> None:
    """Add Holm-adjusted p-values across the available declared contrasts."""
    available = [
        (index, comparison["randomization_test"]["p_value"])
        for index, comparison in enumerate(comparisons)
        if comparison["randomization_test"].get("available")
    ]
    ordered = sorted(available, key=lambda item: item[1])
    running = 0.0
    count = len(ordered)
    for rank, (index, p_value) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * p_value)
        running = max(running, adjusted)
        comparisons[index]["randomization_test"]["adjusted_p_value"] = running


def format_comparisons(comparisons: list[dict]) -> str:
    """Format the central effect and uncertainty for declared contrasts."""
    out = [
        (
            "| data | contrast | selection | mean gain | 95% CI | practical conclusion "
            "| p | Holm p |"
        ),
        "|---|---|---|---:|---:|---|---:|---:|",
    ]
    for comparison in comparisons:
        left = comparison["left"]["label"]
        right = comparison["right"]["label"]
        mean = comparison["effects"]["mean_gain"]
        interval = (
            "—"
            if mean["ci_low"] is None
            else f"[{mean['ci_low']:.4f}, {mean['ci_high']:.4f}]"
        )
        test = comparison["randomization_test"]
        p_value = test.get("p_value")
        adjusted = test.get("adjusted_p_value")
        views = comparison["protocol"]["selection_views"]
        selection = views[left]
        if views[left] != views[right]:
            selection = f"{left}: {views[left]}; {right}: {views[right]}"
        out.append(
            "| "
            + " | ".join(
                [
                    (
                        f"{comparison['data_condition']['dataset']} / "
                        f"{comparison['data_condition']['configuration_hash']}"
                    ),
                    f"{left} − {right}",
                    selection,
                    f"{mean['estimate']:.4f}",
                    interval,
                    _conclusion(comparison),
                    "—" if p_value is None else f"{p_value:.4g}",
                    "—" if adjusted is None else f"{adjusted:.4g}",
                ]
            )
            + " |"
        )
    resources = [
        "",
        "### Resource differences",
        "| data | contrast | communication bytes | FLOPs | wall seconds |",
        "|---|---|---:|---:|---:|",
    ]
    any_resources = False
    for comparison in comparisons:
        left = comparison["left"]["label"]
        right = comparison["right"]["label"]
        resource_values = []
        for name in ("communication_bytes", "flops", "wall_seconds"):
            value = comparison["resources"].get(name, {})
            resource_values.append(
                _format_optional(value.get("difference"))
                if value.get("available")
                else "—"
            )
            any_resources = any_resources or bool(value.get("available"))
        resources.append(
            "| "
            + " | ".join(
                [
                    (
                        f"{comparison['data_condition']['dataset']} / "
                        f"{comparison['data_condition']['configuration_hash']}"
                    ),
                    f"{left} − {right}",
                    *resource_values,
                ]
            )
            + " |"
        )
    if any_resources:
        out.extend(resources)
    return "\n".join(out)


def _format_optional(value) -> str:
    return "—" if value is None else f"{value:.4f}"


def _conclusion(comparison: dict) -> str:
    conclusion = comparison["practical_conclusion"]
    if conclusion == "left_better":
        return f"{comparison['left']['label']} better"
    if conclusion == "right_better":
        return f"{comparison['right']['label']} better"
    return conclusion.replace("_", " ")
