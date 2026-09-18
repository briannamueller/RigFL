"""Client-level transfer relative to a Local reference run."""

from __future__ import annotations

import json
import math

from rigfl.eval.comparison import (
    ConfigurationComparisonError,
    comparison_context,
    paired_client_differences,
    summarize_paired_effects,
)
from rigfl.experiment.config import algorithm_identity


class TransferComparisonError(ValueError):
    """Raised when algorithm and Local results cannot be paired exactly."""


def _local_configuration(records: list[dict]) -> dict:
    configurations = {
        json.dumps(
            algorithm_identity(record.get("config", {}).get("algorithm", {})),
            sort_keys=True,
        )
        for record in records
    }
    if len(configurations) != 1:
        raise TransferComparisonError(
            "negative-transfer comparison requires one Local configuration "
            "per experiment"
        )
    return json.loads(next(iter(configurations)))


def paired_client_gains(
    algorithm_records: list[dict],
    local_records: list[dict],
    metric: str,
    *,
    view: str = "global",
    aggregation: str = "mean",
    tie_break: str = "earliest",
) -> dict:
    """Pair validation-selected test results by replicate and client identifier."""
    if not local_records:
        raise TransferComparisonError(
            "negative-transfer comparison requires matching Local results"
        )

    def models_by_replicate(records: list[dict]) -> dict[tuple, tuple]:
        indexed = {}
        for record in records:
            experiment = record.get("config", {}).get("experiment", {})
            key = (
                comparison_context(record),
                experiment.get("partition_seed"),
                experiment.get("split_seed"),
                experiment.get("seed"),
            )
            indexed[key] = tuple(experiment.get("resolved_models", []))
        return indexed

    algorithm_models = models_by_replicate(algorithm_records)
    local_models = models_by_replicate(local_records)
    mismatched = []
    for key in sorted(algorithm_models.keys() & local_models.keys(), key=repr):
        if algorithm_models[key] == local_models[key]:
            continue
        (dataset, _), partition_seed, split_seed, experiment_seed = key
        mismatched.append(
            f"dataset={dataset!r}, partition_seed={partition_seed!r}, "
            f"split_seed={split_seed!r}, experiment_seed={experiment_seed!r} "
            f"(algorithm={list(algorithm_models[key])!r}, "
            f"local={list(local_models[key])!r})"
        )
    if mismatched:
        raise TransferComparisonError(
            "negative-transfer comparison requires identical resolved client-model "
            "assignments for each run; mismatched: " + "; ".join(mismatched)
        )
    local_configuration = _local_configuration(local_records)
    try:
        paired = paired_client_differences(
            algorithm_records,
            local_records,
            metric,
            left_label="algorithm",
            right_label="local",
            view=view,
            aggregation=aggregation,
            tie_break=tie_break,
            evaluation_split="test",
            align_selection_views=True,
        )
    except ConfigurationComparisonError as error:
        raise TransferComparisonError(str(error)) from error
    views = paired.pop("selection_views")
    pairs = [
        {
            "dataset": pair["dataset"],
            "data_configuration_hash": pair["data_configuration_hash"],
            "partition_id": pair["partition_id"],
            "replicate_condition": pair["replicate_condition"],
            "client_id": pair["client_id"],
            "algorithm_value": pair["left_value"],
            "local_value": pair["right_value"],
            "gain": pair["gain"],
            "sample_count": pair["sample_count"],
        }
        for pair in paired.pop("pairs")
    ]

    return {
        **paired,
        "selection_view": views["algorithm"],
        "local_configuration": local_configuration,
        "source_files": {
            "algorithm": [
                record["_source_file"]
                for record in algorithm_records
                if record.get("_source_file")
            ],
            "local": [
                record["_source_file"]
                for record in local_records
                if record.get("_source_file")
            ],
        },
        "pairs": pairs,
    }


def negative_transfer_summary(
    algorithm_records: list[dict],
    local_records: list[dict],
    metric: str,
    *,
    view: str = "global",
    aggregation: str = "mean",
    tie_break: str = "earliest",
    threshold: float = 0.0,
    profile_thresholds: tuple[float, ...] = (),
    tail_fraction: float = 0.10,
    include_uncertainty: bool = True,
) -> dict:
    """Summarize benefit and harm relative to Local across matched client runs."""
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("negative-transfer threshold must be finite and nonnegative")
    if not 0 < tail_fraction <= 1:
        raise ValueError("negative-transfer tail fraction must be in (0, 1]")
    thresholds = sorted({threshold, *profile_thresholds})
    if any(not math.isfinite(value) or value < 0 for value in thresholds):
        raise ValueError(
            "negative-transfer profile thresholds must be finite and nonnegative"
        )

    paired = paired_client_gains(
        algorithm_records,
        local_records,
        metric,
        view=view,
        aggregation=aggregation,
        tie_break=tie_break,
    )
    pairs = paired.pop("pairs")
    effects, uncertainty = summarize_paired_effects(
        pairs,
        threshold,
        tail_fraction,
        aggregation,
        include_uncertainty=include_uncertainty,
    )
    replicate_count = len(
        {
            tuple(pair["replicate_condition"].values())
            for pair in pairs
        }
    )

    summary = {
        "schema_version": 1,
        "available": True,
        "baseline": "local",
        "comparison_unit": "client_replicate_pair",
        "weighting": "equal_client_replicate_pairs",
        "gain_definition": (
            "algorithm_value - local_value"
            if paired["direction"] == "maximize"
            else "local_value - algorithm_value"
        ),
        **paired,
        "threshold": threshold,
        "tail_fraction": tail_fraction,
        "pair_count": len(pairs),
        "replicate_count": replicate_count,
        "uncertainty": uncertainty,
        "paired_gains": pairs,
        "benefit_rate": effects["benefit_rate"],
        "negative_transfer_rate": effects["harm_rate"],
        "negative_transfer_magnitude": effects["harm_magnitude"],
        "negative_transfer_burden": effects["harm_burden"],
        "worst_tail_gain": effects["worst_tail_gain"],
    }
    summary["threshold_profile"] = []
    for value in thresholds:
        threshold_effects, _ = summarize_paired_effects(
            pairs,
            value,
            tail_fraction,
            aggregation,
            include_uncertainty=include_uncertainty,
        )
        summary["threshold_profile"].append(
            {
                "threshold": value,
                "negative_transfer_rate": threshold_effects["harm_rate"],
            }
        )
    return summary


def _interval(value: dict, *, percent: bool = False) -> str:
    estimate = value.get("estimate")
    if estimate is None:
        return "—"
    scale = 100 if percent else 1
    digits = 1 if percent else 3
    suffix = "%" if percent else ""
    rendered = f"{estimate * scale:.{digits}f}{suffix}"
    low, high = value.get("ci_low"), value.get("ci_high")
    if low is None or high is None:
        return rendered
    return (
        f"{rendered} [{low * scale:.{digits}f}{suffix}, "
        f"{high * scale:.{digits}f}{suffix}]"
    )


def format_negative_transfer_table(rows: dict) -> str:
    """Format one Local-relative transfer summary per algorithm."""
    available = [
        (label, row["negative_transfer"])
        for label, row in rows.items()
        if row.get("negative_transfer") is not None
    ]
    if not available:
        return ""
    computed = [summary for _, summary in available if summary.get("available", True)]
    tail_label = (
        f"worst-{computed[0]['tail_fraction'] * 100:g}% gain"
        if computed
        else "worst-tail gain"
    )
    out = [
        f"| algorithm | selection | pairs | benefit rate | NTR | NTM | NTB | {tail_label} |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in available:
        if not summary.get("available", True):
            out.append(f"| {label} | — | — | — | — | — | — | — |")
            continue
        marker = " §" if summary.get("uncertainty", {}).get("reason") == (
            "experiment seeds are reused across run conditions"
        ) else ""
        out.append(
            "| "
            + " | ".join(
                [
                    label + marker,
                    summary["selection_view"],
                    str(summary["pair_count"]),
                    _interval(summary["benefit_rate"], percent=True),
                    _interval(summary["negative_transfer_rate"], percent=True),
                    _interval(summary["negative_transfer_magnitude"]),
                    _interval(summary["negative_transfer_burden"]),
                    _interval(summary["worst_tail_gain"]),
                ]
            )
            + " |"
        )
    suppressed = any(
        summary.get("uncertainty", {}).get("reason")
        == "experiment seeds are reused across run conditions"
        for summary in computed
    )
    if computed:
        threshold = computed[0]["threshold"]
        note = f"Performance margin: {threshold:g} metric units."
        if not suppressed:
            note += (
                " Intervals resample replicate conditions and clients; they "
                "require more than one of each."
            )
        out.extend(["", note])
    if suppressed:
        out.extend([
            "",
            (
                "§ experiment seeds are reused across run conditions; point "
                "estimates are shown without confidence intervals."
            ),
        ])
    unavailable = [
        (label, summary["reason"])
        for label, summary in available
        if not summary.get("available", True)
    ]
    if unavailable:
        out.extend(
            [
                "",
                *[
                    f"{label}: negative-transfer comparison unavailable — {reason}."
                    for label, reason in unavailable
                ],
            ]
        )
    return "\n".join(out)


def format_negative_transfer_profile(rows: dict) -> str:
    """Format negative-transfer rates over requested practical thresholds."""
    available = [
        (label, row["negative_transfer"])
        for label, row in rows.items()
        if row.get("negative_transfer") is not None
        and row["negative_transfer"].get("available", True)
        and len(row["negative_transfer"]["threshold_profile"]) > 1
    ]
    if not available:
        return ""
    thresholds = [item["threshold"] for item in available[0][1]["threshold_profile"]]
    out = [
        "| algorithm | "
        + " | ".join(f"NTR (δ={threshold:g})" for threshold in thresholds)
        + " |",
        "|---|" + "---:|" * len(thresholds),
    ]
    for label, summary in available:
        marker = " §" if summary.get("uncertainty", {}).get("reason") == (
            "experiment seeds are reused across run conditions"
        ) else ""
        rates = {
            item["threshold"]: item["negative_transfer_rate"]
            for item in summary["threshold_profile"]
        }
        out.append(
            "| "
            + " | ".join(
                [
                    label + marker,
                    *[_interval(rates[value], percent=True) for value in thresholds],
                ]
            )
            + " |"
        )
    if any(
        summary.get("uncertainty", {}).get("reason")
        == "experiment seeds are reused across run conditions"
        for _, summary in available
    ):
        out.extend([
            "",
            (
                "§ experiment seeds are reused across run conditions; point "
                "estimates are shown without confidence intervals."
            ),
        ])
    return "\n".join(out)
