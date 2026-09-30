"""Client-level transfer relative to a Local reference run."""

from __future__ import annotations

import math

from rigfl.eval.metrics import canonical
from rigfl.eval.report import (
    REPLICATE_FIELDS,
    by_condition,
    condition_heading,
    pooled_within_replicate_sd,
    replicate_key,
    replicate_statistics,
    row_cells,
    row_name,
    selection_for,
)
from rigfl.eval.selection import aggregate
from rigfl.experiment.config import (
    result_data_configuration,
    result_data_configuration_id,
)

# worst-tail gain averages the lowest-gaining 10% of clients
TAIL_FRACTION = 0.10


class TransferComparisonError(ValueError):
    """Raised when algorithm and Local results cannot be paired exactly."""


def replicate_condition(record: dict) -> dict:
    """Return the data and training seeds that identify one replicate."""
    experiment = record.get("config", {}).get("experiment", {})
    missing = [field for field in REPLICATE_FIELDS if field not in experiment]
    if missing:
        raise TransferComparisonError(
            "comparison requires partition_seed, split_seed, and training_seed in every "
            f"result; missing: {', '.join(missing)}"
        )
    return dict(zip(REPLICATE_FIELDS, replicate_key(record)))


def comparison_context(record: dict) -> tuple[str, str]:
    """Return the dataset and seed-independent data-configuration identity."""
    dataset = result_data_configuration(record)["dataset"]
    return dataset, result_data_configuration_id(record)


def _index(records: list[dict], label: str) -> dict[tuple, dict]:
    indexed = {}
    for record in records:
        key = (*comparison_context(record), *replicate_condition(record).values())
        if key in indexed:
            raise TransferComparisonError(
                f"{label} contains more than one result for {key}"
            )
        indexed[key] = record
    if not indexed:
        raise TransferComparisonError(f"{label} contains no run results")
    return indexed


def _client_values(selected: dict, metric: str) -> dict[str, float]:
    values = selected.get("test", {}).get(metric, [])
    client_ids = selected.get("client_ids") or [str(i) for i in range(len(values))]
    return {
        str(client_id): value
        for client_id, value in zip(client_ids, values)
        if value is not None
    }


def _client_weights(selected: dict) -> dict[str, float | None]:
    client_ids = selected.get("client_ids") or []
    weights = selected.get("sample_counts", {}).get("test", [])
    return {str(client_id): weight for client_id, weight in zip(client_ids, weights)}


def paired_client_gains(
    algorithm_records: list[dict],
    local_records: list[dict],
    metric: str,
    *,
    view: str = "shared",
    aggregation: str = "uniform",
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
            key = (comparison_context(record), *replicate_key(record))
            indexed[key] = experiment.get("resolved_models", [])
        return indexed

    algorithm_models = models_by_replicate(algorithm_records)
    local_models = models_by_replicate(local_records)
    mismatched = []
    for key in sorted(algorithm_models.keys() & local_models.keys(), key=repr):
        # null: the algorithm doesn't use the setting
        if None in (algorithm_models[key], local_models[key]) or (
            algorithm_models[key] == local_models[key]
        ):
            continue
        (dataset, _), partition_seed, split_seed, training_seed = key
        mismatched.append(
            f"dataset={dataset!r}, partition_seed={partition_seed!r}, "
            f"split_seed={split_seed!r}, training_seed={training_seed!r} "
            f"(algorithm={list(algorithm_models[key])!r}, "
            f"local={list(local_models[key])!r})"
        )
    if mismatched:
        raise TransferComparisonError(
            "negative-transfer comparison requires identical resolved client-model "
            "assignments for each run; mismatched: " + "; ".join(mismatched)
        )

    algorithm_index = _index(algorithm_records, "algorithm")
    local_index = _index(local_records, "local")
    if set(algorithm_index) != set(local_index):
        missing_local = sorted(set(algorithm_index) - set(local_index), key=repr)
        missing_algorithm = sorted(set(local_index) - set(algorithm_index), key=repr)
        details = []
        if missing_local:
            details.append(f"missing from local: {missing_local}")
        if missing_algorithm:
            details.append(f"missing from algorithm: {missing_algorithm}")
        raise TransferComparisonError(
            "comparison requires identical data configurations and replicate "
            "conditions ("
            + "; ".join(details)
            + ")"
        )

    name = canonical(metric)
    pairs = []
    for key in sorted(algorithm_index, key=repr):
        algorithm = algorithm_index[key]
        local = local_index[key]
        experiment = algorithm["config"]["experiment"]
        if experiment.get("partition_id") != local["config"]["experiment"].get(
            "partition_id"
        ):
            raise TransferComparisonError(
                f"paired runs for {key} use different generated partitions"
            )
        replicate = replicate_condition(algorithm)
        selected_algorithm = selection_for(
            algorithm, name, view=view, aggregation=aggregation
        )
        selected_local = selection_for(local, name, view=view, aggregation=aggregation)
        algorithm_values = _client_values(selected_algorithm, name)
        local_values = _client_values(selected_local, name)
        if set(algorithm_values) != set(local_values):
            raise TransferComparisonError(
                f"client IDs differ for dataset={key[0]}, "
                f"replicate={replicate}"
            )
        expected_client_count = experiment.get("num_clients")
        if (
            expected_client_count is not None
            and len(algorithm_values) != expected_client_count
        ):
            raise TransferComparisonError(
                f"dataset={key[0]}, replicate={replicate} contains "
                f"{len(algorithm_values)} clients; expected {expected_client_count}"
            )
        weights = _client_weights(selected_algorithm)
        if weights != _client_weights(selected_local):
            raise TransferComparisonError(
                f"sample counts differ for dataset={key[0]}, "
                f"replicate={replicate}"
            )
        for client_id in sorted(algorithm_values):
            algorithm_value = algorithm_values[client_id]
            local_value = local_values[client_id]
            pairs.append(
                {
                    "dataset": key[0],
                    "data_configuration_hash": key[1],
                    "partition_id": experiment["partition_id"],
                    "replicate_condition": replicate,
                    "client_id": client_id,
                    "algorithm_value": algorithm_value,
                    "local_value": local_value,
                    "gain": algorithm_value - local_value,
                    "sample_count": weights.get(client_id),
                }
            )
    if not pairs:
        raise TransferComparisonError("comparison found no paired test values")

    return {"metric": name, "selection_view": view, "pairs": pairs}


def pairs_by_replicate(pairs: list[dict]) -> dict[tuple, list[dict]]:
    """Group matched client differences by their complete replicate condition."""
    by_run: dict[tuple, list[dict]] = {}
    for pair in pairs:
        replicate = pair["replicate_condition"]
        key = (
            pair["dataset"],
            pair["data_configuration_hash"],
            replicate["partition_seed"],
            replicate["split_seed"],
            replicate["training_seed"],
        )
        by_run.setdefault(key, []).append(pair)
    return by_run


def matched_difference_summary(
    pairs: list[dict],
    aggregation: str = "uniform",
    *,
    include_uncertainty: bool = True,
) -> tuple[dict, dict]:
    """Summarize client-paired differences after reducing within each replicate."""
    by_run = pairs_by_replicate(pairs)
    ordered = [by_run[key] for key in sorted(by_run, key=repr)]
    run_gains = [_run_gain(run_pairs, aggregation) for run_pairs in ordered]
    effect = effect_summary(run_gains, include_uncertainty=include_uncertainty)
    effect["pooled_client_sd"] = pooled_within_replicate_sd(
        [[pair["gain"] for pair in run_pairs] for run_pairs in ordered]
    )
    if include_uncertainty and effect["n"] > 1:
        reason = None
    elif effect["n"] < 2:
        reason = "fewer than two replicate estimates"
    else:
        reason = "training seeds are reused across run conditions"
    return effect, {"reason": reason}


def effect_summary(values: list[float], *, include_uncertainty: bool) -> dict:
    """Replicate statistics for one effect, with the mean named ``estimate``."""
    stats = replicate_statistics(values, confidence_interval=include_uncertainty)
    return {
        "estimate": stats["mean"],
        "sd": stats["sd"],
        "ci_half_width": stats["ci_half_width"],
        "ci_low": stats["ci_low"],
        "ci_high": stats["ci_high"],
        "n": stats["n"],
        "df": stats["df"],
    }


def _run_gain(pairs: list[dict], aggregation: str) -> float:
    weights = [pair.get("sample_count") for pair in pairs]
    if aggregation == "sample_count" and any(weight is None for weight in weights):
        raise TransferComparisonError(
            "weighted comparison requires a sample count for every client"
        )
    return aggregate([pair["gain"] for pair in pairs], weights, aggregation)


def negative_transfer_summary(
    algorithm_records: list[dict],
    local_records: list[dict],
    metric: str,
    *,
    view: str = "shared",
    aggregation: str = "uniform",
    include_uncertainty: bool = True,
) -> dict:
    """Summarize benefit and harm relative to Local across matched client runs."""
    paired = paired_client_gains(
        algorithm_records,
        local_records,
        metric,
        view=view,
        aggregation=aggregation,
    )
    pairs = paired.pop("pairs")
    effects, uncertainty = _local_relative_effects(
        pairs,
        aggregation=aggregation,
        include_uncertainty=include_uncertainty,
    )
    replicate_count = effects["mean_gain"]["n"]

    summary = {
        "available": True,
        "baseline": "local",
        **paired,
        "tail_fraction": TAIL_FRACTION,
        "pair_count": len(pairs),
        "replicate_count": replicate_count,
        "uncertainty": uncertainty,
        "paired_gains": pairs,
        "mean_gain": effects["mean_gain"],
        "benefit_rate": effects["benefit_rate"],
        "negative_transfer_rate": effects["harm_rate"],
        "negative_transfer_magnitude": effects["harm_magnitude"],
        "negative_transfer_burden": effects["harm_burden"],
        "worst_tail_gain": effects["worst_tail_gain"],
    }
    return summary


def _local_relative_effects(
    pairs: list[dict],
    *,
    aggregation: str,
    include_uncertainty: bool,
) -> tuple[dict, dict]:
    """Compute client-level gain quantities within, then across, replicates."""
    mean_gain, uncertainty = matched_difference_summary(
        pairs, aggregation, include_uncertainty=include_uncertainty
    )
    by_replicate = pairs_by_replicate(pairs)
    benefit_rates = []
    harm_rates = []
    harm_burdens = []
    harm_magnitudes = []
    worst_tail_gains = []
    harmed_client_count = 0
    affected_replicate_count = 0
    for key in sorted(by_replicate, key=repr):
        gains = [pair["gain"] for pair in by_replicate[key]]
        harms = [-gain for gain in gains if gain < 0]
        benefit_rates.append(sum(gain > 0 for gain in gains) / len(gains))
        harm_rates.append(len(harms) / len(gains))
        harm_burdens.append(sum(harms) / len(gains))
        tail_count = max(1, math.ceil(TAIL_FRACTION * len(gains)))
        worst_tail_gains.append(sum(sorted(gains)[:tail_count]) / tail_count)
        if harms:
            harm_magnitudes.append(sum(harms) / len(harms))
            harmed_client_count += len(harms)
            affected_replicate_count += 1

    conditional_magnitude = effect_summary(
        harm_magnitudes, include_uncertainty=False
    )
    conditional_magnitude.update(
        {
            "harmed_client_count": harmed_client_count,
            "affected_replicate_count": affected_replicate_count,
            "conditional_on_harm": True,
        }
    )
    return {
        "mean_gain": mean_gain,
        "benefit_rate": _clip_rate(effect_summary(
            benefit_rates, include_uncertainty=include_uncertainty
        )),
        "harm_rate": _clip_rate(effect_summary(
            harm_rates, include_uncertainty=include_uncertainty
        )),
        "harm_magnitude": conditional_magnitude,
        "harm_burden": effect_summary(
            harm_burdens, include_uncertainty=include_uncertainty
        ),
        "worst_tail_gain": effect_summary(
            worst_tail_gains, include_uncertainty=include_uncertainty
        ),
    }, uncertainty


def _clip_rate(effect: dict) -> dict:
    """Keep a t interval for a client fraction inside [0, 1]."""
    for bound in ("ci_low", "ci_high"):
        if effect[bound] is not None:
            effect[bound] = min(1.0, max(0.0, effect[bound]))
    return effect


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
    """Format one client-level gain summary per algorithm, grouped by condition."""
    compared = {
        label: row for label, row in rows.items()
        if row.get("negative_transfer") is not None
    }
    if not compared:
        return ""
    summaries = [row["negative_transfer"] for row in compared.values()]
    computed = [summary for summary in summaries if summary.get("available", True)]
    tail_label = (
        f"worst-{computed[0]['tail_fraction'] * 100:g}% gain"
        if computed
        else "worst-tail gain"
    )
    out = []
    for condition, group in by_condition(compared).items():
        if out:
            out.append("")
        out += condition_heading(condition)
        out += [
            f"| algorithm | configuration | round selection | pairs | mean gain | pooled client-difference SD | benefit rate | NTR | NTM | NTB | {tail_label} |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        unavailable = []
        for label, row in group.items():
            summary = row["negative_transfer"]
            algorithm, configuration = row_cells(label, row)
            if not summary.get("available", True):
                out.append(
                    f"| {algorithm} | {configuration} | — | — | — | — | — | — | — | — | — |"
                )
                unavailable.append((row_name(label, row), summary["reason"]))
                continue
            marker = " §" if summary.get("uncertainty", {}).get("reason") == (
                "training seeds are reused across run conditions"
            ) else ""
            out.append(
                "| "
                + " | ".join(
                    [
                        algorithm + marker,
                        configuration,
                        summary["selection_view"],
                        str(summary["pair_count"]),
                        _interval(summary["mean_gain"]),
                        (
                            f"{summary['mean_gain']['pooled_client_sd']:.3f}"
                            if summary["mean_gain"].get("pooled_client_sd") is not None
                            else "—"
                        ),
                        _interval(summary["benefit_rate"], percent=True),
                        _interval(summary["negative_transfer_rate"], percent=True),
                        _interval(summary["negative_transfer_magnitude"]),
                        _interval(summary["negative_transfer_burden"]),
                        _interval(summary["worst_tail_gain"]),
                    ]
                )
                + " |"
            )
        local = next(iter(group.values()))["negative_transfer"].get("local_configuration")
        if local is not None:
            chosen = ["local" if local == "—" else f"local {local}"]
            chosen += [row_name(label, row) for label, row in group.items()]
            out.extend([
                "",
                "Client-level gains use each method's best validation configuration: "
                + " · ".join(chosen),
            ])
        if unavailable:
            out.append("")
            out.extend(
                f"{name}: negative-transfer comparison unavailable — {reason}."
                for name, reason in unavailable
            )
    suppressed = any(
        summary.get("uncertainty", {}).get("reason")
        == "training seeds are reused across run conditions"
        for summary in computed
    )
    if computed and not suppressed:
        out.extend([
            "",
            "Intervals are two-sided replicate-level t intervals and require "
            "at least two independent replicate estimates. NTM is conditional "
            "on observed harm and is reported without an ordinary replicate CI.",
        ])
    if suppressed:
        out.extend([
            "",
            (
                "§ training seeds are reused across run conditions; point "
                "estimates are shown without confidence intervals."
            ),
        ])
    return "\n".join(out)
