"""Compare named frozen configurations using matched test results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from rigfl.eval.comparison import (
    ConfigurationComparisonError,
    compare_configurations,
    format_comparisons,
    frozen_configuration,
)
from rigfl.eval.metrics import direction_of
from rigfl.eval.report import summarize
from rigfl.eval.selection import resolve_metric
from rigfl.experiment.artifacts import (
    ResultValidationError,
    is_run_result,
    read_json,
    validate_run_record,
)
from rigfl.experiment.reporting_config import (
    ReportingConfigError,
    apply_named_filter,
    human_configuration,
    load_reporting_config,
    record_value,
    selection_defaults,
    write_csv,
)


def _load_results(
    path: Path,
    *,
    required: bool = True,
    ignore_invalid: bool = False,
    invalid: list[str] | None = None,
) -> list[dict]:
    paths = sorted(path.glob("*.json")) if path.is_dir() else [path]
    records = []
    problems = []
    for candidate in paths:
        try:
            record = read_json(candidate)
            if not is_run_result(record):
                continue
            validate_run_record(record, path=candidate)
        except ResultValidationError as error:
            problems.append(f"{candidate}: {error.reason}")
            continue
        record["_source_file"] = str(candidate)
        records.append(record)
    if problems and not ignore_invalid:
        raise ConfigurationComparisonError(
            "invalid result files:\n  " + "\n  ".join(problems)
        )
    if invalid is not None:
        invalid.extend(problems)
    if required and not records:
        raise ConfigurationComparisonError(f"no completed run results found at {path}")
    return records


def main(argv: list[str] | None = None, *, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=prog, description="Run one named matched configuration comparison."
    )
    parser.add_argument("results", nargs="?", default="results/runs")
    parser.add_argument("--config", required=True, help="reporting YAML")
    parser.add_argument("--comparison", required=True, help="named YAML comparison")
    parser.add_argument(
        "--save", nargs="?", const="", metavar="PATH",
        help="save the comparison CSV; optionally set its path",
    )
    args = parser.parse_args(argv)

    try:
        results_dir = Path(args.results)
        reporting = load_reporting_config(args.config)
        defaults = selection_defaults(reporting)
        definition = _comparison_definition(reporting, args.comparison)
        metric = resolve_metric(defaults["metric"], source="defaults.metric")
        invalid = []
        loaded = _load_results(
            results_dir, ignore_invalid=True, invalid=invalid
        )
        filtered, resolved_filter = apply_named_filter(
            loaded, reporting, definition["filter"]
        )
        selected = []
        selection_details = []
        for value in definition["values"]:
            candidates = [
                record
                for record in filtered
                if record_value(record, definition["field"], object()) == value
            ]
            records, details = _select_configuration(
                candidates,
                definition["select"],
                metric,
                view=defaults["view"],
                aggregation=defaults["aggregation"],
                tie_break=defaults["tie_break"],
                side=f"{definition['field']}={value}",
            )
            selected.append(records)
            selection_details.append(details)
        if definition["select"] == "exact":
            _check_exact_confounding(
                selected[0][0], selected[1][0], definition["field"]
            )
        labels = [f"{definition['field']}={value}" for value in definition["values"]]
        comparison = compare_configurations(
            selected[0],
            selected[1],
            metric,
            left_label=labels[0],
            right_label=labels[1],
            view=defaults["view"],
            aggregation=defaults["aggregation"],
            tie_break=defaults["tie_break"],
            practical_threshold=definition["practical_margin"],
        )
    except (ConfigurationComparisonError, ReportingConfigError, ValueError) as error:
        raise SystemExit(f"cannot compare configurations: {error}") from error

    print(
        f"Comparison {args.comparison!r}; filter {definition['filter']!r}: "
        f"{resolved_filter}; {len(filtered)} completed replicate(s)."
    )
    if invalid:
        print(
            f"Warning: excluded {len(invalid)} invalid result file(s): "
            + "; ".join(invalid)
        )
    for label, details in zip(labels, selection_details):
        print(
            f"{label}: {details['candidate_count']} candidate configuration(s); "
            f"selected {details['selected_label']}; validation "
            f"{metric}={details['validation_score']:.6g}; "
            f"replicates={details['replicate_count']}"
            + ("; deterministic tie-break applied" if details["tie"] else "")
            + "."
        )
    print(format_comparisons([comparison]))
    if args.save is not None:
        path = (
            Path(args.save)
            if args.save
            else _default_comparison_path(results_dir, definition)
        )
        write_csv(
            path,
            [_comparison_csv_row(args.comparison, definition, comparison)],
        )
        print(f"wrote {path}")


def _comparison_definition(config: dict, name: str) -> dict:
    comparisons = config.get("comparisons", {})
    if name not in comparisons:
        raise ReportingConfigError(
            f"unknown comparison {name!r}; available comparisons: "
            + (", ".join(sorted(comparisons)) or "none")
        )
    definition = comparisons[name]
    if not isinstance(definition, dict):
        raise ReportingConfigError(f"comparison {name!r} must be a mapping")
    required = {"filter", "field", "values", "select", "practical_margin"}
    missing = sorted(required - set(definition))
    unknown = sorted(set(definition) - required)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise ReportingConfigError(
            f"comparison {name!r} is invalid ({'; '.join(details)})"
        )
    values = definition["values"]
    if not isinstance(values, list) or len(values) != 2 or values[0] == values[1]:
        raise ReportingConfigError(
            f"comparison {name!r} values must contain exactly two distinct entries"
        )
    if definition["select"] not in {"exact", "best_validation"}:
        raise ReportingConfigError(
            f"comparison {name!r} select must be exact or best_validation"
        )
    margin = definition["practical_margin"]
    if not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
        raise ReportingConfigError(
            f"comparison {name!r} practical_margin must be finite and nonnegative"
        )
    return definition


def _configuration_groups(records: list[dict]) -> list[list[dict]]:
    grouped = {}
    for record in records:
        key = json.dumps(frozen_configuration(record), sort_keys=True)
        grouped.setdefault(key, []).append(record)
    return [grouped[key] for key in sorted(grouped)]


def _select_configuration(
    records: list[dict],
    mode: str,
    metric: str,
    *,
    view: str,
    aggregation: str,
    tie_break: str,
    side: str,
) -> tuple[list[dict], dict]:
    groups = _configuration_groups(records)
    if not groups:
        raise ReportingConfigError(f"comparison side {side!r} has no candidates")
    scored = []
    for group in groups:
        summary = summarize(
            group,
            metric,
            view=view,
            aggregation=aggregation,
            tie_break=tie_break,
        )
        if summary["val_mean"] is None:
            raise ReportingConfigError(
                f"comparison side {side!r} has a candidate without validation {metric}"
            )
        label = _human_label(group[0])
        scored.append((summary["val_mean"], label, group))
    if mode == "exact" and len(scored) != 1:
        raise ReportingConfigError(
            f"comparison side {side!r} with select=exact has {len(scored)} "
            "complete configurations: " + "; ".join(item[1] for item in scored)
        )
    sign = -1 if direction_of(metric) == "maximize" else 1
    scored.sort(key=lambda item: (sign * item[0], item[1]))
    score, label, selected = scored[0]
    tied = sum(item[0] == score for item in scored) > 1
    return selected, {
        "candidate_count": len(scored),
        "selected_label": label,
        "validation_score": score,
        "replicate_count": len(selected),
        "tie": tied,
    }


def _human_label(record: dict) -> str:
    fields = human_configuration(record)
    return ", ".join(f"{path}={fields[path]}" for path in sorted(fields))


def _check_exact_confounding(left: dict, right: dict, field: str) -> None:
    if field == "algorithm":
        return
    left_fields = human_configuration(left)
    right_fields = human_configuration(right)
    left_fields.pop(field, None)
    right_fields.pop(field, None)
    differences = sorted(
        path
        for path in set(left_fields) | set(right_fields)
        if left_fields.get(path, object()) != right_fields.get(path, object())
    )
    if differences:
        raise ReportingConfigError(
            "select=exact comparison is confounded by other scientific settings: "
            + ", ".join(differences)
        )


def _default_comparison_path(results: Path, definition: dict) -> Path:
    base = results.parent / "comparisons"
    parts = [definition["field"], *(str(value) for value in definition["values"])]
    filename = "_".join(part.replace("/", "_") for part in parts) + ".csv"
    return base / filename


def _comparison_csv_row(name: str, definition: dict, comparison: dict) -> dict:
    effect = comparison["effects"]["mean_gain"]
    replicates = comparison["run_level_differences"]
    seed_values = {
        component: sorted(
            {
                item["replicate_condition"][field]
                for item in replicates
            },
            key=repr,
        )
        for component, field in (
            ("partition", "partition_seed"),
            ("split", "split_seed"),
            ("training", "experiment_seed"),
        )
    }
    varied = [name for name, values in seed_values.items() if len(values) > 1]
    return {
        "comparison": name,
        "filter": definition["filter"],
        "field": definition["field"],
        "left_value": definition["values"][0],
        "right_value": definition["values"][1],
        "select": definition["select"],
        "metric": comparison["protocol"]["metric"],
        "practical_margin": definition["practical_margin"],
        "mean_difference": effect["estimate"],
        "replicate_sd": effect["sd"],
        "ci_low": effect["ci_low"],
        "ci_high": effect["ci_high"],
        "replicate_count": effect["n"],
        "df": effect["df"],
        "pooled_within_replicate_client_difference_sd": effect[
            "pooled_within_replicate_client_sd"
        ],
        "practical_conclusion": comparison["practical_conclusion"],
        "partition_seed_values": seed_values["partition"],
        "split_seed_values": seed_values["split"],
        "training_seed_values": seed_values["training"],
        "varied_seed_components": varied,
    }


if __name__ == "__main__":
    main()
