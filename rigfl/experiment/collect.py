"""Summarize named sets of completed runs through ``rigfl report``."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from rigfl.eval.metrics import canonical, direction_of
from rigfl.eval.report import (
    format_replicate_details,
    format_resource_table,
    format_table,
    independent_replicates,
    selection_for,
    summarize,
)
from rigfl.eval.selection import resolve_metric
from rigfl.eval.transfer import (
    TransferComparisonError,
    format_negative_transfer_table,
    negative_transfer_summary,
)
from rigfl.experiment.artifacts import (
    ResultValidationError,
    is_run_result,
    read_json,
    validate_run_record,
)
from rigfl.experiment.config import (
    algorithm_identity,
    normalize_early_stopping,
    result_data_configuration_id,
)
from rigfl.experiment.config import hashable as _hashable
from rigfl.experiment.registry import ALL_ALGORITHMS
from rigfl.experiment.reporting_config import (
    ReportingConfigError,
    apply_named_filter,
    configuration_count,
    human_configuration,
    load_reporting_config,
    seed_summary,
    selection_defaults,
    write_csv,
)


def load_results(results_dir: Path, dataset: str | None, *, ignore_invalid: bool = False,
                 invalid: list | None = None) -> dict[str, list[dict]]:
    """Load current-schema run records, grouped by algorithm and optionally filtered."""
    by_algorithm: dict[str, list[dict]] = defaultdict(list)
    problems: list[tuple[str, str]] = [] if invalid is None else invalid
    others: list[str] = []

    for path in sorted(results_dir.glob("*.json")):
        try:
            rec = read_json(path)
        except ResultValidationError as e:
            problems.append((path.name, e.reason))
            continue
        if not is_run_result(rec):
            others.append(path.name)                          # a written artifact, not a run
            continue
        try:
            validate_run_record(rec, path=path)
        except ResultValidationError as e:
            problems.append((path.name, e.reason))
            continue
        rec["_source_file"] = path.name                       # provenance for artifacts
        exp = rec.get("config", {}).get("experiment", {})     # resolved config lives here
        if dataset and exp.get("dataset") != dataset:
            continue
        by_algorithm[rec["algorithm"]].append(rec)

    if others:
        print(f"[report] skipped {len(others)} JSON file(s) that are not run "
              f"results: {', '.join(sorted(others))}")
    if problems:
        listing = "\n".join(f"  {name}\n    {reason}" for name, reason in problems)
        if not ignore_invalid:
            noun = "file" if len(problems) == 1 else "files"
            raise SystemExit(
                f"[report] strict mode stopped: {len(problems)} invalid result "
                f"{noun} in {results_dir}:\n{listing}")
    return by_algorithm


# Algorithm settings are excluded so different algorithms can share one experimental
# condition. Replicate seeds and generated partition IDs are excluded because rows
# aggregate over replicate conditions.
_EXPERIMENT = ("dataset", "data_backend", "partition_scheme",
               "num_clients", "num_classes",
               "validation_fraction", "input_kind",
               "rounds", "shared_dim", "model", "model_family", "batch", "eval_gap",
               "estimate_flops")


def condition_fields(rec: dict) -> dict:
    """The flattened fields that define an experiment."""
    exp = rec.get("config", {}).get("experiment", {})
    fields = {k: _hashable(exp.get(k)) for k in _EXPERIMENT}
    fields["data_configuration"] = result_data_configuration_id(rec)
    fields["estimate_flops"] = bool(exp.get("estimate_flops", False))
    for k, v in normalize_early_stopping(exp.get("early_stopping")).items():
        fields[f"early_stopping.{k}"] = v
    return fields


def experiment_condition(rec: dict) -> tuple:
    """What makes two runs comparable across algorithms.

    Early stopping is part of it: a different control metric, patience or
    aggregation ends training somewhere else, so those runs are different
    experiments rather than extra seeds. Selection stays out, being post-hoc
    analysis over a history that is identical either way.
    """
    return tuple(sorted(condition_fields(rec).items()))


def algorithm_variant(rec: dict) -> tuple:
    """An algorithm's own settings, for telling apart its sweep points.

    Only used within an algorithm -- never across, or nothing would pair. Operational
    settings are excluded through the same helper run identity uses, so two runs
    that differ only in where their cache lives stay one row.
    """
    cfg = algorithm_identity(
        rec.get("config", {}).get("algorithm", {}),
        algorithm=rec.get("algorithm"),
    )
    return tuple(sorted((k, _hashable(v)) for k, v in cfg.items()))


def algorithm_fields(rec: dict) -> dict[str, object]:
    """Flatten an algorithm's settings for row labels."""
    fields: dict[str, object] = {}

    def add(path: str, value: object) -> None:
        if isinstance(value, dict) and value:
            for key, child in value.items():
                add(f"{path}.{key}", child)
        else:
            fields[path] = value

    for key, value in algorithm_identity(
        rec.get("config", {}).get("algorithm", {}),
        algorithm=rec.get("algorithm"),
    ).items():
        add(key, value)
    return fields


def varying_algorithm_fields(records: list[dict]) -> list[str]:
    """Algorithm settings that differ among configurations of one experiment."""
    flattened = [algorithm_fields(rec) for rec in records]
    keys = set().union(*(fields.keys() for fields in flattened))
    return sorted(
        key for key in keys
        if len({(key in fields, _hashable(fields.get(key))) for fields in flattened}) > 1
    )


def varying_fields(records: list[dict]) -> list[str]:
    """Condition fields that differ across these records."""
    seen: dict[str, set] = {}
    for rec in records:
        for k, v in condition_fields(rec).items():
            seen.setdefault(k, set()).add(repr(v))
    return [k for k, vals in seen.items() if len(vals) > 1]


def describe_condition(rec: dict, fields: list[str] | None = None) -> str:
    """Label an experiment by the fields given, or by its headline ones."""
    values = condition_fields(rec)
    keys = fields if fields is not None else [
        k for k in ("dataset", "partition_id", "num_clients")
        if values.get(k) is not None]
    bits = [f"{k}={values.get(k)}" for k in keys]
    return " ".join(bits) or "(unlabelled)"


def _add_row_details(summary: dict, records: list[dict]) -> None:
    representative = records[0]
    summary["algorithm"] = representative["algorithm"]
    summary["dataset"] = representative.get("config", {}).get("experiment", {}).get("dataset")
    summary["data_configuration_id"] = result_data_configuration_id(representative)
    summary["resolved_configuration"] = human_configuration(representative)
    seeds = seed_summary(records)
    summary["partition_seed_values"] = seeds["values"]["partition"]
    summary["split_seed_values"] = seeds["values"]["split"]
    summary["training_seed_values"] = seeds["values"]["training"]
    summary["varied_seed_components"] = seeds["varied"]


def _sort_rows_by_validation(rows: dict[str, dict], metric: str) -> dict[str, dict]:
    """Sort scores only within each seed-independent data configuration."""
    sign = -1 if direction_of(metric) == "maximize" else 1
    return dict(sorted(
        rows.items(),
        key=lambda item: (
            item[1].get("dataset") or "",
            item[1]["data_configuration_id"],
            item[1]["selection_view"],
            sign * item[1]["val_mean"],
            item[0],
        ),
    ))


def _rows_by_algorithm(by_algorithm: dict[str, list[dict]], metric: str, *, view: str,
                    aggregation: str, tie_break: str,
                    include_resources: bool = False,
                    include_transfer: bool = True,
                    transfer_threshold: float = 0.0,
                    transfer_profile: tuple[float, ...] = (),
                    transfer_tail: float = 0.10) -> dict[str, dict]:
    """One row per algorithm configuration and experiment."""
    flat = [r for recs in by_algorithm.values() for r in recs]
    experiments = {experiment_condition(r) for r in flat}
    multi = len(experiments) > 1
    fields = varying_fields(flat) if multi else []
    if multi:
        print(f"[report] {len(experiments)} distinct experiments present "
              f"(differing in {', '.join(fields)}); reported separately")

    rows: dict[str, dict] = {}
    for exp in sorted(experiments, key=str):
        here = [r for r in flat if experiment_condition(r) == exp]
        local_records = [r for r in here if r["algorithm"] == "local"]
        exp_suffix = f"  [{describe_condition(here[0], fields)}]" if multi else ""

        for name in ALL_ALGORITHMS:
            recs = [r for r in here if r["algorithm"] == name]
            if not recs:
                continue
            variants: dict[tuple, list[dict]] = defaultdict(list)
            for r in recs:
                variants[algorithm_variant(r)].append(r)
            changed = varying_algorithm_fields([records[0] for records in variants.values()])
            for _, vrecs in sorted(variants.items(), key=lambda kv: str(kv[0])):
                algorithm_values = algorithm_fields(vrecs[0])
                details = " ".join(
                    f"{key}={algorithm_values.get(key, '<unset>')}"
                    for key in changed
                )
                label = name + (f" {details}" if details else "") + exp_suffix
                summary = summarize(vrecs, metric, view=view, aggregation=aggregation,
                                    tie_break=tie_break,
                                    include_resources=include_resources)
                _add_row_details(summary, vrecs)
                if include_transfer and local_records and name != "local":
                    summary["negative_transfer"] = _negative_transfer(
                        vrecs,
                        local_records,
                        metric,
                        view=view,
                        aggregation=aggregation,
                        tie_break=tie_break,
                        threshold=transfer_threshold,
                        profile_thresholds=transfer_profile,
                        tail_fraction=transfer_tail,
                        include_uncertainty=independent_replicates(vrecs),
                    )
                rows[label] = summary
    return _sort_rows_by_validation(rows, metric)


def _collection_rows(rows: dict) -> tuple[dict, dict]:
    """Separate predictive summaries from Local-relative transfer summaries."""
    performance, transfer = {}, {}
    for label, row in rows.items():
        saved = dict(row)
        comparison = saved.pop("negative_transfer", None)
        performance[label] = saved
        if comparison is not None:
            transfer[label] = comparison
    return performance, transfer


def _negative_transfer(algorithm_records: list[dict], local_records: list[dict],
                       metric: str, **options) -> dict:
    try:
        return negative_transfer_summary(
            algorithm_records, local_records, metric, **options
        )
    except TransferComparisonError as error:
        return {
            "available": False,
            "baseline": "local",
            "reason": str(error),
        }


def main(argv: list[str] | None = None, *, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=prog, description="Summarize completed RigFL results."
    )
    parser.add_argument("results", nargs="?", default="results/runs")
    parser.add_argument("--config", required=True, help="reporting YAML")
    parser.add_argument("--filter", required=True, help="named YAML filter")
    parser.add_argument(
        "--resources", action="store_true", help="include measured resource results"
    )
    parser.add_argument(
        "--per-client",
        action="store_true",
        help="include one selected result row per run and client",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const="",
        metavar="PATH",
        help="save CSV tables; optionally set the main table path",
    )
    args = parser.parse_args(argv)

    invalid: list[tuple[str, str]] = []
    try:
        reporting = load_reporting_config(args.config)
        defaults = selection_defaults(reporting)
        metric = resolve_metric(defaults["metric"], source="defaults.metric")
        by_algorithm = load_results(
            Path(args.results), None, ignore_invalid=True, invalid=invalid
        )
        loaded = [record for records in by_algorithm.values() for record in records]
        if not loaded:
            raise ReportingConfigError(
                f"no completed run results found in {args.results}"
            )
        filtered, resolved_filter = apply_named_filter(
            loaded, reporting, args.filter
        )
    except (ReportingConfigError, ValueError) as error:
        raise SystemExit(f"cannot report results: {error}") from error

    by_algorithm = defaultdict(list)
    for record in filtered:
        by_algorithm[record["algorithm"]].append(record)
    rows = _rows_by_algorithm(
        by_algorithm,
        metric,
        view=defaults["view"],
        aggregation=defaults["aggregation"],
        tie_break=defaults["tie_break"],
        include_resources=args.resources,
    )
    print(
        f"Filter {args.filter!r}: {resolved_filter}; "
        f"{configuration_count(filtered)} configuration(s), "
        f"{len(filtered)} completed replicate(s)."
    )
    for label, summary in rows.items():
        varied = ", ".join(summary["varied_seed_components"]) or "none"
        fixed = ", ".join(
            component
            for component in ("partition", "split", "training")
            if component not in summary["varied_seed_components"]
        ) or "none"
        print(f"Seeds for {label}: varied={varied}; fixed={fixed}.")
    if invalid:
        print(
            f"Warning: excluded {len(invalid)} invalid result file(s): "
            + ", ".join(name for name, _ in invalid)
        )

    print(
        f"\n### results (metric={metric}, selection={defaults['view']}, "
        f"aggregation={defaults['aggregation']}, tie_break={defaults['tie_break']})"
    )
    print(format_table(rows, metric))
    print("\n" + format_replicate_details(rows))
    transfer_table = format_negative_transfer_table(rows)
    if transfer_table:
        print("\n### Local-relative client impact")
        print(transfer_table)
    if args.resources:
        print("\n### resources")
        print(format_resource_table(rows))

    client_rows = (
        _per_client_rows(
            filtered,
            metric,
            view=defaults["view"],
            aggregation=defaults["aggregation"],
            tie_break=defaults["tie_break"],
        )
        if args.per_client
        else []
    )
    if args.per_client:
        print(f"\nPer-client rows available: {len(client_rows)}.")

    if args.save is not None:
        paths = _report_paths(Path(args.results), args.filter, args.save)
        performance, transfer = _collection_rows(rows)
        write_csv(
            paths["performance"],
            _performance_csv_rows(performance, args.filter, metric),
        )
        print(f"wrote {paths['performance']}")
        if transfer:
            write_csv(paths["transfer"], _transfer_csv_rows(transfer, args.filter))
            print(f"wrote {paths['transfer']}")
        if args.resources:
            write_csv(paths["resources"], _resource_csv_rows(rows, args.filter))
            print(f"wrote {paths['resources']}")
        if args.per_client:
            write_csv(paths["per_client"], client_rows)
            print(f"wrote {paths['per_client']}")


def _report_paths(results: Path, filter_name: str, custom: str) -> dict[str, Path]:
    if custom:
        main = Path(custom)
        stem = main.with_suffix("")
        return {
            "performance": main,
            "transfer": Path(f"{stem}_negative_transfer_analysis.csv"),
            "resources": Path(f"{stem}_resources.csv"),
            "per_client": Path(f"{stem}_per_client.csv"),
        }
    base = results.parent / "reports"
    return {
        "performance": base / f"{filter_name}.csv",
        "transfer": base / f"{filter_name}_negative_transfer_analysis.csv",
        "resources": base / f"{filter_name}_resources.csv",
        "per_client": base / f"{filter_name}_per_client.csv",
    }


def _performance_csv_rows(rows: dict, filter_name: str, metric: str) -> list[dict]:
    output = []
    for label, summary in rows.items():
        row = {
            "filter": filter_name,
            "configuration": label,
            "algorithm": summary["algorithm"],
            "dataset": summary["dataset"],
            "data_configuration_id": summary["data_configuration_id"],
            "metric": metric,
            "selection_view": summary["selection_view"],
            "validation_mean": summary["val_mean"],
            "validation_sd": summary["val_std"],
            "validation_ci_low": summary["val_ci_low"],
            "validation_ci_high": summary["val_ci_high"],
            "test_mean": summary["test_mean"],
            "test_sd": summary["test_std"],
            "test_ci_low": summary["test_ci_low"],
            "test_ci_high": summary["test_ci_high"],
            "replicate_count": summary["test_n"],
            "df": summary["test_df"],
            "pooled_within_replicate_client_sd": summary[
                "pooled_within_replicate_client_sd"
            ],
            "partition_seed_values": summary["partition_seed_values"],
            "split_seed_values": summary["split_seed_values"],
            "training_seed_values": summary["training_seed_values"],
            "varied_seed_components": summary["varied_seed_components"],
        }
        row.update(summary["resolved_configuration"])
        output.append(row)
    return output


def _transfer_csv_rows(rows: dict, filter_name: str) -> list[dict]:
    output = []
    fields = (
        "mean_gain",
        "benefit_rate",
        "negative_transfer_rate",
        "negative_transfer_magnitude",
        "negative_transfer_burden",
        "worst_tail_gain",
    )
    for label, summary in rows.items():
        if not summary.get("available", True):
            output.append(
                {"filter": filter_name, "configuration": label, "available": False,
                 "reason": summary.get("reason")}
            )
            continue
        row = {
            "filter": filter_name,
            "configuration": label,
            "available": True,
            "baseline": "local",
            "metric": summary["metric"],
            "threshold": summary["threshold"],
            "tail_fraction": summary["tail_fraction"],
            "pair_count": summary["pair_count"],
            "replicate_count": summary["replicate_count"],
        }
        for field in fields:
            effect = summary[field]
            for statistic in ("estimate", "sd", "ci_low", "ci_high", "n", "df"):
                row[f"{field}_{statistic}"] = effect.get(statistic)
        magnitude = summary["negative_transfer_magnitude"]
        row["harmed_client_count"] = magnitude["harmed_client_count"]
        row["affected_replicate_count"] = magnitude["affected_replicate_count"]
        output.append(row)
    return output


def _resource_csv_rows(rows: dict, filter_name: str) -> list[dict]:
    output = []
    for label, summary in rows.items():
        resource = summary.get("resources", {})
        row = {
            "filter": filter_name,
            "configuration": label,
            "algorithm": summary["algorithm"],
            "available": resource.get("available", False),
        }
        row.update(resource)
        output.append(row)
    return output


def _per_client_rows(
    records: list[dict],
    metric: str,
    *,
    view: str,
    aggregation: str,
    tie_break: str,
) -> list[dict]:
    name = canonical(metric)
    rows = []
    for record in records:
        selected = selection_for(
            record,
            name,
            view=view,
            aggregation=aggregation,
            tie_break=tie_break,
        )
        ids = selected.get("client_ids", [])
        validation = selected.get("validation", {}).get(name, [])
        test = selected.get("test", {}).get(name, [])
        validation_counts = selected.get("sample_counts", {}).get("validation", [])
        test_counts = selected.get("sample_counts", {}).get("test", [])
        experiment = record["config"]["experiment"]
        for index, client_id in enumerate(ids):
            selected_round = selected.get("selected_round")
            if selected_round is None:
                selected_round = selected.get("selected_rounds", {}).get(client_id)
            rows.append(
                {
                    "source_file": record.get("_source_file"),
                    "algorithm": record["algorithm"],
                    "dataset": experiment.get("dataset"),
                    "data_configuration_id": result_data_configuration_id(record),
                    "partition_id": experiment.get("partition_id"),
                    "partition_seed": experiment.get("partition_seed"),
                    "split_seed": experiment.get("split_seed"),
                    "training_seed": experiment.get("seed"),
                    "client_id": client_id,
                    "metric": name,
                    "selection_view": selected["selection_view"],
                    "selected_round": selected_round,
                    "validation_value": validation[index] if index < len(validation) else None,
                    "test_value": test[index] if index < len(test) else None,
                    "validation_sample_count": (
                        validation_counts[index]
                        if index < len(validation_counts)
                        else None
                    ),
                    "test_sample_count": (
                        test_counts[index] if index < len(test_counts) else None
                    ),
                }
            )
    return rows


if __name__ == "__main__":
    main()
