"""Summarize named sets of completed runs through ``rigfl report``."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rigfl.eval.metrics import canonical
from rigfl.eval.report import (
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
    normalize_early_stopping,
    result_data_configuration_id,
)
from rigfl.experiment.identity import hashable
from rigfl.experiment.registry import ALL_ALGORITHMS
from rigfl.experiment.reporting_config import (
    ReportingConfigError,
    apply_named_filter,
    configuration_count,
    load_reporting_config,
    reporting_settings,
    setting_columns,
    write_csv,
)
from rigfl.experiment.storage import records_for_grid, run_store
from rigfl.experiment.tuning import MANIFEST_KIND, MANIFEST_NAME

DEFAULT_REPORTING_CONFIG = "configs/reporting.yaml"


def load_results(runs_dir: Path, dataset: str | None, *, ignore_invalid: bool = False,
                 invalid: list | None = None) -> dict[str, list[dict]]:
    """Load current-schema run records, grouped by algorithm and optionally filtered."""
    by_algorithm: dict[str, list[dict]] = defaultdict(list)
    problems: list[tuple[str, str]] = [] if invalid is None else invalid
    others: list[str] = []

    for path in sorted(runs_dir.glob("*.json")):
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
        rec["_source_file"] = path.name
        exp = rec.get("config", {}).get("experiment", {})
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
                f"{len(problems)} invalid result {noun} in {runs_dir}:\n{listing}")
    return by_algorithm


# Algorithm settings are excluded so different algorithms can share one experimental
# condition. Replicate seeds and generated partition IDs are excluded because rows
# aggregate over replicate conditions.
_EXPERIMENT = ("dataset", "data_backend", "partition_scheme",
               "num_clients", "num_classes",
               "validation_fraction", "input_kind",
               "rounds", "model_arch", "batch", "eval_gap")


def condition_fields(rec: dict) -> dict:
    """The flattened fields that define an experiment.

    A null field is one the algorithm ignores; it matches any value.
    """
    exp = rec.get("config", {}).get("experiment", {})
    fields = {k: hashable(exp.get(k)) for k in _EXPERIMENT}
    fields["data_configuration"] = result_data_configuration_id(rec)
    if "early_stopping" in exp and exp["early_stopping"] is None:
        fields["early_stopping.enabled"] = None
    else:
        for k, v in normalize_early_stopping(exp.get("early_stopping")).items():
            fields[f"early_stopping.{k}"] = v
    return fields


def experiment_condition(rec: dict) -> tuple:
    """Experiment fields shared by comparable runs, including early stopping (not selection).

    Early stopping changes the executed history; reporting selection is post-hoc.
    """
    return tuple(sorted(condition_fields(rec).items()))


def _in_condition(fields: dict, condition: dict) -> bool:
    """Whether a run's condition fields fit a condition; null matches anything."""
    return all(
        fields.get(key) is None or fields.get(key) == condition.get(key)
        for key in fields.keys() | condition.keys()
    )


def experiment_conditions(records: list[dict]) -> list[dict]:
    """The conditions set by concrete values.

    A run with null fields belongs to every condition it fits and forms its
    own condition only when it fits none.
    """
    distinct = {experiment_condition(r): condition_fields(r) for r in records}
    return [
        fields for key, fields in distinct.items()
        if not any(
            other != key and _in_condition(fields, distinct[other])
            for other in distinct
        )
    ]


def algorithm_settings(rec: dict) -> dict:
    """An algorithm's own settings, plus ``shared_dim`` when the run used it.

    ``shared_dim`` only affects algorithms that exchange representations, so it
    separates their variants without splitting the experiment condition shared
    with native-output algorithms.
    """
    settings = dict(rec.get("config", {}).get("algorithm", {}))
    identity_exp = (
        (rec.get("identity") or {}).get("fingerprint_input") or {}
    ).get("experiment", {})
    if "shared_dim" in identity_exp:
        settings["shared_dim"] = identity_exp["shared_dim"]
    return settings


def algorithm_variant(rec: dict) -> tuple:
    """An algorithm's own settings; compared only within one algorithm."""
    cfg = algorithm_settings(rec)
    return tuple(sorted((k, hashable(v)) for k, v in cfg.items()))


def algorithm_fields(rec: dict) -> dict[str, object]:
    """Flatten an algorithm's settings for row labels."""
    fields: dict[str, object] = {}

    def add(path: str, value: object) -> None:
        if isinstance(value, dict) and value:
            for key, child in value.items():
                add(f"{path}.{key}", child)
        else:
            fields[path] = value

    for key, value in algorithm_settings(rec).items():
        add(key, value)
    return fields


def varying_algorithm_fields(records: list[dict]) -> list[str]:
    """Algorithm settings that differ among configurations of one experiment."""
    flattened = [algorithm_fields(rec) for rec in records]
    keys = set().union(*(fields.keys() for fields in flattened))
    return sorted(
        key for key in keys
        if len({(key in fields, hashable(fields.get(key))) for fields in flattened}) > 1
    )


def _varying(conditions: list[dict]) -> list[str]:
    seen: dict[str, set] = {}
    for fields in conditions:
        for k, v in fields.items():
            values = seen.setdefault(k, set())
            if v is not None:
                values.add(repr(v))
    return [k for k, vals in seen.items() if len(vals) > 1]


def varying_fields(records: list[dict]) -> list[str]:
    """Condition fields with more than one concrete value across these records."""
    return _varying([condition_fields(rec) for rec in records])


def describe_condition(condition: dict, fields: list[str]) -> str:
    """Label an experiment condition by the given fields."""
    bits = [
        f"{k}={'—' if condition.get(k) is None else condition.get(k)}"
        for k in fields
    ]
    return " ".join(bits) or "(unlabelled)"


def _add_row_details(summary: dict, records: list[dict]) -> None:
    representative = records[0]
    summary["algorithm"] = representative["algorithm"]
    summary["dataset"] = representative.get("config", {}).get("experiment", {}).get("dataset")
    summary["data_configuration_id"] = result_data_configuration_id(representative)
    summary["experiment_settings"], summary["algorithm_settings"] = (
        setting_columns(representative)
    )
    summary["source_files"] = [record.get("_source_file") for record in records]


# Derived from the dataset or internal, so left out of the shared-settings line.
_UNSHARED = ("data_backend", "num_classes", "input_kind", "data_configuration")


def shared_settings(records: list[dict]) -> str:
    """Condition fields every record agrees on, for one line above the tables."""
    varying = set(varying_fields(records))
    values: dict = {}
    for record in records:
        for key, value in condition_fields(record).items():
            if values.get(key) is None:
                values[key] = value
    hide_stopping = (
        "early_stopping.enabled" in varying
        or values.get("early_stopping.enabled") is False
    )
    return ", ".join(
        f"{key}={value}"
        for key, value in values.items()
        if key not in varying and key not in _UNSHARED and value is not None
        and not (hide_stopping and key.startswith("early_stopping."))
    )


def _heading_fields(conditions: list[dict]) -> list[str]:
    """Fields that tell conditions apart; the data-configuration hash only when needed."""
    fields = _varying(conditions)
    readable = [field for field in fields if field != "data_configuration"]
    labels = {
        tuple(condition.get(field) for field in readable) for condition in conditions
    }
    return readable if readable and len(labels) == len(conditions) else fields


def _by_validation(labels: list[str], rows: dict[str, dict]) -> list[str]:
    """Best validation mean first; ties keep the given order. Test values never matter."""
    return sorted(labels, key=lambda label: -rows[label]["val_mean"])


def _rows_by_algorithm(by_algorithm: dict[str, list[dict]], metric: str, *, view: str,
                    aggregation: str,
                    include_resources: bool = False,
                    best_only: bool = False) -> dict[str, dict]:
    """One row per algorithm configuration and experiment condition.

    Conditions are grouped, methods follow ``ALL_ALGORITHMS``, and each method's
    configurations are ordered best validation mean first. ``best_only`` keeps
    only that first configuration, the one the gain analysis uses. A
    configuration with null condition fields is listed in every condition it
    fits.
    """
    flat = [r for recs in by_algorithm.values() for r in recs]
    experiments = experiment_conditions(flat)
    multi = len(experiments) > 1
    fields = _heading_fields(experiments) if multi else []
    if multi:
        print(f"[report] {len(experiments)} distinct experiments present "
              f"(differing in {', '.join(varying_fields(flat))}); reported separately")

    conditions = []
    for condition in experiments:
        here = [r for r in flat if _in_condition(condition_fields(r), condition)]
        heading = describe_condition(condition, fields) if multi else ""
        exp = tuple(sorted(condition.items()))
        conditions.append((heading, str(exp), exp, here))

    rows: dict[str, dict] = {}
    for heading, _, exp, here in sorted(conditions, key=lambda item: item[:2]):
        suffix = f"  [{heading}]" if multi else ""
        # algorithm -> labels ordered best validation first
        ordered: dict[str, list[str]] = {}
        candidates: dict[str, dict] = {}
        records_for: dict[str, list[dict]] = {}

        for name in ALL_ALGORITHMS:
            recs = [r for r in here if r["algorithm"] == name]
            if not recs:
                continue
            variants: dict[tuple, list[dict]] = defaultdict(list)
            for r in recs:
                variants[algorithm_variant(r)].append(r)
            changed = varying_algorithm_fields([records[0] for records in variants.values()])
            labels = []
            for _, vrecs in sorted(variants.items(), key=lambda kv: str(kv[0])):
                algorithm_values = algorithm_fields(vrecs[0])
                details = " ".join(
                    f"{key}={algorithm_values.get(key, '<unset>')}"
                    for key in changed
                )
                label = name + (f" {details}" if details else "") + suffix
                summary = summarize(
                    vrecs,
                    metric,
                    view=view,
                    aggregation=aggregation,
                    include_resources=include_resources,
                )
                _add_row_details(summary, vrecs)
                summary["condition"] = heading
                summary["configuration"] = details or "—"
                candidates[label] = summary
                records_for[label] = vrecs
                labels.append(label)
            ordered[name] = _by_validation(labels, candidates)
            if best_only:
                ordered[name] = ordered[name][:1]

        for labels in ordered.values():
            rows.update((label, candidates[label]) for label in labels)
        if "local" not in ordered:
            for labels in ordered.values():
                for label in labels:
                    rows[label]["missing_local_condition"] = exp
            continue
        local_label = ordered["local"][0]
        for name, labels in ordered.items():
            if name == "local":
                continue
            label = labels[0]
            comparison = _negative_transfer(
                records_for[label],
                records_for[local_label],
                metric,
                view=view,
                aggregation=aggregation,
                include_uncertainty=independent_replicates(records_for[label]),
            )
            comparison["algorithm"] = name
            comparison["configuration"] = candidates[label]["configuration"]
            comparison["local_configuration"] = candidates[local_label]["configuration"]
            comparison["local_label"] = local_label
            comparison["condition"] = heading
            rows[label]["negative_transfer"] = comparison
    return rows


def _missing_local_note(rows: dict, records: list[dict]) -> str:
    """One line counting configurations left out of the gain analysis."""
    missing = [
        row["missing_local_condition"]
        for row in rows.values()
        if "missing_local_condition" in row
    ]
    if not missing:
        return ""
    conditions = len(set(missing))
    fields = varying_fields(records)
    detail = f" (conditions differ in {', '.join(fields)})" if fields else ""
    configurations = "configuration" + ("s" if len(missing) > 1 else "")
    verb = "have" if len(missing) > 1 else "has"
    in_conditions = f"condition{'s' if conditions > 1 else ''}"
    return (
        f"Not shown: {len(missing)} {configurations} in {conditions} experiment "
        f"{in_conditions} {verb} no Local run{detail}."
    )


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


def records_for_study(records: list[dict], study_path: str | Path) -> list[dict]:
    """Keep the runs an HPO study evaluated, as listed in its study.json."""
    path = Path(study_path)
    if path.is_dir():
        path = path / MANIFEST_NAME
    if not path.is_file():
        raise ValueError(f"HPO study not found: {path}")
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError:
        manifest = None
    if not isinstance(manifest, dict) or manifest.get("kind") != MANIFEST_KIND:
        raise ValueError(f"{path} is not a RigFL study.json")
    sources = {
        Path(source).name
        for evaluation in manifest.get("evaluations", [])
        for source in evaluation.get("source_results", [])
    }
    return [record for record in records if record["_source_file"] in sources]


def main(argv: list[str] | None = None, *, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=prog, description="Summarize completed RigFL results."
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="base directory containing runs and reports (default: results)",
    )
    parser.add_argument(
        "--config", default=DEFAULT_REPORTING_CONFIG, help="reporting YAML"
    )
    parser.add_argument("--filter", help="named YAML filter")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--grid",
        metavar="PATH",
        help="report only the runs of this sweep grid (grid.jsonl)",
    )
    source.add_argument(
        "--study",
        metavar="PATH",
        help="report only the runs of this HPO study (study directory or study.json)",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="show only each method's best validation configuration per condition",
    )
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
    results_root = Path(args.results_root)
    runs_dir = run_store(results_root)

    invalid: list[tuple[str, str]] = []
    try:
        reporting = load_reporting_config(args.config)
        settings = reporting_settings(reporting)
        metric = resolve_metric(settings["metric"])
        by_algorithm = load_results(
            runs_dir, None, ignore_invalid=True, invalid=invalid
        )
        loaded = [record for records in by_algorithm.values() for record in records]
        if not loaded:
            raise ReportingConfigError(
                f"no completed run results found in {runs_dir}"
            )
        if args.grid is not None:
            if Path(args.grid).suffix != ".jsonl":
                raise ValueError(f"--grid expects a grid.jsonl file, got {args.grid}")
            loaded = records_for_grid(loaded, args.grid)
        elif args.study is not None:
            loaded = records_for_study(loaded, args.study)
        if args.filter is None:
            filtered, resolved_filter = loaded, None
        else:
            filtered, resolved_filter = apply_named_filter(
                loaded, reporting, args.filter
            )
    except (ReportingConfigError, ValueError) as error:
        raise SystemExit(f"cannot report results: {error}") from error

    by_algorithm = defaultdict(list)
    for record in filtered:
        by_algorithm[record["algorithm"]].append(record)
    selection_view = settings["round_selection"]
    aggregation = settings["client_weighting"]
    rows = _rows_by_algorithm(
        by_algorithm,
        metric,
        view=selection_view,
        aggregation=aggregation,
        include_resources=args.resources,
        best_only=args.best,
    )
    scopes = []
    if args.grid is not None:
        scopes.append(f"Sweep grid {args.grid}")
    elif args.study is not None:
        scopes.append(f"HPO study {args.study}")
    if args.filter is not None:
        scopes.append(f"Filter {args.filter!r}: {resolved_filter}")
    scope = "; ".join(scopes) or "All completed runs"
    print(
        f"{scope}; {configuration_count(filtered)} configuration(s), "
        f"{len(filtered)} completed replicate(s)."
    )
    if invalid:
        print(
            f"Warning: excluded {len(invalid)} invalid result file(s): "
            + ", ".join(name for name, _ in invalid)
        )

    print(
        f"\n===== AGGREGATE PERFORMANCE (metric={metric}, "
        f"round_selection={settings['round_selection']}, "
        f"client_weighting={settings['client_weighting']}) ====="
    )
    print(f"Shared settings: {shared_settings(filtered)}\n")
    print(format_table(rows, metric))
    transfer_table = format_negative_transfer_table(rows)
    missing_local = _missing_local_note(rows, filtered)
    if transfer_table or missing_local:
        print("\n===== CLIENT-LEVEL GAIN ANALYSIS =====\n")
        print("\n\n".join(part for part in (transfer_table, missing_local) if part))
    if args.resources:
        print("\n===== RESOURCES =====\n")
        print(format_resource_table(rows))

    shown = {name for row in rows.values() for name in row["source_files"]}
    client_rows = (
        _per_client_rows(
            [record for record in filtered if record.get("_source_file") in shown],
            metric,
            view=selection_view,
            aggregation=aggregation,
        )
        if args.per_client
        else []
    )
    if args.per_client:
        print(f"\nPer-client rows available: {len(client_rows)}.")

    if args.save is not None:
        try:
            _save_csvs(
                _report_paths(results_root, args.filter, args.save),
                rows, args.filter or "", metric,
                resources=args.resources,
                client_rows=client_rows if args.per_client else None,
            )
        except ReportingConfigError as error:
            raise SystemExit(f"cannot save report: {error}") from error


def _save_csvs(paths: dict, rows: dict, filter_name: str, metric: str, *,
               resources: bool, client_rows: list[dict] | None) -> None:
    """Write the tidy CSVs: each configuration once, each Local comparison once."""
    configurations = _unique_configurations(rows)
    write_csv(
        paths["performance"],
        _performance_csv_rows(configurations, filter_name, metric),
    )
    print(f"wrote {paths['performance']}")
    transfer = _transfer_csv_rows(rows, filter_name)
    if transfer:
        write_csv(paths["transfer"], transfer)
        print(f"wrote {paths['transfer']}")
    if resources:
        write_csv(
            paths["resources"], _resource_csv_rows(configurations, filter_name)
        )
        print(f"wrote {paths['resources']}")
    if client_rows is not None:
        write_csv(paths["per_client"], client_rows)
        print(f"wrote {paths['per_client']}")


def _report_paths(
    results_root: Path, filter_name: str | None, custom: str
) -> dict[str, Path]:
    if custom:
        main = Path(custom)
        stem = main.with_suffix("")
        return {
            "performance": main,
            "transfer": Path(f"{stem}_client_level_gain_analysis.csv"),
            "resources": Path(f"{stem}_resources.csv"),
            "per_client": Path(f"{stem}_per_client.csv"),
        }
    base = results_root / "reports"
    stem = filter_name or "summary"
    return {
        "performance": base / f"{stem}.csv",
        "transfer": base / f"{stem}_client_level_gain_analysis.csv",
        "resources": base / f"{stem}_resources.csv",
        "per_client": base / f"{stem}_per_client.csv",
    }


def _unique_configurations(rows: dict) -> list[dict]:
    """Each configuration once, though the terminal lists one with null
    settings under every condition it fits."""
    unique = {}
    for summary in rows.values():
        unique.setdefault(tuple(summary["source_files"]), summary)
    return list(unique.values())


def _with_settings(row: dict, *settings: dict) -> dict:
    for values in settings:
        for key, value in values.items():
            if key in row:
                raise ReportingConfigError(
                    f"setting {key!r} has the same CSV column name as another "
                    "column of the table"
                )
            row[key] = value
    return row


def _performance_csv_rows(
    configurations: list[dict], filter_name: str, metric: str
) -> list[dict]:
    output = []
    for summary in configurations:
        row = {
            "filter": filter_name,
            "algorithm": summary["algorithm"],
            "configuration": summary["configuration"],
            "dataset": summary["dataset"],
            "data_configuration_id": summary["data_configuration_id"],
            "metric": metric,
            "round_selection": summary["selection_view"],
            "client_weighting": summary["selection_aggregation"],
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
            "pooled_client_sd": summary["pooled_client_sd"],
            # aligned lists, one entry per replicate
            **{
                f"{component}_seeds": [
                    replicate[f"{component}_seed"]
                    for replicate in summary["replicate_conditions"]
                ]
                for component in ("partition", "split", "training")
            },
        }
        output.append(_with_settings(
            row, summary["experiment_settings"], summary["algorithm_settings"]
        ))
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
    seen = set()
    for result in rows.values():
        summary = result.get("negative_transfer")
        if summary is None:
            continue
        local = rows[summary["local_label"]]
        pair = (tuple(result["source_files"]), tuple(local["source_files"]))
        if pair in seen:
            continue
        seen.add(pair)
        # a setting the algorithm ignores takes the value of the Local it was paired with
        settings = {
            **result["experiment_settings"],
            **{
                key: value
                for key, value in local["experiment_settings"].items()
                if result["experiment_settings"].get(key) is None
            },
        }
        row = {
            "filter": filter_name,
            "algorithm": result["algorithm"],
            "configuration": result["configuration"],
            "local_configuration": summary["local_configuration"],
        }
        if not summary.get("available", True):
            row.update(available=False, reason=summary.get("reason"))
            output.append(
                _with_settings(row, settings, result["algorithm_settings"])
            )
            continue
        row.update({
            "available": True,
            "baseline": "local",
            "metric": summary["metric"],
            "tail_fraction": summary["tail_fraction"],
            "pair_count": summary["pair_count"],
            "replicate_count": summary["replicate_count"],
        })
        for field in fields:
            effect = summary[field]
            for statistic in ("estimate", "sd", "ci_low", "ci_high", "n", "df"):
                row[f"{field}_{statistic}"] = effect.get(statistic)
        row["pooled_client_difference_sd"] = summary["mean_gain"].get(
            "pooled_client_sd"
        )
        magnitude = summary["negative_transfer_magnitude"]
        row["harmed_client_count"] = magnitude["harmed_client_count"]
        row["affected_replicate_count"] = magnitude["affected_replicate_count"]
        output.append(
            _with_settings(row, settings, result["algorithm_settings"])
        )
    return output


def _resource_csv_rows(configurations: list[dict], filter_name: str) -> list[dict]:
    output = []
    for summary in configurations:
        resource = summary.get("resources", {})
        row = {
            "filter": filter_name,
            "algorithm": summary["algorithm"],
            "configuration": summary["configuration"],
            "available": resource.get("available", False),
        }
        row.update(resource)
        output.append(_with_settings(row, summary["experiment_settings"]))
    return output


def _per_client_rows(
    records: list[dict],
    metric: str,
    *,
    view: str,
    aggregation: str,
) -> list[dict]:
    name = canonical(metric)
    rows = []
    for record in records:
        selected = selection_for(
            record,
            name,
            view=view,
            aggregation=aggregation,
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
                    "training_seed": experiment.get("training_seed"),
                    "client_id": client_id,
                    "metric": name,
                    "round_selection": selected["selection_view"],
                    "client_weighting": aggregation,
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
