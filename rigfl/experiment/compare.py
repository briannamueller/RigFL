"""Compare named frozen configurations using matched test results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from rigfl.eval.comparison import (
    ConfigurationComparisonError,
    compare_configurations,
    comparison_context,
    format_comparisons,
    frozen_configuration,
)
from rigfl.eval.selection import resolve_metric
from rigfl.experiment.artifacts import (
    ResultValidationError,
    atomic_write_json,
    atomic_write_text,
    is_run_result,
    read_json,
    validate_run_record,
)
from rigfl.experiment.config import fingerprint
from rigfl.experiment.storage import run_store
from rigfl.experiment.tuning import (
    SELECTION_KIND,
    TuningError,
    load_manifest,
    load_selection,
)


def _contrast(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("use LEFT:RIGHT")
    left, right = value.split(":", 1)
    if not left.strip() or not right.strip() or left == right:
        raise argparse.ArgumentTypeError("contrast labels must be distinct")
    return left.strip(), right.strip()


def _load_results(path: Path, *, required: bool = True) -> list[dict]:
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
    if problems:
        raise ConfigurationComparisonError(
            "invalid result files:\n  " + "\n  ".join(problems)
        )
    if required and not records:
        raise ConfigurationComparisonError(f"no completed run results found at {path}")
    return records


def _selection_result_paths(selection_path: Path) -> set[Path]:
    try:
        artifact = load_selection(selection_path)
    except TuningError as error:
        raise ConfigurationComparisonError(str(error)) from error
    selected = {
        (selection_path.parent / source).resolve()
        for group in artifact.get("groups", [])
        for source in group.get("selected_result_files", [])
    }
    if not selected:
        raise ConfigurationComparisonError(
            f"selection contains no selected result files: {selection_path}"
        )
    return selected


def _load_selected_results(path: Path) -> list[dict] | None:
    if path.is_file():
        artifact = read_json(path)
        if not isinstance(artifact, dict) or artifact.get("kind") != SELECTION_KIND:
            return None
        artifact_paths = [path]
    elif path.is_dir():
        artifact_paths = sorted(path.rglob("selection.json"))
    else:
        artifact_paths = []

    selected_paths: set[Path] = set()
    found = False
    for artifact_path in artifact_paths:
        found = True
        selected_paths.update(_selection_result_paths(artifact_path))

    if not found:
        return None
    if not selected_paths:
        raise ConfigurationComparisonError(
            f"selection artifacts contain no selected result files below {path}"
        )
    records = []
    for result_path in sorted(selected_paths):
        records.extend(_load_results(result_path))
    return records


def _tuning_result_names(path: Path) -> set[str]:
    if not path.is_dir():
        return set()
    names = set()
    for manifest_path in path.rglob("study.json"):
        try:
            artifact = load_manifest(manifest_path.parent)
        except TuningError as error:
            raise ConfigurationComparisonError(str(error)) from error
        if artifact is None:
            continue
        for evaluation in artifact.get("evaluations", []):
            names.update(
                Path(source).name
                for source in evaluation.get("source_results", [])
                if source
            )
        for trial in artifact.get("optimization", {}).get("trials", []):
            names.update(
                Path(source).name
                for source in trial.get("source_results", [])
                if source
            )
    return names


def _comparison_results(path: Path) -> list[dict]:
    scope = (
        path.parent
        if path.is_dir() and path.name == "runs"
        else path
    )
    selected = _load_selected_results(scope)
    raw_tuning = path.is_dir() and (path / "study.json").exists()
    if raw_tuning and selected is None:
        raise ConfigurationComparisonError(
            "a raw tuning directory cannot be used for final test comparison; "
            "complete selection first"
        )
    ordinary = []
    if not raw_tuning:
        locations = [path]
        shared_runs = run_store(path)
        if path.is_dir() and shared_runs != path and shared_runs.is_dir():
            locations.append(shared_runs)
        for location in locations:
            ordinary.extend(_load_results(location, required=False))
        tuned = _tuning_result_names(scope)
        ordinary = [
            record
            for record in ordinary
            if Path(record.get("_source_file", "")).name not in tuned
        ]
    records = (selected or []) + ordinary
    unique = {}
    for record in records:
        source = str(Path(record["_source_file"]).resolve())
        unique[source] = record
    if not unique:
        raise ConfigurationComparisonError(
            f"no selected or ordinary run results found at {path}"
        )
    return list(unique.values())


def _group_results(records: list[dict]) -> dict[tuple[str, str], dict[str, list[dict]]]:
    by_context = {}
    for record in records:
        by_context.setdefault(comparison_context(record), []).append(record)

    return {
        context: _group_context(context_records)
        for context, context_records in sorted(by_context.items())
    }


def _group_context(records: list[dict]) -> dict[str, list[dict]]:
    grouped = {}
    identities = {}
    for record in records:
        identity = frozen_configuration(record)
        key = json.dumps(identity, sort_keys=True)
        grouped.setdefault(key, []).append(record)
        identities[key] = identity
    algorithm_counts = Counter(
        identity.get("algorithm") for identity in identities.values()
    )
    labelled = {}
    for key in sorted(grouped):
        identity = identities[key]
        algorithm = str(identity.get("algorithm"))
        label = algorithm
        if algorithm_counts[identity.get("algorithm")] > 1:
            label = f"{algorithm}@{fingerprint(_variant_identity(identity))}"
        if label in labelled:
            raise ConfigurationComparisonError(
                f"configuration label collision for {label}"
            )
        labelled[label] = grouped[key]
    return labelled


def _variant_identity(identity: dict) -> dict:
    experiment = dict(identity.get("experiment", {}))
    for field in (
        "data_backend",
        "dataset",
        "input_kind",
        "input_spec",
        "num_classes",
        "num_clients",
        "partition_id",
        "partition_scheme",
        "validation_fraction",
    ):
        experiment.pop(field, None)
    return {
        "algorithm": identity.get("algorithm"),
        "experiment": experiment,
        "algorithm_config": identity.get("algorithm_config", {}),
    }


def _contrasts(
    records: dict[str, list[dict]],
    *,
    reference: str | None,
    declared: list[tuple[str, str]],
    all_pairs: bool,
) -> list[tuple[str, str]]:
    if reference:
        if reference not in records:
            raise ConfigurationComparisonError(
                f"unknown reference {reference!r}; available labels: "
                + ", ".join(records)
            )
        contrasts = [(label, reference) for label in records if label != reference]
    elif all_pairs:
        contrasts = list(combinations(records, 2))
    elif declared:
        contrasts = declared
    elif len(records) == 2:
        contrasts = list(combinations(records, 2))
    else:
        raise ConfigurationComparisonError(
            "more than two configurations were found; use --reference, "
            "--contrast, or --all-pairs. Available labels: " + ", ".join(records)
        )
    unknown = sorted({label for pair in contrasts for label in pair} - set(records))
    if unknown:
        raise ConfigurationComparisonError(
            "unknown contrast label(s): " + ", ".join(unknown)
        )
    if len({tuple(sorted(pair)) for pair in contrasts}) != len(contrasts):
        raise ConfigurationComparisonError("each contrast must be declared once")
    return contrasts


def main(argv: list[str] | None = None, *, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Compare frozen configurations on matched test results."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="results or selection artifacts (default: results)",
    )
    parser.add_argument(
        "--contrast",
        action="append",
        default=[],
        type=_contrast,
        metavar="LEFT:RIGHT",
        help=(
            "predeclared comparison; required when more than two "
            "configurations are given"
        ),
    )
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument(
        "--reference",
        help="compare every other discovered configuration with this label",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="compare every pair of discovered configurations",
    )
    parser.add_argument(
        "--selection-view", choices=["global", "per-client"], default="global"
    )
    parser.add_argument(
        "--selection-aggregation",
        choices=["mean", "weighted_mean"],
        default="mean",
    )
    parser.add_argument(
        "--tie-break", choices=["earliest", "latest"], default="earliest"
    )
    parser.add_argument("--practical-threshold", type=float, required=True)
    parser.add_argument("--out", help="write the Markdown table")
    parser.add_argument(
        "--out-json",
        help="output path (default: comparison.json inside the results directory)",
    )
    args = parser.parse_args(argv)

    try:
        results_dir = Path(args.results_dir)
        loaded = _comparison_results(results_dir)
        contexts = _group_results(loaded)
        comparison_modes = sum(
            bool(value) for value in (args.reference, args.contrast, args.all_pairs)
        )
        if comparison_modes > 1:
            raise ConfigurationComparisonError(
                "choose one of --reference, --contrast, or --all-pairs"
            )
        metric = resolve_metric(args.metric, source="--metric")
        comparisons = []
        labels_by_context = {}
        for context, records in contexts.items():
            if len(records) < 2:
                raise ConfigurationComparisonError(
                    f"dataset={context[0]}, data_configuration={context[1]} "
                    "contains fewer than two frozen configurations"
                )
            labels_by_context[f"{context[0]}:{context[1]}"] = list(records)
            for left, right in _contrasts(
                records,
                reference=args.reference,
                declared=args.contrast,
                all_pairs=args.all_pairs,
            ):
                comparisons.append(
                    compare_configurations(
                        records[left],
                        records[right],
                        metric,
                        left_label=left,
                        right_label=right,
                        view=args.selection_view,
                        aggregation=args.selection_aggregation,
                        tie_break=args.tie_break,
                        practical_threshold=args.practical_threshold,
                    )
                )
    except (ConfigurationComparisonError, ValueError) as error:
        raise SystemExit(f"cannot compare configurations: {error}") from error

    table = format_comparisons(comparisons)
    print(table)
    if args.out:
        atomic_write_text(Path(args.out), table + "\n")
        print(f"wrote {args.out}")
    out_json = Path(args.out_json) if args.out_json else (
        results_dir / "comparison.json"
        if results_dir.is_dir()
        else results_dir.parent / "comparison.json"
    )
    atomic_write_json(
        out_json,
        {
            "kind": "rigfl.comparison_collection",
            "results_dir": str(results_dir),
            "configuration_labels": labels_by_context,
            "comparisons": comparisons,
        },
    )
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
