"""Describe performance sensitivity across a fully crossed seed sweep."""

from __future__ import annotations

import argparse
import itertools
import statistics
from collections import defaultdict
from pathlib import Path

from rigfl.eval.report import (
    REPLICATE_FIELDS,
    replicate_key,
    selected_metric_value,
    selection_for,
)
from rigfl.eval.selection import (
    SelectionError,
    resolve_metric,
)
from rigfl.experiment.artifacts import atomic_write_json, atomic_write_text
from rigfl.experiment.collect import (
    algorithm_settings,
    algorithm_variant,
    condition_fields,
    experiment_condition,
    load_results,
    varying_fields,
)
from rigfl.experiment.paths import flatten_mapping
from rigfl.experiment.storage import records_for_grid, run_store

SEED_LABELS = {
    "partition_seed": "partition seed",
    "split_seed": "split seed",
    "training_seed": "training seed",
}


class VariancePilotError(ValueError):
    """Raised when results do not form an analyzable crossed seed design."""


def _seed_condition(record: dict) -> tuple[int, int, int]:
    values = replicate_key(record)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values
    ):
        source = record.get("_source_file", "result record")
        raise VariancePilotError(
            f"{source} must record nonnegative partition_seed, split_seed, and training_seed"
        )
    return values


def _validation_score(
    record: dict,
    metric: str,
    *,
    view: str,
    aggregation: str,
) -> dict:
    selected = selection_for(
        record,
        metric,
        view=view,
        aggregation=aggregation,
        include_test=False,
    )
    value = selected_metric_value(
        selected, metric, "validation", aggregation
    )
    if value is None:
        source = record.get("_source_file", "result record")
        raise VariancePilotError(f"{source} has no validation {metric}")
    result = {
        "validation_score": float(value),
        "mixed_rounds": bool(selected.get("mixed_rounds")),
    }
    for field in ("selected_round", "selected_rounds"):
        if field in selected:
            result[field] = selected[field]
    return result


def _source_ranking(sources: dict) -> tuple[list[str], list[list[str]]]:
    ordered = sorted(
        sources,
        key=lambda field: (-sources[field]["marginal_spread"], field),
    )
    tied = []
    rank = 0
    prior_display = None
    for position, field in enumerate(ordered, 1):
        spread = sources[field]["marginal_spread"]
        displayed = f"{spread:.4f}"
        if displayed != prior_display:
            rank = position
        sources[field]["rank"] = rank
        prior_display = displayed
    for rank_value in sorted({sources[field]["rank"] for field in ordered}):
        fields = [field for field in ordered if sources[field]["rank"] == rank_value]
        if len(fields) > 1:
            tied.append(fields)
    return ordered, tied


def _analyze_group(
    records: list[dict],
    metric: str,
    *,
    view: str,
    aggregation: str,
) -> dict:
    indexed = {}
    for record in records:
        condition = _seed_condition(record)
        if condition in indexed:
            sources = [
                indexed[condition].get("_source_file", "result record"),
                record.get("_source_file", "result record"),
            ]
            raise VariancePilotError(
                f"duplicate seed condition {condition}: {', '.join(sources)}"
            )
        indexed[condition] = record

    levels = [sorted({condition[index] for condition in indexed}) for index in range(3)]
    insufficient = [
        SEED_LABELS[field]
        for field, values in zip(REPLICATE_FIELDS, levels)
        if len(values) < 2
    ]
    if insufficient:
        raise VariancePilotError(
            "crossed seed sensitivity needs at least two values for each seed; "
            "insufficient: " + ", ".join(insufficient)
        )

    expected = set(itertools.product(*levels))
    missing = sorted(expected - set(indexed))
    if missing:
        rendered = ", ".join(str(condition) for condition in missing)
        raise VariancePilotError(
            f"crossed seed grid is incomplete; missing {len(missing)} condition(s): "
            + rendered
        )

    cells = []
    mixed_rounds = []
    for condition in sorted(indexed):
        record = indexed[condition]
        selection = _validation_score(
            record,
            metric,
            view=view,
            aggregation=aggregation,
        )
        mixed_rounds.append(selection["mixed_rounds"])
        cell = {
            "partition_seed": condition[0],
            "split_seed": condition[1],
            "training_seed": condition[2],
            "source_file": record.get("_source_file"),
            **selection,
        }
        cells.append(cell)

    sources = {}
    for index, field in enumerate(REPLICATE_FIELDS):
        marginal_means = []
        for level in levels[index]:
            scores = [
                cell["validation_score"]
                for cell in cells
                if cell[field] == level
            ]
            marginal_means.append(
                {"seed": level, "mean": statistics.mean(scores), "runs": len(scores)}
            )
        means = [item["mean"] for item in marginal_means]
        sources[field] = {
            "marginal_means": marginal_means,
            "marginal_spread": max(means) - min(means),
        }

    first = records[0]
    ranking, ties = _source_ranking(sources)
    return {
        "algorithm": first["algorithm"],
        "experiment": condition_fields(first),
        "algorithm_configuration": algorithm_settings(first),
        "selection_view": view,
        "mixed_rounds": any(mixed_rounds),
        "runs": len(cells),
        "levels": {
            "partition_seed": levels[0],
            "split_seed": levels[1],
            "training_seed": levels[2],
        },
        "overall_mean": statistics.mean(
            cell["validation_score"] for cell in cells
        ),
        "overall_range": (
            max(cell["validation_score"] for cell in cells)
            - min(cell["validation_score"] for cell in cells)
        ),
        "seed_sources": sources,
        "ranked_seed_sources": ranking,
        "source_ranking_ties": ties,
        "cells": cells,
    }


def _add_group_labels(groups: list[dict], records: list[dict]) -> None:
    experiment_fields = varying_fields(records)
    if len(experiment_fields) > 1 and "data_configuration" in experiment_fields:
        experiment_fields.remove("data_configuration")
    by_algorithm: dict[str, list[dict]] = defaultdict(list)
    for group in groups:
        by_algorithm[group["algorithm"]].append(group)

    for algorithm_groups in by_algorithm.values():
        flattened = [
            flatten_mapping(group["algorithm_configuration"])
            for group in algorithm_groups
        ]
        algorithm_fields = sorted({
            field
            for config in flattened
            for field in config
            if len({repr(item.get(field)) for item in flattened}) > 1
        })
        for group, config in zip(algorithm_groups, flattened):
            details = [
                f"{field}={group['experiment'].get(field)}"
                for field in experiment_fields
            ]
            details.extend(
                f"{field}={config.get(field)}" for field in algorithm_fields
            )
            group["label"] = group["algorithm"] + (
                " (" + ", ".join(details) + ")" if details else ""
            )


def analyze_variance_pilot(
    records: list[dict],
    metric: str = "accuracy",
    *,
    view: str = "shared",
    aggregation: str = "uniform",
) -> dict:
    """Analyze complete Cartesian combinations of the three RigFL seed controls."""
    name = resolve_metric(metric)
    grouped = defaultdict(list)
    for record in records:
        key = (
            record.get("algorithm"),
            experiment_condition(record),
            algorithm_variant(record),
        )
        grouped[key].append(record)
    if not grouped:
        raise VariancePilotError("no run results were provided")

    groups = [
        _analyze_group(
            grouped[key],
            name,
            view=view,
            aggregation=aggregation,
        )
        for key in sorted(grouped, key=repr)
    ]
    _add_group_labels(groups, records)
    return {
        "kind": "rigfl.seed_sensitivity",
        "selection": {
            "metric": name,
            "direction": "maximize",
            "split": "validation",
            "round_selection": view,
            "client_weighting": aggregation,
        },
        "interpretation": (
            "Marginal spreads are descriptive diagnostics, not variance-component "
            "estimates, significance tests, or causal attributions, and do not "
            "measure interactions between seed sources."
        ),
        "groups": groups,
    }


def format_variance_pilot(artifact: dict) -> str:
    lines = ["# Crossed seed sensitivity", "", artifact["interpretation"]]
    metric = artifact["selection"]["metric"]
    for index, group in enumerate(artifact["groups"], 1):
        lines.extend(
            [
                "",
                f"## {index}. {group['label']}",
                "",
                (
                    f"Overall mean validation {metric}: "
                    f"{group['overall_mean']:.4f} across {group['runs']} runs "
                    f"({len(group['levels']['partition_seed'])} partition, "
                    f"{len(group['levels']['split_seed'])} split, and "
                    f"{len(group['levels']['training_seed'])} training seed "
                    "values)."
                ),
                (
                    "Seed values: "
                    f"partition={group['levels']['partition_seed']}; "
                    f"split={group['levels']['split_seed']}; "
                    f"training={group['levels']['training_seed']}."
                ),
                "",
                "| rank | seed source | marginal means | spread |",
                "|---:|---|---|---:|",
            ]
        )
        for field in group["ranked_seed_sources"]:
            source = group["seed_sources"][field]
            means = ", ".join(
                f"{item['seed']}: {item['mean']:.4f}"
                for item in source["marginal_means"]
            )
            lines.append(
                f"| {source['rank']} | {field} | {means} | "
                f"{source['marginal_spread']:.4f} |"
            )
        if group["source_ranking_ties"]:
            rendered = "; ".join(", ".join(fields)
                                 for fields in group["source_ranking_ties"])
            lines.extend(["", f"Tied marginal spreads: {rendered}."])
        if group["mixed_rounds"]:
            lines.extend([
                "",
                "Each cell aggregates clients from their separately selected rounds.",
            ])
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Analyze a complete crossed sweep over RigFL's three seed controls."
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="base directory containing completed runs (default: results)",
    )
    parser.add_argument("--grid", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument(
        "--round-selection",
        choices=["shared", "client-specific"],
        default="shared",
    )
    parser.add_argument(
        "--client-weighting",
        choices=["uniform", "sample_count"],
        default="uniform",
    )
    parser.add_argument("--out-json")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    by_algorithm = load_results(run_store(args.results_root), args.dataset)
    records = [record for values in by_algorithm.values() for record in values]
    try:
        records = records_for_grid(records, args.grid)
    except ValueError as error:
        raise SystemExit(f"[seed-sensitivity] {error}") from error
    try:
        artifact = analyze_variance_pilot(
            records,
            args.metric,
            view=args.round_selection,
            aggregation=args.client_weighting,
        )
    except (VariancePilotError, SelectionError) as error:
        raise SystemExit(f"[seed-sensitivity] {error}") from error

    rendered = format_variance_pilot(artifact)
    print(rendered)
    if args.out_json:
        atomic_write_json(Path(args.out_json), artifact)
    if args.out:
        atomic_write_text(Path(args.out), rendered + "\n")


if __name__ == "__main__":
    main()
