"""Intensify a tuning search on additional replicate conditions."""

from __future__ import annotations

import argparse
import json
import statistics
from copy import deepcopy
from pathlib import Path

from rigfl.eval.comparison import apply_holm, compare_configurations
from rigfl.eval.report import selection_for
from rigfl.eval.selection import aggregate
from rigfl.experiment.artifacts import atomic_write_json, read_json
from rigfl.experiment.launch import (
    _validate_biosilo_partitions,
    _write_grid,
    expand,
    run_config,
)
from rigfl.experiment.tuning import _check_ranking, _write_final_selection


def _validation_value(record: dict, selection: dict, view: str) -> float:
    selected = selection_for(
        record,
        selection["metric"],
        view=view,
        aggregation=selection["client_aggregation"],
        tie_break=selection["round_tie_break"],
        include_test=False,
    )
    values = selected.get("validation", {}).get(selection["metric"], [])
    weights = (selected.get("sample_counts") or {}).get("validation")
    value = aggregate(values, weights, selection["client_aggregation"])
    if value is None:
        raise ValueError(
            f"run produced no validation {selection['metric']}"
        )
    return float(value)


def _resource_mean(records: list[dict], preference: str) -> float | None:
    values = []
    for record in records:
        resources = record.get("resources", {})
        if preference == "communication":
            value = (
                resources.get("observed", {})
                .get("communication_bytes", {})
                .get("total")
            )
        else:
            key = "wall_seconds" if preference == "time" else "flops"
            attributed = resources.get("attributed_training", {})
            value = attributed.get(key)
            if preference == "time" and not attributed.get(
                "wall_seconds_comparable"
            ):
                return None
        if value is None:
            return None
        values.append(float(value))
    return statistics.mean(values) if values else None


def _plan(ranking_path: Path) -> tuple[dict, dict]:
    artifact = read_json(ranking_path)
    _check_ranking(artifact)
    plan = artifact.get("intensification_plan")
    if not plan:
        raise SystemExit(
            "the ranking does not contain an intensification plan; declare "
            "tuning.intensification in the search configuration"
        )
    return artifact, plan


def _tasks(plan: dict) -> list[dict]:
    tasks = []
    seen = set()
    for shortlist in plan["shortlists"]:
        for item in shortlist["configurations"]:
            expanded, manifest = expand(item["configuration"])
            if manifest is not None:
                raise ValueError(
                    "an intensification configuration cannot define another search"
                )
            for task in expanded:
                identity = json.dumps(task, sort_keys=True)
                if identity not in seen:
                    seen.add(identity)
                    tasks.append(task)
    return tasks


def prepare_intensification(ranking_path: Path) -> Path:
    ranking_path = Path(ranking_path)
    _, plan = _plan(ranking_path)
    output_dir = ranking_path.parent / "intensification"
    tasks = _tasks(plan)
    _validate_biosilo_partitions(tasks)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / "grid.jsonl"
    _write_grid(grid_path, tasks)
    return grid_path


def _run_configuration(
    configuration: dict,
    results_dir: Path,
    force: bool,
    *,
    run_missing: bool,
) -> list[dict]:
    tasks, manifest = expand(configuration)
    if manifest is not None:
        raise ValueError("an intensification configuration cannot define another search")
    return [
        run_config(
            task,
            results_dir,
            force=force,
            task_label=f"intensification task {index}",
            run_missing=run_missing,
        )
        for index, task in enumerate(tasks, 1)
    ]


def _intensify_shortlist(
    shortlist: dict,
    plan: dict,
    selection: dict,
    results_dir: Path,
    force: bool,
    *,
    run_missing: bool,
) -> dict:
    candidates = []
    for item in shortlist["configurations"]:
        records = _run_configuration(
            item["configuration"],
            results_dir,
            force,
            run_missing=run_missing,
        )
        candidate = dict(item)
        validation_by_replicate = []
        for record in records:
            experiment = record["config"]["experiment"]
            validation_by_replicate.append(
                {
                    "partition_seed": experiment.get("partition_seed"),
                    "split_seed": experiment.get("split_seed"),
                    "experiment_seed": experiment["seed"],
                    "value": _validation_value(
                        record, selection, shortlist["selection_view"]
                    ),
                }
            )
        candidate["intensification_validation"] = {
            "mean": statistics.mean(
                replicate["value"] for replicate in validation_by_replicate
            ),
            "per_replicate": validation_by_replicate,
        }
        candidate["intensification_result_files"] = [
            record["_source_file"] for record in records
        ]
        candidate["_records"] = records
        candidates.append(candidate)

    reverse = selection["direction"] == "maximize"
    candidates.sort(
        key=lambda item: item["intensification_validation"]["mean"],
        reverse=reverse,
    )
    leader = candidates[0]
    comparisons = []
    for candidate in candidates[1:]:
        comparisons.append(
            compare_configurations(
                leader["_records"],
                candidate["_records"],
                selection["metric"],
                left_label=f"candidate_{leader['candidate_id']}",
                right_label=f"candidate_{candidate['candidate_id']}",
                view=shortlist["selection_view"],
                aggregation=selection["client_aggregation"],
                tie_break=selection["round_tie_break"],
                practical_threshold=plan["practical_threshold"],
                tail_fraction=plan["tail_fraction"],
                evaluation_split="validation",
            )
        )
    apply_holm(comparisons)

    equivalent_ids = {leader["candidate_id"]}
    for comparison, candidate in zip(comparisons, candidates[1:]):
        if comparison["practical_conclusion"] == "practically_equivalent":
            equivalent_ids.add(candidate["candidate_id"])

    warnings = []
    selected = leader
    if plan["prefer"] != "validation" and len(equivalent_ids) > 1:
        scored = []
        if plan["prefer"] == "time":
            hardware = {
                json.dumps(
                    record.get("resources", {})
                    .get("measurement", {})
                    .get("timing", {})
                    .get("hardware"),
                    sort_keys=True,
                )
                for candidate in candidates
                if candidate["candidate_id"] in equivalent_ids
                for record in candidate["_records"]
            }
            if len(hardware) != 1:
                warnings.append(
                    "time measurements were not collected on matching hardware; "
                    "validation ranking was used"
                )
                scored = None
        if scored is not None:
            for candidate in candidates:
                if candidate["candidate_id"] not in equivalent_ids:
                    continue
                value = _resource_mean(candidate["_records"], plan["prefer"])
                if value is None:
                    warnings.append(
                        f"{plan['prefer']} measurements were not available for every "
                        "practically equivalent candidate; validation ranking was used"
                    )
                    scored = []
                    break
                scored.append((value, candidate))
        if scored:
            selected = min(
                scored, key=lambda item: (item[0], item[1]["candidate_id"])
            )[1]

    selected_because = (
        f"lowest {plan['prefer']} among candidates demonstrated to be practically "
        "equivalent to the validation leader"
        if selected["candidate_id"] != leader["candidate_id"]
        else "highest intensification validation score"
        if selection["direction"] == "maximize"
        else "lowest intensification validation score"
    )
    clean_candidates = []
    for candidate in candidates:
        saved = dict(candidate)
        saved.pop("_records")
        clean_candidates.append(saved)
    return {
        "group_id": shortlist["group_id"],
        "group_key": shortlist["group_key"],
        "label": shortlist["label"],
        "algorithm": shortlist["algorithm"],
        "condition": shortlist["condition"],
        "selection_view": shortlist["selection_view"],
        "candidates": clean_candidates,
        "comparisons_to_validation_leader": comparisons,
        "validation_leader": leader["candidate_id"],
        "practically_equivalent_candidates": sorted(equivalent_ids),
        "selected_candidate": selected["candidate_id"],
        "selected_configuration": selected["configuration"],
        "selected_result_files": selected["intensification_result_files"],
        "selected_because": selected_because,
        "warnings": warnings,
    }


def finalize_intensification(
    ranking_path: Path,
    *,
    force: bool = False,
    run_missing: bool = False,
) -> dict:
    if force and not run_missing:
        raise SystemExit("force requires run_missing=True")
    ranking_path = Path(ranking_path)
    ranking_artifact, plan = _plan(ranking_path)

    output_dir = ranking_path.parent / "intensification"
    results_dir = (ranking_path.parent / ranking_artifact["run_store"]).resolve()
    groups = [
        _intensify_shortlist(
            shortlist,
            plan,
            ranking_artifact["selection_protocol"],
            results_dir,
            force,
            run_missing=run_missing,
        )
        for shortlist in plan["shortlists"]
    ]
    run_store_path = Path(ranking_artifact["run_store"])
    for group in groups:
        group["selected_result_files"] = [
            str(Path("..") / run_store_path / Path(path).name)
            for path in group["selected_result_files"]
        ]
    artifact = {
        "schema_version": 1,
        "kind": "rigfl.tuning_intensification_evaluation",
        "ranking": str(ranking_path),
        "selection_protocol": {
            **ranking_artifact["selection_protocol"],
            "intensification_replicates": plan["intensification_replicates"],
            "shortlist_size": plan["top_k"],
            "practical_threshold": plan["practical_threshold"],
            "tail_fraction": plan["tail_fraction"],
            "equivalent_preference": plan["prefer"],
        },
        "groups": groups,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "evaluation.json", artifact)
    selected_groups = deepcopy(groups)
    for group in selected_groups:
        group["selected_result_files"] = [
            str(run_store_path / Path(path).name)
            for path in group["selected_result_files"]
        ]
    _write_final_selection(
        ranking_artifact,
        selected_groups,
        ranking_path.parent,
        selected_from="intensification",
    )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize an intensified tuning selection."
    )
    parser.add_argument("--ranking", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--run",
        action="store_true",
        help="run missing intensification tasks locally before finalizing",
    )
    args = parser.parse_args()
    if args.force and not args.run:
        parser.error("--force requires --run")
    artifact = finalize_intensification(
        Path(args.ranking),
        force=args.force,
        run_missing=args.run,
    )
    print(f"intensified {len(artifact['groups'])} tuning group(s)")


if __name__ == "__main__":
    main()
