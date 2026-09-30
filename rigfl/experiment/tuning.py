"""Rank an HPO study's candidates on validation scores over its replicates.

A candidate is one joint assignment of the searched parameters. The replicate
conditions are excluded from candidate identity; a candidate is ranked only
when every declared replicate produced a validation score.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import yaml

from rigfl.eval.report import (
    replicate_statistics,
    selected_metric_value,
    selection_for,
)
from rigfl.eval.selection import SelectionError
from rigfl.experiment.artifacts import (
    atomic_write_json,
    atomic_write_text,
    dumps,
    loads,
)
from rigfl.experiment.config import experiment_fields, fingerprint

MANIFEST_NAME = "study.json"
MANIFEST_KIND = "rigfl.tuning_study"

ARTIFACT_KIND = "rigfl.tuning_ranking"


class TuningError(ValueError):
    """Candidate selection cannot be completed."""


@dataclass
class Candidate:
    """One complete joint assignment of the searched parameters."""
    id: int
    algorithm: str
    parameters: dict            # "experiment.<field>" or "algorithm.<field>" -> value
    config_hash: str

    def to_dict(self) -> dict:
        return {"id": self.id, "algorithm": self.algorithm,
                "parameters": dict(self.parameters), "config_hash": self.config_hash}


def candidate_hash(algorithm: str, parameters: dict) -> str:
    """A stable hash recorded beside a candidate's integer id."""
    payload = {"algorithm": algorithm,
               "parameters": {k: parameters[k] for k in sorted(parameters)}}
    return fingerprint(payload)


def write_manifest(manifest: dict, sweep_dir: Path) -> Path:
    return atomic_write_json(Path(sweep_dir) / MANIFEST_NAME, manifest)


def validation_score(
    record: dict, metric: str, *, view: str, aggregation: str
) -> tuple[float, dict]:
    """One run's validation score at its selected round(s), and the selection."""
    selected = selection_for(
        record, metric, view=view, aggregation=aggregation, include_test=False
    )
    score = selected_metric_value(selected, metric, "validation", aggregation)
    if score is None:
        raise ValueError(f"run produced no validation {metric}")
    return score, selected


def candidate_runs(records: list[dict], manifest: dict) -> dict[int, dict]:
    """``{candidate id: {training seed: record}}`` from the study's evaluations."""
    by_file = {record["_source_file"]: record for record in records}
    return {
        evaluation["candidate_id"]: {
            by_file[name]["config"]["experiment"]["training_seed"]: by_file[name]
            for name in evaluation["source_results"]
            if name in by_file
        }
        for evaluation in manifest["evaluations"]
    }


def rank(records: list[dict], manifest: dict, *, metric: str, view: str,
         aggregation: str = "uniform") -> dict:
    """Rank the study's candidates on validation only.

    Returns the ranking artifact. Test values are not read or included.
    """
    expected = [c["training_seed"] for c in manifest["replicate_conditions"]]
    runs = candidate_runs(records, manifest)
    rows = [
        _candidate_row(c, runs.get(c["id"], {}), expected, metric, view, aggregation)
        for c in manifest["candidates"]
    ]
    return {
        "kind": ARTIFACT_KIND,
        "study": {
            "path": manifest.get("_path"),
            "name": manifest.get("name"),
            "engine": manifest.get("engine"),
        },
        "selection_protocol": {
            "metric": metric,
            "split": "validation",
            "selection_view": view,
            "client_weighting": aggregation,
            "seed_aggregation": "mean",
            "direction": "maximize",
        },
        "expected_seeds": expected,
        "candidates": rows,
        "ranking": _order(rows),
    }


def _candidate_row(candidate: dict, runs: dict, expected: list,
                   metric: str, view: str, aggregation: str) -> dict:
    """One candidate's per-replicate validation scores and their mean."""
    per_seed: dict[str, dict] = {}
    scores = []
    errors: dict[str, str] = {}
    for seed in sorted(runs, key=str):
        try:
            value, selected = validation_score(
                runs[seed], metric, view=view, aggregation=aggregation
            )
        except (SelectionError, ValueError) as e:
            errors[str(seed)] = str(e)
            continue
        s = {"validation": value}
        if view == "shared":
            s["selected_round"] = selected["selected_round"]
        else:
            s["selected_rounds"] = selected["selected_rounds"]
        s["source_file"] = runs[seed].get("_source_file")
        per_seed[str(seed)] = s
        scores.append(value)

    missing = [s for s in expected if str(s) not in per_seed]
    reason = None
    if missing:
        reason = f"missing replicate(s) {missing} of expected {expected}"
        if errors:
            reason += "; " + "; ".join(f"training seed {k}: {v}" for k, v in errors.items())
    stats = replicate_statistics(scores)
    return {
        "id": candidate["id"],
        "algorithm": candidate["algorithm"],
        "parameters": dict(candidate["parameters"]),
        "config_hash": candidate["config_hash"],
        "validation": {"mean": stats["mean"],
                       "std": 0.0 if stats["n"] == 1 else stats["sd"],
                       "ci": stats["ci_half_width"], "n": stats["n"]},
        "observed_seeds": [s for s in expected if str(s) in per_seed],
        "missing_seeds": missing,
        "eligible": not missing,
        "ineligible_reason": reason,
        "errors": errors,
        "per_seed": per_seed,
    }


def _order(rows: list[dict]) -> dict:
    """Order eligible candidates by mean validation score; equal scores share a rank."""
    pool = sorted((r for r in rows if r["eligible"]),
                  key=lambda r: (-r["validation"]["mean"], r["id"]))
    previous_score = object()
    current_rank = 0
    for position, row in enumerate(pool, 1):
        score = row["validation"]["mean"]
        if score != previous_score:
            current_rank = position
            previous_score = score
        row["rank"] = current_rank
    for r in rows:
        r.setdefault("rank", None)

    best = [r["id"] for r in pool if r["rank"] == 1]
    return {
        "order": [r["id"] for r in pool],
        "best_candidate_ids": best,
        "selected_candidate_id": best[0] if len(best) == 1 else None,
        "excluded": [{"id": r["id"], "reason": r["ineligible_reason"]}
                     for r in rows if not r["eligible"]],
    }


def selected_configuration(record: dict, manifest: dict) -> dict:
    """A directly runnable sweep spec for the winning configuration.

    Built from a completed run's resolved config; seeds go under ``replicates``.
    """
    # Keep only user-settable experiment fields; resolved data facts are
    # recomputed from the dataset registry. Null fields don't apply to the
    # algorithm.
    exp = {
        key: value
        for key, value in experiment_fields(
            record.get("config", {}).get("experiment", {})
        ).items()
        if value is not None
    }
    acfg = deepcopy(record.get("config", {}).get("algorithm", {}))
    for name in ("partition_seed", "split_seed", "training_seed"):
        exp.pop(name, None)
    return {
        "base": {**exp, **acfg},
        "sweep": {"algorithm": [record["algorithm"]]},
        "replicates": list(manifest["replicate_conditions"]),
    }


def write_ranking(
    artifact: dict, records: list[dict], manifest: dict, out_dir: Path
) -> list[Path]:
    """Write the validation ranking and, for a unique winner, selected.yaml."""
    out_dir = Path(out_dir)
    normalized = loads(dumps(artifact, indent=None))
    normalized["run_store"] = manifest["run_store"]
    ranking = normalized["ranking"]
    selected = None
    candidate_id = ranking["selected_candidate_id"]
    if candidate_id is not None:
        runs = candidate_runs(records, manifest)[candidate_id]
        selected = selected_configuration(runs[min(runs, key=str)], manifest)

    out_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = out_dir / "ranking.json"
    atomic_write_json(ranking_path, normalized)
    written = [ranking_path]
    selected_path = out_dir / "selected.yaml"
    if selected is not None:
        atomic_write_text(selected_path, _dump(selected))
        written.append(selected_path)
    elif ranking["best_candidate_ids"]:
        selected_path.unlink(missing_ok=True)
        ids = ", ".join(str(value) for value in ranking["best_candidate_ids"])
        print(
            f"candidates {ids} tied for the best aggregate validation score; "
            "no selected.yaml was written"
        )
    else:
        raise TuningError("no eligible configuration can be selected")
    return written


def _dump(payload: dict) -> str:
    return yaml.safe_dump(loads(dumps(payload, indent=None)), sort_keys=False)
