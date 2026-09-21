"""Rank complete hyperparameter candidates over replicate runs.

A candidate is one joint assignment of its tuned parameters. The replicate axis
is excluded from candidate identity, and validation scores rank candidates.
"""

from __future__ import annotations

import json
import statistics
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rigfl.eval.metrics import canonical, direction_of
from rigfl.eval.report import mean_ci, selection_for
from rigfl.eval.selection import SelectionError, aggregate
from rigfl.experiment.artifacts import (
    ResultValidationError,
    atomic_write_json,
    atomic_write_text,
    dumps,
    loads,
    read_json,
)
from rigfl.experiment.config import (
    ExperimentConfig,
    IntensificationConfig,
    ReplicateCondition,
    replicate_conditions_from_count,
    algorithm_identity,
    fingerprint,
    hashable,
)
from rigfl.experiment.paths import (
    flatten_mapping,
    model_has_path,
    nested_delete,
    nested_get,
)
from rigfl.experiment.registry import config_class, ignores_experiment_field

#: Bumped when the study layout changes in a way a reader must notice.
MANIFEST_SCHEMA_VERSION = 6
MANIFEST_NAME = "study.json"
MANIFEST_KIND = "rigfl.tuning_study"

ARTIFACT_SCHEMA_VERSION = 2
ARTIFACT_KIND = "rigfl.tuning_ranking"
SELECTION_SCHEMA_VERSION = 1
SELECTION_KIND = "rigfl.tuning_selection"

#: Tie-breaks for equal aggregate validation scores.
CANDIDATE_TIE_BREAKS = ("lowest_id",)

SEED_AGGREGATION = "mean"

class TuningError(ValueError):
    """Raised when candidate selection is asked for and cannot be done honestly."""


# ── Axis paths ───────────────────────────────────────────────────────────────

def canonical_axis(path: str) -> str:
    """Require an explicit experiment or algorithm configuration path."""
    p = str(path).strip()
    if p.startswith(("experiment.", "algorithm.")):
        return p
    raise SystemExit(
        f"Configuration path {p!r} must start with 'experiment.' or 'algorithm.'."
    )


def _split(path: str) -> tuple[str, str]:
    section, field_name = canonical_axis(path).split(".", 1)
    return section, field_name


def _norm(v):
    """A value comparable across the YAML -> grid -> pydantic -> JSON round trip.

    ``--sweep batch=16,32`` puts the string ``"16"`` on the axis, and the
    result file holds the integer ``16`` that ``ExperimentConfig`` parsed. They
    are the same experimental setting and must match.
    """
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return tuple(_norm(x) for x in v)
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    try:
        return float(s)
    except ValueError:
        return s


# ── The tuning declaration ───────────────────────────────────────────────────

def parse_intensification(
    value, screening_conditions: list[dict[str, int]] | None
) -> IntensificationConfig | None:
    if value is None:
        return None
    if isinstance(value, IntensificationConfig):
        parsed = value
    else:
        from pydantic import ValidationError

        try:
            parsed = IntensificationConfig.model_validate(value)
        except ValidationError as error:
            first = error.errors(include_url=False)[0]
            location = ".".join(str(part) for part in first["loc"])
            raise SystemExit(
                f"invalid tuning.intensification at {location}: {first['msg']}"
            ) from error
    if screening_conditions is None:
        raise SystemExit(
            "tuning.intensification requires top-level replicates so its data and "
            "training conditions can be checked against the screening conditions"
        )
    if isinstance(parsed.replicates, int):
        # A count continues the screening seeds, so the conditions are new by
        # construction rather than by the check below.
        parsed = parsed.model_copy(
            update={
                "replicates": [
                    ReplicateCondition(**condition)
                    for condition in replicate_conditions_from_count(
                        parsed.replicates, start=len(screening_conditions)
                    )
                ]
            }
        )
    replicates = [condition.model_dump() for condition in parsed.replicates]
    screening_data = {
        (item["partition_seed"], item["split_seed"])
        for item in screening_conditions
    }
    repeated_data = sorted(
        screening_data
        & {
            (item["partition_seed"], item["split_seed"])
            for item in replicates
        }
    )
    if repeated_data:
        raise SystemExit(
            "intensification replicates must use data-seed pairs not used for "
            f"screening; repeated pairs: {repeated_data}"
        )
    return parsed


# ── Candidates ───────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    """One complete joint assignment of an algorithm's tuning parameters."""
    id: int
    algorithm: str
    parameters: dict            # canonical axis path -> value
    config_hash: str

    def to_dict(self) -> dict:
        return {"id": self.id, "algorithm": self.algorithm,
                "parameters": dict(self.parameters), "config_hash": self.config_hash}

    @property
    def key(self) -> tuple:
        return tuple(sorted((k, _norm(v)) for k, v in self.parameters.items()))


def candidate_hash(algorithm: str, parameters: dict) -> str:
    """A stable hash of a candidate, recorded *beside* its integer id.

    It survives renumbering and is convenient for cross-referencing, but it is
    not the identity: the integer id and the parameter mapping are.
    """
    payload = {"algorithm": algorithm,
               "parameters": {k: parameters[k] for k in sorted(parameters)}}
    return fingerprint(payload)


def applicable_parameters(algorithm: str, parameters: list[str]) -> list[str]:
    """The tuning parameters this algorithm actually has, in declaration order.

    Algorithm fields apply only where defined, and experiment fields apply only
    where used. Other axes do not mint duplicate "not applicable" candidates.
    """
    out = []
    for p in parameters:
        section, name = _split(p)
        if section == "algorithm" and not model_has_path(
            config_class(algorithm), name
        ):
            continue
        if section == "experiment" and ignores_experiment_field(algorithm, name):
            continue
        out.append(p)
    return out


def candidate_index(candidates: list[Candidate]) -> dict[tuple, int]:
    """``(algorithm, normalised parameter key) -> candidate id``."""
    return {(c.algorithm, c.key): c.id for c in candidates}


def write_manifest(manifest: dict, sweep_dir: Path) -> Path:
    """Install the manifest atomically, checking it reads back as one.

    A half-written manifest is worse than none: ``load_manifest`` would refuse
    it, and every task in the sweep would then look untuned.
    """
    path = Path(sweep_dir) / MANIFEST_NAME
    return atomic_write_json(path, manifest, validate=_check_manifest)


def _check_manifest(parsed: dict) -> None:
    if parsed.get("kind") != MANIFEST_KIND:
        raise TuningError(f"written manifest has kind {parsed.get('kind')!r}")
    if parsed.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise TuningError("written manifest has the wrong schema version")
    for key in (
        "algorithm",
        "candidates",
        "search_space",
        "tuning_parameters",
        "replicate_axis",
        "replicate_values",
        "condition_axes",
        "selection_protocol",
    ):
        if key not in parsed:
            raise TuningError(f"written manifest is missing {key!r}")


def load_manifest(results_dir: Path) -> Optional[dict]:
    """The tuning manifest in a results directory, or None if the sweep had none."""
    path = Path(results_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    if manifest.get("kind") != MANIFEST_KIND:
        raise TuningError(f"{path} is not a RigFL tuning manifest "
                          f'(kind={manifest.get("kind")!r}).')
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise TuningError(
            f"{path} uses manifest schema {manifest.get('schema_version')}, and this "
            f"version reads {MANIFEST_SCHEMA_VERSION}. Re-declare the sweep with the "
            f"current launcher rather than reading it under the wrong schema.")
    manifest["_path"] = str(path.resolve())
    return manifest


def manifest_candidates(manifest: dict) -> list[Candidate]:
    return [Candidate(int(c["id"]), c["algorithm"], dict(c["parameters"]),
                      c.get("config_hash", ""))
            for c in manifest["candidates"]]


# ── Placing a completed run ──────────────────────────────────────────────────

def run_parameters(record: dict, parameters: list[str], algorithm: str) -> dict:
    """The tuning-parameter values a completed run actually used."""
    cfg = record.get("config", {})
    out = {}
    for p in applicable_parameters(algorithm, parameters):
        section, name = _split(p)
        source = cfg.get("algorithm", {}) if section == "algorithm" else cfg.get("experiment", {})
        out[p] = nested_get(source, name)
    return out


def replicate_of(record: dict, manifest: dict):
    """The run's value on the replicate axis."""
    conditions = manifest.get("replicate_conditions")
    if conditions:
        experiment = record.get("config", {}).get("experiment", {})
        actual = {
            "partition_seed": experiment.get("partition_seed"),
            "split_seed": experiment.get("split_seed"),
            "experiment_seed": experiment.get("seed"),
        }
        if not any(
            all(_norm(actual[key]) == _norm(condition[key]) for key in actual)
            for condition in conditions
        ):
            return None
    section, name = _split(manifest["replicate_axis"])
    cfg = record.get("config", {})
    source = cfg.get("algorithm", {}) if section == "algorithm" else cfg.get("experiment", {})
    return _norm(nested_get(source, name))


def candidate_of(record: dict, manifest: dict, index: dict[tuple, int]) -> Optional[int]:
    """The candidate a completed run belongs to, or None if it matches none.

    Placement is derived from the resolved configuration recorded by the run.
    The tuning manifest defines the candidate combinations; result files do not
    carry a second, potentially contradictory candidate identity.
    """
    algorithm = record["algorithm"]
    params = run_parameters(record, manifest["tuning_parameters"], algorithm)
    key = (algorithm, tuple(sorted((k, _norm(v)) for k, v in params.items())))
    return index.get(key)


def effective_condition(record: dict, manifest: dict) -> dict:
    """The experimental condition a candidate is ranked *within*.

    Everything that makes two runs incomparable stays: dataset-partition
    identity, client count, early stopping, and any swept field the user did not
    declare as a tuning parameter. What is being searched comes out, and so does
    the replicate axis.

    Collector condition fields support comparisons across algorithms. Tuning
    also retains non-tuned algorithm settings so distinct configurations are
    not combined.
    """
    from rigfl.experiment.collect import condition_fields

    tuned = list(manifest["tuning_parameters"]) + list(
        manifest.get("replicate_fields", [manifest["replicate_axis"]])
    )
    drop_exp = {_split(p)[1] for p in tuned if _split(p)[0] == "experiment"}
    drop_algorithm = {_split(p)[1] for p in tuned
                      if _split(p)[0] == "algorithm"}

    cond = {k: v for k, v in condition_fields(record).items() if k not in drop_exp}
    exp = record.get("config", {}).get("experiment", {})
    acfg = algorithm_identity(record.get("config", {}).get("algorithm", {}))
    flat_acfg = flatten_mapping(acfg)
    for name, value in flat_acfg.items():
        if name not in drop_algorithm:
            cond[f"algorithm.{name}"] = hashable(value)
    for axis in manifest["condition_axes"]:
        section, name = _split(axis)
        if section == "algorithm":
            if name in flat_acfg:
                cond[f"algorithm.{name}"] = hashable(flat_acfg[name])
        elif (
            axis in applicable_parameters(record["algorithm"], [axis])
            and name not in cond
        ):
            cond[name] = hashable(exp.get(name))   # collector's headline list
    return cond


def group_key(record: dict, manifest: dict) -> tuple:
    return (record["algorithm"],) + tuple(sorted((k, repr(v))
                                              for k, v in effective_condition(record, manifest).items()))


# ── Ranking ──────────────────────────────────────────────────────────────────

def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "ci": None, "n": 0}
    mean, ci = mean_ci(values)
    return {"mean": mean,
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "ci": ci, "n": len(values)}


#: Fields worth naming when a single group needs a label at all.
_HEADLINE = ("dataset", "partition_id", "num_clients")


def label_keys(conditions: list[dict]) -> list[str]:
    """The condition fields worth putting in a label: the ones that differ.

    The full condition is the group's identity and is written out in full; this
    is only what a human reads. Labelling by a fixed list is what let two
    experiments render identically and collide, so the keys are derived from the
    groups actually present -- exactly as the collector does for its rows.
    """
    if len(conditions) > 1:
        seen: dict[str, set] = {}
        for cond in conditions:
            for k, v in cond.items():
                seen.setdefault(k, set()).add(repr(v))
        varying = [k for k, vals in seen.items() if len(vals) > 1]
        if varying:
            return sorted(varying)
    return [k for k in _HEADLINE
            if any(c.get(k) is not None for c in conditions)]


def _label(condition: dict, keys: list[str] | None = None) -> str:
    keys = sorted(condition) if keys is None else keys
    bits = [f"{k}={condition.get(k)}" for k in keys]
    return ", ".join(bits) or "(single condition)"


def place_records(records: list[dict], manifest: dict,
                  index: Optional[dict] = None) -> tuple[dict, dict, list]:
    """Put every completed run under its (tuning group, candidate, replicate).

    One function so that ranking and config writing cannot disagree about which
    run produced which candidate.
    """
    if index is None:
        index = candidate_index(manifest_candidates(manifest))
    placed: dict[tuple, dict[int, dict]] = {}
    conditions: dict[tuple, dict] = {}
    unassigned: list[dict] = []
    expected_replicates = {_norm(value) for value in manifest["replicate_values"]}
    for rec in records:
        cid = candidate_of(rec, manifest, index)
        if cid is None:
            unassigned.append({
                "algorithm": rec["algorithm"],
                "parameters": run_parameters(rec, manifest["tuning_parameters"],
                                             rec["algorithm"]),
                "reason": "no manifest candidate has these tuning-parameter values"})
            continue
        replicate = replicate_of(rec, manifest)
        if replicate is None:
            unassigned.append({
                "algorithm": rec["algorithm"],
                "parameters": run_parameters(
                    rec, manifest["tuning_parameters"], rec["algorithm"]
                ),
                "reason": "the run does not match a declared replicate condition",
            })
            continue
        if replicate not in expected_replicates:
            unassigned.append({
                "algorithm": rec["algorithm"],
                "parameters": run_parameters(
                    rec, manifest["tuning_parameters"], rec["algorithm"]
                ),
                "reason": "the run does not match a declared replicate condition",
            })
            continue
        gk = group_key(rec, manifest)
        conditions.setdefault(gk, effective_condition(rec, manifest))
        slot = placed.setdefault(gk, {}).setdefault(cid, {})
        if replicate in slot:
            first = slot[replicate].get("_source_file", "an earlier result")
            second = rec.get("_source_file", "another result")
            raise TuningError(
                f"Duplicate results for {rec['algorithm']} candidate {cid}, "
                f"replicate {replicate!r}: {first} and {second}."
            )
        slot[replicate] = rec
    return placed, conditions, unassigned


def rank(records: list[dict], manifest: dict, *, metric: str, views: list[str],
         aggregation: str = "mean", tie_break: str = "earliest",
         allow_incomplete: bool = False,
         candidate_tie_break: str = "lowest_id") -> dict:
    """Rank every candidate within every tuning group, on validation only.

    Returns the selection artifact. Test values are not read or included.
    """
    if candidate_tie_break not in CANDIDATE_TIE_BREAKS:
        raise TuningError(f'Unknown candidate tie-break "{candidate_tie_break}"; '
                          f'known: {", ".join(CANDIDATE_TIE_BREAKS)}.')

    candidates = manifest_candidates(manifest)
    index = candidate_index(candidates)
    by_algorithm: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_algorithm.setdefault(c.algorithm, []).append(c)

    # Matched on normalised values (the sweep may declare "0" where the result
    # holds 0), displayed as declared.
    display = {_norm(v): v for v in manifest["replicate_values"]}
    expected = [_norm(v) for v in manifest["replicate_values"]]
    direction = direction_of(metric)
    warnings: list[str] = []

    placed, conditions, unassigned = place_records(records, manifest, index)
    if unassigned:
        warnings.append(
            f"{len(unassigned)} result(s) match no candidate in the manifest and were "
            f"excluded from ranking; they are listed under 'unassigned_records'.")

    order = sorted(placed, key=str)
    keys = label_keys([conditions[gk] for gk in order])
    groups = []
    for gid, gk in enumerate(order):
        condition = conditions[gk]
        algorithm = gk[0]
        group_records = [record for candidate in placed[gk].values()
                         for record in candidate.values()]
        group_views = list(views)
        if len(views) > 1 and group_records:
            group_views = [
                view for view in views
                if all(view in record["result"].get(
                    "selection_views_supported", ["global", "per-client"])
                       for record in group_records)
            ]
            omitted = [view for view in views if view not in group_views]
            if omitted:
                warnings.append(
                    f"{algorithm} does not support selection view(s) "
                    f"{', '.join(omitted)}; those views were omitted for its "
                    "tuning group rather than duplicating a fallback view.")
        rows = []
        for cand in by_algorithm.get(algorithm, []):
            recs = placed[gk].get(cand.id, {})
            row = {**cand.to_dict(),
                   "expected_seeds": [display[s] for s in expected],
                   "observed_seeds": [display.get(s, s) for s in sorted(recs, key=str)],
                   "missing_seeds": [display[s] for s in expected if s not in recs],
                   "views": {}}
            for v in group_views:
                row["views"][v] = _candidate_view(recs, expected, display, metric, v,
                                                  aggregation, tie_break)
            rows.append(row)

        rankings = {}
        for v in group_views:
            rankings[v] = _rank_view(rows, v, direction, allow_incomplete,
                                     candidate_tie_break, warnings, algorithm,
                                     _label(condition, keys))
        groups.append({
            "group_id": gid,
            "group_key": str(gk),
            "algorithm": algorithm,
            "condition": condition,
            "label": f"{algorithm} [{_label(condition, keys)}]",
            "label_fields": keys,
            "expected_seeds": [display[s] for s in expected],
            "candidates": rows,
            "rankings": rankings,
        })

    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "kind": ARTIFACT_KIND,
        "study": {"path": manifest.get("_path"), "name": manifest.get("name"),
                     "engine": manifest.get("engine"),
                     "schema_version": manifest.get("schema_version")},
        "selection_protocol": {
            "metric": metric,
            "split": "validation",
            # Plural and unordered: neither view is primary, and one may pick a
            # different candidate than the other.
            "views": list(views),
            "client_aggregation": aggregation,
            "seed_aggregation": SEED_AGGREGATION,
            "direction": direction,
            "round_tie_break": tie_break,
            "candidate_tie_break": candidate_tie_break,
            "allow_incomplete": allow_incomplete,
            "ranked_on": "validation only",
        },
        "groups": groups,
        "unassigned_records": unassigned,
        "warnings": warnings,
    }


def _candidate_view(recs: dict, expected: list, display: dict, metric: str, view: str,
                    aggregation: str, tie_break: str) -> dict:
    """One candidate's per-seed scores and their aggregate, for one view."""
    per_seed: dict[str, dict] = {}
    val_scores = []
    errors: dict[str, str] = {}
    for seed in sorted(recs, key=str):
        try:
            selected = selection_for(
                recs[seed], metric, view=view, aggregation=aggregation,
                tie_break=tie_break, include_test=False,
            )
            name = canonical(metric)
            values = selected.get("validation", {}).get(name, [])
            weights = (selected.get("sample_counts") or {}).get("validation")
            value = aggregate(values, weights, aggregation) if values else None
        except (SelectionError, ValueError) as e:
            errors[str(seed)] = str(e)
            continue
        s = {
            "selection_view": selected.get("selection_view", view),
            "validation": value,
        }
        if selected.get("selected_round") is not None:
            s["selected_round"] = selected["selected_round"]
        elif "selected_rounds" in selected:
            s["selected_rounds"] = selected["selected_rounds"]
        s["source_file"] = recs[seed].get("_source_file")
        per_seed[str(display.get(seed, seed))] = s
        if s["validation"] is not None:
            val_scores.append(s["validation"])

    scored = [s for s in recs if str(display.get(s, s)) in per_seed]
    missing = [display.get(s, s) for s in expected if s not in scored]
    scored_display = [display.get(s, s) for s in scored]
    eligible = bool(val_scores) and not missing
    reason = None
    if not eligible:
        bits = []
        if missing:
            bits.append(f"missing replicate(s) {missing} of expected "
                        f"{[display.get(s, s) for s in expected]}")
        if not val_scores:
            bits.append("no run produced a validation score")
        if errors:
            bits.append("; ".join(f"seed {k}: {v}" for k, v in errors.items()))
        reason = "; ".join(bits)

    return {
        "validation": _stats(val_scores),
        "observed_seeds": scored_display,
        "missing_seeds": missing,
        "eligible": eligible,
        "ineligible_reason": reason,
        "errors": errors,
        "per_seed": per_seed,
    }


def _rank_view(rows: list[dict], view: str, direction: str, allow_incomplete: bool,
               candidate_tie_break: str, warnings: list[str], algorithm: str,
               label: str) -> dict:
    """Order one view's candidates by their aggregate validation score."""
    pool = [r for r in rows
            if r["views"][view]["validation"]["mean"] is not None
            and (r["views"][view]["eligible"] or allow_incomplete)]

    sign = -1.0 if direction == "maximize" else 1.0
    pool.sort(key=lambda r: (sign * r["views"][view]["validation"]["mean"], r["id"]))
    for i, r in enumerate(pool, 1):
        r["views"][view]["rank"] = i
    for r in rows:
        r["views"][view].setdefault("rank", None)

    counts = {len(r["views"][view]["observed_seeds"]) for r in pool}
    included_incomplete = [r["id"] for r in pool if not r["views"][view]["eligible"]]
    if included_incomplete:
        warnings.append(
            f"Incomplete ranking for {algorithm} [{label}] / {view}: candidate(s) "
            f"{included_incomplete} were ranked with missing replicates "
            f"(" + "; ".join(
                f"id {r['id']} has {len(r['views'][view]['observed_seeds'])} of "
                f"{len(r['expected_seeds'])}, missing {r['views'][view]['missing_seeds']}"
                for r in pool if not r["views"][view]["eligible"]) + ")")
    if len(counts) > 1:
        warnings.append(
            f"Incomplete ranking for {algorithm} [{label}] / {view}: candidates were "
            f"ranked on unequal replicate counts {sorted(counts)}; the comparison "
            f"is not between equal amounts of evidence.")

    ranked_ids = {r["id"] for r in pool}
    excluded = [{"id": r["id"], "reason": r["views"][view]["ineligible_reason"]}
                for r in rows if r["id"] not in ranked_ids]
    return {
        "order": [r["id"] for r in pool],
        "selected_candidate": pool[0]["id"] if pool else None,
        "ranked_on": "validation",
        "direction": direction,
        "candidate_tie_break": candidate_tie_break,
        "ranked_count": len(pool),
        "excluded": excluded,
        "incomplete_ranked": included_incomplete,
        "replicate_counts": sorted(counts),
    }


# ── The runnable configuration a selection produces ──────────────────────────

def selected_configuration(
    record: dict,
    manifest: dict,
    *,
    replicate_values: list | None = None,
    replicate_conditions: list[dict[str, int]] | None = None,
) -> dict:
    """A directly runnable sweep spec for the winning configuration.

    Built from the resolved config of a run that produced the candidate, so it
    is complete rather than a diff against defaults. The replicate axis is
    lifted back out into ``sweep`` -- baking in whichever seed happened to be
    read would present a replicate as though it were part of what was tuned.
    """
    # Completed records hold resolved data facts for provenance and comparison.
    # A selected YAML must contain only fields users are allowed to put in an
    # ExperimentConfig; the dataset registry will resolve those facts again.
    exp = {
        key: value
        for key, value in record.get("config", {}).get("experiment", {}).items()
        if key in ExperimentConfig.model_fields
    }
    acfg = deepcopy(record.get("config", {}).get("algorithm", {}))
    if manifest.get("replicate_conditions"):
        for name in ("partition_seed", "split_seed", "seed"):
            exp.pop(name, None)
        conditions = (
            list(manifest["replicate_conditions"])
            if replicate_conditions is None
            else [dict(condition) for condition in replicate_conditions]
        )
        return {
            "algorithms": [record["algorithm"]],
            "base": {"experiment": exp, "algorithm": acfg},
            "replicates": conditions,
        }
    section, name = _split(manifest["replicate_axis"])
    values = (
        list(manifest["replicate_values"])
        if replicate_values is None
        else list(replicate_values)
    )
    if section == "algorithm":
        nested_delete(acfg, name)
        sweep = {f"algorithm.{name}": values}
    else:
        nested_delete(exp, name)
        sweep = {name: values}
    return {"algorithms": [record["algorithm"]],
            "base": {"experiment": exp, "algorithm": acfg},
            "sweep": sweep}


def build_intensification_plan(
    artifact: dict, records: list[dict], manifest: dict
) -> dict | None:
    intensification = manifest.get("intensification")
    if not intensification:
        return None
    placed, _, _ = place_records(records, manifest)
    by_key = {str(key): value for key, value in placed.items()}
    shortlists = []
    for group in artifact["groups"]:
        for view, ranking in group["rankings"].items():
            candidate_ids = ranking["order"][: intensification["top_k"]]
            if not candidate_ids:
                raise TuningError(
                    f"{group['label']} / {view} has no eligible candidates for "
                    "intensification"
                )
            configurations = []
            for screening_rank, candidate_id in enumerate(candidate_ids, 1):
                runs = by_key[group["group_key"]][candidate_id]
                record = runs[min(runs, key=str)]
                candidate = next(
                    item
                    for item in group["candidates"]
                    if item["id"] == candidate_id
                )
                configurations.append(
                    {
                        "screening_rank": screening_rank,
                        "candidate_id": candidate_id,
                        "candidate_parameters": candidate["parameters"],
                        "candidate_config_hash": candidate["config_hash"],
                        "screening_validation": candidate["views"][view][
                            "validation"
                        ],
                        "screening_result_files": [
                            runs[seed].get("_source_file")
                            for seed in sorted(runs, key=str)
                        ],
                        "configuration": selected_configuration(
                            record,
                            manifest,
                            replicate_conditions=intensification["replicates"],
                        ),
                    }
                )
            shortlists.append(
                {
                    "group_id": group["group_id"],
                    "group_key": group["group_key"],
                    "label": group["label"],
                    "algorithm": group["algorithm"],
                    "condition": group["condition"],
                    "selection_view": view,
                    "configurations": configurations,
                }
            )
    return {
        "top_k": intensification["top_k"],
        "intensification_replicates": [
            dict(condition) for condition in intensification["replicates"]
        ],
        "practical_threshold": intensification["practical_threshold"],
        "tail_fraction": intensification["tail_fraction"],
        "prefer": intensification["prefer"],
        "ranking": intensification["ranking"],
        "shortlists": shortlists,
    }


def _slug(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_.") else "-" for c in str(text)]
    out = "".join(keep).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out[:80] or "x"


def _selected_groups(artifact: dict, records: list[dict], manifest: dict) -> list[dict]:
    """Build final choices from an initial ranking."""
    placed, _, _ = place_records(records, manifest)
    by_key = {str(gk): v for gk, v in placed.items()}
    selected = []
    for group in artifact["groups"]:
        for view, ranking in group["rankings"].items():
            cid = ranking["selected_candidate"]
            if cid is None:
                continue
            runs = by_key.get(group["group_key"], {}).get(cid, {})
            if not runs:
                continue
            record = runs[min(runs, key=str)]
            candidate = next(item for item in group["candidates"] if item["id"] == cid)
            selected.append(
                {
                    "group_id": group["group_id"],
                    "group_key": group["group_key"],
                    "label": group["label"],
                    "algorithm": group["algorithm"],
                    "condition": group["condition"],
                    "selection_view": view,
                    "candidate_id": cid,
                    "candidate_parameters": candidate["parameters"],
                    "candidate_config_hash": candidate["config_hash"],
                    "validation": candidate["views"][view]["validation"],
                    "selected_configuration": selected_configuration(record, manifest),
                    "selected_result_files": [
                        str(
                            Path(manifest.get("run_store", "../runs"))
                            / Path(runs[seed].get("_source_file", "")).name
                        )
                        for seed in sorted(runs, key=str)
                    ],
                    "selected_because": (
                        f'{"highest" if artifact["selection_protocol"]["direction"] == "maximize" else "lowest"} '
                        f'aggregate validation {artifact["selection_protocol"]["metric"]} among '
                        f'{ranking["ranked_count"]} ranked candidate(s); test performance '
                        "was not consulted"
                    ),
                }
            )
    return selected


def _write_final_selection(
    artifact: dict,
    groups: list[dict],
    out_dir: Path,
    *,
    selected_from: str,
) -> list[Path]:
    if not groups:
        raise TuningError("no eligible configuration can be selected")
    selection = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "kind": SELECTION_KIND,
        "study": artifact["study"],
        "ranking": "ranking.json",
        "selected_from": selected_from,
        "selection_protocol": artifact["selection_protocol"],
        "groups": groups,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_path = out_dir / "selection.json"
    atomic_write_json(selection_path, selection, validate=_check_selection)
    written = [selection_path]
    if len(groups) == 1:
        selected_path = out_dir / "selected.yaml"
        atomic_write_text(
            selected_path,
            _dump(groups[0]["selected_configuration"]),
            validate=_check_selected_config,
        )
        written.append(selected_path)
    else:
        configs_dir = out_dir / "selected"
        configs_dir.mkdir(parents=True, exist_ok=True)
        for group in groups:
            target = configs_dir / (
                f"group{group['group_id']}_{_slug(group['algorithm'])}_"
                f"{_slug(group['selection_view'])}.yaml"
            )
            atomic_write_text(
                target,
                _dump(group["selected_configuration"]),
                validate=_check_selected_config,
            )
            written.append(target)
    return written


def write_ranking(
    artifact: dict, records: list[dict], manifest: dict, out_dir: Path
) -> list[Path]:
    """Write the initial ranking and either select or prepare intensification."""
    out_dir = Path(out_dir)
    normalized = loads(dumps(artifact, indent=None))
    normalized["run_store"] = manifest.get("run_store", "../runs")
    normalized["intensification_plan"] = build_intensification_plan(
        artifact, records, manifest
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = out_dir / "ranking.json"
    atomic_write_json(ranking_path, normalized, validate=_check_ranking)
    written = [ranking_path]
    if normalized["intensification_plan"] is not None:
        from rigfl.experiment.intensification import prepare_intensification
        written.append(prepare_intensification(ranking_path))
    else:
        written.extend(
            _write_final_selection(
                normalized,
                _selected_groups(normalized, records, manifest),
                out_dir,
                selected_from="initial_ranking",
            )
        )
    return written


def _check_selected_config(text: str) -> None:
    payload = _load(text)
    if not isinstance(payload, dict) or not payload.get("algorithms") or "base" not in payload:
        raise TuningError("written configuration is not a runnable sweep spec")


def _check_ranking(parsed: dict) -> None:
    if not isinstance(parsed, dict):
        raise TuningError(f"ranking document is {type(parsed).__name__}, not an object")
    if parsed.get("kind") != ARTIFACT_KIND:
        raise TuningError(f"written ranking has kind {parsed.get('kind')!r}")
    if parsed.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise TuningError("written ranking has the wrong schema version")
    if not isinstance(parsed.get("groups"), list) or not isinstance(
            parsed.get("selection_protocol"), dict):
        raise TuningError("written ranking is missing its groups or protocol")


def _check_selection(parsed: dict) -> None:
    if not isinstance(parsed, dict):
        raise TuningError(f"selection document is {type(parsed).__name__}, not an object")
    if parsed.get("kind") != SELECTION_KIND:
        raise TuningError(f"written selection has kind {parsed.get('kind')!r}")
    if parsed.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise TuningError("written selection has the wrong schema version")
    if not isinstance(parsed.get("groups"), list) or not isinstance(
        parsed.get("selection_protocol"), dict
    ):
        raise TuningError("written selection is missing its groups or protocol")


def load_ranking(out_dir: Path) -> dict:
    """Load a tuning ranking artifact."""
    marker = Path(out_dir) / "ranking.json"
    try:
        parsed = read_json(marker)
    except ResultValidationError as exc:
        raise TuningError(f"cannot load tuning ranking: {exc}") from exc
    _check_ranking(parsed)
    return parsed


def _load(text: str) -> dict:
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        return json.loads(text)


def _dump(payload: dict) -> str:
    normalized = loads(dumps(payload, indent=None))
    try:
        import yaml
        return yaml.safe_dump(normalized, sort_keys=False)
    except ImportError:            # JSON is valid YAML; the file still loads
        return dumps(normalized)


# ── Printing ─────────────────────────────────────────────────────────────────

def format_ranking(artifact: dict) -> str:
    """Format validation-only candidate rankings."""
    sel = artifact["selection_protocol"]
    m = sel["metric"]
    out = [f"### hyperparameter candidates  (engine={artifact['study']['engine']}, "
           f"ranked on VALIDATION {m}, direction={sel['direction']}, "
           f"seeds aggregated by {sel['seed_aggregation']}, "
           f"candidate tie-break={sel['candidate_tie_break']})"]
    for group in artifact["groups"]:
        out.append(f"\n#### group {group['group_id']}: {group['label']}   "
                   f"expected replicates: {group['expected_seeds']}")
        for view, ranking in group["rankings"].items():
            out.append(f"\nselection-view: {view}   "
                       f"selected candidate: {ranking['selected_candidate']}")
            out.append(f"| rank (val) | candidate | parameters | val {m} "
                       f"| seeds | eligible |")
            out.append("|---|---|---|---|---|---|")
            order = ranking["order"] + [c["id"] for c in group["candidates"]
                                        if c["id"] not in ranking["order"]]
            for cid in order:
                c = next(x for x in group["candidates"] if x["id"] == cid)
                v = c["views"][view]
                params = " ".join(f"{k}={val}" for k, val in c["parameters"].items()) or "(defaults)"
                out.append("| " + " | ".join([
                    str(v["rank"]) if v["rank"] is not None else "—",
                    str(cid), params,
                    _fmt(v["validation"]),
                    f"{len(v['observed_seeds'])}/{len(c['expected_seeds'])}",
                    "yes" if v["eligible"] else f"no — {v['ineligible_reason']}",
                ]) + " |")
    for w in artifact["warnings"]:
        out.append(f"\n! {w}")
    if artifact["unassigned_records"]:
        out.append(f"\n! {len(artifact['unassigned_records'])} result(s) matched no "
                   f"candidate and were not ranked.")
    return "\n".join(out)


def _fmt(stats: dict) -> str:
    if stats.get("mean") is None:
        return "—"
    if stats.get("ci") is None:
        return f"{stats['mean']:.4f}"
    return f"{stats['mean']:.4f} ± {stats['ci']:.4f}"
