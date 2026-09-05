"""Rank complete hyperparameter candidates over replicate runs.

A candidate is one joint assignment of its tuned parameters. The replicate axis
is excluded from candidate identity, and validation scores rank candidates.
"""

from __future__ import annotations

import itertools
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rigfl.eval.metrics import direction_of
from rigfl.eval.report import mean_ci, run_score
from rigfl.eval.selection import SelectionError
from rigfl.experiment.artifacts import (ResultValidationError, atomic_write_json,
                                        atomic_write_text, dumps, loads, read_json)
from rigfl.experiment.config import (ExperimentConfig, _ALGORITHM_ENV_IRRELEVANT,
                                     fingerprint, hashable)
from rigfl.experiment.registry import config_class

#: Bumped when the manifest layout changes in a way a reader must notice.
MANIFEST_SCHEMA_VERSION = 2
MANIFEST_NAME = "tuning_manifest.json"
MANIFEST_KIND = "rigfl.tuning_manifest"

ARTIFACT_SCHEMA_VERSION = 2
ARTIFACT_KIND = "rigfl.tuning_selection"

#: Supported tuning strategies.
STRATEGIES = ("grid",)

#: Tie-breaks for equal aggregate validation scores.
CANDIDATE_TIE_BREAKS = ("lowest_id",)

SEED_AGGREGATION = "mean"


class TuningError(ValueError):
    """Raised when candidate selection is asked for and cannot be done honestly."""


# ── Axis paths ───────────────────────────────────────────────────────────────

def canonical_axis(path: str) -> str:
    """``batch`` / ``exp.batch`` -> ``exp.batch``; ``algorithm.lr`` stays unchanged.

    Sweep files may spell an experiment axis either way. One spelling internally
    means a tuning parameter and the axis it names cannot fail to match because
    of how they were written.
    """
    p = str(path).strip()
    if p.startswith("algorithm."):
        return p
    if p.startswith("exp."):
        return p
    return f"exp.{p}"


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

@dataclass
class TuningSpec:
    """The ``tuning:`` block of a sweep file, validated against its sweep axes."""
    strategy: str
    parameters: list[str]            # canonical axis paths, in declaration order
    replicate_axis: str              # canonical axis path
    declared: dict = field(default_factory=dict)   # the block as written

    def to_dict(self) -> dict:
        return {"strategy": self.strategy, "parameters": list(self.parameters),
                "replicate_axis": self.replicate_axis}


def parse_tuning(block: dict | None, axis_values: dict[str, list]) -> Optional[TuningSpec]:
    """Validate a ``tuning:`` block against the axes the sweep actually declares.

    ``axis_values`` maps canonical axis path -> its declared values. Every error
    here is a launch-time ``SystemExit``: a tuning parameter naming an axis that
    is not swept would otherwise produce one candidate and a sweep that reports
    a search it never ran.
    """
    if not block:
        return None
    if not isinstance(block, dict):
        raise SystemExit("tuning: must be a mapping with 'parameters' and 'replicate_axis'.")

    unknown = sorted(set(block) - {"strategy", "parameters", "replicate_axis"})
    if unknown:
        raise SystemExit(
            f"Unknown key(s) in tuning: {', '.join(unknown)}\n"
            f"Known: strategy, parameters, replicate_axis")

    strategy = str(block.get("strategy", "grid"))
    if strategy not in STRATEGIES:
        raise SystemExit(
            f'Unknown tuning strategy "{strategy}". '
            f'Only {", ".join(STRATEGIES)} is implemented -- random, Bayesian and '
            f'bandit search are not, and naming one must not quietly run a grid.')

    raw_params = block.get("parameters") or []
    if isinstance(raw_params, str):
        raw_params = [raw_params]
    if not raw_params:
        raise SystemExit(
            "tuning.parameters is empty. Name the axes that jointly define a "
            "candidate; every other swept axis stays an experimental condition.")

    known = ", ".join(sorted(axis_values)) or "(none declared)"

    params: list[str] = []
    for p in raw_params:
        path = canonical_axis(p)
        if path in params:
            raise SystemExit(
                f"Duplicate tuning parameter: {p} (already declared as {path}).")
        if path not in axis_values:
            raise SystemExit(
                f"Unknown tuning parameter: {p}\n"
                f'"{path}" is not a swept axis, so it has nothing to search over.\n\n'
                f"Declared sweep axes: {known}")
        params.append(path)

    if "replicate_axis" not in block:
        raise SystemExit(
            "tuning.replicate_axis is unset. State which axis is a replicate "
            "rather than part of the search -- usually 'seed'. Without it, seeds "
            "would be ranked against each other as if they were configurations.\n\n"
            f"Declared sweep axes: {known}")
    rep = canonical_axis(block["replicate_axis"])
    if rep not in axis_values:
        raise SystemExit(
            f"Unknown replicate axis: {block['replicate_axis']}\n"
            f'"{rep}" is not a swept axis, so there are no replicates to aggregate.\n\n'
            f"Declared sweep axes: {known}")
    if rep in params:
        raise SystemExit(
            f'"{rep}" is both a tuning parameter and the replicate axis. It is '
            f"either part of what is being searched or what is averaged over, "
            f"not both.")

    return TuningSpec(strategy=strategy, parameters=params, replicate_axis=rep,
                      declared=dict(block))


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

    An ``algorithm.x`` axis applies only to algorithms defining ``x``. Dropping the rest
    is what keeps a FedDES-only axis from minting duplicate "not applicable"
    candidates for every other algorithm.
    """
    fields = set(config_class(algorithm).model_fields)
    out = []
    for p in parameters:
        section, name = _split(p)
        if section == "algorithm" and name not in fields:
            continue
        out.append(p)
    return out


def build_candidates(algorithms: list[str], parameters: list[str],
                     axis_values: dict[str, list]) -> list[Candidate]:
    """Every complete assignment, numbered from zero.

    Ordering is algorithm order, then declared parameter order, then declared value
    order -- so the same sweep file always produces the same ids. The replicate
    axis is not among ``parameters``, so seeds never reach this function and
    cannot split one candidate into several.
    """
    out: list[Candidate] = []
    seen: set[tuple] = set()
    for algorithm in algorithms:
        applicable = applicable_parameters(algorithm, parameters)
        for combo in itertools.product(*(axis_values[p] for p in applicable)):
            params = dict(zip(applicable, combo))
            key = (algorithm, tuple(sorted((k, _norm(v))
                                        for k, v in params.items())))
            if key in seen:
                continue
            seen.add(key)
            out.append(Candidate(len(out), algorithm, params,
                                 candidate_hash(algorithm, params)))
    return out


def candidate_index(candidates: list[Candidate]) -> dict[tuple, int]:
    """``(algorithm, normalised parameter key) -> candidate id``."""
    return {(c.algorithm, c.key): c.id for c in candidates}


# ── The manifest ─────────────────────────────────────────────────────────────

def build_manifest(*, name: str, tuning: TuningSpec, algorithms: list[str],
                   base: dict, axis_values: dict[str, list],
                   declared_axes: list[dict], candidates: list[Candidate],
                   tasks: list[dict],
                   declared_tuning_parameters: list[str] | None = None) -> dict:
    """Everything needed to reconstruct the search without parsing a label."""
    condition_axes = [a for a in axis_values
                      if a not in tuning.parameters and a != tuning.replicate_axis]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "name": name,
        "strategy": tuning.strategy,
        "algorithms": list(algorithms),
        "base": base,
        # Declaration order, because candidate numbering follows it.
        "sweep_axes": declared_axes,
        "tuning_parameters": list(tuning.parameters),
        "declared_tuning_parameters": list(
            declared_tuning_parameters or tuning.parameters),
        "replicate_axis": tuning.replicate_axis,
        "replicate_values": list(axis_values[tuning.replicate_axis]),
        "condition_axes": condition_axes,
        "candidates": [c.to_dict() for c in candidates],
        # Task ids are 1-based array indices; candidate ids are 0-based.
        "tasks": tasks,
        "candidate_tasks": _candidate_tasks(tasks),
    }


def _candidate_tasks(tasks: list[dict]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for t in tasks:
        out.setdefault(str(t["candidate_id"]), []).append(t["task_id"])
    return out


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
    for key in ("candidates", "tasks", "tuning_parameters", "replicate_axis",
                "replicate_values", "condition_axes"):
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
    manifest["_path"] = str(path)
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
        out[p] = source.get(name)
    return out


def replicate_of(record: dict, manifest: dict):
    """The run's value on the replicate axis."""
    section, name = _split(manifest["replicate_axis"])
    cfg = record.get("config", {})
    source = cfg.get("algorithm", {}) if section == "algorithm" else cfg.get("experiment", {})
    return _norm(source.get(name))


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

    Collector condition fields support comparisons across algorithms; tuning adds
    the swept algorithm settings needed to distinguish candidates within an algorithm.
    """
    from rigfl.experiment.collect import condition_fields

    tuned = list(manifest["tuning_parameters"]) + [manifest["replicate_axis"]]
    drop_exp = {_split(p)[1] for p in tuned if _split(p)[0] == "exp"}

    cond = {k: v for k, v in condition_fields(record).items() if k not in drop_exp}
    exp = record.get("config", {}).get("experiment", {})
    acfg = record.get("config", {}).get("algorithm", {})
    for axis in manifest["condition_axes"]:
        section, name = _split(axis)
        if section == "algorithm":
            if name in _ALGORITHM_ENV_IRRELEVANT:      # a location, not a condition
                continue
            if name in acfg:
                cond[f"algorithm.{name}"] = hashable(acfg[name])
        elif name not in cond:          # a swept experiment field outside the
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
    for rec in records:
        cid = candidate_of(rec, manifest, index)
        if cid is None:
            unassigned.append({
                "algorithm": rec["algorithm"],
                "parameters": run_parameters(rec, manifest["tuning_parameters"],
                                             rec["algorithm"]),
                "reason": "no manifest candidate has these tuning-parameter values"})
            continue
        gk = group_key(rec, manifest)
        conditions.setdefault(gk, effective_condition(rec, manifest))
        placed.setdefault(gk, {}).setdefault(cid, {})[replicate_of(rec, manifest)] = rec
    return placed, conditions, unassigned


def rank(records: list[dict], manifest: dict, *, metric: str, views: list[str],
         aggregation: str = "mean", tie_break: str = "earliest",
         allow_incomplete: bool = False,
         candidate_tie_break: str = "lowest_id") -> dict:
    """Rank every candidate within every tuning group, on validation only.

    Returns the selection artifact. Test statistics are computed and carried for
    every candidate; nothing in the ordering, the eligibility test or the tie
    break reads them.
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
        "manifest": {"path": manifest.get("_path"), "name": manifest.get("name"),
                     "strategy": manifest.get("strategy"),
                     "schema_version": manifest.get("schema_version")},
        "selection": {
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
            "ranked_on": "validation only; test statistics are reported, never ranked",
        },
        "groups": groups,
        "unassigned_records": unassigned,
        "warnings": warnings,
    }


def _candidate_view(recs: dict, expected: list, display: dict, metric: str, view: str,
                    aggregation: str, tie_break: str) -> dict:
    """One candidate's per-seed scores and their aggregate, for one view."""
    per_seed: dict[str, dict] = {}
    val_scores, test_scores = [], []
    errors: dict[str, str] = {}
    for seed in sorted(recs, key=str):
        try:
            s = run_score(recs[seed], metric, view=view, aggregation=aggregation,
                          tie_break=tie_break)
        except (SelectionError, ValueError) as e:
            errors[str(seed)] = str(e)
            continue
        s["source_file"] = recs[seed].get("_source_file")
        per_seed[str(display.get(seed, seed))] = s
        if s["validation"] is not None:
            val_scores.append(s["validation"])
        if s["test"] is not None:
            test_scores.append(s["test"])

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
        # Reported so every candidate's test performance stays visible. Reading
        # it to choose a candidate is test-based tuning, whatever it is called.
        "test": _stats(test_scores),
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
            f"--allow-incomplete: {algorithm} [{label}] / {view}: candidate(s) "
            f"{included_incomplete} were ranked with missing replicates "
            f"(" + "; ".join(
                f"id {r['id']} has {len(r['views'][view]['observed_seeds'])} of "
                f"{len(r['expected_seeds'])}, missing {r['views'][view]['missing_seeds']}"
                for r in pool if not r["views"][view]["eligible"]) + ")")
    if len(counts) > 1:
        warnings.append(
            f"--allow-incomplete: {algorithm} [{label}] / {view}: candidates were "
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

def selected_configuration(record: dict, manifest: dict) -> dict:
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
    acfg = dict(record.get("config", {}).get("algorithm", {}))
    section, name = _split(manifest["replicate_axis"])
    if section == "algorithm":
        acfg.pop(name, None)
        sweep = {f"algorithm.{name}": list(manifest["replicate_values"])}
    else:
        exp.pop(name, None)
        sweep = {name: list(manifest["replicate_values"])}
    return {"algorithms": [record["algorithm"]],
            "base": {"experiment": exp, "algorithm": acfg},
            "sweep": sweep}


def _slug(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_.") else "-" for c in str(text)]
    out = "".join(keep).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out[:80] or "x"


def write_selection(artifact: dict, records: list[dict], manifest: dict,
                    out_dir: Path) -> list[Path]:
    """Write the artifact plus one runnable config per (group, view).

    Separate files, named by group id and view, because a sweep over several
    algorithms or conditions selects several configurations and there is no sense
    in which one of them is *the* result. The names are checked for collision
    before anything is written rather than after one has overwritten another.
    """
    out_dir = Path(out_dir)
    placed, _, _ = place_records(records, manifest)
    by_key = {str(gk): v for gk, v in placed.items()}
    planned: dict[str, dict] = {}
    for group in artifact["groups"]:
        for view, ranking in group["rankings"].items():
            cid = ranking["selected_candidate"]
            if cid is None:
                continue
            runs = by_key.get(group["group_key"], {}).get(cid, {})
            if not runs:
                continue
            rec = runs[sorted(runs, key=str)[0]]        # any run of the candidate:
            #                                             they differ only in replicate
            cand = next(c for c in group["candidates"] if c["id"] == cid)
            name = (
                f"group{group['group_id']}_{_slug(group['algorithm'])}"
                f"_{_slug(_label(group['condition'], group.get('label_fields')))}"
                f"_{_slug(view)}.yaml")
            if name in planned:
                raise TuningError(
                    f"Two selected configurations would be written as {name}. "
                    f"Refusing rather than overwriting one with the other.")
            planned[name] = {
                "selected_configuration": selected_configuration(rec, manifest),
                "provenance": {
                    "manifest": artifact["manifest"],
                    "group_id": group["group_id"],
                    "algorithm": group["algorithm"],
                    "condition": group["condition"],
                    "selection_view": view,
                    "candidate_id": cid,
                    "candidate_parameters": cand["parameters"],
                    "candidate_config_hash": cand["config_hash"],
                    "selection": artifact["selection"],
                    "expected_seeds": group["expected_seeds"],
                    "observed_seeds": cand["views"][view]["observed_seeds"],
                    "validation": cand["views"][view]["validation"],
                    "selected_because": (
                        f'highest aggregate VALIDATION {artifact["selection"]["metric"]} '
                        f'among {ranking["ranked_count"]} ranked candidate(s) '
                        if artifact["selection"]["direction"] == "maximize" else
                        f'lowest aggregate VALIDATION {artifact["selection"]["metric"]} '
                        f'among {ranking["ranked_count"]} ranked candidate(s) ')
                        + "(test performance was not consulted)",
                    "source_results": [
                        (cand["views"][view]["per_seed"].get(str(s)) or {}).get("source_file")
                        for s in cand["views"][view]["observed_seeds"]],
                },
            }

    # The JSON artifact is authoritative and contains the selected configurations
    # directly. YAML files are convenient runnable copies and may be regenerated.
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized = loads(dumps(artifact, indent=None))
    normalized["selected_configurations"] = planned

    configs_dir = out_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, payload in sorted(planned.items()):
        target = configs_dir / name
        atomic_write_text(target, _dump(payload), validate=_check_selected_config)
        written.append(target)

    selection = out_dir / "selection.json"
    atomic_write_json(selection, normalized, validate=_check_selection)
    return written + [selection]


def _check_selected_config(text: str) -> None:
    payload = _load(text)
    spec = payload.get("selected_configuration")
    if not isinstance(spec, dict) or not spec.get("algorithms") or "base" not in spec:
        raise TuningError("written configuration is not a runnable sweep spec")
    if "candidate_id" not in (payload.get("provenance") or {}):
        raise TuningError("written configuration carries no candidate provenance")


def _check_selection(parsed: dict) -> None:
    if not isinstance(parsed, dict):
        raise TuningError(f"selection document is {type(parsed).__name__}, not an object")
    if parsed.get("kind") != ARTIFACT_KIND:
        raise TuningError(f"written selection has kind {parsed.get('kind')!r}")
    if parsed.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise TuningError("written selection has the wrong schema version")
    if not isinstance(parsed.get("groups"), list) or not isinstance(
            parsed.get("selection"), dict):
        raise TuningError("written selection is missing its groups or protocol")
    if not isinstance(parsed.get("selected_configurations"), dict):
        raise TuningError("written selection is missing selected configurations")


def load_selection(out_dir: Path) -> dict:
    """Load the latest tuning selection artifact."""
    marker = Path(out_dir) / "selection.json"
    try:
        parsed = read_json(marker)
    except ResultValidationError as exc:
        raise TuningError(f"cannot load tuning selection: {exc}") from exc
    _check_selection(parsed)
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
    """The tables a human reads: validation ranks, with test alongside."""
    sel = artifact["selection"]
    m = sel["metric"]
    out = [f"### hyperparameter candidates  (strategy={artifact['manifest']['strategy']}, "
           f"ranked on VALIDATION {m}, direction={sel['direction']}, "
           f"seeds aggregated by {sel['seed_aggregation']}, "
           f"candidate tie-break={sel['candidate_tie_break']})",
           "test columns are reported for inspection and take no part in ranking."]
    for group in artifact["groups"]:
        out.append(f"\n#### group {group['group_id']}: {group['label']}   "
                   f"expected replicates: {group['expected_seeds']}")
        for view, ranking in group["rankings"].items():
            out.append(f"\nselection-view: {view}   "
                       f"selected candidate: {ranking['selected_candidate']}")
            out.append(f"| rank (val) | candidate | parameters | val {m} | test {m} "
                       f"| seeds | eligible |")
            out.append("|---|---|---|---|---|---|---|")
            order = ranking["order"] + [c["id"] for c in group["candidates"]
                                        if c["id"] not in ranking["order"]]
            for cid in order:
                c = next(x for x in group["candidates"] if x["id"] == cid)
                v = c["views"][view]
                params = " ".join(f"{k}={val}" for k, val in c["parameters"].items()) or "(defaults)"
                out.append("| " + " | ".join([
                    str(v["rank"]) if v["rank"] is not None else "—",
                    str(cid), params,
                    _fmt(v["validation"]), _fmt(v["test"]),
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
    return f"{stats['mean']:.4f} ± {stats['ci']:.4f}"
