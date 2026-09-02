"""Aggregate completed runs into multi-seed tables and tuning rankings.

    python -m rigfl.experiment.collect --results-dir results

Use ``--group-by`` to label hyperparameter variants:

    python -m rigfl.experiment.collect --results-dir results/feddes_tune \
        --group-by algorithm.graph_k algorithm.gnn_arch
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from rigfl.eval.report import format_table, summarize, win_rate
from rigfl.eval.metrics import direction_of
from rigfl.eval.selection import resolve_metric
from rigfl.experiment.config import (algorithm_identity, hashable as _hashable,
                                     normalize_early_stopping)
from rigfl.experiment.artifacts import (ResultValidationError, atomic_write_json,
                                        atomic_write_text, is_run_result, read_json,
                                        validate_run_record)
from rigfl.experiment.registry import ALL_ALGORITHMS
from rigfl.experiment.tuning import (MANIFEST_NAME, TuningError, format_ranking,
                                     load_manifest,
                                     rank as rank_tuning, write_selection)


def load_results(results_dir: Path, dataset: str | None, alpha: float | None,
                 *, ignore_invalid: bool = False,
                 invalid: list | None = None) -> dict[str, list[dict]]:
    """Load current-schema run records, grouped by algorithm and optionally filtered."""
    by_algorithm: dict[str, list[dict]] = defaultdict(list)
    problems: list[tuple[str, str]] = [] if invalid is None else invalid
    others: list[str] = []

    for path in sorted(results_dir.glob("*.json")):
        if path.name == MANIFEST_NAME:                        # the sweep's own manifest
            continue
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
        if alpha is not None and exp.get("alpha") != alpha:
            continue
        by_algorithm[rec["algorithm"]].append(rec)

    if others:
        print(f"[collect] skipped {len(others)} JSON file(s) that are not run "
              f"results: {', '.join(sorted(others))}")
    if problems:
        listing = "\n".join(f"  {name}\n    {reason}" for name, reason in problems)
        if not ignore_invalid:
            raise SystemExit(
                f"[collect] {len(problems)} result file(s) in {results_dir} cannot be "
                f"read as completed runs:\n{listing}\n\n"
                f"No table or ranking was produced: a summary computed over an "
                f"unknown subset of the runs is not the summary it claims to be. "
                f"The files were not modified. Re-run those configurations, or pass "
                f"--ignore-invalid to proceed with the rest -- the ignored files and "
                f"reasons are then recorded in every artifact this produces.")
        print(f"\n!! [collect] --ignore-invalid: {len(problems)} result file(s) were "
              f"NOT read:\n{listing}\n!! Any candidate whose replicate is among "
              f"them stays missing, not complete.\n")
    return by_algorithm


# Algorithm settings are excluded so different algorithms can share one experimental
# condition. Seed is excluded because rows aggregate over seeds.
_EXPERIMENT = ("dataset", "partition", "scheme", "alpha", "num_clients",
               "num_classes", "rounds", "shared_dim", "model_architectures",
               "train_per_client", "test_per_client", "val_frac", "batch",
               "eval_gap")


def condition_fields(rec: dict) -> dict:
    """The flattened fields that define an experiment."""
    exp = rec.get("config", {}).get("experiment", {})
    fields = {k: _hashable(exp.get(k)) for k in _EXPERIMENT}
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
    cfg = algorithm_identity(rec.get("config", {}).get("algorithm", {}))
    return tuple(sorted((k, _hashable(v)) for k, v in cfg.items()))


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
        k for k in ("dataset", "partition", "num_clients")
        if values.get(k) is not None]
    bits = [f"{k}={values.get(k)}" for k in keys]
    return " ".join(bits) or "(unlabelled)"


def _field(rec: dict, key: str):
    """Value of a group-by key: 'algorithm' | 'algorithm.<f>' | 'exp.<f>' | bare exp field."""
    if key == "algorithm":
        return rec["algorithm"]
    section, field = key.split(".", 1) if "." in key else ("exp", key)
    cfg = rec["config"]["algorithm"] if section == "algorithm" else rec["config"]["experiment"]
    return cfg.get(field)


def _rows_by_algorithm(by_algorithm: dict[str, list[dict]], metric: str, *, view: str,
                    aggregation: str, tie_break: str) -> dict[str, dict]:
    """One row per algorithm per experiment, with win% vs Local from that experiment."""
    flat = [r for recs in by_algorithm.values() for r in recs]
    experiments = {experiment_condition(r) for r in flat}
    multi = len(experiments) > 1
    fields = varying_fields(flat) if multi else []
    if multi:
        print(f"[collect] {len(experiments)} distinct experiments present "
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
            for i, (_, vrecs) in enumerate(sorted(variants.items(), key=lambda kv: str(kv[0]))):
                label = name + (f" (variant {i + 1})" if len(variants) > 1 else "") + exp_suffix
                summary = summarize(vrecs, metric, view=view, aggregation=aggregation,
                                    tie_break=tie_break)
                if local_records and name != "local":
                    summary["win"] = win_rate(vrecs, local_records, metric, view=view,
                                              aggregation=aggregation, tie_break=tie_break)
                rows[label] = summary
    return rows


def _rows_by_group(by_algorithm: dict[str, list[dict]], group_by: list[str], metric: str,
                   *, view: str, aggregation: str, tie_break: str) -> dict[str, dict]:
    """Grouped view: one row per (experiment + algorithm + selected fields) setting."""
    flat = [r for recs in by_algorithm.values() for r in recs]
    extra = [k for k in group_by if k != "algorithm"]
    experiments = {experiment_condition(r) for r in flat}
    multi = len(experiments) > 1
    fields = varying_fields(flat) if multi else []
    if multi:
        print(f"[collect] {len(experiments)} distinct experiments present "
              f"(differing in {', '.join(fields)}); grouped separately")

    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in flat:
        label = rec["algorithm"] + "".join(f" {k.split('.')[-1]}={_field(rec, k)}" for k in extra)
        if multi:
            label += f"  [{describe_condition(rec, fields)}]"
        groups[label].append(rec)
    return {label: summarize(groups[label], metric, view=view, aggregation=aggregation,
                             tie_break=tie_break)
            for label in sorted(groups)}


def _records_supporting(by_algorithm: dict[str, list[dict]], view: str) -> dict[str, list[dict]]:
    """Filter records for ``both``, where fallback would duplicate a one-view method."""
    return {
        algorithm: supporting
        for algorithm, records in by_algorithm.items()
        if (supporting := [
            record for record in records
            if view in record["result"].get(
                "selection_views_supported", ["global", "per-client"])
        ])
    }


def rank_candidates(rows: dict, metric: str, direction: str) -> list[tuple[str, float]]:
    """Order hyperparameter candidates by validation score."""
    scored = [(label, s["val_mean"]) for label, s in rows.items()]
    return sorted(scored, key=lambda kv: kv[1], reverse=(direction == "maximize"))


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate RigFL result JSONs into a table.")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--dataset", default=None)
    p.add_argument("--group-by", nargs="*", default=None,
                   help="fields to group rows by, e.g. algorithm.graph_k algorithm.gnn_arch "
                        "(default: one row per algorithm)")
    p.add_argument("--selection-metric", default=None,
                   help="metric that chooses the reported round, on VALIDATION. "
                        "Default: accuracy. accuracy | balanced_accuracy | "
                        "macro_f1 | loss")
    p.add_argument("--selection-view", choices=["global", "per-client", "both"],
                   default=None,
                   help="global: one round for every client (a real checkpoint). "
                        "per-client: each client's own best round (mixes rounds). "
                        "both: show both. Default: global.")
    p.add_argument("--selection-aggregation", choices=["mean", "weighted_mean"],
                   default=None, help="how the global view combines clients")
    p.add_argument("--tie-break", choices=["earliest", "latest"], default=None)
    p.add_argument("--rank", action="store_true",
                   help="order candidates by validation score. With a tuning "
                        "manifest in the results directory this is candidate-aware: "
                        "one ranking per tuning group, seeds aggregated as "
                        "replicates. Without one it orders the table's rows.")
    p.add_argument("--select-out", default=None, metavar="DIR",
                   help="write the tuning selection artifact and one runnable "
                        "config per (group, selection view) into DIR. Needs a "
                        "tuning manifest.")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="rank candidates that are missing expected replicates. Off "
                        "by default: a three-seed candidate and a one-seed candidate "
                        "are not comparable. The output records what was missing.")
    p.add_argument("--candidate-tie-break", choices=["lowest_id"], default="lowest_id",
                   help="how equal validation scores are broken between candidates")
    p.add_argument("--ignore-invalid", action="store_true",
                   help="proceed when some result files cannot be read as completed "
                        "runs. Off by default: a table over an unknown subset of the "
                        "runs is not the table it claims to be. The ignored files and "
                        "reasons are recorded in every artifact produced.")
    p.add_argument("--out", default=None, help="also write the markdown table here")
    p.add_argument("--out-json", default=None,
                   help="write the collection artifact (both views, full provenance)")
    args = p.parse_args()

    invalid: list[tuple[str, str]] = []
    by_algorithm = load_results(Path(args.results_dir), args.dataset, None,
                                ignore_invalid=args.ignore_invalid, invalid=invalid)
    ignored = [{"file": name, "reason": reason} for name, reason in invalid]
    if not by_algorithm:
        print(f"no results found in {args.results_dir}")
        return

    flat = [r for recs in by_algorithm.values() for r in recs]
    metric, view, aggregation, tie_break = _resolve_selection(args, flat)

    manifest = load_manifest(Path(args.results_dir))
    if args.select_out and manifest is None:
        raise SystemExit(
            f"--select-out needs a tuning manifest, and {args.results_dir} has no "
            f"{MANIFEST_NAME}. Candidates are defined by the sweep that produced "
            f"the runs; reconstructing them from result filenames or table labels "
            f"would be a guess. Re-declare the sweep with a tuning: block "
            f"(rigfl.experiment.launch writes the manifest), or drop --select-out.")

    views = ["global", "per-client"] if view == "both" else [view]
    tables = {}
    for v in views:
        source = (_records_supporting(by_algorithm, v)
                  if view == "both" else by_algorithm)
        rows = (_rows_by_group(source, args.group_by, metric, view=v,
                               aggregation=aggregation, tie_break=tie_break)
                if args.group_by else
                _rows_by_algorithm(source, metric, view=v,
                                   aggregation=aggregation, tie_break=tie_break))
        tables[v] = rows
        print(f"\n### selection-view: {v}  (metric={metric}, split=validation, "
              f"direction={direction_of(metric)}, aggregation={aggregation}, "
              f"tie_break={tie_break})")
        print(format_table(rows, metric))
        if args.rank and not manifest:
            print("\nranked by VALIDATION (test never ranks):")
            for i, (label, score) in enumerate(rank_candidates(rows, metric,
                                                               direction_of(metric)), 1):
                print(f"  {i}. {label}  val {metric}={score:.4f}")

    if manifest and (args.rank or args.select_out):
        try:
            artifact = rank_tuning(flat, manifest, metric=metric, views=views,
                                   aggregation=aggregation, tie_break=tie_break,
                                   allow_incomplete=args.allow_incomplete,
                                   candidate_tie_break=args.candidate_tie_break)
        except TuningError as e:      # a refusal, not a crash: say why and stop
            raise SystemExit(f"[collect] cannot select a configuration: {e}")
        print()
        print(format_ranking(artifact))
        # Carried into the selection artifact: a candidate that looks short of a
        # replicate because its file was unreadable must say so where the
        # selection is read, not only in this terminal.
        artifact["ignored_invalid_results"] = ignored
        if ignored:
            artifact["warnings"].append(
                f"--ignore-invalid: {len(ignored)} result file(s) were excluded from "
                f"this collection; any replicate among them is missing, and its "
                f"candidate is incomplete rather than complete.")
        if args.select_out:
            written = write_selection(artifact, flat, manifest, Path(args.select_out))
            print("\nwrote:")
            for w in written:
                print(f"  {w}")

    if args.out:
        body = "\n\n".join(f"### selection-view: {v}\n" + format_table(rows, metric)
                            for v, rows in tables.items())
        if ignored:
            body += ("\n\n**Ignored (--ignore-invalid):**\n"
                     + "\n".join(f"- `{i['file']}` — {i['reason']}" for i in ignored))
        atomic_write_text(Path(args.out), body)
        print(f"\nwrote {args.out}")

    if args.out_json:
        # Both views always, whatever was displayed: the artifact is the record,
        # and which view was looked at should not change what was computed.
        artifact = {
            "schema_version": 2,
            "selection": {"metric": metric, "split": "validation",
                          "direction": direction_of(metric),
                          "aggregation": aggregation, "tie_break": tie_break},
            "views": {},
            # Every artifact this collection produces states what it could not
            # read, so a number from it is never quietly a number over a subset.
            "ignored_invalid_results": ignored,
        }
        for v in ("global", "per-client"):
            source = _records_supporting(by_algorithm, v)
            rows = (_rows_by_group(source, args.group_by, metric, view=v,
                                   aggregation=aggregation, tie_break=tie_break)
                    if args.group_by else
                    _rows_by_algorithm(source, metric, view=v,
                                       aggregation=aggregation, tie_break=tie_break))
            artifact["views"][v] = rows
        atomic_write_json(Path(args.out_json), artifact)
        print(f"wrote {args.out_json}")


def _resolve_selection(args, records: list[dict]) -> tuple[str, str, str, str]:
    """Resolve collection-time reporting choices."""
    return (resolve_metric(args.selection_metric, source="--selection-metric"),
            args.selection_view or "global",
            args.selection_aggregation or "mean",
            args.tie_break or "earliest")


if __name__ == "__main__":
    main()
