"""Aggregate completed runs into multi-seed tables and tuning rankings.

    python -m rigfl.experiment.collect

Use ``--group-by`` to label hyperparameter variants:

    python -m rigfl.experiment.collect --results-dir results/runs \
        --group-by algorithm.graphroute.graph.k algorithm.graphroute.gnn.arch
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from rigfl.eval.metrics import direction_of
from rigfl.eval.report import (
    format_resource_table,
    format_table,
    independent_replicates,
    summarize,
)
from rigfl.eval.selection import resolve_metric
from rigfl.eval.transfer import (
    TransferComparisonError,
    format_negative_transfer_profile,
    format_negative_transfer_table,
    negative_transfer_summary,
)
from rigfl.experiment.artifacts import (
    ResultValidationError,
    atomic_write_json,
    atomic_write_text,
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
from rigfl.experiment.paths import nested_get
from rigfl.experiment.registry import ALL_ALGORITHMS
from rigfl.experiment.storage import records_for_grid


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
        k for k in ("dataset", "partition_id", "num_clients")
        if values.get(k) is not None]
    bits = [f"{k}={values.get(k)}" for k in keys]
    return " ".join(bits) or "(unlabelled)"


def _field(rec: dict, key: str):
    """Value of a group-by key: 'algorithm' | 'algorithm.<f>' | 'exp.<f>' | bare exp field."""
    if key == "algorithm":
        return rec["algorithm"]
    section, field = key.split(".", 1) if "." in key else ("exp", key)
    cfg = rec["config"]["algorithm"] if section == "algorithm" else rec["config"]["experiment"]
    return nested_get(cfg, field)


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
                                    tie_break=tie_break,
                                    include_resources=include_resources)
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
    return rows


def _rows_by_group(by_algorithm: dict[str, list[dict]], group_by: list[str], metric: str,
                   *, view: str, aggregation: str, tie_break: str,
                   include_resources: bool = False,
                   include_transfer: bool = True,
                   transfer_threshold: float = 0.0,
                   transfer_profile: tuple[float, ...] = (),
                   transfer_tail: float = 0.10) -> dict[str, dict]:
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
    rows = {}
    for label in sorted(groups):
        records = groups[label]
        if len({algorithm_variant(record) for record in records}) > 1:
            raise ValueError(
                f"group {label!r} combines multiple algorithm configurations; "
                "include the differing algorithm fields in --group-by"
            )
        summary = summarize(
            records, metric, view=view, aggregation=aggregation,
            tie_break=tie_break, include_resources=include_resources,
        )
        if include_transfer and records[0]["algorithm"] != "local":
            condition = experiment_condition(records[0])
            local_records = [
                record for record in flat
                if record["algorithm"] == "local"
                and experiment_condition(record) == condition
            ]
            if local_records:
                summary["negative_transfer"] = _negative_transfer(
                    records,
                    local_records,
                    metric,
                    view=view,
                    aggregation=aggregation,
                    tie_break=tie_break,
                    threshold=transfer_threshold,
                    profile_thresholds=transfer_profile,
                    tail_fraction=transfer_tail,
                    include_uncertainty=independent_replicates(records),
                )
        rows[label] = summary
    return rows


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


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return parsed


def _tail_fraction(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return parsed


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
            "schema_version": 1,
            "available": False,
            "baseline": "local",
            "reason": str(error),
        }


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate RigFL result JSONs into a table.")
    p.add_argument("--results-dir", default="results/runs")
    p.add_argument("--grid", help="include only runs in this saved grid.jsonl")
    p.add_argument("--dataset", default=None)
    p.add_argument("--group-by", nargs="*", default=None,
                   help="fields to group rows by, e.g. algorithm.graphroute.graph.k "
                        "algorithm.graphroute.gnn.arch "
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
                   help="order the displayed result rows by validation score")
    p.add_argument("--ignore-invalid", action="store_true",
                   help="proceed when some result files cannot be read as completed "
                        "runs. Off by default: a table over an unknown subset of the "
                        "runs is not the table it claims to be. The ignored files and "
                        "reasons are recorded in every artifact produced.")
    p.add_argument("--out", default=None, help="also write the markdown table here")
    p.add_argument("--out-json", default=None,
                   help="write the collection artifact (both views, full provenance)")
    p.add_argument("--include-resources", action="store_true",
                   help="print communication, estimated FLOPs, and training time")
    p.add_argument("--negative-transfer-threshold", type=_nonnegative_float,
                   default=0.0, metavar="DELTA",
                   help="smallest change treated as a benefit or harm, in metric "
                        "units (default: 0)")
    p.add_argument("--negative-transfer-profile", nargs="*", type=_nonnegative_float,
                   default=[], metavar="DELTA",
                   help="additional thresholds for the negative-transfer rate profile")
    p.add_argument("--negative-transfer-tail", type=_tail_fraction, default=0.10,
                   metavar="FRACTION",
                   help="fraction used for worst-tail relative gain (default: 0.10)")
    args = p.parse_args()

    invalid: list[tuple[str, str]] = []
    by_algorithm = load_results(Path(args.results_dir), args.dataset,
                                ignore_invalid=args.ignore_invalid, invalid=invalid)
    if args.grid:
        try:
            filtered = records_for_grid(
                [record for records in by_algorithm.values() for record in records],
                args.grid,
            )
        except ValueError as error:
            raise SystemExit(f"[collect] {error}") from error
        by_algorithm = defaultdict(list)
        for record in filtered:
            by_algorithm[record["algorithm"]].append(record)
    ignored = [{"file": name, "reason": reason} for name, reason in invalid]
    if not by_algorithm:
        print(f"no results found in {args.results_dir}")
        return

    flat = [r for recs in by_algorithm.values() for r in recs]
    metric, view, aggregation, tie_break = _resolve_selection(
        args, flat, manifest=None
    )

    views = ["global", "per-client"] if view == "both" else [view]
    tables = {}
    for v in views:
        source = (_records_supporting(by_algorithm, v)
                  if view == "both" else by_algorithm)
        rows = (_rows_by_group(
                    source, args.group_by, metric, view=v,
                    aggregation=aggregation, tie_break=tie_break,
                    transfer_threshold=args.negative_transfer_threshold,
                    transfer_profile=tuple(args.negative_transfer_profile),
                    transfer_tail=args.negative_transfer_tail)
                if args.group_by else
                _rows_by_algorithm(
                    source, metric, view=v,
                    aggregation=aggregation, tie_break=tie_break,
                    transfer_threshold=args.negative_transfer_threshold,
                    transfer_profile=tuple(args.negative_transfer_profile),
                    transfer_tail=args.negative_transfer_tail))
        tables[v] = rows
        print(f"\n### selection-view: {v}  (metric={metric}, split=validation, "
              f"direction={direction_of(metric)}, aggregation={aggregation}, "
              f"tie_break={tie_break})")
        print(format_table(rows, metric))
        transfer_table = format_negative_transfer_table(rows)
        if transfer_table:
            print("\n### negative transfer relative to Local")
            print(transfer_table)
        transfer_profile = format_negative_transfer_profile(rows)
        if transfer_profile:
            print("\n### negative-transfer rate profile")
            print(transfer_profile)
        if args.rank:
            print("\nranked by VALIDATION (test never ranks):")
            for i, (label, score) in enumerate(rank_candidates(
                    rows, metric, direction_of(metric)), 1):
                print(f"  {i}. {label}  val {metric}={score:.4f}")

    resource_rows = None
    if args.include_resources:
        resource_rows = (
            _rows_by_group(by_algorithm, args.group_by, metric, view="global",
                           aggregation=aggregation, tie_break=tie_break,
                           include_resources=True, include_transfer=False)
            if args.group_by else
            _rows_by_algorithm(by_algorithm, metric, view="global",
                               aggregation=aggregation, tie_break=tie_break,
                               include_resources=True, include_transfer=False)
        )
        print("\n### resources: attributed training")
        print(format_resource_table(resource_rows))

    if args.out:
        sections = []
        for v, rows in tables.items():
            sections.append(f"### selection-view: {v}\n" + format_table(rows, metric))
            transfer_table = format_negative_transfer_table(rows)
            if transfer_table:
                sections.append(
                    f"### negative transfer relative to Local: {v}\n"
                    + transfer_table
                )
            transfer_profile = format_negative_transfer_profile(rows)
            if transfer_profile:
                sections.append(
                    f"### negative-transfer rate profile: {v}\n"
                    + transfer_profile
                )
        body = "\n\n".join(sections)
        if resource_rows is not None:
            body += ("\n\n### resources: attributed training\n" +
                     format_resource_table(resource_rows))
        if ignored:
            body += ("\n\n**Ignored (--ignore-invalid):**\n"
                     + "\n".join(
                         f"- `{i['file']}` — {i['reason']}" for i in ignored))
        atomic_write_text(Path(args.out), body)
        print(f"\nwrote {args.out}")

    if args.out_json:
        # Both views always, whatever was displayed: the artifact is the record,
        # and which view was looked at should not change what was computed.
        artifact = {
            "schema_version": 3,
            "selection": {"metric": metric, "split": "validation",
                          "direction": direction_of(metric),
                          "aggregation": aggregation, "tie_break": tie_break},
            "views": {},
            "negative_transfer": {},
            # Every artifact this collection produces states what it could not
            # read, so a number from it is never quietly a number over a subset.
            "ignored_invalid_results": ignored,
        }
        for v in ("global", "per-client"):
            source = _records_supporting(by_algorithm, v)
            rows = (_rows_by_group(source, args.group_by, metric, view=v,
                                   aggregation=aggregation, tie_break=tie_break,
                                   include_resources=args.include_resources,
                                   transfer_threshold=args.negative_transfer_threshold,
                                   transfer_profile=tuple(args.negative_transfer_profile),
                                   transfer_tail=args.negative_transfer_tail)
                    if args.group_by else
                    _rows_by_algorithm(source, metric, view=v,
                                       aggregation=aggregation, tie_break=tie_break,
                                       include_resources=args.include_resources,
                                       transfer_threshold=args.negative_transfer_threshold,
                                       transfer_profile=tuple(args.negative_transfer_profile),
                                       transfer_tail=args.negative_transfer_tail))
            performance, transfer = _collection_rows(rows)
            artifact["views"][v] = performance
            artifact["negative_transfer"][v] = transfer
        atomic_write_json(Path(args.out_json), artifact)
        print(f"wrote {args.out_json}")


def _resolve_selection(args, records: list[dict],
                       manifest: dict | None = None) -> tuple[str, str, str, str]:
    """Resolve collection-time reporting choices."""
    optimization = (manifest or {}).get("optimization", {})
    protocol = (manifest or {}).get("selection_protocol", {})
    defaults = {
        "metric": protocol.get("metric", optimization.get("metric", "accuracy")),
        "view": protocol.get(
            "selection_view", optimization.get("selection_view", "global")
        ),
        "aggregation": protocol.get(
            "selection_aggregation",
            optimization.get("selection_aggregation", "mean"),
        ),
        "tie_break": protocol.get(
            "round_tie_break", optimization.get("round_tie_break", "earliest")
        ),
    }
    resolved = {
        "metric": resolve_metric(
            args.selection_metric or defaults["metric"],
            source="--selection-metric",
        ),
        "view": args.selection_view or defaults["view"],
        "aggregation": args.selection_aggregation or defaults["aggregation"],
        "tie_break": args.tie_break or defaults["tie_break"],
    }
    if optimization or protocol.get("fixed"):
        expected = dict(defaults)
        expected["metric"] = resolve_metric(expected["metric"])
        changed = [name for name in expected if resolved[name] != expected[name]]
        if changed:
            details = ", ".join(
                f"{name}={resolved[name]!r} (study used {expected[name]!r})"
                for name in changed
            )
            raise SystemExit(
                "a tuning search must be selected with the objective protocol "
                f"that guided its trials: {details}"
            )
    return (
        resolved["metric"], resolved["view"], resolved["aggregation"],
        resolved["tie_break"],
    )


if __name__ == "__main__":
    main()
