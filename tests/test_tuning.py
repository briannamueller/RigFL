"""Hyperparameter selection: candidates, tuning groups, replicates, artifacts.

The layer above round selection. Round selection picks a round inside one run;
this picks a configuration across runs and seeds. The properties that matter and
that these tests pin down:

* a candidate is a **complete joint assignment**, so a 2x2x3 space is twelve
  candidates and never three independent per-axis rankings;
* seeds are **replicates**, so they change the number of runs and nothing about
  candidate identity, and a candidate short of one is not silently compared
  against candidates that have all of theirs;
* **validation ranks and test never does**, in either selection view.
"""

from __future__ import annotations

import json
import pytest

from rigfl.eval.metrics import register, unregister
from rigfl.experiment.config import ExperimentConfig
from rigfl.experiment.launch import expand
from rigfl.experiment.registry import config_class
from rigfl.experiment.run import resolve_experiment_architectures
from rigfl.experiment.tuning import (TuningError, candidate_index, candidate_of,
                                     load_manifest, manifest_candidates,
                                     place_records, rank, write_manifest,
                                     write_selection)
from rigfl.models.registry import MODEL_ARCHITECTURE_FAMILIES

VIEWS = ["global", "per-client"]


# ── fixtures ─────────────────────────────────────────────────────────────────

def _spec(**over) -> dict:
    spec = {
        "name": "tune",
        "algorithms": ["feddes"],
        # One round, so the fabricated one-round histories below are what a run
        # of this configuration actually produces -- the validator checks that.
        "base": {"experiment": {"rounds": 1, "eval_gap": 1,
                                  "num_clients": 2}},
        "sweep": {"seed": [0, 1, 2],
                  "algorithm.base_lr": [0.01, 0.1],
                  "algorithm.graph_k": [3, 5]},
        "tuning": {"strategy": "grid",
                   "parameters": ["algorithm.base_lr", "algorithm.graph_k"],
                   "replicate_axis": "seed"},
    }
    spec.update(over)
    return spec


def _hist(val: dict, test: dict, counts=None) -> dict:
    """``{metric: [per-client [per-round values]]}`` -> a schema-2 result."""
    any_metric = next(iter(val))
    n_clients, n_rounds = len(val[any_metric]), len(val[any_metric][0])
    clients = {
        str(c): {"validation": {m: list(series[c]) for m, series in val.items()},
                 "test": {m: list(series[c]) for m, series in test.items()}}
        for c in range(n_clients)}
    w = counts or [10] * n_clients
    sample_counts = {s: {str(c): [w[c]] * n_rounds for c in range(n_clients)}
                     for s in ("validation", "test")}
    return {"schema_version": 3,
            "selection_views_supported": ["global", "per-client"],
            "evaluation_history": {"evaluation_rounds": list(range(n_rounds)),
                                   "clients": clients,
                                   "client_sample_counts": sample_counts},
            "early_stopping": _no_stopping(n_rounds)}


def _no_stopping(n_rounds: int) -> dict:
    """The block iterative writes when stopping is disabled."""
    return {"enabled": False, "termination_reason": "completed_all_rounds",
            "stopped_at_round": n_rounds - 1, "metric": None, "split": None,
            "direction": None, "aggregation": None, "patience": None,
            "min_delta": None, "best_round": None, "best_value": None}


def _flat(val: float, test: float, metric="accuracy", clients=2, rounds=1,
          counts=None, also=()) -> dict:
    """A history where every client scores the same at every round."""
    names = [metric, *also]
    return _hist({m: [[val] * rounds for _ in range(clients)] for m in names},
                 {m: [[test] * rounds for _ in range(clients)] for m in names}, counts)


def _records(grid, manifest, scores, *, skip=()):
    """One record per grid task, scored by ``scores[candidate_id] -> history``."""
    out = []
    metadata = {t["task_id"]: t for t in manifest["tasks"]}
    for i, t in enumerate(grid, 1):
        meta = metadata[i]
        cid, replicate = meta["candidate_id"], meta["replicate"]
        if (cid, replicate) in skip:
            continue
        rec = {"algorithm": t["algorithm"],
               "config": {"experiment": dict(t["experiment"]),
                          "algorithm": dict(t["algorithm_config"])},
               "result": scores(cid),
               "_source_file": f"task{i}.json"}
        out.append(rec)
    return out


def _rank(recs, manifest, metric="accuracy", views=("global",), **kw):
    return rank(recs, dict(manifest, _path="(test)"), metric=metric,
                views=list(views), **kw)


def _selected(artifact, group=0, view="global"):
    return artifact["groups"][group]["rankings"][view]["selected_candidate"]


# ── 1-4: candidates are complete joint assignments; seeds are replicates ─────

def test_a_2x2x3_tuning_space_produces_twelve_candidates():
    _, manifest = expand(_spec(sweep={"seed": [0, 1, 2],
                                      "algorithm.base_lr": [0.01, 0.1],
                                      "algorithm.base_epochs": [1, 5],
                                      "algorithm.graph_k": [3, 5, 10]},
                               tuning={"strategy": "grid",
                                       "parameters": ["algorithm.base_lr", "algorithm.base_epochs",
                                                      "algorithm.graph_k"],
                                       "replicate_axis": "seed"}))
    assert len(manifest["candidates"]) == 2 * 2 * 3 == 12
    # every candidate names every tuning parameter -- not one axis at a time
    for c in manifest["candidates"]:
        assert set(c["parameters"]) == {"algorithm.base_lr", "algorithm.base_epochs",
                                        "algorithm.graph_k"}


def test_three_seeds_make_36_tasks_without_changing_the_12_candidates():
    base = dict(sweep={"seed": [0], "algorithm.base_lr": [0.01, 0.1],
                       "algorithm.base_epochs": [1, 5], "algorithm.graph_k": [3, 5, 10]},
                tuning={"strategy": "grid",
                        "parameters": ["algorithm.base_lr", "algorithm.base_epochs",
                                       "algorithm.graph_k"],
                        "replicate_axis": "seed"})
    one_grid, one = expand(_spec(**base))
    three = dict(base, sweep=dict(base["sweep"], seed=[0, 1, 2]))
    three_grid, many = expand(_spec(**three))

    assert len(one_grid) == 12 and len(three_grid) == 36
    assert one["candidates"] == many["candidates"]      # identity, ids and hashes


def test_candidate_ids_are_stable_zero_based_integers():
    _, a = expand(_spec())
    _, b = expand(_spec())
    ids = [c["id"] for c in a["candidates"]]
    assert ids == list(range(len(ids)))
    assert all(isinstance(i, int) and not isinstance(i, bool) for i in ids)
    assert a["candidates"] == b["candidates"]           # deterministic re-expansion


def test_seed_is_excluded_from_candidate_identity():
    _, manifest = expand(_spec())
    for c in manifest["candidates"]:
        assert "exp.seed" not in c["parameters"] and "seed" not in c["parameters"]
    # and every candidate is reached by every seed
    grid, manifest = expand(_spec())
    by_candidate: dict[int, set] = {}
    for t in manifest["tasks"]:
        by_candidate.setdefault(t["candidate_id"], set()).add(t["replicate"])
    assert all(v == {0, 1, 2} for v in by_candidate.values())


def test_architecture_family_candidates_match_recorded_lists(monkeypatch):
    monkeypatch.setitem(
        MODEL_ARCHITECTURE_FAMILIES, "image_pair",
        ["fedavg_cnn", "cifar_resnet18"])
    spec = {
        "algorithms": ["local"],
        "base": {"experiment": {"rounds": 1, "num_clients": 2}},
        "sweep": {
            "seed": [0, 1],
            "model_architecture_family": [
                "image_heterogeneous_3", "image_pair"],
        },
        "tuning": {
            "parameters": ["model_architecture_family"],
            "replicate_axis": "seed",
        },
    }
    grid, manifest = expand(spec)
    index = candidate_index(manifest_candidates(manifest))
    expected = {task["task_id"]: task["candidate_id"]
                for task in manifest["tasks"]}
    observed = set()

    for task_id, task in enumerate(grid, 1):
        exp = resolve_experiment_architectures(
            ExperimentConfig(**task["experiment"]), input_kind="image")
        record = {
            "algorithm": "local",
            "config": {
                "experiment": exp.model_dump(mode="json"),
                "algorithm": config_class("local")().model_dump(mode="json"),
            },
        }
        candidate = candidate_of(record, manifest, index)
        assert candidate == expected[task_id]
        observed.add(candidate)

    assert all("model_architecture_family" not in task["experiment"]
               for task in grid)
    assert manifest["tuning_parameters"] == ["exp.model_architectures"]
    assert manifest["declared_tuning_parameters"] == [
        "exp.model_architecture_family"]
    assert observed == {0, 1}


def test_architecture_family_and_explicit_lists_make_same_candidates(monkeypatch):
    pair = ["fedavg_cnn", "cifar_resnet18"]
    monkeypatch.setitem(MODEL_ARCHITECTURE_FAMILIES, "image_pair", pair)
    common = {
        "algorithms": ["local"],
        "base": {"experiment": {"rounds": 1, "num_clients": 2}},
    }
    family_grid, family_manifest = expand({
        **common,
        "sweep": {
            "seed": [0, 1],
            "model_architecture_family": [
                "image_heterogeneous_3", "image_pair"],
        },
        "tuning": {
            "parameters": ["model_architecture_family"],
            "replicate_axis": "seed",
        },
    })
    list_grid, list_manifest = expand({
        **common,
        "sweep": {
            "seed": [0, 1],
            "model_architectures": [
                list(MODEL_ARCHITECTURE_FAMILIES["image_heterogeneous_3"]),
                pair],
        },
        "tuning": {
            "parameters": ["model_architectures"],
            "replicate_axis": "seed",
        },
    })

    assert family_grid == list_grid
    assert family_manifest["candidates"] == list_manifest["candidates"]
    assert family_manifest["tasks"] == list_manifest["tasks"]


def test_untuned_architecture_axis_remains_a_separate_condition(monkeypatch):
    monkeypatch.setitem(
        MODEL_ARCHITECTURE_FAMILIES, "image_pair",
        ["fedavg_cnn", "cifar_resnet18"])
    _, manifest = expand({
        "algorithms": ["feddes"],
        "base": {"experiment": {"rounds": 1, "num_clients": 2}},
        "sweep": {
            "seed": [0, 1],
            "model_architecture_family": [
                "image_heterogeneous_3", "image_pair"],
            "algorithm.graph_k": [3, 5],
        },
        "tuning": {
            "parameters": ["algorithm.graph_k"],
            "replicate_axis": "seed",
        },
    })

    assert "exp.model_architectures" in manifest["condition_axes"]
    conditions = {
        tuple(task["condition"]["exp.model_architectures"])
        for task in manifest["tasks"]
    }
    assert conditions == {
        tuple(MODEL_ARCHITECTURE_FAMILIES["image_heterogeneous_3"]),
        tuple(MODEL_ARCHITECTURE_FAMILIES["image_pair"]),
    }


def test_task_ids_and_candidate_ids_are_distinct_numbers():
    _, manifest = expand(_spec())
    assert [t["task_id"] for t in manifest["tasks"]] == list(range(1, 13))   # 1-based
    assert min(c["id"] for c in manifest["candidates"]) == 0                 # 0-based


# ── 5-6: the winner is one complete combination, not a per-axis argmax ───────

def _interaction_case():
    """A grid where choosing each axis independently picks the wrong point.

    Marginal means favour lr=0.1 (0.775 vs 0.725) and graph_k=5 (0.85 vs 0.65),
    so a per-axis search lands on (0.1, 5) = 0.75. The best joint assignment is
    (0.01, 5) = 0.95.
    """
    grid, manifest = expand(_spec())
    val = {}
    for c in manifest["candidates"]:
        p = c["parameters"]
        val[c["id"]] = {(0.01, 3): 0.50, (0.01, 5): 0.95,
                        (0.1, 3): 0.80, (0.1, 5): 0.75}[(p["algorithm.base_lr"], p["algorithm.graph_k"])]
    recs = _records(grid, manifest, lambda cid: _flat(val[cid], 0.5))
    return grid, manifest, recs, val


def test_the_selected_candidate_is_one_complete_parameter_combination():
    grid, manifest, recs, val = _interaction_case()
    art = _rank(recs, manifest)
    cid = _selected(art)
    params = next(c["parameters"] for c in art["groups"][0]["candidates"]
                  if c["id"] == cid)
    assert set(params) == {"algorithm.base_lr", "algorithm.graph_k"}       # complete
    assert params == {"algorithm.base_lr": 0.01, "algorithm.graph_k": 5}   # the joint optimum


def test_values_are_not_selected_independently_per_parameter():
    grid, manifest, recs, val = _interaction_case()
    art = _rank(recs, manifest)
    params = next(c["parameters"] for c in art["groups"][0]["candidates"]
                  if c["id"] == _selected(art))

    # what a per-axis search would have chosen, computed the same way
    def marginal(axis):
        buckets: dict = {}
        for c in art["groups"][0]["candidates"]:
            buckets.setdefault(c["parameters"][axis], []).append(val[c["id"]])
        return max(buckets, key=lambda k: sum(buckets[k]) / len(buckets[k]))

    per_axis = {a: marginal(a) for a in ("algorithm.base_lr", "algorithm.graph_k")}
    assert per_axis == {"algorithm.base_lr": 0.1, "algorithm.graph_k": 5}
    assert params != per_axis          # the joint winner is a different point


# ── 7-10: tuning groups ──────────────────────────────────────────────────────

def test_non_tuned_alpha_creates_separate_tuning_groups():
    spec = _spec(sweep={"seed": [0, 1, 2], "alpha": [0.1, 0.5],
                        "algorithm.base_lr": [0.01, 0.1], "algorithm.graph_k": [3, 5]})
    grid, manifest = expand(spec)
    assert manifest["condition_axes"] == ["exp.alpha"]
    assert len(manifest["candidates"]) == 4          # alpha is not part of a candidate

    # alpha=0.1 favours candidate 0; alpha=0.5 favours candidate 3
    metadata = {t["task_id"]: t for t in manifest["tasks"]}
    recs = []
    for task_id, t in enumerate(grid, 1):
        cid = metadata[task_id]["candidate_id"]
        best = 0 if t["experiment"]["alpha"] == 0.1 else 3
        recs.append({"algorithm": t["algorithm"],
                     "config": {"experiment": dict(t["experiment"]),
                                "algorithm": dict(t["algorithm_config"])},
                     "result": _flat(0.9 if cid == best else 0.5, 0.5)})
    art = _rank(recs, manifest)
    assert len(art["groups"]) == 2
    assert [g["label"] for g in art["groups"]] == ["feddes [alpha=0.1]",
                                                   "feddes [alpha=0.5]"]
    assert [_selected(art, i) for i in (0, 1)] == [0, 3]


def test_whatever_is_declared_the_replicate_axis_does_not_split_groups():
    """Replicates are averaged over, not compared. That holds for the axis the
    user declares, not only for one called ``seed`` -- ``seed`` happens to sit
    outside the collector's condition fields already, so the rule would look
    satisfied while doing nothing."""
    spec = _spec(sweep={"batch": [16, 32], "algorithm.base_lr": [0.01, 0.1],
                        "algorithm.graph_k": [3, 5]},
                 tuning={"strategy": "grid",
                         "parameters": ["algorithm.base_lr", "algorithm.graph_k"],
                         "replicate_axis": "batch"})
    grid, manifest = expand(spec)
    assert manifest["replicate_axis"] == "exp.batch"
    assert manifest["condition_axes"] == []
    recs = _records(grid, manifest, lambda cid: _flat(0.5 + 0.01 * cid, 0.5))
    art = _rank(recs, manifest)
    assert len(art["groups"]) == 1                     # not one group per batch
    assert art["groups"][0]["expected_seeds"] == [16, 32]
    for c in art["groups"][0]["candidates"]:
        assert c["views"]["global"]["observed_seeds"] == [16, 32]


def test_declaring_alpha_as_tuned_makes_it_part_of_the_candidate():
    spec = _spec(sweep={"seed": [0, 1, 2], "alpha": [0.1, 0.5],
                        "algorithm.base_lr": [0.01, 0.1], "algorithm.graph_k": [3, 5]},
                 tuning={"strategy": "grid",
                         "parameters": ["exp.alpha", "algorithm.base_lr", "algorithm.graph_k"],
                         "replicate_axis": "seed"})
    grid, manifest = expand(spec)
    assert len(manifest["candidates"]) == 8          # 2 alphas x 2 lr x 2 graph_k
    assert manifest["condition_axes"] == []
    assert all("exp.alpha" in c["parameters"] for c in manifest["candidates"])

    recs = _records(grid, manifest, lambda cid: _flat(0.5 + 0.01 * cid, 0.5))
    art = _rank(recs, manifest)
    assert len(art["groups"]) == 1                   # one group, alpha is searched
    assert len(art["groups"][0]["candidates"]) == 8


def test_different_algorithms_are_ranked_separately():
    spec = _spec(algorithms=["feddes", "fedproto"],
                 sweep={"seed": [0], "algorithm.base_lr": [0.01, 0.1],
                        "algorithm.graph_k": [3, 5]})
    grid, manifest = expand(spec)
    recs = _records(grid, manifest, lambda cid: _flat(0.5 + 0.01 * cid, 0.5))
    art = _rank(recs, manifest)
    assert {g["algorithm"] for g in art["groups"]} == {"feddes", "fedproto"}
    for g in art["groups"]:
        assert {c["algorithm"] for c in g["candidates"]} == {g["algorithm"]}
        # every ranked candidate belongs to this group's algorithm
        assert set(g["rankings"]["global"]["order"]) <= {c["id"] for c in g["candidates"]}


def test_algorithm_specific_axis_makes_no_duplicate_candidates_for_other_algorithms():
    """Each algorithm gets only the candidate axes its configuration defines."""
    spec = _spec(algorithms=["feddes", "local"],
                 sweep={"seed": [0], "algorithm.base_lr": [0.01, 0.1],
                        "algorithm.lr": [0.01, 0.1],
                        "algorithm.graph_k": [3, 5, 10]},
                 tuning={"strategy": "grid",
                         "parameters": ["algorithm.base_lr", "algorithm.lr",
                                        "algorithm.graph_k"],
                         "replicate_axis": "seed"})
    _, manifest = expand(spec)
    per_algorithm: dict[str, list] = {}
    for c in manifest["candidates"]:
        per_algorithm.setdefault(c["algorithm"], []).append(c["parameters"])
    assert len(per_algorithm["feddes"]) == 6            # 2 lr x 3 graph_k
    assert len(per_algorithm["local"]) == 2             # lr only
    assert all("algorithm.graph_k" not in p for p in per_algorithm["local"])
    assert len({json.dumps(p, sort_keys=True) for p in per_algorithm["local"]}) == 2


# ── 11-13: what ranks, and in which direction ────────────────────────────────

def test_changing_only_test_metrics_cannot_change_ranking_or_selection():
    grid, manifest = expand(_spec())
    val = {c["id"]: 0.5 + 0.01 * c["id"] for c in manifest["candidates"]}

    a = _rank(_records(grid, manifest, lambda cid: _flat(val[cid], 0.10 * cid)), manifest,
              views=VIEWS)
    # test values reversed -- the candidate with the worst test is now the best
    b = _rank(_records(grid, manifest, lambda cid: _flat(val[cid], 1.0 - 0.10 * cid)), manifest,
              views=VIEWS)

    for view in VIEWS:
        ra = a["groups"][0]["rankings"][view]
        rb = b["groups"][0]["rankings"][view]
        assert ra["order"] == rb["order"]
        assert ra["selected_candidate"] == rb["selected_candidate"]
    # ...and the test numbers really did change, so the test is not vacuous
    ta = a["groups"][0]["candidates"][0]["views"]["global"]["test"]["mean"]
    tb = b["groups"][0]["candidates"][0]["views"]["global"]["test"]["mean"]
    assert ta != tb


def test_different_validation_metrics_can_select_different_candidates():
    grid, manifest = expand(_spec())
    order = {c["id"]: i for i, c in enumerate(manifest["candidates"])}

    def hist(cid):
        i = order[cid]
        return _hist({"accuracy": [[0.5 + 0.01 * i]] * 2,
                      "macro_f1": [[0.9 - 0.01 * i]] * 2},
                     {"accuracy": [[0.5]] * 2, "macro_f1": [[0.5]] * 2})

    recs = _records(grid, manifest, hist)
    by_acc = _selected(_rank(recs, manifest, metric="accuracy"))
    by_f1 = _selected(_rank(recs, manifest, metric="macro_f1"))
    assert by_acc != by_f1


def test_minimize_direction_metrics_rank_correctly():
    register("tuning_loss", "minimize")               # a name + a direction only
    try:
        grid, manifest = expand(_spec())
        order = {c["id"]: i for i, c in enumerate(manifest["candidates"])}
        recs = _records(grid, manifest, lambda cid: _hist(
            {"tuning_loss": [[1.0 - 0.1 * order[cid]]] * 2},
            {"tuning_loss": [[0.5]] * 2}))
        art = _rank(recs, manifest, metric="tuning_loss")
        assert art["selection"]["direction"] == "minimize"
        ranked = art["groups"][0]["rankings"]["global"]["order"]
        means = [next(c["views"]["global"]["validation"]["mean"]
                      for c in art["groups"][0]["candidates"] if c["id"] == cid)
                 for cid in ranked]
        assert means == sorted(means)                # smallest loss first
        assert art["groups"][0]["rankings"]["global"]["selected_candidate"] == ranked[0]
    finally:
        unregister("tuning_loss")


# ── 14-16: client aggregation and the two views ──────────────────────────────

def test_weighted_and_unweighted_aggregation_can_rank_differently():
    grid, manifest = expand(_spec())
    ids = [c["id"] for c in manifest["candidates"]]
    a, b = ids[0], ids[1]
    counts = [1, 99]

    def hist(cid):
        if cid == a:        # good on the tiny client, poor on the large one
            v, t = [[0.90], [0.50]], [[0.5], [0.5]]
        elif cid == b:
            v, t = [[0.50], [0.60]], [[0.5], [0.5]]
        else:
            v, t = [[0.10], [0.10]], [[0.5], [0.5]]
        return _hist({"accuracy": v}, {"accuracy": t}, counts)

    recs = _records(grid, manifest, hist)
    unweighted = _selected(_rank(recs, manifest, aggregation="mean"))
    weighted = _selected(_rank(recs, manifest, aggregation="weighted_mean"))
    assert unweighted == a                # 0.70 vs 0.55
    assert weighted == b                  # 0.504 vs 0.599
    assert unweighted != weighted


def _view_disagreement():
    """Candidate A peaks together; candidate B peaks in different rounds."""
    grid, manifest = expand(_spec())
    ids = [c["id"] for c in manifest["candidates"]]
    a, b = ids[0], ids[1]

    def hist(cid):
        if cid == a:
            v = [[0.80, 0.80], [0.80, 0.80]]      # global 0.80, per-client 0.80
        elif cid == b:
            v = [[0.90, 0.10], [0.10, 0.90]]      # global 0.50, per-client 0.90
        else:
            v = [[0.20, 0.20], [0.20, 0.20]]
        return _hist({"accuracy": v}, {"accuracy": [[0.5, 0.5], [0.5, 0.5]]})

    return _records(grid, manifest, hist), manifest, a, b


def test_global_and_per_client_views_can_select_different_candidates():
    recs, manifest, a, b = _view_disagreement()
    art = _rank(recs, manifest, views=VIEWS)
    assert _selected(art, view="global") == a
    assert _selected(art, view="per-client") == b


def test_both_views_are_recorded_and_neither_is_primary():
    recs, manifest, a, b = _view_disagreement()
    art = _rank(recs, manifest, views=VIEWS)
    rankings = art["groups"][0]["rankings"]
    assert set(rankings) == {"global", "per-client"}
    assert art["selection"]["views"] == VIEWS
    assert "view" not in art["selection"]             # no singular, primary view
    for view in VIEWS:
        assert rankings[view]["selected_candidate"] is not None
        assert rankings[view]["order"]
    # each candidate carries its own statistics under each view
    for c in art["groups"][0]["candidates"]:
        assert set(c["views"]) == {"global", "per-client"}


def test_both_omits_unsupported_global_view_for_one_shot_tuning_group():
    grid, manifest = expand(_spec())
    recs = _records(grid, manifest, lambda cid: _flat(0.75, 0.5))
    for record in recs:
        record["result"]["selection_views_supported"] = ["per-client"]
        record["result"]["selection_provenance"] = {
            "view": "per-client", "stage": "local_computation",
            "metric": "accuracy",
            "clients": {
                "0": {"selected_step": 4, "validation_value": 0.75},
                "1": {"selected_step": 7, "validation_value": 0.75},
            },
        }

    artifact = _rank(recs, manifest, views=VIEWS)
    group = artifact["groups"][0]
    assert set(group["rankings"]) == {"per-client"}
    assert all(set(candidate["views"]) == {"per-client"}
               for candidate in group["candidates"])
    assert any("does not support selection view(s) global" in warning
               for warning in artifact["warnings"])


# ── 17-19: ties and replicate completeness ──────────────────────────────────

def test_candidate_ties_resolve_to_the_lowest_id_deterministically():
    grid, manifest = expand(_spec())
    recs = _records(grid, manifest, lambda cid: _flat(0.75, 0.5))       # every candidate equal
    orders = {tuple(_rank(recs, manifest)["groups"][0]["rankings"]["global"]["order"])
              for _ in range(5)}
    assert len(orders) == 1                                    # deterministic
    order = orders.pop()
    assert list(order) == sorted(order)                        # ...by ascending id
    art = _rank(recs, manifest)
    assert _selected(art) == min(c["id"] for c in manifest["candidates"])
    assert art["selection"]["candidate_tie_break"] == "lowest_id"


def test_missing_seeds_make_a_candidate_ineligible_by_default():
    grid, manifest = expand(_spec())
    best = manifest["candidates"][0]["id"]
    # the best candidate on validation is missing one of its three seeds
    recs = _records(grid, manifest, lambda cid: _flat(0.99 if cid == best else 0.5, 0.5),
                    skip={(best, 2)})
    art = _rank(recs, manifest)
    row = next(c for c in art["groups"][0]["candidates"] if c["id"] == best)
    view = row["views"]["global"]
    assert view["eligible"] is False
    assert view["missing_seeds"] == [2]
    assert "missing replicate(s) [2] of expected [0, 1, 2]" in view["ineligible_reason"]
    assert best not in art["groups"][0]["rankings"]["global"]["order"]
    assert _selected(art) != best
    assert {"id": best, "reason": view["ineligible_reason"]} in \
        art["groups"][0]["rankings"]["global"]["excluded"]


def test_allow_incomplete_ranks_it_and_records_what_was_missing():
    grid, manifest = expand(_spec())
    best = manifest["candidates"][0]["id"]
    recs = _records(grid, manifest, lambda cid: _flat(0.99 if cid == best else 0.5, 0.5),
                    skip={(best, 2)})
    art = _rank(recs, manifest, allow_incomplete=True)
    ranking = art["groups"][0]["rankings"]["global"]
    assert _selected(art) == best                    # now eligible to win
    assert ranking["incomplete_ranked"] == [best]
    assert ranking["replicate_counts"] == [2, 3]
    assert art["selection"]["allow_incomplete"] is True
    joined = " ".join(art["warnings"])
    assert "--allow-incomplete" in joined
    assert f"id {best} has 2 of 3, missing [2]" in joined
    assert "unequal replicate counts [2, 3]" in joined
    # the candidate is still marked for what it is
    row = next(c for c in art["groups"][0]["candidates"] if c["id"] == best)
    assert row["views"]["global"]["eligible"] is False


# ── 20: declaration errors ───────────────────────────────────────────────────

def test_unknown_tuning_parameter_fails_clearly():
    with pytest.raises(SystemExit) as e:
        expand(_spec(tuning={"strategy": "grid",
                             "parameters": ["algorithm.base_lr", "algorithm.hidden"],
                             "replicate_axis": "seed"}))
    msg = str(e.value)
    assert "Unknown tuning parameter: algorithm.hidden" in msg
    assert "not a swept axis" in msg
    assert "algorithm.base_lr" in msg and "algorithm.graph_k" in msg  # axes it could name


def test_tuning_parameter_naming_an_unswept_field_fails():
    with pytest.raises(SystemExit, match="not a swept axis"):
        expand(_spec(tuning={"strategy": "grid", "parameters": ["exp.batch"],
                             "replicate_axis": "seed"}))


def test_invalid_replicate_axis_fails_clearly():
    with pytest.raises(SystemExit) as e:
        expand(_spec(tuning={"strategy": "grid", "parameters": ["algorithm.base_lr"],
                             "replicate_axis": "repetition"}))
    assert "Unknown replicate axis: repetition" in str(e.value)


def test_missing_replicate_axis_fails_clearly():
    with pytest.raises(SystemExit, match="tuning.replicate_axis is unset"):
        expand(_spec(tuning={"strategy": "grid", "parameters": ["algorithm.base_lr"]}))


def test_replicate_axis_cannot_also_be_a_tuning_parameter():
    with pytest.raises(SystemExit, match="both a tuning parameter and the replicate"):
        expand(_spec(tuning={"strategy": "grid",
                             "parameters": ["algorithm.base_lr", "seed"],
                             "replicate_axis": "seed"}))


def test_duplicate_tuning_parameters_are_rejected():
    with pytest.raises(SystemExit, match="Duplicate tuning parameter"):
        expand(_spec(tuning={"strategy": "grid",
                             "parameters": ["algorithm.base_lr", "exp.alpha", "alpha"],
                             "replicate_axis": "seed",
                             },
                     sweep={"seed": [0], "alpha": [0.1, 0.5],
                            "algorithm.base_lr": [0.01, 0.1]}))


def test_only_grid_strategy_is_accepted():
    with pytest.raises(SystemExit, match="Only grid is implemented"):
        expand(_spec(tuning={"strategy": "bayesian", "parameters": ["algorithm.base_lr"],
                             "replicate_axis": "seed"}))


def test_unknown_sweep_axis_still_fails_before_tuning_is_parsed():
    with pytest.raises(SystemExit, match="Unknown sweep axis"):
        expand(_spec(sweep={"seed": [0], "algorithm.graf_k": [3, 5]},
                     tuning={"strategy": "grid", "parameters": ["algorithm.graf_k"],
                             "replicate_axis": "seed"}))


def test_a_sweep_without_a_tuning_block_writes_no_manifest():
    grid, manifest = expand({"algorithms": ["feddes"], "sweep": {"seed": [0, 1]}})
    assert manifest is None
    assert all("candidate_id" not in t for t in grid)


def test_declaring_tuning_does_not_change_the_run_tasks():
    spec = _spec()
    tuned, manifest = expand(spec)
    plain, none = expand({k: v for k, v in spec.items() if k != "tuning"})
    assert manifest is not None and none is None
    assert tuned == plain
    assert all("candidate_id" not in task and "replicate" not in task
               for task in tuned)


# ── 21-22: the artifact and the configurations it writes ────────────────────

def test_the_selected_configuration_is_complete_and_directly_runnable(tmp_path):
    from rigfl.experiment.config import ExperimentConfig
    from rigfl.experiment.registry import config_class

    grid, manifest = expand(_spec())
    recs = []
    metadata = {t["task_id"]: t for t in manifest["tasks"]}
    for task_id, t in enumerate(grid, 1):  # resolved configs, as run_task writes them
        meta = metadata[task_id]
        cid, replicate = meta["candidate_id"], meta["replicate"]
        exp = ExperimentConfig(**t["experiment"])
        cfg = config_class(t["algorithm"])(**t["algorithm_config"])
        recs.append({"algorithm": t["algorithm"],
                     "config": {"experiment": exp.model_dump(),
                                "algorithm": cfg.model_dump()},
                     "result": _flat(0.5 + 0.01 * cid, 0.5),
                     "_source_file": f"c{cid}_s{replicate}.json"})

    art = _rank(recs, manifest)
    written = write_selection(art, recs, manifest, tmp_path)
    cfg_files = [p for p in written if p.suffix == ".yaml"]
    assert len(cfg_files) == 1

    import yaml
    payload = yaml.safe_load(cfg_files[0].read_text())
    spec = payload["selected_configuration"]

    # complete: it re-expands, and every task it produces validates
    regrid, remanifest = expand(dict(spec, name="final"))
    assert remanifest is None                     # a plain sweep, not another search
    assert len(regrid) == len(manifest["replicate_values"])
    for t in regrid:
        ExperimentConfig(**t["experiment"])
        config_class(t["algorithm"])(**t["algorithm_config"])

    # the tuned values are the winning candidate's, jointly
    winner = next(c for c in manifest["candidates"] if c["id"] == _selected(art))
    for path, value in winner["parameters"].items():
        assert regrid[0]["algorithm_config"][path.split(".", 1)[1]] == value

    # seeds stay parameterised rather than baked in as one arbitrary replicate
    assert spec["sweep"]["seed"] == manifest["replicate_values"]
    assert "seed" not in spec["base"]["experiment"]

    prov = payload["provenance"]
    assert prov["candidate_id"] == _selected(art)
    assert prov["candidate_parameters"] == winner["parameters"]
    assert prov["expected_seeds"] == [0, 1, 2] == prov["observed_seeds"]
    assert prov["selection"]["split"] == "validation"
    assert "test performance was not consulted" in prov["selected_because"]
    assert len(prov["source_results"]) == 3
    assert "test" not in prov                     # no test number cited as a reason


def test_multiple_output_groups_cannot_silently_overwrite_one_another(tmp_path):
    spec = _spec(algorithms=["feddes", "fedproto"],
                 sweep={"seed": [0], "alpha": [0.1, 0.5],
                        "algorithm.base_lr": [0.01, 0.1],
                        "algorithm.graph_k": [3, 5]})
    grid, manifest = expand(spec)
    recs = _records(grid, manifest, lambda cid: _flat(0.5 + 0.01 * cid, 0.5))
    art = _rank(recs, manifest, views=VIEWS)

    assert len(art["groups"]) == 4                       # 2 algorithms x 2 alphas
    written = write_selection(art, recs, manifest, tmp_path)
    cfgs = [p for p in written if p.suffix == ".yaml"]
    assert len(cfgs) == 4 * 2                            # one per (group, view)
    assert len({p.name for p in cfgs}) == len(cfgs)      # all distinct filenames
    assert len({p.read_text() for p in cfgs}) == len(cfgs)   # and distinct contents

    # and the guard itself: two groups that would land on one path are refused
    art["groups"][1]["group_id"] = art["groups"][0]["group_id"]
    art["groups"][1]["algorithm"] = art["groups"][0]["algorithm"]
    art["groups"][1]["condition"] = art["groups"][0]["condition"]
    art["groups"][1]["label_fields"] = art["groups"][0]["label_fields"]
    with pytest.raises(TuningError, match="Refusing rather than overwriting"):
        write_selection(art, recs, manifest, tmp_path / "again")


def test_the_artifact_records_the_full_selection_protocol():
    grid, manifest = expand(_spec())
    recs = _records(grid, manifest, lambda cid: _flat(0.5 + 0.01 * cid, 0.4))
    art = _rank(recs, manifest, views=VIEWS, aggregation="weighted_mean",
                tie_break="latest")
    sel = art["selection"]
    assert sel == {"metric": "accuracy", "split": "validation", "views": VIEWS,
                   "client_aggregation": "weighted_mean", "seed_aggregation": "mean",
                   "direction": "maximize", "round_tie_break": "latest",
                   "candidate_tie_break": "lowest_id", "allow_incomplete": False,
                   "ranked_on": "validation only; test statistics are reported, "
                                "never ranked"}
    assert art["schema_version"] == 2 and art["kind"] == "rigfl.tuning_selection"
    # test statistics are present for every candidate, in every view
    for c in art["groups"][0]["candidates"]:
        for v in VIEWS:
            assert c["views"][v]["test"]["mean"] is not None


# ── placement derives from recorded configurations ──────────────────────────

def test_results_are_placed_by_their_recorded_parameter_values():
    grid, manifest = expand(_spec())
    recs = _records(grid, manifest, lambda cid: _flat(0.6, 0.5))
    assert all("tuning" not in r for r in recs)
    placed, _, unassigned = place_records(recs, manifest)
    assert not unassigned
    assert sum(len(v) for group in placed.values() for v in group.values()) == len(recs)


def test_a_result_matching_no_candidate_is_reported_not_invented():
    grid, manifest = expand(_spec())
    recs = _records(grid, manifest, lambda cid: _flat(0.6, 0.5))
    recs[0]["config"]["algorithm"]["graph_k"] = 999            # never swept
    art = _rank(recs, manifest)
    assert len(art["unassigned_records"]) == 1
    assert art["unassigned_records"][0]["parameters"]["algorithm.graph_k"] == 999
    assert any("match no candidate" in w for w in art["warnings"])


def test_string_and_numeric_spellings_of_a_value_match(tmp_path):
    """``--sweep alpha=0.1,0.5`` puts strings on the axis; results hold floats."""
    grid, manifest = expand(_spec(sweep={"seed": "0-2", "algorithm.base_lr": "0.01,0.1",
                                         "algorithm.graph_k": "3,5"}))
    assert manifest["replicate_values"] == [0, 1, 2]
    recs = []
    for t in grid:
        recs.append({"algorithm": t["algorithm"],
                     "config": {"experiment": dict(t["experiment"], seed=int(t["experiment"]["seed"])),
                                "algorithm": {k: float(v) if k == "base_lr" else int(v)
                                           for k, v in t["algorithm_config"].items()}},
                     "result": _flat(0.6, 0.5)})
    _, _, unassigned = place_records(recs, manifest)
    assert not unassigned


def test_manifest_round_trips_through_the_results_directory(tmp_path):
    _, manifest = expand(_spec())
    write_manifest(manifest, tmp_path)
    loaded = load_manifest(tmp_path)
    assert loaded["candidates"] == manifest["candidates"]
    assert loaded["schema_version"] == 2
    assert load_manifest(tmp_path / "empty") is None


def test_a_manifest_from_a_future_schema_is_refused(tmp_path):
    from rigfl.experiment.tuning import MANIFEST_NAME

    _, manifest = expand(_spec())
    # Written directly: write_manifest validates what it installs, so a manifest
    # from a future schema cannot be produced through it. This is about the
    # reader meeting one that another version wrote.
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(dict(manifest, schema_version=99)))
    with pytest.raises(TuningError, match="manifest schema 99"):
        load_manifest(tmp_path)


# ── the collector's interface ────────────────────────────────────────────────

def _sweep_dir(tmp_path, spec, *, with_manifest=True, val=None):
    """A results directory holding real result files for a declared sweep."""
    from rigfl.experiment.config import ExperimentConfig, result_filename, run_fingerprint
    from rigfl.experiment.registry import config_class
    from rigfl.experiment.artifacts import make_run_record

    grid, manifest = expand(spec)
    d = tmp_path / spec.get("name", "tune")
    d.mkdir(parents=True, exist_ok=True)
    if manifest and with_manifest:
        write_manifest(manifest, d)
    metadata = ({t["task_id"]: t for t in manifest["tasks"]} if manifest else {})
    for task_id, t in enumerate(grid, 1):
        exp = ExperimentConfig(**t["experiment"])
        cfg = config_class(t["algorithm"])(**t["algorithm_config"])
        cid = metadata.get(task_id, {}).get("candidate_id", 0)
        fp = run_fingerprint(exp, cfg.model_dump())
        rec = make_run_record(
            algorithm=t["algorithm"], experiment=exp.model_dump(),
            algorithm_config=cfg.model_dump(), run_fingerprint=fp,
            result=_flat((val or (lambda c: 0.5 + 0.01 * c))(cid), 0.4,
                         clients=exp.num_clients, also=("balanced_accuracy",)))
        (d / result_filename(exp, t["algorithm"], fp)).write_text(json.dumps(rec))
    return d, manifest


def _collect(monkeypatch, *argv):
    from rigfl.experiment import collect
    monkeypatch.setattr("sys.argv", ["collect", *map(str, argv)])
    collect.main()


def test_select_out_without_a_manifest_fails_clearly(tmp_path, monkeypatch):
    d, _ = _sweep_dir(tmp_path, {"name": "plain", "algorithms": ["feddes"],
                                 "base": {"experiment": {"rounds": 1, "eval_gap": 1}},
                                 "sweep": {"seed": [0, 1], "algorithm.graph_k": [3, 5]}})
    with pytest.raises(SystemExit) as e:
        _collect(monkeypatch, "--results-dir", d, "--selection-metric", "accuracy",
                 "--rank", "--select-out", tmp_path / "out")
    msg = str(e.value)
    assert "--select-out needs a tuning manifest" in msg
    assert "would be a guess" in msg


def test_rank_without_a_manifest_keeps_the_ordinary_table_ranking(tmp_path,
                                                                  monkeypatch, capsys):
    d, _ = _sweep_dir(tmp_path, {"name": "plain", "algorithms": ["feddes"],
                                 "base": {"experiment": {"rounds": 1, "eval_gap": 1}},
                                 "sweep": {"seed": [0, 1], "algorithm.graph_k": [3, 5]}})
    _collect(monkeypatch, "--results-dir", d, "--selection-metric", "accuracy", "--rank")
    out = capsys.readouterr().out
    assert "ranked by VALIDATION (test never ranks):" in out
    assert "hyperparameter candidates" not in out


def test_rank_with_a_manifest_is_candidate_aware(tmp_path, monkeypatch, capsys):
    d, manifest = _sweep_dir(tmp_path, _spec())
    _collect(monkeypatch, "--results-dir", d, "--selection-metric", "accuracy",
             "--selection-view", "both", "--rank",
             "--select-out", tmp_path / "selected")
    out = capsys.readouterr().out
    assert "ranked on VALIDATION accuracy" in out
    assert "test columns are reported for inspection and take no part in ranking." in out
    assert "selection-view: global" in out and "selection-view: per-client" in out
    assert "ranked by VALIDATION (test never ranks):" not in out    # not the old path

    artifact = json.loads((tmp_path / "selected" / "selection.json").read_text())
    assert artifact["kind"] == "rigfl.tuning_selection"
    assert artifact["selection"]["views"] == ["global", "per-client"]
    written = sorted(artifact["selected_configurations"])
    assert len(written) == 2                         # one group, two views
    assert all(n.startswith("group0_feddes_") for n in written)
    assert [n.endswith("_global.yaml") for n in written].count(True) == 1
    assert [n.endswith("_per-client.yaml") for n in written].count(True) == 1
    for name in written:
        assert (tmp_path / "selected" / "configs" / name).is_file()


def test_old_schema_results_are_refused_by_the_collector(tmp_path, monkeypatch):
    d, _ = _sweep_dir(tmp_path, _spec())
    victim = sorted(d.glob("cifar10_*.json"))[0]
    rec = json.loads(victim.read_text())
    rec["result"] = {"best_round": 3, "val_bacc": 0.7, "test": {"acc": 0.6, "bacc": 0.6}}
    victim.write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="cannot be read as completed runs"):
        _collect(monkeypatch, "--results-dir", d, "--selection-metric",
                 "balanced_accuracy", "--rank")


def test_the_manifest_is_not_read_as_a_result_file(tmp_path, monkeypatch, capsys):
    from rigfl.experiment.collect import load_results
    d, _ = _sweep_dir(tmp_path, _spec())
    by_algorithm = load_results(d, None, None)
    assert sum(len(v) for v in by_algorithm.values()) == 12       # 4 candidates x 3 seeds
    assert "skipped" not in capsys.readouterr().out


def _stub_run_one(name, exp, cfg, device, **kw):
    """What run_one returns, without training: a real envelope over a real history."""
    from rigfl.experiment.artifacts import make_run_record
    from rigfl.experiment.config import run_fingerprint

    return make_run_record(
        algorithm=name, experiment=exp.model_dump(), algorithm_config=cfg.model_dump(),
        run_fingerprint=run_fingerprint(exp, cfg.model_dump()),
        result=_flat(0.5, 0.5, clients=exp.num_clients, rounds=exp.rounds),
        device=str(device), wall_seconds=0.0)


def test_run_task_does_not_embed_tuning_identity_in_the_result(tmp_path, monkeypatch):
    from rigfl.experiment import launch

    grid, manifest = expand(_spec())
    grid_path = tmp_path / "grid.jsonl"
    grid_path.write_text("".join(json.dumps(t) + "\n" for t in grid))

    monkeypatch.setattr(launch, "run_one", _stub_run_one)
    monkeypatch.setattr(launch, "resolve_device", lambda d: "cpu")
    monkeypatch.setattr(launch, "resolve_experiment_data", lambda exp: (exp, None))
    launch.run_task(str(grid_path), 5, tmp_path)

    written = [p for p in tmp_path.glob("*.json")]
    assert len(written) == 1
    rec = json.loads(written[0].read_text())
    assert "tuning" not in rec


def test_run_task_writes_no_tuning_identity(tmp_path, monkeypatch):
    from rigfl.experiment import launch

    grid, manifest = expand({"algorithms": ["feddes"], "sweep": {"seed": [0, 1]}})
    assert manifest is None
    grid_path = tmp_path / "grid.jsonl"
    grid_path.write_text("".join(json.dumps(t) + "\n" for t in grid))
    monkeypatch.setattr(launch, "run_one", _stub_run_one)
    monkeypatch.setattr(launch, "resolve_device", lambda d: "cpu")
    monkeypatch.setattr(launch, "resolve_experiment_data", lambda exp: (exp, None))
    launch.run_task(str(grid_path), 1, tmp_path)
    rec = json.loads(next(iter(tmp_path.glob("*.json"))).read_text())
    assert "tuning" not in rec
