"""Saved test predictions match the rounds reporting selects."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.algorithms.local import Local, LocalConfig
from rigfl.core import Client, ClientModel, LearnedProjection, independent, iterative
from rigfl.eval.predictions import SelectedPredictions, predictions_at
from rigfl.eval.selection import select_client_specific, select_shared
from rigfl.eval.transfer import _client_flips, flip_summary

NC, DIM = 3, 8
_CENTERS = torch.randn(NC, DIM, generator=torch.Generator().manual_seed(0)) * 2.5
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def _loader(n, shuffle=False):
    xs = [_CENTERS[c] + 2.0 * torch.randn(n, DIM) for c in range(NC)]
    ys = [torch.full((n,), c) for c in range(NC)]
    return DataLoader(TensorDataset(torch.cat(xs), torch.cat(ys)),
                      batch_size=16, shuffle=shuffle)


def _clients():
    return [Client(model=ClientModel(nn.Sequential(nn.Linear(DIM, hidden), nn.ReLU()),
                                     LearnedProjection(hidden, 6), nn.Linear(6, NC)),
                   train_loader=_loader(24, shuffle=True),
                   val_loader=_loader(n),
                   test_loader=_loader(n))
            for hidden, n in ((16, 6), (24, 10))]


def _run(runner):
    torch.manual_seed(0)
    return runner(Local(LocalConfig(local_epochs=1, lr=0.05)), _clients(),
                  num_rounds=8, device=torch.device("cpu"), num_classes=NC,
                  eval_gap=1, verbose=False, save_test_predictions=True)


def _check(record, selected, rounds):
    for i, cid in enumerate(selected["client_ids"]):
        saved = predictions_at(record, cid, rounds(cid))
        assert saved is not None
        correct = sum(p == y for p, y in zip(saved["predicted"], saved["labels"]))
        assert correct / len(saved["labels"]) == pytest.approx(
            selected["test"]["accuracy"][i])
        assert len(saved["true_class_probability"]) == len(saved["labels"])


@pytest.mark.parametrize("metric", METRICS)
def test_iterative_saves_predictions_at_every_selectable_round(metric):
    result = _run(iterative)
    record = {"result": result}
    history = result["evaluation_history"]
    selected = select_client_specific(history, metric)
    _check(record, selected, lambda cid: selected["selected_rounds"][cid])
    for how in ("uniform", "sample_count"):
        selected = select_shared(history, metric, aggregation=how)
        _check(record, selected, lambda cid: selected["selected_round"])


@pytest.mark.parametrize("metric", METRICS)
def test_local_saves_each_clients_selected_epoch(metric):
    result = _run(independent)
    selected = select_client_specific(result["evaluation_history"], metric)
    _check({"result": result}, selected, lambda cid: selected["selected_rounds"][cid])


@pytest.mark.parametrize("runner", (independent, iterative))
def test_predictions_are_not_saved_by_default(runner):
    result = runner(Local(LocalConfig(local_epochs=1, lr=0.05)), _clients(),
                    num_rounds=2, device=torch.device("cpu"), num_classes=NC,
                    verbose=False)
    assert "test_predictions" not in result


def test_a_tie_keeps_the_first_round_and_drops_superseded_rounds():
    kept = SelectedPredictions()
    test = {"labels": [0, 1], "predicted": [0, 1], "true_class_probability": [0.9, 0.8]}
    for rnd, value in ((0, 0.5), (1, 0.7), (2, 0.7)):
        kept.observe_client(0, rnd, {name: value for name in METRICS}, test)
    assert list(kept.to_dict()["clients"]["0"]["rounds"]) == ["1"]


def test_balanced_accuracy_flips_average_the_classes():
    # class 0: 1 of 8 samples flips negative; class 1: 1 of 2 flips positive
    labels = [0] * 8 + [1] * 2
    local = [0] * 8 + [0, 1]
    ours = [1] + [0] * 7 + [1, 1]

    def record(predicted):
        return {"result": {"test_predictions": {"clients": {"0": {
            "labels": labels,
            "rounds": {"0": {"predicted": predicted,
                             "true_class_probability": None}}}}}}}

    selected = {"selected_round": 0}
    pair = {"replicate_condition": {"partition_seed": 0, "split_seed": 0,
                                    "training_seed": 0},
            "data_configuration_hash": "h",
            "dataset": "d", "partition_id": 0,
            "flips": _client_flips(record(ours), selected, record(local), selected, "0")}
    accuracy = flip_summary([pair], "accuracy", include_uncertainty=False)
    balanced = flip_summary([pair], "balanced_accuracy", include_uncertainty=False)
    assert accuracy["negative_flip_rate"]["estimate"] == pytest.approx(0.1)
    assert accuracy["positive_flip_rate"]["estimate"] == pytest.approx(0.1)
    assert balanced["negative_flip_rate"]["estimate"] == pytest.approx(1 / 16)
    assert balanced["positive_flip_rate"]["estimate"] == pytest.approx(1 / 4)
    assert {c: (v["n"], v["negative"], v["positive"])
            for c, v in pair["flips"]["per_class"].items()} == {"0": (8, 1, 0),
                                                               "1": (2, 0, 1)}
