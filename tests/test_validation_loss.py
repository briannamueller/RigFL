"""Validation loss end to end: history, early stopping, selection, tuning.

Log loss earns its place by separating rounds that the hard-label metrics cannot
tell apart. A model can spend rounds getting steadily more (or less) confident
without a single prediction changing class -- accuracy is flat across all of
them, and stopping on accuracy therefore stops on a tie-break. These tests pin
that difference down, and then check that having it never lets test data reach a
decision.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.core import Client, ClientModel, LearnedProjection, iterative
from rigfl.core.interfaces import Algorithm
from rigfl.eval.selection import select_global, select_per_client
from rigfl.prediction import Predictions

DEVICE = torch.device("cpu")
NUM_CLASSES = 2
VAL_TAG, TEST_TAG = 0.0, 1.0


# ── an algorithm whose confidence is scripted round by round ─────────────────────

def _tagged_loader(tag: float, n: int = 4) -> DataLoader:
    """Inputs whose first feature says which split they are, so a scripted
    algorithm can answer differently for validation and test without the loop
    having to tell it."""
    x = torch.full((n, 2), tag)
    y = torch.arange(n) % NUM_CLASSES
    return DataLoader(TensorDataset(x, y), batch_size=n)


def _scripted_client(n: int = 4) -> Client:
    model = ClientModel(nn.Sequential(nn.Linear(2, 2), nn.ReLU()),
                        LearnedProjection(2, 2), nn.Linear(2, NUM_CLASSES))
    return Client(model=model, train_loader=_tagged_loader(VAL_TAG, n),
                  val_loader=_tagged_loader(VAL_TAG, n),
                  test_loader=_tagged_loader(TEST_TAG, n))


class Scripted(Algorithm):
    """Predicts the true class every round, with a confidence given per round.

    ``val`` and ``test`` are lists of "probability assigned to the true class",
    indexed by round. Every prediction is correct, so accuracy, balanced accuracy
    and macro F1 are 1.0 at every round and cannot separate them. Only the
    distribution differs.
    """

    def __init__(self, val: list[float], test: list[float] | None = None):
        self.val, self.test = val, test or list(val)
        self._round = 0

    def init_globals(self):
        return 0

    def local_train(self, client, shared):
        return None

    def aggregate(self, uploads, shared):
        self._round = self.round_idx
        return self.round_idx                  # shared == the round being evaluated

    def predict(self, client, x, shared) -> Predictions:
        rnd = int(shared)
        p = (self.test if float(x[0, 0]) == TEST_TAG else self.val)[rnd]
        n = x.shape[0]
        truth = torch.arange(n) % NUM_CLASSES
        probs = torch.full((n, NUM_CLASSES), 1.0 - p)
        probs[torch.arange(n), truth] = p
        return Predictions.from_probabilities(probs)


def _run(val, test=None, **kw):
    torch.manual_seed(0)
    return iterative(Scripted(val, test), [_scripted_client()],
                         num_rounds=len(val), device=DEVICE,
                         num_classes=NUM_CLASSES, verbose=False, **kw)


# ── 12: loss is in the history, for both splits ──────────────────────────────

def test_validation_and_test_loss_appear_in_the_evaluation_history():
    hist = _run([0.6, 0.9], test=[0.5, 0.8])["evaluation_history"]
    client = hist["clients"]["0"]
    assert "loss" in client["validation"] and "loss" in client["test"]
    assert client["validation"]["loss"] == pytest.approx(
        [-math.log(0.6), -math.log(0.9)], abs=1e-5)
    assert client["test"]["loss"] == pytest.approx(
        [-math.log(0.5), -math.log(0.8)], abs=1e-5)
    # dense and aligned with every other metric, as the format requires
    n = len(hist["evaluation_rounds"])
    for split in ("validation", "test"):
        assert len(client[split]["loss"]) == n
        assert all(v is not None for v in client[split]["loss"])


# ── the concrete case: same accuracy, different loss ─────────────────────────

SAME_ACCURACY = [0.60, 0.90, 0.70]         # correct every round; confidence differs


def test_loss_separates_rounds_that_accuracy_cannot():
    hist = _run(SAME_ACCURACY)["evaluation_history"]
    client = hist["clients"]["0"]
    assert client["validation"]["accuracy"] == [1.0, 1.0, 1.0]        # indistinguishable
    losses = client["validation"]["loss"]
    assert losses == pytest.approx([0.5108, 0.1054, 0.3567], abs=1e-3)
    assert len(set(losses)) == 3                                      # all distinct

    on_loss = _run(SAME_ACCURACY,
                   early_stopping={"enabled": True, "metric": "loss", "patience": 5})
    on_acc = _run(SAME_ACCURACY,
                  early_stopping={"enabled": True, "metric": "accuracy", "patience": 5})
    assert on_loss["early_stopping"]["metric"] == "loss"
    assert on_loss["early_stopping"]["best_round"] == 1               # the best model
    assert on_acc["early_stopping"]["best_round"] == 0                # a tie-break
    assert on_loss["early_stopping"]["best_value"] == pytest.approx(0.1054, abs=1e-3)


def test_accuracy_based_stopping_halts_where_loss_based_stopping_continues():
    """Same runs, patience 1. Accuracy sees no improvement after round 0 and
    stops before round 2 is ever reached; loss sees round 1 improve and keeps
    going, so the better model gets evaluated at all."""
    on_acc = _run(SAME_ACCURACY, early_stopping={"enabled": True, "metric": "accuracy",
                                                 "patience": 1})
    on_loss = _run(SAME_ACCURACY, early_stopping={"enabled": True, "metric": "loss", "patience": 1})
    assert on_acc["early_stopping"]["stopped_at_round"] == 1
    assert on_acc["evaluation_history"]["evaluation_rounds"] == [0, 1]
    assert on_loss["early_stopping"]["stopped_at_round"] == 2
    assert on_loss["evaluation_history"]["evaluation_rounds"] == [0, 1, 2]
    assert on_loss["early_stopping"]["best_round"] == 1


# ── 13-15: the early-stopping default ────────────────────────────────────────

def test_enabling_early_stopping_defaults_to_validation_loss():
    from rigfl.experiment.config import EarlyStoppingConfig

    cfg = EarlyStoppingConfig(enabled=True, patience=10)
    assert cfg.metric == "loss" and cfg.direction == "minimize"
    record = _run([0.6, 0.9], early_stopping={"enabled": True,
                                               "patience": 10})["early_stopping"]
    assert (record["metric"], record["split"], record["direction"]) == \
        ("loss", "validation", "minimize")


def test_explicit_loss_is_valid_and_resolves_its_direction():
    from rigfl.experiment.config import EarlyStoppingConfig

    cfg = EarlyStoppingConfig(enabled=True, metric="loss", patience=10)
    assert cfg.model_dump() == {"enabled": True, "split": "validation", "metric": "loss",
                                "direction": "minimize", "aggregation": "mean",
                                "patience": 10, "min_delta": 0.0}
    record = _run([0.6, 0.9], early_stopping={"enabled": True, "metric": "loss",
                                              "patience": 10})["early_stopping"]
    assert (record["metric"], record["split"], record["direction"]) == \
        ("loss", "validation", "minimize")
    assert record["best_round"] == 1                       # the lower loss


def test_explicit_accuracy_is_valid_and_resolves_its_direction():
    from rigfl.experiment.config import EarlyStoppingConfig

    cfg = EarlyStoppingConfig(enabled=True, metric="accuracy", patience=10)
    assert (cfg.metric, cfg.direction) == ("accuracy", "maximize")
    record = _run(SAME_ACCURACY,
                  early_stopping={"enabled": True, "metric": "macro_f1"})["early_stopping"]
    assert record["metric"] == "macro_f1" and record["direction"] == "maximize"


class Overfitting(Algorithm):
    """Predicts more classes right each round, and is more wrong about the rest.

    The shape a real run takes when it overfits: accuracy climbs while the
    errors become confident, so validation loss rises. Observed for real on a
    Local run -- accuracy 0.417 -> 0.583 while loss went 1.246 -> 2.508.
    ``script`` is ``(n_correct, p_true_when_right, p_true_when_wrong)`` per round.
    """

    def __init__(self, script):
        self.script = script

    def init_globals(self):
        return 0

    def local_train(self, client, shared):
        return None

    def aggregate(self, uploads, shared):
        return self.round_idx

    def predict(self, client, x, shared) -> Predictions:
        k, right, wrong = self.script[int(shared)]
        n = x.shape[0]
        truth = torch.arange(n) % NUM_CLASSES
        p = torch.where(torch.arange(n) < k, right, wrong)      # p assigned to the truth
        probs = torch.zeros(n, NUM_CLASSES)
        probs[torch.arange(n), truth] = p
        probs[torch.arange(n), 1 - truth] = 1 - p
        return Predictions.from_probabilities(probs)


def test_the_two_control_metrics_stop_the_same_run_in_different_places():
    """The reason neither can be a default, as a test rather than an assertion.

    Accuracy improves while loss worsens. Stopping on loss halts the run;
    stopping on accuracy lets it continue. Neither reading is wrong -- they
    optimize different outcomes -- so neither can stand in for the other.
    """
    script = [(4, 0.60, 0.40),        # 4/8 right, mildly confident either way
              (5, 0.99, 0.001),       # 5/8 right, and badly wrong about the rest
              (6, 0.99, 0.001),
              (6, 0.99, 0.001)]

    def run(**kw):
        torch.manual_seed(0)
        return iterative(Overfitting(script), [_scripted_client(n=8)],
                             num_rounds=len(script), device=DEVICE,
                             num_classes=NUM_CLASSES, verbose=False, **kw)

    curve = run()["evaluation_history"]["clients"]["0"]["validation"]
    assert curve["accuracy"] == [0.5, 0.625, 0.75, 0.75]        # improving
    assert curve["loss"][1] > curve["loss"][0]                  # ...and worsening

    on_acc = run(early_stopping={"enabled": True, "metric": "accuracy", "patience": 2})
    on_loss = run(early_stopping={"enabled": True, "metric": "loss", "patience": 2})
    assert on_acc["early_stopping"]["termination_reason"] == "completed_all_rounds"
    assert on_acc["early_stopping"]["best_round"] == 2
    assert on_loss["early_stopping"]["termination_reason"] == "early_stopping"
    assert on_loss["early_stopping"]["best_round"] == 0
    assert on_loss["early_stopping"]["stopped_at_round"] < len(script) - 1


def test_disabled_early_stopping_records_no_active_metric():
    record = _run([0.6, 0.9], early_stopping={"enabled": False, "patience": 3})["early_stopping"]
    assert record["enabled"] is False
    assert record["metric"] is None and record["direction"] is None
    assert record["patience"] is None and record["best_round"] is None


# ── 16-17: what stopping may and may not see ─────────────────────────────────

def test_changing_only_test_loss_cannot_change_the_stopping_round():
    a = _run(SAME_ACCURACY, test=[0.99, 0.10, 0.99],
             early_stopping={"enabled": True, "metric": "loss", "patience": 1})
    b = _run(SAME_ACCURACY, test=[0.10, 0.99, 0.10],
             early_stopping={"enabled": True, "metric": "loss", "patience": 1})

    for key in ("metric", "best_round", "best_value", "stopped_at_round",
                "termination_reason"):
        assert a["early_stopping"][key] == b["early_stopping"][key], key
    # the test curves really were different -- opposite at every round, so the
    # test-best round is round 1 in one run and rounds 0/2 in the other
    ta = a["evaluation_history"]["clients"]["0"]["test"]["loss"]
    tb = b["evaluation_history"]["clients"]["0"]["test"]["loss"]
    assert all(x != y for x, y in zip(ta, tb))
    assert ta.index(min(ta)) != tb.index(min(tb))


# ── 18: round selection on loss ──────────────────────────────────────────────

def test_round_selection_on_loss_minimizes_in_both_views():
    hist = _run(SAME_ACCURACY)["evaluation_history"]

    g = select_global(hist, "loss")
    assert g["selection_direction"] == "minimize"
    assert g["selected_round"] == 1                       # the lowest loss, not the highest
    assert g["validation"]["loss"][0] == pytest.approx(-math.log(0.90), abs=1e-5)

    p = select_per_client(hist, "loss")
    assert p["selection_direction"] == "minimize"
    assert set(p["selected_rounds"].values()) == {1}

    # ...and accuracy, being tied at every round, falls to the tie-break
    assert select_global(hist, "accuracy")["selected_round"] == 0
    assert select_global(hist, "accuracy", tie_break="latest")["selected_round"] == 2


def test_selecting_on_loss_reads_test_from_the_validation_chosen_round():
    hist = _run(SAME_ACCURACY, test=[0.10, 0.20, 0.99])["evaluation_history"]
    g = select_global(hist, "loss")
    assert g["selected_round"] == 1                       # chosen on validation
    # round 2 has the best test loss and is not the one reported
    assert g["test"]["loss"][0] == pytest.approx(-math.log(0.20), abs=1e-5)
