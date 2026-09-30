"""Validation loss as the early-stopping metric."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.core import Client, ClientModel, LearnedProjection, iterative
from rigfl.core.interfaces import Algorithm
from rigfl.eval.selection import SelectionError, select_shared
from rigfl.experiment.config import EarlyStoppingConfig
from rigfl.prediction import Predictions

DEVICE = torch.device("cpu")
NUM_CLASSES = 2
VAL_TAG, TEST_TAG = 0.0, 1.0


# An algorithm whose confidence is scripted round by round

def _tagged_loader(tag: float, n: int = 4) -> DataLoader:
    """Inputs tagged with their split in the first feature."""
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
    """Always predicts the true class, with a scripted per-round confidence."""

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


# Same accuracy, different loss

SAME_ACCURACY = [0.60, 0.90, 0.70]         # correct every round; confidence differs


def test_loss_separates_rounds_that_accuracy_cannot():
    hist = _run(SAME_ACCURACY)["evaluation_history"]
    client = hist["clients"]["0"]
    assert client["validation"]["accuracy"] == [1.0, 1.0, 1.0]        # indistinguishable
    losses = client["validation"]["loss"]
    assert losses == pytest.approx([0.5108, 0.1054, 0.3567], abs=1e-3)
    assert len(set(losses)) == 3                                      # all distinct

    on_loss = _run(SAME_ACCURACY,
                   early_stopping=EarlyStoppingConfig(enabled=True, metric="loss", patience=5))
    on_acc = _run(SAME_ACCURACY,
                  early_stopping=EarlyStoppingConfig(enabled=True, metric="accuracy", patience=5))
    assert on_loss["early_stopping"]["metric"] == "loss"
    assert on_loss["early_stopping"]["best_round"] == 1               # the best model
    assert on_acc["early_stopping"]["best_round"] == 0                # a tie-break
    assert on_loss["early_stopping"]["best_value"] == pytest.approx(0.1054, abs=1e-3)


# Early-stopping defaults

def test_enabling_early_stopping_defaults_to_validation_loss():
    cfg = EarlyStoppingConfig(enabled=True, patience=10)
    assert cfg.metric == "loss"
    record = _run([0.6, 0.9], early_stopping=cfg)["early_stopping"]
    assert (record["metric"], record["direction"]) == ("loss", "minimize")


# Validation/test separation during stopping

def test_changing_only_test_loss_cannot_change_the_stopping_round():
    a = _run(SAME_ACCURACY, test=[0.99, 0.10, 0.99],
             early_stopping=EarlyStoppingConfig(enabled=True, metric="loss", patience=1))
    b = _run(SAME_ACCURACY, test=[0.10, 0.99, 0.10],
             early_stopping=EarlyStoppingConfig(enabled=True, metric="loss", patience=1))

    for key in ("metric", "best_round", "best_value", "stopped_at_round",
                "termination_reason"):
        assert a["early_stopping"][key] == b["early_stopping"][key], key
    # the test curves really were different -- opposite at every round, so the
    # test-best round is round 1 in one run and rounds 0/2 in the other
    ta = a["evaluation_history"]["clients"]["0"]["test"]["loss"]
    tb = b["evaluation_history"]["clients"]["0"]["test"]["loss"]
    assert all(x != y for x, y in zip(ta, tb))
    assert ta.index(min(ta)) != tb.index(min(tb))


# Loss is not a round-selection metric

def test_selecting_on_loss_is_refused():
    hist = _run(SAME_ACCURACY)["evaluation_history"]
    with pytest.raises(SelectionError, match="only for early stopping"):
        select_shared(hist, "log_loss")
