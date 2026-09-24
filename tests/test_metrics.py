"""Metrics on hand-constructed preds/labels with known answers."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from rigfl.core import Client
from rigfl.eval.metrics import (
    accuracy,
    auroc,
    average_precision,
    balanced_accuracy,
)
from rigfl.eval.protocol import evaluate_split
from rigfl.prediction import Predictions
from tests.helpers import resolved_experiment

# accuracy/balanced_accuracy reduce float32 tensors, so compare with a tolerance.
approx = pytest.approx


def test_accuracy_matches_a_hand_computed_example():
    preds = torch.tensor([0, 0, 1, 1])
    labels = torch.tensor([0, 1, 0, 1])
    assert accuracy(preds, labels) == approx(0.5)


def test_balanced_accuracy_differs_under_imbalance():
    # 4 samples of class 0, 1 sample of class 1; predict everything class 0.
    # plain accuracy = 4/5 = 0.8, but balanced (macro recall) = (1.0 + 0.0)/2 = 0.5.
    labels = torch.tensor([0, 0, 0, 0, 1])
    preds = torch.zeros(5, dtype=torch.long)
    assert accuracy(preds, labels) == approx(0.8)
    assert balanced_accuracy(preds, labels, num_classes=2) == approx(0.5)


def test_balanced_accuracy_skips_absent_classes():
    # num_classes=3 but only classes 0 and 1 appear in labels: the absent class
    # is skipped, so the average is over the two present classes only.
    labels = torch.tensor([0, 0, 1, 1])
    preds = torch.tensor([0, 1, 1, 1])  # class0 recall 0.5, class1 recall 1.0
    assert balanced_accuracy(preds, labels, num_classes=3) == approx(0.75)


def test_balanced_accuracy_empty_is_zero():
    # No labels at all -> defined as 0.0 (no per-class recalls to average).
    preds = torch.empty(0, dtype=torch.long)
    labels = torch.empty(0, dtype=torch.long)
    assert balanced_accuracy(preds, labels, num_classes=2) == approx(0.0)


def test_binary_auroc_and_average_precision_use_the_configured_positive_class():
    labels = torch.tensor([0, 0, 1, 1])
    positive_scores = torch.tensor([0.10, 0.40, 0.35, 0.80])
    probabilities = torch.stack([1 - positive_scores, positive_scores], dim=1)

    assert auroc(probabilities, labels, 2, positive_class=1) == approx(0.75)
    assert average_precision(
        probabilities, labels, 2, positive_class=1
    ) == approx(5 / 6)
    assert auroc(probabilities, labels, 2, positive_class=0) == approx(0.75)
    assert average_precision(
        probabilities, labels, 2, positive_class=0
    ) == approx(5 / 6)


def test_multiclass_ranking_metrics_are_one_vs_rest_macro_averages():
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    probabilities = torch.tensor([
        [0.90, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
        [0.80, 0.10, 0.10],
        [0.10, 0.80, 0.10],
        [0.10, 0.10, 0.80],
    ])

    assert auroc(probabilities, labels, 3) == approx(1.0)
    assert average_precision(probabilities, labels, 3) == approx(1.0)


def test_ranking_metrics_are_unavailable_without_required_outcomes():
    binary_probabilities = torch.tensor([[0.8, 0.2], [0.7, 0.3]])
    binary_labels = torch.tensor([0, 0])
    assert auroc(
        binary_probabilities, binary_labels, 2, positive_class=1
    ) is None
    assert average_precision(
        binary_probabilities, binary_labels, 2, positive_class=1
    ) is None

    multiclass_probabilities = torch.tensor([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
    ])
    multiclass_labels = torch.tensor([0, 1])
    assert auroc(multiclass_probabilities, multiclass_labels, 3) is None
    assert average_precision(multiclass_probabilities, multiclass_labels, 3) is None


def test_tied_ranking_scores_receive_threshold_level_credit():
    probabilities = torch.full((4, 2), 0.5)
    labels = torch.tensor([0, 1, 0, 1])

    assert auroc(probabilities, labels, 2, positive_class=1) == approx(0.5)
    assert average_precision(
        probabilities, labels, 2, positive_class=1
    ) == approx(0.5)


def test_run_level_ranking_metrics_pool_predictions_instead_of_client_scores():
    class ProbabilityAlgorithm:
        def predict(self, client, inputs, shared):
            return Predictions.from_probabilities(inputs)

    def client(probabilities, labels):
        loader = DataLoader(
            TensorDataset(torch.tensor(probabilities), torch.tensor(labels)),
            batch_size=2,
        )
        return Client(
            model=None,
            train_loader=loader,
            val_loader=loader,
            test_loader=loader,
        )

    evaluated = evaluate_split(
        ProbabilityAlgorithm(),
        [
            client([[0.9, 0.1], [0.8, 0.2]], [0, 0]),
            client([[0.7, 0.3], [0.1, 0.9]], [1, 1]),
        ],
        shared=None,
        device=torch.device("cpu"),
        split="test",
        num_classes=2,
        positive_class=1,
    )

    assert evaluated["clients"]["0"]["auroc"] is None
    assert evaluated["clients"]["1"]["auroc"] is None
    assert evaluated["clients"]["0"]["auprc"] is None
    assert evaluated["clients"]["1"]["auprc"] is None
    assert evaluated["aggregate"]["auroc"] == approx(1.0)
    assert evaluated["aggregate"]["auprc"] == approx(1.0)


def test_positive_class_is_valid_only_for_a_binary_target():
    assert resolved_experiment(num_classes=2, positive_class=1).positive_class == 1
    with pytest.raises(ValueError, match="only valid for binary"):
        resolved_experiment(num_classes=3, positive_class=1)
    with pytest.raises(ValueError, match=r"must be in \[0, 2\)"):
        resolved_experiment(num_classes=2, positive_class=2)
