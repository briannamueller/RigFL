"""Metrics on hand-constructed preds/labels with known answers."""

from __future__ import annotations

import pytest
import torch

from rigfl.eval.metrics import (
    accuracy,
    auroc,
    average_precision,
    balanced_accuracy,
    require_task_metric,
)

# accuracy/balanced_accuracy reduce float32 tensors, so compare with a tolerance.
approx = pytest.approx


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


def test_binary_auroc_and_average_precision_treat_label_1_as_positive():
    labels = torch.tensor([0, 0, 1, 1])
    positive_scores = torch.tensor([0.10, 0.40, 0.35, 0.80])
    probabilities = torch.stack([1 - positive_scores, positive_scores], dim=1)

    assert auroc(probabilities, labels, 2) == approx(0.75)
    assert average_precision(probabilities, labels, 2) == approx(5 / 6)


def test_ranking_metrics_are_unavailable_without_required_outcomes():
    probabilities = torch.tensor([[0.8, 0.2], [0.7, 0.3]])
    labels = torch.tensor([0, 0])
    assert auroc(probabilities, labels, 2) is None
    assert average_precision(probabilities, labels, 2) is None

    with pytest.raises(ValueError, match="binary"):
        require_task_metric("auroc", 3)


def test_tied_ranking_scores_receive_threshold_level_credit():
    probabilities = torch.full((4, 2), 0.5)
    labels = torch.tensor([0, 1, 0, 1])

    assert auroc(probabilities, labels, 2) == approx(0.5)
    assert average_precision(probabilities, labels, 2) == approx(0.5)
