"""Metrics on hand-constructed preds/labels with known answers."""

from __future__ import annotations

import pytest
import torch

from rigfl.eval.metrics import accuracy, balanced_accuracy

# accuracy/balanced_accuracy reduce float32 tensors, so compare with a tolerance.
approx = pytest.approx


def test_accuracy_all_correct():
    preds = torch.tensor([0, 1, 2, 1])
    labels = torch.tensor([0, 1, 2, 1])
    assert accuracy(preds, labels) == approx(1.0)


def test_accuracy_half_correct():
    preds = torch.tensor([0, 0, 1, 1])
    labels = torch.tensor([0, 1, 0, 1])
    assert accuracy(preds, labels) == approx(0.5)


def test_accuracy_all_wrong():
    preds = torch.tensor([1, 1, 1])
    labels = torch.tensor([0, 0, 0])
    assert accuracy(preds, labels) == approx(0.0)


def test_balanced_accuracy_differs_under_imbalance():
    # 4 samples of class 0, 1 sample of class 1; predict everything class 0.
    # plain accuracy = 4/5 = 0.8, but balanced (macro recall) = (1.0 + 0.0)/2 = 0.5.
    labels = torch.tensor([0, 0, 0, 0, 1])
    preds = torch.zeros(5, dtype=torch.long)
    assert accuracy(preds, labels) == approx(0.8)
    assert balanced_accuracy(preds, labels, num_classes=2) == approx(0.5)


def test_balanced_accuracy_equals_accuracy_when_balanced_and_symmetric():
    # Balanced classes, symmetric per-class recall -> the two metrics coincide.
    labels = torch.tensor([0, 0, 1, 1])
    preds = torch.tensor([0, 0, 1, 1])
    assert balanced_accuracy(preds, labels, num_classes=2) == approx(1.0)
    assert accuracy(preds, labels) == approx(1.0)


def test_balanced_accuracy_perfect():
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    preds = labels.clone()
    assert balanced_accuracy(preds, labels, num_classes=3) == approx(1.0)


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
