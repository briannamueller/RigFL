"""Balanced class weighting for supervised training losses."""

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rigfl.core.interfaces import Algorithm
from rigfl.core.losses import (
    apply_class_weighting,
    classification_loss,
    distribution_classification_loss,
)
from rigfl.core.round import Client
from rigfl.eval.protocol import evaluate_split
from rigfl.experiment.config import ExperimentConfig
from rigfl.prediction import Predictions


def _client(labels, seed):
    labels = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(torch.arange(len(labels)).float().unsqueeze(1), labels)
    generator = torch.Generator().manual_seed(seed)
    return Client(
        model=None,
        train_loader=DataLoader(
            dataset, batch_size=2, shuffle=True, generator=generator
        ),
    )


def test_global_balanced_weights_use_pooled_labels_without_advancing_loaders():
    clients = [_client([0, 0], 1), _client([0, 1], 2)]
    states = [client.train_loader.generator.get_state() for client in clients]

    apply_class_weighting(clients, "global_balanced", num_classes=2)

    for client, state in zip(clients, states):
        assert client.class_weights.tolist() == pytest.approx([2 / 3, 2.0])
        assert torch.equal(client.train_loader.generator.get_state(), state)


def test_local_balanced_weights_use_each_clients_training_labels():
    clients = [_client([0, 0, 1], 1), _client([0, 1, 1, 1], 2)]

    apply_class_weighting(clients, "local_balanced", num_classes=2)

    assert clients[0].class_weights.tolist() == pytest.approx([0.75, 1.5])
    assert clients[1].class_weights.tolist() == pytest.approx([2.0, 2 / 3])


def test_classification_losses_match_pytorch_weighted_reductions():
    logits = torch.tensor([[2.0, -1.0], [0.0, 1.0], [1.0, 0.0]])
    labels = torch.tensor([0, 1, 1])
    weights = torch.tensor([0.6, 3.0])

    expected = F.cross_entropy(logits, labels, weight=weights)
    assert classification_loss(logits, labels, weights) == pytest.approx(expected)
    assert distribution_classification_loss(
        logits.softmax(dim=1), labels, weights
    ) == pytest.approx(expected)


@pytest.mark.parametrize("mode", ["global_balanced", "local_balanced"])
def test_balanced_weighting_rejects_a_missing_training_class(mode):
    with pytest.raises(ValueError, match="missing classes: \\[1\\]"):
        apply_class_weighting([_client([0, 0], 1)], mode, num_classes=2)


def test_experiment_config_exposes_balanced_class_weighting():
    assert ExperimentConfig().class_weighting == "none"
    for mode in ("global_balanced", "local_balanced"):
        assert ExperimentConfig(class_weighting=mode).class_weighting == mode


class _ProbabilityPredictor(Algorithm):
    def predict(self, client, x, shared):
        return Predictions.from_probabilities(x)


def test_evaluation_loss_uses_training_class_weights():
    probabilities = torch.tensor([[0.9, 0.1], [0.6, 0.4], [0.4, 0.6]])
    labels = torch.tensor([0, 0, 1])
    loader = DataLoader(TensorDataset(probabilities, labels), batch_size=3)
    client = Client(model=None, train_loader=loader, val_loader=loader)
    client.class_weights = torch.tensor([0.75, 1.5])

    evaluated = evaluate_split(
        _ProbabilityPredictor(None), [client], None, torch.device("cpu"), "val", 2
    )
    expected = F.nll_loss(
        probabilities.log(), labels, weight=client.class_weights
    ).item()

    assert evaluated["clients"]["0"]["loss"] == pytest.approx(expected)
