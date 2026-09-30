"""Prediction contract: labels, probabilities, and log loss."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.core import Client, ClientModel, LearnedProjection, iterative
from rigfl.eval.metrics import (
    LOG_LOSS_EPS,
    log_loss,
    macro_f1,
)
from rigfl.prediction import PredictionError, Predictions, check_predictions

NUM_CLASSES = 3
INPUT_DIM = 8
SHARED_DIM = 4
DEVICE = torch.device("cpu")

CENTERS = torch.randn(NUM_CLASSES, INPUT_DIM,
                      generator=torch.Generator().manual_seed(12345)) * 3.0


def _loader(n_per_class: int = 12, spread: float = 1.5, batch: int = 16) -> DataLoader:
    xs, ys = [], []
    for c in range(NUM_CLASSES):
        xs.append(CENTERS[c] + spread * torch.randn(n_per_class, INPUT_DIM))
        ys.append(torch.full((n_per_class,), c))
    return DataLoader(TensorDataset(torch.cat(xs), torch.cat(ys)),
                      batch_size=batch, shuffle=True)


def _client(hidden: int) -> Client:
    model = ClientModel(
        nn.Sequential(nn.Linear(INPUT_DIM, hidden), nn.ReLU()),
        LearnedProjection(hidden, SHARED_DIM),
        nn.Linear(SHARED_DIM, NUM_CLASSES),
    )
    return Client(model=model, train_loader=_loader(),
                  val_loader=_loader(6), test_loader=_loader(6))


def _clients():
    return [_client(h) for h in (8, 12)]


# Metric behavior

def test_log_loss_matches_a_hand_computed_example():
    probs = torch.tensor([[0.7, 0.2, 0.1],
                          [0.1, 0.5, 0.4]])
    y = torch.tensor([0, 2])
    expected = (-math.log(0.7) - math.log(0.4)) / 2
    assert log_loss(probs, y, 3) == pytest.approx(expected, abs=1e-6)


def test_macro_f1_penalizes_a_class_predicted_but_absent_from_labels():
    labels = torch.tensor([0, 0])
    predictions = torch.tensor([0, 1])
    # class 0 F1 = 2/3; predicted-only class 1 F1 = 0
    assert macro_f1(predictions, labels, 2) == pytest.approx(1 / 3)


def test_zero_probability_on_the_true_class_is_clamped_not_infinite():
    """Zero probability on the true class gives a finite, clamped loss."""
    probs = torch.tensor([[1.0, 0.0, 0.0]])
    value = log_loss(probs, torch.tensor([1]), 3)
    assert math.isfinite(value)
    assert value == pytest.approx(-math.log(LOG_LOSS_EPS), rel=1e-6)


# Binary logits

def test_a_one_logit_binary_head_is_normalized_to_two_columns():
    logits = torch.tensor([[0.0], [2.0], [-2.0]])
    out = Predictions.from_logits(logits)
    p = torch.sigmoid(logits.squeeze(1))
    assert out.probabilities.shape == (3, 2)
    assert torch.allclose(out.probabilities[:, 1], p)
    assert torch.allclose(out.probabilities[:, 0], 1 - p)
    assert torch.allclose(out.probabilities.sum(1), torch.ones(3))
    assert out.labels.tolist() == [0, 1, 0]      # 0.5 ties to column 0; >0.5 -> 1

    flat = Predictions.from_logits(torch.tensor([3.0, -3.0]))
    assert flat.labels.tolist() == [1, 0]


# Invalid predictions

@pytest.mark.parametrize("probs, match", [
    (torch.tensor([[-0.5, 1.5]]), "negative values"),
    (torch.tensor([[0.3, 0.3]]), "must sum to 1"),
])
def test_invalid_probabilities_are_refused(probs, match):
    with pytest.raises(PredictionError, match=match):
        Predictions.from_probabilities(probs)


# Built-in algorithms

def _built_in_algorithms():
    """One instance of each prediction-style algorithm under test, with the pieces each needs."""
    from rigfl.algorithms.fedgh import FedGH, FedGHConfig
    from rigfl.algorithms.fedproto import FedProto, FedProtoConfig
    from rigfl.algorithms.fedtgp import FedTGP, FedTGPConfig
    from rigfl.algorithms.fml import FML, FMLConfig
    from rigfl.algorithms.global_ensemble import GlobalEnsemble, GlobalEnsembleConfig
    from rigfl.algorithms.local import Local, LocalConfig

    def aux():
        b = nn.Sequential(nn.Linear(INPUT_DIM, 8), nn.ReLU())
        b.out_dim = 8
        return ClientModel(b, LearnedProjection(8, SHARED_DIM),
                           nn.Linear(SHARED_DIM, NUM_CLASSES))

    return {
        "local": Local(LocalConfig(local_epochs=1, lr=0.05)),
        "global_ensemble": GlobalEnsemble(
            GlobalEnsembleConfig(local_epochs=1, lr=0.05)),
        "fedproto": FedProto(
            FedProtoConfig(lamda=1.0, local_epochs=1, lr=0.05)),
        "fedgh": FedGH(
            FedGHConfig(local_epochs=1, lr=0.05), SHARED_DIM, NUM_CLASSES),
        "fml": FML(FMLConfig(local_epochs=1, lr=0.05), aux),
        "fedtgp": FedTGP(
            FedTGPConfig(
                lamda=1.0, local_epochs=1, lr=0.05, server_epochs=1
            ),
            NUM_CLASSES, SHARED_DIM),
    }


def _predictions(algorithm, rounds: int = 2):
    """Run the algorithm for real, then collect one batch of predictions per client."""
    torch.manual_seed(0)
    clients = _clients()
    algorithm.device = DEVICE
    algorithm.total_rounds = rounds
    for cid, client in enumerate(clients):
        client.client_id = cid
        client.state.clear()
    shared = algorithm.init_globals()
    for rnd in range(rounds):
        algorithm.round_idx = rnd
        uploads = [algorithm.local_train(client, shared) for client in clients]
        shared = algorithm.aggregate(uploads, shared)

    outs = []
    with torch.no_grad():
        for cid, c in enumerate(clients):
            for m in ([c.model] + ([shared] if isinstance(shared, nn.Module) else [])):
                m.eval()
            x, _ = next(iter(c.test_loader))
            outs.append((c.model, x, shared, c.state,
                         check_predictions(algorithm.predict(c, x, shared))))
    return outs


@pytest.mark.parametrize("name", sorted(_built_in_algorithms()))
def test_every_built_in_algorithm_preserves_the_prediction_contract(name):
    import torch.nn.functional as F

    for model, x, shared, state, out in _predictions(
        _built_in_algorithms()[name]
    ):
        assert out.probabilities is not None, f"{name} returned no probabilities"
        p = out.probabilities
        assert p.shape == (len(out.labels), NUM_CLASSES)
        assert torch.isfinite(p).all()
        assert (p >= 0).all()
        assert torch.allclose(p.sum(1), torch.ones(p.shape[0]), atol=1e-5)
        assert torch.equal(out.labels, out.probabilities.argmax(dim=1)), name
        if name in ("local", "fml", "fedgh"):
            old = model(x).argmax(dim=1)
        elif name == "global_ensemble":
            old = (sum(F.softmax(m(x), dim=1) for m in shared) / len(shared)).argmax(dim=1)
        else:                                   # fedproto / fedtgp: nearest prototype
            protos_map = shared["protos"] if name == "fedtgp" else shared
            classes = sorted(protos_map)
            protos = torch.stack([protos_map[c] for c in classes])
            nearest = torch.cdist(model.rep(x), protos).argmin(dim=1).tolist()
            old = torch.tensor([classes[i] for i in nearest])
        assert torch.equal(out.labels, old), f"{name} hard labels changed"


# Prototype algorithms

def test_prototype_probabilities_follow_the_euclidean_distances():
    from rigfl.algorithms.fedproto import prototype_prediction

    rep = torch.tensor([[0.0, 0.0]])
    protos = {0: torch.tensor([1.0, 0.0]),     # d = 1
              1: torch.tensor([3.0, 0.0]),     # d = 3
              2: torch.tensor([2.0, 0.0])}     # d = 2
    out = prototype_prediction(rep, protos, 3)
    expected = torch.softmax(torch.tensor([[-1.0, -3.0, -2.0]]), dim=1)
    assert torch.allclose(out.probabilities, expected, atol=1e-6)
    assert out.labels.tolist() == [0]                       # nearest prototype
    # ordering of probabilities is the reverse ordering of distances
    assert out.probabilities[0, 0] > out.probabilities[0, 2] > out.probabilities[0, 1]

    # a class with no global prototype gets zero probability
    missing = prototype_prediction(
        torch.tensor([[0.5, 0.5]]),
        {0: torch.tensor([1.0, 0.0]), 2: torch.tensor([0.0, 1.0])}, 3)
    assert missing.probabilities[0, 1].item() == 0.0


# Label-only algorithms

class LabelOnlyAlgorithm:
    """A minimal algorithm that intentionally returns only class IDs."""

    def init_globals(self):
        return None

    def local_train(self, client, shared):
        return None

    def aggregate(self, uploads, shared):
        return None

    def predict(self, client, x, shared):
        return Predictions.labels_only(client.model(x).argmax(dim=1))


def test_a_label_only_algorithm_still_gets_hard_label_metrics():
    torch.manual_seed(0)
    result = iterative(LabelOnlyAlgorithm(), _clients(), num_rounds=2, device=DEVICE,
                           num_classes=NUM_CLASSES, verbose=False)
    hist = result["evaluation_history"]
    for client in hist["clients"].values():
        for hard in ("accuracy", "balanced_accuracy", "macro_f1"):
            assert all(v is not None for v in client["validation"][hard])
        assert all(v is None for v in client["validation"]["loss"])


def test_batched_predictions_and_labels_share_loader_order():
    from rigfl.eval.protocol import evaluate_split

    dataset = TensorDataset(torch.arange(12).float().unsqueeze(1),
                            torch.arange(12) % NUM_CLASSES)
    reversed_loader = DataLoader(dataset, batch_size=4,
                                 sampler=list(reversed(range(len(dataset)))))

    class BatchAlgorithm:
        def predict(self, client, x, shared):
            labels = x.squeeze(1).long() % NUM_CLASSES
            return Predictions.from_probabilities(
                torch.nn.functional.one_hot(labels, NUM_CLASSES).float())

    client = Client(nn.Linear(1, NUM_CLASSES), reversed_loader,
                    reversed_loader, reversed_loader)
    result = evaluate_split(BatchAlgorithm(), [client], None, DEVICE,
                            "val", NUM_CLASSES)
    assert result["clients"]["0"]["accuracy"] == 1.0
