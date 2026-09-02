"""The prediction contract: labels, probabilities, and the log loss they enable.

Hard-label metrics cannot tell a confident model from a lucky one -- two algorithms
predicting the identical classes score identically on accuracy, balanced accuracy
and macro F1 however differently calibrated they are. Log loss can, which is why
the evaluation boundary carries the distribution and not just the argmax.

These tests cover the contract itself, the loss definition, and then every
built-in algorithm: that it produces a real normalized distribution, and that
gaining one did not move a single hard label.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.core import (Client, ClientModel, LearnedProjection, iterative,
                        p2p_one_shot)
from rigfl.eval.metrics import (LOG_LOSS_EPS, MetricInputUnavailable, compute_all,
                                log_loss, macro_f1, metric_input, register, spec,
                                unregister)
from rigfl.prediction import PredictionError, Predictions, as_predictions

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


# ── 1-4: what log loss is ────────────────────────────────────────────────────

def test_identical_labels_with_different_confidence_give_different_loss():
    """The property the whole change exists for."""
    y = torch.tensor([0, 1, 0])
    confident = torch.tensor([[0.98, 0.01, 0.01], [0.01, 0.98, 0.01], [0.98, 0.01, 0.01]])
    hesitant = torch.tensor([[0.40, 0.35, 0.25], [0.30, 0.40, 0.30], [0.40, 0.35, 0.25]])

    a = compute_all(Predictions.from_probabilities(confident), y, NUM_CLASSES)
    b = compute_all(Predictions.from_probabilities(hesitant), y, NUM_CLASSES)

    assert torch.equal(confident.argmax(1), hesitant.argmax(1))     # same predictions
    for hard in ("accuracy", "balanced_accuracy", "macro_f1"):
        assert a[hard] == b[hard]                                   # hard metrics cannot tell
    assert a["loss"] < b["loss"]                                    # loss can


def test_log_loss_matches_a_hand_computed_example():
    probs = torch.tensor([[0.7, 0.2, 0.1],
                          [0.1, 0.5, 0.4]])
    y = torch.tensor([0, 2])
    expected = (-math.log(0.7) - math.log(0.4)) / 2
    assert log_loss(probs, y, 3) == pytest.approx(expected, abs=1e-6)


def test_a_uniform_prediction_scores_log_c():
    probs = torch.full((5, 4), 0.25)
    assert log_loss(probs, torch.tensor([0, 1, 2, 3, 0]), 4) == pytest.approx(math.log(4),
                                                                              abs=1e-6)


def test_perfect_confident_predictions_have_low_loss():
    probs = torch.tensor([[0.999, 0.0005, 0.0005], [0.0005, 0.999, 0.0005]])
    assert log_loss(probs, torch.tensor([0, 1]), 3) < 0.002


def test_macro_f1_penalizes_a_class_predicted_but_absent_from_labels():
    labels = torch.tensor([0, 0])
    predictions = torch.tensor([0, 1])
    # class 0 F1 = 2/3; predicted-only class 1 F1 = 0
    assert macro_f1(predictions, labels, 2) == pytest.approx(1 / 3)


def test_confidently_wrong_predictions_have_high_loss():
    probs = torch.tensor([[0.999, 0.0005, 0.0005], [0.0005, 0.999, 0.0005]])
    right = log_loss(probs, torch.tensor([0, 1]), 3)
    wrong = log_loss(probs, torch.tensor([1, 0]), 3)
    assert wrong > 7.0 > right
    # ...and being wrong while uncertain costs far less than being wrong loudly
    hedged = log_loss(torch.full((2, 3), 1 / 3), torch.tensor([1, 0]), 3)
    assert hedged < wrong


def test_zero_probability_on_the_true_class_is_clamped_not_infinite():
    """A hard-vote ensemble routinely puts zero mass on a class."""
    probs = torch.tensor([[1.0, 0.0, 0.0]])
    value = log_loss(probs, torch.tensor([1]), 3)
    assert math.isfinite(value)
    assert value == pytest.approx(-math.log(LOG_LOSS_EPS), rel=1e-6)


# ── 5: binary one-logit heads ────────────────────────────────────────────────

def test_a_one_logit_binary_head_is_normalized_to_two_columns():
    logits = torch.tensor([[0.0], [2.0], [-2.0]])
    out = Predictions.from_logits(logits)
    p = torch.sigmoid(logits.squeeze(1))
    assert out.probabilities.shape == (3, 2)
    assert torch.allclose(out.probabilities[:, 1], p)
    assert torch.allclose(out.probabilities[:, 0], 1 - p)
    assert torch.allclose(out.probabilities.sum(1), torch.ones(3))
    assert out.labels.tolist() == [0, 1, 0]      # 0.5 ties to column 0; >0.5 -> 1


def test_a_flat_binary_logit_vector_is_accepted():
    out = Predictions.from_logits(torch.tensor([3.0, -3.0]))
    assert out.probabilities.shape == (2, 2)
    assert out.labels.tolist() == [1, 0]


# ── 6: invalid probabilities fail clearly ────────────────────────────────────

@pytest.mark.parametrize("probs, match", [
    (torch.rand(4), "must be \\[N, C\\]"),
    (torch.rand(2, 2, 2), "must be \\[N, C\\]"),
    (torch.tensor([[float("nan"), 1.0]]), "NaN or inf"),
    (torch.tensor([[float("inf"), 0.0]]), "NaN or inf"),
    (torch.tensor([[-0.5, 1.5]]), "negative values"),
    (torch.tensor([[0.3, 0.3]]), "must sum to 1"),
    (torch.tensor([[2.0, 3.0]]), "must sum to 1"),
])
def test_invalid_probabilities_are_refused(probs, match):
    with pytest.raises(PredictionError, match=match):
        Predictions.from_probabilities(probs)


def test_probabilities_must_have_one_row_per_label():
    with pytest.raises(PredictionError, match="2 rows for 3 label"):
        Predictions(labels=torch.tensor([0, 1, 2]),
                         probabilities=torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


def test_labels_must_be_one_dimensional():
    with pytest.raises(PredictionError, match="labels must be 1-D"):
        Predictions(labels=torch.zeros(2, 2, dtype=torch.long), probabilities=None)


def test_log_loss_refuses_a_wrong_width_distribution():
    with pytest.raises(PredictionError, match="2 columns for 3 classes"):
        log_loss(torch.tensor([[0.5, 0.5]]), torch.tensor([0]), 3)


# ── 7-9: every built-in algorithm ───────────────────────────────────────────────

def _built_in_algorithms():
    """One instance of every shipped algorithm, with the pieces each needs."""
    from rigfl.algorithms.fedgh import FedGH, FedGHConfig
    from rigfl.algorithms.fedkd import FedKD, FedKDConfig
    from rigfl.algorithms.fedproto import FedProto, FedProtoConfig
    from rigfl.algorithms.fedtgp import FedTGP, FedTGPConfig
    from rigfl.algorithms.fml import FML, FMLConfig
    from rigfl.algorithms.global_ensemble import (GlobalEnsemble,
                                                  GlobalEnsembleConfig)
    from rigfl.algorithms.lgfedavg import LGFedAvg, LGFedAvgConfig
    from rigfl.algorithms.local import Local, LocalConfig

    def aux():
        b = nn.Sequential(nn.Linear(INPUT_DIM, 8), nn.ReLU())
        b.out_dim = 8
        return ClientModel(b, LearnedProjection(8, SHARED_DIM),
                           nn.Linear(SHARED_DIM, NUM_CLASSES))

    return {
        "local": Local(LocalConfig(local_epochs=1, lr=0.05)),
        "global": GlobalEnsemble(
            GlobalEnsembleConfig(local_epochs=1, lr=0.05)),
        "fedproto": FedProto(
            FedProtoConfig(lamda=1.0, local_epochs=1, lr=0.05)),
        "fedgh": FedGH(
            FedGHConfig(local_epochs=1, lr=0.05), SHARED_DIM, NUM_CLASSES),
        "lgfedavg": LGFedAvg(
            LGFedAvgConfig(local_epochs=1, lr=0.05), SHARED_DIM, NUM_CLASSES),
        "fml": FML(FMLConfig(local_epochs=1, lr=0.05), aux),
        "fedkd": FedKD(
            FedKDConfig(local_epochs=1, lr=0.05), aux, SHARED_DIM),
        "fedtgp": FedTGP(
            FedTGPConfig(lamda=1.0, local_epochs=1, lr=0.05),
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
                         as_predictions(algorithm.predict(c, x, shared))))
    return outs


@pytest.mark.parametrize("name", sorted(_built_in_algorithms()))
def test_every_built_in_algorithm_returns_a_valid_distribution(name):
    for _, x, _, _, out in _predictions(_built_in_algorithms()[name]):
        assert out.probabilities is not None, f"{name} returned no probabilities"
        p = out.probabilities
        assert p.shape == (len(out.labels), NUM_CLASSES)
        assert torch.isfinite(p).all()
        assert (p >= 0).all()
        assert torch.allclose(p.sum(1), torch.ones(p.shape[0]), atol=1e-5)


@pytest.mark.parametrize("name", sorted(_built_in_algorithms()))
def test_every_built_in_algorithm_labels_equal_the_probability_argmax(name):
    for _, _, _, _, out in _predictions(_built_in_algorithms()[name]):
        assert torch.equal(out.labels, out.probabilities.argmax(dim=1)), name


@pytest.mark.parametrize("name", sorted(_built_in_algorithms()))
def test_hard_labels_are_unchanged_by_the_new_interface(name):
    """The labels the old label-only implementations produced, recomputed here.

    Not a re-run of the new code under another name: each expression below is the
    decision rule as it was written before probabilities existed.
    """
    import torch.nn.functional as F

    for model, x, shared, state, out in _predictions(_built_in_algorithms()[name]):
        if name in ("local", "fml", "fedkd"):
            old = model(x).argmax(dim=1)
        elif name in ("fedgh", "lgfedavg"):
            model.head.load_state_dict(shared.state_dict())
            old = model(x).argmax(dim=1)
        elif name == "global":
            old = (sum(F.softmax(m(x), dim=1) for m in shared) / len(shared)).argmax(dim=1)
        else:                                   # fedproto / fedtgp: nearest prototype
            protos_map = shared["protos"] if name == "fedtgp" else shared
            classes = sorted(protos_map)
            protos = torch.stack([protos_map[c] for c in classes])
            nearest = torch.cdist(model.rep(x), protos).argmin(dim=1).tolist()
            old = torch.tensor([classes[i] for i in nearest])
        assert torch.equal(out.labels, old), f"{name} hard labels changed"


# ── 11: prototype algorithms specifically ───────────────────────────────────────

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


def test_prototype_confidence_depends_on_the_representation_scale():
    from rigfl.algorithms.fedproto import prototype_prediction

    torch.manual_seed(0)
    rep = torch.randn(16, SHARED_DIM)
    protos = {c: torch.randn(SHARED_DIM) for c in range(NUM_CLASSES)}
    y = torch.randint(0, NUM_CLASSES, (16,))
    c = 4.0

    base = prototype_prediction(rep, protos, NUM_CLASSES)
    scaled = prototype_prediction(c * rep, {k: c * v for k, v in protos.items()},
                                  NUM_CLASSES)

    assert torch.equal(base.labels, scaled.labels)           # labels never move
    # Scaling distances changes confidence and loss while hard metrics stay fixed.
    a = compute_all(base, y, NUM_CLASSES)
    b = compute_all(scaled, y, NUM_CLASSES)
    for hard in ("accuracy", "balanced_accuracy", "macro_f1"):
        assert a[hard] == b[hard]
    assert a["loss"] != pytest.approx(b["loss"], abs=1e-4)


def test_a_class_with_no_global_prototype_gets_zero_probability():
    from rigfl.algorithms.fedproto import prototype_prediction

    protos = {0: torch.tensor([1.0, 0.0]), 2: torch.tensor([0.0, 1.0])}   # no class 1
    out = prototype_prediction(torch.tensor([[0.5, 0.5]]), protos, 3)
    assert out.probabilities[0, 1].item() == 0.0
    assert out.probabilities.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert out.labels.item() in (0, 2)


# ── 10: FedDES keeps GraphRoute's soft_probs ─────────────────────────────────

def test_feddes_returns_graphroutes_soft_probs():
    """The distribution ``evaluate_ensemble`` already computes, not a re-derivation."""
    pytest.importorskip("graphroute")
    from rigfl.algorithms.feddes import FedDES, FedDESConfig

    algorithm = FedDES(
        FedDESConfig(calibrate=False, cache_dir=""),
        [lambda: nn.Linear(INPUT_DIM, NUM_CLASSES)],
        NUM_CLASSES,
    )     # calibration is not what this tests
    captured = {}
    import graphroute.selection as gsel
    real = gsel.evaluate_ensemble

    def spy(*a, **kw):
        soft, hard = real(*a, **kw)
        captured["soft"], captured["hard"] = soft.clone(), hard.clone()
        return soft, hard

    gsel.evaluate_ensemble = spy
    try:
        torch.manual_seed(0)
        clients = _clients()
        result = p2p_one_shot(algorithm, clients, num_rounds=1, device=DEVICE,
                              num_classes=NUM_CLASSES, verbose=False)
    finally:
        gsel.evaluate_ensemble = real

    assert captured, "evaluate_ensemble was never called"
    assert torch.allclose(captured["soft"].sum(1), torch.ones_like(captured["soft"].sum(1)),
                          atol=1e-4)
    # Preparation is complete before the first evaluated round, so round 0
    # already carries the actual GraphRoute/FedDES distribution.
    hist = result["evaluation_history"]
    assert hist["evaluation_rounds"] == [0]
    i = 0
    losses = [c["validation"]["loss"][i] for c in hist["clients"].values()]
    assert all(v is not None and math.isfinite(v) for v in losses)


# ── 21: label-only algorithms ───────────────────────────────────────────────────

class LabelOnlyAlgorithm:
    """An external algorithm written against the older contract."""

    def init_globals(self):
        return None

    def local_train(self, client, shared):
        return None

    def aggregate(self, uploads, shared):
        return None

    def predict(self, client, x, shared):
        return client.model(x).argmax(dim=1)          # a bare tensor, as before


def test_a_label_only_algorithm_still_gets_hard_label_metrics():
    torch.manual_seed(0)
    result = iterative(LabelOnlyAlgorithm(), _clients(), num_rounds=2, device=DEVICE,
                           num_classes=NUM_CLASSES, verbose=False)
    hist = result["evaluation_history"]
    for client in hist["clients"].values():
        for hard in ("accuracy", "balanced_accuracy", "macro_f1"):
            assert all(v is not None for v in client["validation"][hard])
        assert all(v is None for v in client["validation"]["loss"])


def test_a_label_only_algorithm_runs_with_verbose_reporting():
    """The per-round print reduces every computed metric, loss included.

    An unavailable metric is present in the client's dict with a ``None`` value,
    which is not the same absence as a client having no data -- summing it
    crashed a run that was otherwise perfectly valid.
    """
    torch.manual_seed(0)
    result = iterative(LabelOnlyAlgorithm(), _clients(), num_rounds=2, device=DEVICE,
                           num_classes=NUM_CLASSES, verbose=True)
    assert result["evaluation_history"]["evaluation_rounds"] == [0, 1]


def test_verbose_reporting_prints_the_metrics_that_are_available(capsys):
    torch.manual_seed(0)
    iterative(LabelOnlyAlgorithm(), _clients(), num_rounds=1, device=DEVICE,
                  num_classes=NUM_CLASSES, verbose=True)
    out = capsys.readouterr().out
    assert "accuracy" in out and "macro_f1" in out
    assert "loss" not in out                    # unavailable, so not reported as a number


def test_mean_over_clients_skips_unavailable_values():
    from rigfl.eval.protocol import mean_over_clients

    evaluated = {"clients": {"0": {"accuracy": 0.8, "loss": None},
                             "1": {"accuracy": 0.6, "loss": None},
                             "2": None},                       # no data for this split
                 "sample_counts": {"0": 10, "1": 10, "2": None}}
    assert mean_over_clients(evaluated, "accuracy") == pytest.approx(0.7)
    assert mean_over_clients(evaluated, "loss") is None

    mixed = {"clients": {"0": {"loss": 0.4}, "1": {"loss": None}}}
    assert mean_over_clients(mixed, "loss") == pytest.approx(0.4)


def test_the_tracker_reduction_path_handles_a_label_only_algorithm():
    """WandbTracker.log_round shares mean_over_clients with the verbose print."""
    from rigfl.eval.protocol import evaluate_split
    from rigfl.experiment.tracking import WandbTracker

    torch.manual_seed(0)
    clients = _clients()
    algorithm = LabelOnlyAlgorithm()
    val = evaluate_split(algorithm, clients, None, DEVICE, "val", NUM_CLASSES)
    test = evaluate_split(algorithm, clients, None, DEVICE, "test", NUM_CLASSES)

    logged = {}

    class _Run:
        def log(self, payload, step=None):
            logged.update(payload)

    tracker = object.__new__(WandbTracker)     # the real algorithm, no wandb install
    tracker.run = _Run()
    tracker.log_round(3, val, test)

    assert logged["round"] == 3
    assert "val/accuracy" in logged and "val/macro_f1" in logged
    assert "val/loss" not in logged             # unavailable -> omitted, not crashed


def test_the_tracker_logs_loss_when_the_algorithm_provides_it():
    from rigfl.eval.protocol import evaluate_split
    from rigfl.experiment.tracking import WandbTracker
    from rigfl.algorithms.local import Local, LocalConfig

    torch.manual_seed(0)
    clients = _clients()
    algorithm = Local(LocalConfig(local_epochs=1, lr=0.05))
    val = evaluate_split(algorithm, clients, None, DEVICE, "val", NUM_CLASSES)

    logged = {}

    class _Run:
        def log(self, payload, step=None):
            logged.update(payload)

    tracker = object.__new__(WandbTracker)
    tracker.run = _Run()
    tracker.log_round(0, val, val)
    assert math.isfinite(logged["val/loss"])


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


def test_a_label_only_algorithm_gets_a_clear_error_when_loss_is_requested():
    from rigfl.eval.selection import SelectionError, select_global

    torch.manual_seed(0)
    result = iterative(LabelOnlyAlgorithm(), _clients(), num_rounds=2, device=DEVICE,
                           num_classes=NUM_CLASSES, verbose=False)
    with pytest.raises(SelectionError) as e:
        select_global(result["evaluation_history"], "loss")
    msg = str(e.value)
    assert "needs probabilities" in msg
    assert "labels_only" in msg
    assert "one-hot" in msg                    # says why it is not manufactured


def test_early_stopping_on_loss_refuses_a_label_only_algorithm():
    with pytest.raises(ValueError, match="needs probabilities"):
        iterative(LabelOnlyAlgorithm(), _clients(), num_rounds=2, device=DEVICE,
                      num_classes=NUM_CLASSES, verbose=False,
                      early_stopping={"enabled": True, "metric": "loss",
                                      "patience": 1})


def test_no_one_hot_probabilities_are_manufactured_from_labels():
    out = as_predictions(torch.tensor([0, 1, 2]))
    assert out.probabilities is None
    with pytest.raises(MetricInputUnavailable):
        metric_input(spec("loss"), out)


# ── the registry's compatibility path ────────────────────────────────────────

def test_a_custom_hard_label_metric_keeps_the_old_signature():
    register("half_of_accuracy", "maximize",
             fn=lambda preds, labels, n: (preds == labels).float().mean().item() / 2)
    try:
        assert spec("half_of_accuracy").needs == "labels"
        y = torch.tensor([0, 1])
        out = compute_all(Predictions.from_probabilities(
            torch.tensor([[0.9, 0.1], [0.2, 0.8]])), y, 2)
        assert out["half_of_accuracy"] == 0.5
        # ...and a bare tensor still works, exactly as it used to
        assert compute_all(torch.tensor([0, 1]), y, 2)["half_of_accuracy"] == 0.5
    finally:
        unregister("half_of_accuracy")


def test_a_custom_probability_metric_can_ask_for_probabilities():
    register("mean_confidence", "maximize", needs="probabilities",
             fn=lambda probs, labels, n: probs.max(dim=1).values.mean().item())
    try:
        probs = torch.tensor([[0.9, 0.1], [0.6, 0.4]])
        out = compute_all(Predictions.from_probabilities(probs), torch.tensor([0, 0]), 2)
        assert out["mean_confidence"] == pytest.approx(0.75)
        # unavailable, not wrong, for a label-only prediction
        assert compute_all(torch.tensor([0, 0]), torch.tensor([0, 0]), 2)["mean_confidence"] is None
    finally:
        unregister("mean_confidence")


def test_register_rejects_an_unknown_input_kind():
    with pytest.raises(ValueError, match="needs must be one of"):
        register("bad", "maximize", fn=lambda *a: 0.0, needs="embeddings")
