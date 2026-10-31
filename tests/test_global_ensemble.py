"""Shared Local and Global Ensemble training behavior."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.algorithms.global_ensemble import GlobalEnsemble, GlobalEnsembleConfig
from rigfl.algorithms.local import Local, LocalConfig
from rigfl.core import Client, independent
from rigfl.experiment.config import EarlyStoppingConfig
from rigfl.experiment.registry import algorithm_spec


def _client():
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    y = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    model = nn.Linear(1, 2, bias=False)
    nn.init.zeros_(model.weight)
    return Client(model, loader, loader, loader)


def test_local_and_global_ensemble_share_the_independent_runner():
    assert algorithm_spec("local").runner is independent
    assert algorithm_spec("global_ensemble").runner is independent


def test_independent_runner_restores_client_checkpoints_and_soft_votes(monkeypatch):
    clients = [_client(), _client()]
    algorithm = GlobalEnsemble(GlobalEnsembleConfig())
    scores = {0: [0.9, 0.8], 1: [0.2, 0.9, 0.8]}

    def train_one_round(client, optimizer):
        client.state["step"] = client.state.get("step", 0) + 1
        with torch.no_grad():
            client.model.weight[0, 0].add_(1.0)

    def validation_metrics(model, client, device, num_classes):
        return {"accuracy": scores[client.client_id][client.state["step"] - 1]}

    monkeypatch.setattr(algorithm, "train_one_round", train_one_round)
    monkeypatch.setattr("rigfl.core.round._model_validation_metrics", validation_metrics)
    result = independent(
        algorithm,
        clients,
        num_rounds=4,
        device=torch.device("cpu"),
        num_classes=2,
        verbose=False,
        early_stopping=EarlyStoppingConfig(
            enabled=True, metric="accuracy", patience=1
        ),
    )

    selections = result["selection_provenance"]["clients"]
    assert [selections[str(i)]["selected_round"] for i in range(2)] == [0, 1]
    assert [float(model.weight[0, 0]) for model in algorithm.members] == [1.0, 2.0]
    assert result["evaluation_history"]["evaluation_rounds"] == [0]

    x = torch.tensor([[1.0]])
    actual = algorithm.predict(clients[0], x).probabilities
    expected = sum(
        torch.softmax(model(x), dim=1) for model in algorithm.members
    ) / len(algorithm.members)
    assert torch.allclose(actual, expected)


def test_local_and_global_ensemble_train_identical_member_models():
    local_clients = [_client(), _client()]
    ensemble_clients = [_client(), _client()]
    local = Local(LocalConfig(local_epochs=1, lr=0.05))
    ensemble = GlobalEnsemble(GlobalEnsembleConfig(local_epochs=1, lr=0.05))

    common = dict(
        num_rounds=3,
        device=torch.device("cpu"),
        num_classes=2,
        verbose=False,
    )
    independent(local, local_clients, **common)
    independent(ensemble, ensemble_clients, **common)

    for local_client, ensemble_member in zip(local_clients, ensemble.members):
        for name, value in local_client.model.state_dict().items():
            assert torch.equal(value, ensemble_member.state_dict()[name])


def test_local_records_every_epoch_of_each_clients_own_training():
    clients = [_client(), _client()]
    clients[1].train_loader = DataLoader(
        TensorDataset(torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
                      torch.tensor([1, 1, 0, 0])), batch_size=4)
    result = independent(
        Local(LocalConfig(local_epochs=1, lr=0.5)), clients, num_rounds=8,
        device=torch.device("cpu"), num_classes=2, verbose=False,
        early_stopping=EarlyStoppingConfig(enabled=True, metric="loss", patience=2),
    )

    assert result["round_selection"] == "client-specific"
    history = result["evaluation_history"]
    rounds = history["evaluation_rounds"]
    for cid, selection in result["selection_provenance"]["clients"].items():
        losses = history["clients"][cid]["validation"]["loss"]
        tests = history["clients"][cid]["test"]["loss"]
        trained = [r <= selection["stopped_at_round"] for r in rounds]
        assert [v is not None for v in losses] == trained
        assert [v is not None for v in tests] == trained
        assert losses[rounds.index(selection["selected_round"])] == min(
            v for v in losses if v is not None)


def test_local_and_global_ensemble_select_identical_members_with_early_stopping():
    local_clients = [_client(), _client()]
    ensemble_clients = [_client(), _client()]
    local = Local(LocalConfig(local_epochs=1, lr=0.5))
    ensemble = GlobalEnsemble(GlobalEnsembleConfig(local_epochs=1, lr=0.5))
    common = dict(
        num_rounds=8, device=torch.device("cpu"), num_classes=2, verbose=False,
        early_stopping=EarlyStoppingConfig(enabled=True, metric="loss", patience=2),
    )
    local_result = independent(local, local_clients, **common)
    ensemble_result = independent(ensemble, ensemble_clients, **common)

    assert (local_result["selection_provenance"]
            == ensemble_result["selection_provenance"])
    assert "round_selection" not in ensemble_result
    for local_client, ensemble_member in zip(local_clients, ensemble.members):
        for name, value in local_client.model.state_dict().items():
            assert torch.equal(value, ensemble_member.state_dict()[name])


def test_global_ensemble_members_are_the_local_models_local_reports():
    from rigfl.eval.metrics import compute_all
    from rigfl.eval.report import selection_for
    from rigfl.prediction import Predictions

    def clients():
        pair = [_client(), _client()]
        pair[1].train_loader = DataLoader(
            TensorDataset(torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
                          torch.tensor([1, 1, 0, 0])), batch_size=4)
        return pair

    common = dict(
        num_rounds=8, device=torch.device("cpu"), num_classes=2, verbose=False,
        early_stopping=EarlyStoppingConfig(enabled=True, metric="loss", patience=2),
    )
    local = independent(Local(LocalConfig(lr=0.5)), clients(), **common)
    ensemble = GlobalEnsemble(GlobalEnsembleConfig(lr=0.5, member_metric="accuracy"))
    ensemble_clients = clients()
    result = independent(ensemble, ensemble_clients, **common)

    record = {"config": {"experiment": {"num_classes": 2}}, "result": local}
    reported = selection_for(record, "accuracy")["selected_rounds"]
    history = local["evaluation_history"]
    for cid, member in enumerate(ensemble.members):
        selection = result["selection_provenance"]["clients"][str(cid)]
        assert selection["selected_round"] == reported[str(cid)]
        x, y = next(iter(ensemble_clients[cid].test_loader))
        with torch.no_grad():
            loss = compute_all(Predictions.from_logits(member(x)), y, 2)["loss"]
        index = history["evaluation_rounds"].index(reported[str(cid)])
        assert loss == history["clients"][str(cid)]["test"]["loss"][index]
    assert result["selection_provenance"]["metric"] == "accuracy"
    assert result["early_stopping"]["metric"] == "loss"
