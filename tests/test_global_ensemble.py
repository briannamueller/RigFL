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
