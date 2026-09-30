"""Resource accounting at runner boundaries."""

from dataclasses import dataclass

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.algorithms.fedtgp import FedTGP, FedTGPConfig
from rigfl.algorithms.global_ensemble import GlobalEnsemble, GlobalEnsembleConfig
from rigfl.core import Algorithm, Client, Predictions
from rigfl.core.config import AlgorithmConfig
from rigfl.core.round import iterative
from rigfl.eval.report import summarize_resources
from rigfl.eval.resources import ResourceMonitor, payload_bytes

DEVICE = torch.device("cpu")


def _clients(count=2):
    dataset = TensorDataset(torch.zeros(4, 1), torch.zeros(4, dtype=torch.long))
    return [
        Client(nn.Linear(1, 2), DataLoader(dataset, batch_size=2),
               DataLoader(dataset, batch_size=2),
               DataLoader(dataset, batch_size=2))
        for _ in range(count)
    ]


@dataclass
class _Upload:
    tensor: torch.Tensor
    samples: int


def test_payload_size_counts_only_tensor_contents():
    tensor = torch.zeros(3, dtype=torch.float64)
    payload = {"upload": _Upload(tensor, 10), "same_tensor": tensor, "note": "x"}
    assert payload_bytes(payload) == 24

    model = nn.Linear(3, 2)
    expected = sum(value.numel() * value.element_size()
                   for value in model.state_dict().values())
    assert payload_bytes(model) == expected
    assert payload_bytes({"encoded": b"1234"}) == 4
    assert payload_bytes(memoryview(torch.zeros(3, dtype=torch.float32).numpy())) == 12


class _Iterative(Algorithm):
    def __init__(self):
        super().__init__(AlgorithmConfig())

    def init_globals(self):
        return torch.zeros(2)

    def local_train(self, client, shared):
        return torch.zeros(1)

    def aggregate(self, uploads, shared):
        return shared

    def predict(self, client, x, shared):
        return Predictions.from_logits(client.model(x))


def test_iterative_runner_counts_each_logical_transfer():
    monitor = ResourceMonitor(DEVICE)
    with monitor:
        iterative(_Iterative(), _clients(), num_rounds=2, device=DEVICE,
                  num_classes=2, verbose=False, resource_monitor=monitor)
    resources = monitor.to_dict()

    assert resources["observed"]["communication_bytes"] == {
        "client_to_server": 16,
        "server_to_client": 32,
        "total": 48,
    }


def test_global_ensemble_does_not_count_an_unused_server_broadcast():
    algorithm = GlobalEnsemble(GlobalEnsembleConfig())
    monitor = ResourceMonitor(DEVICE)
    with monitor:
        iterative(algorithm, _clients(), num_rounds=2, device=DEVICE,
                  num_classes=2, verbose=False, resource_monitor=monitor)
    communication = monitor.to_dict()["observed"]["communication_bytes"]
    assert communication["client_to_server"] > 0
    assert communication["server_to_client"] == 0


def test_fedtgp_broadcast_counts_prototypes_not_the_private_generator():
    algorithm = FedTGP(FedTGPConfig(), num_classes=3, feature_dim=4)
    shared = algorithm.init_globals()
    assert algorithm.communication_payload_bytes(
        shared, kind="server_to_client") == 0

    shared["protos"] = {label: torch.zeros(4) for label in range(3)}
    assert algorithm.communication_payload_bytes(
        shared, kind="server_to_client") == 48
    assert payload_bytes(shared) > 48


def test_flop_estimation_is_opt_in_and_separates_operation_categories():
    pytest.importorskip("torch.utils.flop_counter")
    monitor = ResourceMonitor(DEVICE, estimate_flops=True)
    with monitor:
        with monitor.measure("matmul", category="algorithm"):
            torch.mm(torch.ones(2, 3), torch.ones(3, 4))
        with monitor.measure("validation", category="evaluation"):
            torch.mm(torch.ones(1, 2), torch.ones(2, 1))
    resources = monitor.to_dict()

    # Matrix multiplication uses 2*m*n*k FLOPs under multiply-add=2:
    # 2*2*3*4 = 48 and 2*1*2*1 = 4.
    assert resources["operations"]["matmul"]["flops"] == 48
    assert resources["operations"]["validation"]["flops"] == 4
    assert resources["observed"]["flops"] == {
        "algorithm_operations": 48,
        "evaluation": 4,
        "total": 52,
    }


def test_resource_summary_requires_compatible_hardware():
    monitor = ResourceMonitor(DEVICE)
    first = monitor.to_dict()
    second = monitor.to_dict()
    second["measurement"]["timing"]["hardware"] = {
        **second["measurement"]["timing"]["hardware"],
        "device_name": "another device",
    }
    records = [{"resources": first}, {"resources": second}]
    summary = summarize_resources(records)
    assert summary["available"] is True
    assert summary["wall_time_comparable"] is False
    assert summary["algorithm_wall_seconds_mean"] is None
