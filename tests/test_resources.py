"""Resource accounting at runner and artifact boundaries."""

import builtins
from dataclasses import dataclass

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.algorithms.fedtgp import FedTGP, FedTGPConfig
from rigfl.algorithms.global_ensemble import GlobalEnsemble, GlobalEnsembleConfig
from rigfl.core import Algorithm, Client, LocalSelection, Predictions
from rigfl.core.config import AlgorithmConfig
from rigfl.core.round import iterative, p2p_one_shot
from rigfl.eval.report import format_resource_table, summarize_resources
from rigfl.eval.resources import (
    ResourceDelta,
    ResourceMonitor,
    load_cached_measurement,
    payload_bytes,
    write_cached_measurement,
)

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
        "peer_to_peer": 0,
        "total": 48,
    }
    assert resources["operations"]["local_train"]["calls"] == 4
    assert [item["round"] for item in resources["checkpoints"]] == [0, 1]
    assert resources["observed"]["flops"]["total"] is None
    assert set(resources["clients"]["0"]["wall_seconds"]) == {
        "algorithm_operations", "evaluation"
    }
    assert resources["clients"]["0"]["wall_seconds"]["algorithm_operations"] > 0
    assert resources["clients"]["0"]["wall_seconds"]["evaluation"] > 0


def test_resource_monitoring_preserves_the_existing_tracker_interface():
    class ExistingTracker:
        def __init__(self):
            self.rounds = []

        def log_round(self, rnd, val, test):
            self.rounds.append(rnd)

    tracker = ExistingTracker()
    monitor = ResourceMonitor(DEVICE)
    with monitor:
        iterative(
            _Iterative(), _clients(), num_rounds=1, device=DEVICE,
            num_classes=2, verbose=False, tracker=tracker,
            resource_monitor=monitor,
        )
    assert tracker.rounds == [0]


@pytest.mark.parametrize("size", [-1, 1.5, True])
def test_transfer_size_must_be_a_non_negative_integer(size):
    monitor = ResourceMonitor(DEVICE)
    with pytest.raises(ValueError, match="non-negative integer"):
        monitor.record_transfer("client_to_server", size)


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


class _OneShot(Algorithm):
    def __init__(self):
        super().__init__(AlgorithmConfig())

    def prepare(self, model, train_loader, ctx):
        return torch.zeros(1)

    def one_shot_communication(self, outgoing):
        return [outgoing for _ in outgoing]

    def local_computation(self, model, incoming, train_loader, ctx):
        return LocalSelection(0, "accuracy", 1.0)

    def predict(self, client, x, shared):
        return Predictions.from_logits(client.model(x))


def test_one_shot_runner_counts_each_peer_delivery():
    monitor = ResourceMonitor(DEVICE)
    with monitor:
        p2p_one_shot(_OneShot(), _clients(3), num_rounds=1, device=DEVICE,
                     num_classes=2, verbose=False, resource_monitor=monitor)
    communication = monitor.to_dict()["observed"]["communication_bytes"]
    assert communication["peer_to_peer"] == 24
    assert communication["total"] == 24


def test_flop_estimation_is_opt_in_and_separates_operation_categories():
    monitor = ResourceMonitor(DEVICE, estimate_flops=True)
    with monitor:
        with monitor.measure("matmul", category="algorithm"):
            torch.mm(torch.ones(2, 3), torch.ones(3, 4))
        with monitor.measure("validation", category="evaluation"):
            torch.mm(torch.ones(1, 2), torch.ones(2, 1))
    resources = monitor.to_dict()

    assert resources["operations"]["matmul"]["flops"] == 48
    assert resources["operations"]["validation"]["flops"] == 4
    assert resources["observed"]["flops"] == {
        "algorithm_operations": 48,
        "evaluation": 4,
        "total": 52,
    }


def test_flop_estimation_explains_its_minimum_pytorch_version(monkeypatch):
    original_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "torch.utils.flop_counter":
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with (
        pytest.raises(RuntimeError, match="PyTorch 2.1 or newer"),
        ResourceMonitor(DEVICE, estimate_flops=True),
    ):
        pass


def test_cached_measurement_is_carried_into_resource_totals(tmp_path):
    first = ResourceMonitor(DEVICE, estimate_flops=True)
    measurement = {
        "wall_seconds": 3.0,
        "flops": 120,
        "measurement": first.measurement_signature(),
    }
    path = tmp_path / "resources.json"
    write_cached_measurement(path, fingerprint="pool-a", measurement=measurement)
    saved = load_cached_measurement(path, fingerprint="pool-a")

    reused = ResourceMonitor(DEVICE, estimate_flops=True)
    reused.add_reused("base_pool/client_0", saved)
    resources = reused.to_dict()
    assert resources["attributed_training"]["wall_seconds"] == 3.0
    assert resources["attributed_training"]["flops"] == 120

    assert load_cached_measurement(path, fingerprint="pool-b") is None

    path.write_text(path.read_text().replace('"wall_seconds": 3.0',
                                             '"wall_seconds": -1.0'))
    assert load_cached_measurement(path, fingerprint="pool-a") is None


def test_attributed_time_replaces_cache_access_with_saved_creation_time():
    times = iter([0.0, 5.0])
    monitor = ResourceMonitor(DEVICE, clock=lambda: next(times))
    with monitor.measure("local_computation", category="algorithm"):
        pass
    measurement = {
        "wall_seconds": 3.0,
        "flops": None,
        "measurement": monitor.measurement_signature(),
    }
    monitor.add_reused(
        "pool_output/client_0/train_logits", measurement,
        access=ResourceDelta(wall_seconds=1.0),
    )
    resources = monitor.to_dict()
    assert resources["observed"]["wall_seconds"]["algorithm_operations"] == 5.0
    assert resources["attributed_training"]["wall_seconds"] == 7.0


def test_cached_wall_time_is_not_combined_across_hardware():
    monitor = ResourceMonitor(DEVICE)
    measurement = {
        "wall_seconds": 3.0,
        "flops": None,
        "measurement": monitor.measurement_signature(),
    }
    measurement["measurement"]["hardware"] = {
        **measurement["measurement"]["hardware"], "device_name": "another device"
    }
    monitor.add_reused("base_pool/client_0", measurement)
    resources = monitor.to_dict()
    assert resources["attributed_training"]["wall_seconds"] is None
    assert resources["attributed_training"]["wall_seconds_comparable"] is False


def test_missing_cached_measurement_disables_attribution():
    monitor = ResourceMonitor(DEVICE, estimate_flops=True)
    monitor.add_reused(
        "base_pool/client_0", None,
        access=ResourceDelta(wall_seconds=1.0, flops=0),
    )
    resources = monitor.to_dict()
    assert resources["missing_reused_measurements"] == ["base_pool/client_0"]
    assert resources["attributed_training"] == {
        "flops": None,
        "wall_seconds": None,
        "wall_seconds_comparable": False,
    }


def test_mixed_cache_results_are_reported_as_partial():
    monitor = ResourceMonitor(DEVICE)
    monitor.record_cache("base_pools", "hit")
    monitor.record_cache("base_pools", "miss")
    assert monitor.to_dict()["cache_reuse"]["base_pools"] == "partial"


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
    assert summary["attributed_training_wall_seconds_mean"] is None

    table = format_resource_table({"fedavg": {"resources": summary}})
    assert "communication (GiB)" in table
    assert "attributed training FLOPs (TFLOPs)" in table
    assert "| fedavg | 0.000 ± 0.000 | — | — | none |" in table


def test_resource_summary_requires_the_same_flop_estimator():
    monitor = ResourceMonitor(DEVICE, estimate_flops=True)
    first = monitor.to_dict()
    second = monitor.to_dict()
    second["measurement"]["flop_estimation"] = {
        **second["measurement"]["flop_estimation"],
        "torch_version": "different",
    }
    summary = summarize_resources([{"resources": first}, {"resources": second}])
    assert summary["flops_comparable"] is False
    assert summary["attributed_training_flops_mean"] is None


def test_resource_summary_handles_malformed_legacy_resources():
    resources = ResourceMonitor(DEVICE).to_dict()
    del resources["measurement"]["timing"]
    assert summarize_resources([{"resources": resources}]) == {
        "available": False
    }

    resources = ResourceMonitor(DEVICE).to_dict()
    resources["measurement"]["timing"]["hardware"] = []
    assert summarize_resources([{"resources": resources}]) == {
        "available": False
    }
