"""RigFL <-> BioSilo connection, validated on BioSilo's synthetic multi-input
dataset (``seq`` + ``flat``, the same shape family as eICU's ``ts`` + ``static``).

Proves the whole path with no real data: BioSilo generation -> load_partition ->
RigFL's generic build_clients -> multi-input collate (MultiTensor) -> a
heterogeneous temporal backbone pool -> the invariant federated round. The test
swaps ``synthetic`` for ``eicu``.
"""

from __future__ import annotations

def __sel(history_or_result):
    """Selection is explicit now; these tests only need *a* reported round."""
    from rigfl.eval.selection import select_global
    hist = history_or_result.get("evaluation_history", history_or_result)
    return select_global(hist, "accuracy")



import tempfile

import pytest
import torch

biosilo = pytest.importorskip("biosilo")   # optional dep (editable-installed from ../BioSilo)

from rigfl.core import iterative
from rigfl.data.biosilo import build_biosilo_clients, temporal_dims
from rigfl.algorithms.fedproto import FedProto, FedProtoConfig
from rigfl.algorithms.fedtgp import FedTGP, FedTGPConfig
from rigfl.algorithms.local import Local, LocalConfig
from rigfl.models.eicu import eicu_pool

SHARED_DIM = 32


def _generate(root: str) -> str:
    """Generate a partition and return its id.

    A natural run must name the partition it loaded -- otherwise its filename,
    fingerprint and pool identity all say None while the numbers came from a
    specific one -- so every fixture below passes the id back in.
    """
    biosilo.generate("synthetic", root=root, n_inputs=2, with_groups=True,
                     n_clients=5, n_per_client=60, n_classes=3, n_features=5,
                     group_size=4, seed=0)
    return biosilo.load("synthetic", root=root).partition_id


def _clients(root: str, n_ts: int, n_static: int):
    return build_biosilo_clients(
        "synthetic", shared_dim=SHARED_DIM, backbones=eicu_pool(n_ts, n_static),
        root=root, val_frac=0.25, batch=16)


def test_multiinput_batch_is_a_multitensor():
    with tempfile.TemporaryDirectory() as root:
        _generate(root)
        handle = biosilo.load("synthetic", root=root)
        n_ts, n_static = temporal_dims(handle)
        assert (n_ts, n_static) == (5, 4)                 # seq (T=8, n_ts=5), flat (n_static=4)
        clients, handle = _clients(root, n_ts, n_static)
        assert len(clients) == handle.num_clients == 5
        xb, yb = next(iter(clients[0].train_loader))
        assert isinstance(xb, tuple) and len(xb) == 2      # MultiTensor(ts, static)
        assert xb[0].shape[1:] == (8, 5) and xb[1].shape[1:] == (4,)
        assert xb.to(torch.device("cpu"))[0].shape == xb[0].shape   # .to() works


def test_federated_round_over_temporal_pool():
    """Local, FedProto (rep-space + groups), FedTGP (server-trained protos +
    adaptive margin) each run end to end on the multi-input partition."""
    dev = torch.device("cpu")
    with tempfile.TemporaryDirectory() as root:
        _generate(root)
        handle = biosilo.load("synthetic", root=root)
        n_ts, n_static = temporal_dims(handle)
        nc = handle.num_classes
        algorithms = [
            Local(LocalConfig(local_epochs=1, lr=0.01)),
            FedProto(FedProtoConfig(lamda=0.1, local_epochs=1, lr=0.01)),
            FedTGP(
                FedTGPConfig(
                    lamda=0.1, local_epochs=1, lr=0.01,
                    server_epochs=1, server_lr=0.01, margin_cap=100.0,
                ),
                nc,
                SHARED_DIM,
            ),
        ]
        for algorithm in algorithms:
            clients, _ = _clients(root, n_ts, n_static)
            best = iterative(algorithm, clients, num_rounds=2, device=dev,
                                 num_classes=nc, eval_gap=1, verbose=False)
            assert 0.0 <= __sel(best)["test"]["accuracy"][0] <= 1.0
            assert 0.0 <= __sel(best)["test"]["balanced_accuracy"][0] <= 1.0


def test_adapter_factory_uses_each_algorithms_own_alignment():
    """FedTGP's paper aligns widths by pooling; every other algorithm's uses a
    learned projection."""
    from rigfl.core.adapters import AdaptivePool, LearnedProjection
    from rigfl.experiment.registry import adapter_factory

    assert isinstance(adapter_factory("fedtgp")(128, 32), AdaptivePool)
    assert isinstance(adapter_factory("fedproto")(128, 32), LearnedProjection)
    assert isinstance(adapter_factory("fedgh")(128, 32), LearnedProjection)
    assert isinstance(adapter_factory("local")(128, 32), LearnedProjection)


def test_natural_scheme_runs_through_the_experiment_infra():
    """scheme='natural' routes run_one to BioSilo and derives num_clients /
    num_classes from the partition (the #2 partition abstraction)."""
    from rigfl.experiment.config import ExperimentConfig
    from rigfl.experiment.registry import config_class
    from rigfl.experiment.run import run_one

    with tempfile.TemporaryDirectory() as root:
        partition = _generate(root)
        exp = ExperimentConfig(dataset="synthetic", scheme="natural", data_root=root,
                               partition=partition,
                               rounds=2, eval_gap=1, shared_dim=SHARED_DIM, batch=16,
                               val_frac=0.25, quiet=True)
        cfg = config_class("fedproto")(lamda=0.1)
        rec = run_one("fedproto", exp, cfg, torch.device("cpu"))
        assert rec["config"]["experiment"]["num_clients"] == 5    # derived from the data
        assert rec["config"]["experiment"]["num_classes"] == 3
        assert 0.0 <= __sel(rec["result"]["evaluation_history"])["test"]["accuracy"][0] <= 1.0


def test_fml_fedkd_build_a_temporal_aux_on_multiinput():
    """FML/FedKD add a shared meme/mentee model; run_one gives it a TEMPORAL
    backbone for multi-input data, so both run there like the six aux-free
    algorithms."""
    from rigfl.experiment.config import ExperimentConfig
    from rigfl.experiment.registry import config_class
    from rigfl.experiment.run import run_one

    with tempfile.TemporaryDirectory() as root:
        partition = _generate(root)
        for name in ("fml", "fedkd"):
            exp = ExperimentConfig(dataset="synthetic", scheme="natural", data_root=root,
                                   partition=partition,
                                   rounds=2, eval_gap=1, shared_dim=SHARED_DIM, batch=16,
                                   val_frac=0.25, quiet=True)
            rec = run_one(name, exp, config_class(name)(), torch.device("cpu"))
            assert 0.0 <= __sel(rec["result"]["evaluation_history"])["test"]["accuracy"][0] <= 1.0


def test_feddes_runs_on_multiinput():
    """FedDES's base pool (the ported eICU-3 classifiers) + GraphRoute's pipeline
    handle multi-input (ts, static) -- via the collate_fn passthrough into
    train_pool and the multi-input _BatchDataset. Needs graphroute."""
    pytest.importorskip("graphroute")
    from rigfl.experiment.config import ExperimentConfig
    from rigfl.experiment.registry import config_class
    from rigfl.experiment.run import run_one

    with tempfile.TemporaryDirectory() as root:
        partition = _generate(root)
        exp = ExperimentConfig(dataset="synthetic", scheme="natural", data_root=root,
                               partition=partition,
                               rounds=2, eval_gap=1, shared_dim=SHARED_DIM, batch=16,
                               val_frac=0.3, quiet=True)
        # calibrate=False: this test is about the eICU pool running through
        # GraphRoute, not about calibration, which GraphRoute covers itself.
        cfg = config_class("feddes")(base_epochs=2, gnn_epochs=5, gnn_patience=3,
                                     cache_dir="", calibrate=False)
        rec = run_one("feddes", exp, cfg, torch.device("cpu"))
        assert 0.0 <= __sel(rec["result"]["evaluation_history"])["test"]["accuracy"][0] <= 1.0
