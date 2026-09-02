"""FedDES base-pool reuse: the part of it RigFL owns.

RigFL decides *which* pool a directory holds -- the federated partition, the
client, the factories it built and the base-training settings, folded into
``_pool_fp``. GraphRoute decides what a stored pool contains and how it is
loaded, locked and published; its own tests cover that.

So the property here is the identity one: ``_pool_fp`` is invariant to graph and
GNN settings, which is what lets a sweep over them reuse one trained pool, and
sensitive to everything that changes the pool itself.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.core.interfaces import OneShotContext
from rigfl.core.round import Client, p2p_one_shot
from rigfl.algorithms.feddes import FedDES, FedDESConfig

DATA_ID = "cifar10-partition-a"
MODEL_IDS = ["linear-a", "linear-b"]


class _ScaledLinear(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x) * self.scale


def _feddes(tmp, **over):
    torch.manual_seed(0)
    factories = [lambda: nn.Linear(4, 3), lambda: nn.Linear(4, 3)]
    validation_fraction = over.pop("validation_fraction", 0.2)
    return FedDES(
        FedDESConfig(cache_dir=tmp, **over), factories, 3,
        data_id=DATA_ID, model_ids=MODEL_IDS,
        validation_fraction=validation_fraction,
    )


def test_pool_fp_is_deterministic_and_client_paths_are_distinct():
    with tempfile.TemporaryDirectory() as tmp:
        m = _feddes(tmp)
        assert m._pool_fp() == m._pool_fp()
        root = Path(tmp) / DATA_ID / f"pool_{m._pool_fp()}" / "clients"
        assert root / "client_0" != root / "client_1"


def test_pool_fp_ignores_gnn_but_tracks_base():
    with tempfile.TemporaryDirectory() as tmp:
        base = _feddes(tmp)._pool_fp()
        # graph/GNN settings must NOT change the pool identity (that's the reuse win)
        assert _feddes(tmp, gnn_arch="mlp", gnn_epochs=99, graph_k=9)._pool_fp() == base
        # base-training settings MUST change it
        assert _feddes(tmp, base_lr=0.1)._pool_fp() != base
        assert _feddes(tmp, base_epochs=7)._pool_fp() != base
        assert _feddes(tmp, validation_fraction=0.4)._pool_fp() != base


def test_pool_fp_tracks_templates_separately_from_readable_names():
    a = _ScaledLinear(1.0)
    b = _ScaledLinear(2.0)
    b.load_state_dict(a.state_dict())

    cfg = FedDESConfig(cache_dir="")
    first = FedDES(cfg, [a], 3, model_ids=["same-readable-name"])
    second = FedDES(cfg, [b], 3, model_ids=["same-readable-name"])

    assert first._pool_fp() != second._pool_fp()


def test_natural_validation_fraction_reaches_the_pool_identity():
    from rigfl.experiment.config import ExperimentConfig
    from rigfl.experiment.registry import build_algorithm, config_class

    factories = [lambda: nn.Linear(4, 3) for _ in range(3)]
    cfg = config_class("feddes")(cache_dir="pool_cache")
    partition_id = "mortality_24h_n0_size_s1_abc123"
    common = dict(dataset="eicu", scheme="natural", partition=partition_id,
                  num_classes=3)
    a = build_algorithm("feddes", ExperimentConfig(**common, val_frac=0.2), cfg,
                     base_pool=factories)
    b = build_algorithm("feddes", ExperimentConfig(**common, val_frac=0.4), cfg,
                     base_pool=factories)

    assert a.data_id == b.data_id == f"eicu-{partition_id}"
    assert a._pool_fp() != b._pool_fp()


def test_generated_partition_identity_reaches_feddes_cache():
    from rigfl.experiment.config import ExperimentConfig
    from rigfl.experiment.registry import build_algorithm, config_class

    exp = ExperimentConfig(
        dataset="cifar10",
        scheme="generated",
        partition="partition-fingerprint",
        num_clients=3,
        num_classes=3,
    )
    algorithm = build_algorithm(
        "feddes",
        exp,
        config_class("feddes")(),
        base_pool=[lambda: nn.Linear(4, 3) for _ in range(3)],
    )
    assert algorithm.data_id == "cifar10-partition-fingerprint"


def test_train_or_load_reuses_pool():
    """First call trains and caches; second reuses -- through GraphRoute.

    RigFL no longer implements the reuse itself; this checks that it hands
    GraphRoute a directory derived from its own fingerprint and gets one training
    call out of two requests. Stubs ``_train`` so no base training happens.
    """
    with tempfile.TemporaryDirectory() as tmp:
        m = _feddes(tmp, base_split_mode="in_sample")
        calls = {"n": 0}

        def fake_train(tr, va, dev, client_id):
            calls["n"] += 1
            return [f() for f in m.base_factories], None   # (models, oof_logits)

        m._train = fake_train                              # instance stub: called as (tr, va, dev)
        dev = torch.device("cpu")
        first = m._train_or_load_pool(None, None, dev, 0)
        second = m._train_or_load_pool(None, None, dev, 0)
        assert calls["n"] == 1                             # trained once, reused thereafter
        for a, b in zip(first.load_models(dev), second.load_models(dev)):
            assert torch.allclose(a.weight, b.weight)      # reused pool == trained pool
        # ...and it landed under the fingerprint RigFL resolved, not somewhere
        # GraphRoute chose
        expected = (Path(tmp) / DATA_ID / f"pool_{m._pool_fp()}"
                    / "clients" / "client_0" / "models" / "model_0.pt")
        assert expected.exists()


def test_feddes_uses_rigfls_official_validation_split():
    train = TensorDataset(torch.randn(12, 4), torch.randint(0, 3, (12,)))
    validation = TensorDataset(torch.randn(5, 4), torch.randint(0, 3, (5,)))
    train_loader = DataLoader(train, batch_size=4)
    validation_loader = DataLoader(validation, batch_size=5)
    algorithm = FedDES(
        FedDESConfig(calibrate=False, cache_dir=""),
        [lambda: nn.Linear(4, 3)], 3)
    captured = {}
    artifact = object()

    def fake_prepare(train_dataset, validation_dataset, device, client_id):
        captured.update(train=train_dataset, validation=validation_dataset)
        return artifact

    algorithm._train_or_load_pool = fake_prepare
    algorithm._union_pools = lambda uploads: uploads[0]
    state = {}
    ctx = OneShotContext(torch.device("cpu"), 0, state, validation_loader)
    outgoing = algorithm.prepare(None, train_loader, ctx)
    assert outgoing is artifact
    assert state["local_pool"] is artifact
    assert captured == {"train": train, "validation": validation}


def test_feddes_publishes_the_training_artifact_layout(tmp_path):
    def dataset(n):
        labels = torch.arange(n) % 3
        return TensorDataset(torch.randn(n, 4), labels)

    clients = [
        Client(nn.Linear(4, 3), DataLoader(dataset(30), batch_size=6),
               DataLoader(dataset(9), batch_size=9),
               DataLoader(dataset(9), batch_size=9))
        for _ in range(2)
    ]
    algorithm = FedDES(
        FedDESConfig(
            base_epochs=1, gnn_arch="mlp", gnn_epochs=1, gnn_patience=1,
            base_oof_folds=2, graph_k=2, hidden_dim=8, calibrate=False,
            cache_dir=str(tmp_path),
        ),
        [lambda: nn.Linear(4, 3)], 3, data_id=DATA_ID,
        model_ids=["linear"], seed=0)
    p2p_one_shot(algorithm, clients, num_rounds=1, device=torch.device("cpu"),
                 num_classes=3, verbose=False)

    root = tmp_path / DATA_ID / f"pool_{algorithm._pool_fp()}"
    assert (root / "manifest.json").exists()
    for cid in range(2):
        assert (root / "clients" / f"client_{cid}" / "models" / "model_0.pt").exists()
        assert (root / "clients" / f"client_{cid}" / "oof_logits.pt").exists()
        outputs = root / "outputs" / f"client_{cid}"
        assert (outputs / "train_logits.pt").exists()
        assert (outputs / "validation_logits.pt").exists()


def test_cache_dir_is_operational_not_scientific():
    """/scratch/a and /scratch/b train the identical configuration.

    It used to be part of the fingerprint, so the two produced different result
    filenames and two rows in the collected table.
    """
    from rigfl.experiment.collect import algorithm_variant
    from rigfl.experiment.config import ExperimentConfig, result_filename, run_fingerprint
    from rigfl.experiment.registry import config_class

    exp = ExperimentConfig(rounds=2)
    Cfg = config_class("feddes")
    a = Cfg(cache_dir="/scratch/a").model_dump()
    b = Cfg(cache_dir="/scratch/b").model_dump()
    other = Cfg(cache_dir="/scratch/a", graph_k=9).model_dump()

    assert run_fingerprint(exp, a) == run_fingerprint(exp, b)
    assert result_filename(exp, "feddes", run_fingerprint(exp, a)) == \
        result_filename(exp, "feddes", run_fingerprint(exp, b))
    assert algorithm_variant({"config": {"algorithm": a}}) == \
        algorithm_variant({"config": {"algorithm": b}})

    # ...while a real setting still separates them
    assert run_fingerprint(exp, a) != run_fingerprint(exp, other)
    assert algorithm_variant({"config": {"algorithm": a}}) != \
        algorithm_variant({"config": {"algorithm": other}})

    assert a["cache_dir"] == "/scratch/a"          # still recorded, still carried
