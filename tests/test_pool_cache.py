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

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rigfl.algorithms.feddes import FedDES, FedDESConfig, _MeasuredPoolOutputs
from rigfl.core.interfaces import OneShotContext
from rigfl.core.round import Client, p2p_one_shot
from rigfl.eval.resources import ResourceMonitor, load_cached_measurement
from tests.helpers import resolved_experiment

DATA_ID = "cifar10-partition-a"
MODEL_IDS = ["linear-a", "linear-b"]


class _ScaledLinear(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x) * self.scale


def _feddes(tmp, *, base=None, graph=None, gnn=None, validation_fraction=0.2):
    torch.manual_seed(0)
    factories = [lambda: nn.Linear(4, 3), lambda: nn.Linear(4, 3)]
    return FedDES(
        FedDESConfig(
            cache_dir=tmp,
            graphroute={"base": base or {}, "graph": graph or {}, "gnn": gnn or {}},
        ),
        factories, 3,
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
        assert _feddes(
            tmp, graph={"k": 9}, gnn={"arch": "mlp", "epochs": 99}
        )._pool_fp() == base
        assert _feddes(tmp, gnn={"arch": "hetero_gat"})._pool_fp() == base
        assert _feddes(
            tmp, gnn={"use_edge_attr": True, "use_sample_residual": True,
                      "fallback": "wacc"}
        )._pool_fp() == base
        # base-training settings MUST change it
        assert _feddes(tmp, base={"lr": 0.1})._pool_fp() != base
        assert _feddes(tmp, base={"epochs": 7})._pool_fp() != base
        assert _feddes(tmp, base={"weighted_by_class": False})._pool_fp() != base
        assert _feddes(tmp, validation_fraction=0.4)._pool_fp() != base


def test_graphroute_config_forwards_modeling_settings():
    with tempfile.TemporaryDirectory() as tmp:
        model = _feddes(
            tmp,
            base={"es_patience": 7},
            graph={
                "node_feature_source": "feature_space",
                "edge_feature_source": "embedding_mean",
                "neighbor_mode": "class_balanced",
            },
            gnn={"use_edge_attr": True, "use_sample_residual": True,
                 "fallback": "wacc"},
        )
        cfg = model._graphroute_config(0, torch.device("cpu"))

    assert cfg.base.es_patience == 7
    assert cfg.graph.node_feature_source == "feature_space"
    assert cfg.graph.edge_feature_source == "embedding_mean"
    assert cfg.graph.neighbor_mode == "class_balanced"
    assert cfg.gnn.use_edge_attr is True
    assert cfg.gnn.use_sample_residual is True
    assert cfg.gnn.fallback == "wacc"


def test_base_training_uses_the_nested_graphroute_settings(monkeypatch):
    captured = {}

    def fake_train(*args, **kwargs):
        captured.update(kwargs)
        return [nn.Linear(4, 3)], torch.zeros(4, 1, 3), torch.zeros(4)

    monkeypatch.setattr("graphroute.pool.train_pool_oof", fake_train)
    model = _feddes(
        "",
        base={
            "oof_folds": 4,
            "batch_size": 17,
            "epochs": 23,
            "es_patience": 6,
            "lr": 0.002,
            "optimizer": "SGD",
            "weight_decay": 0.03,
            "weighted_by_class": False,
            "es_metric": "val_bacc",
        },
    )
    dataset = TensorDataset(torch.randn(4, 4), torch.randint(0, 3, (4,)))

    model._train(dataset, dataset, torch.device("cpu"), client_id=2)

    assert captured == {
        "n_folds": 4,
        "inner_val_ratio": 0.2,
        "batch_size": 17,
        "max_epochs": 23,
        "patience": 6,
        "lr": 0.002,
        "optimizer_name": "SGD",
        "weight_decay": 0.03,
        "task": "classification",
        "num_classes": 3,
        "weighted_by_class": False,
        "es_metric": "val_bacc",
        "seed": 2,
        "collate_fn": captured["collate_fn"],
    }


def test_feddes_graphroute_defaults_and_partial_overrides():
    cfg = FedDESConfig(graphroute={"graph": {"k": 11}})

    assert cfg.graphroute.base.epochs == 100
    assert cfg.graphroute.base.oof_folds == 3
    assert cfg.graphroute.base.batch_size == 64
    assert cfg.graphroute.graph.k == 11
    assert cfg.graphroute.gnn.epochs == 500
    assert cfg.graphroute.gnn.patience == 50
    assert cfg.graphroute.gnn.ens_combination_mode == "hard_weighted_voting"
    assert cfg.graphroute.gnn.voting_weight_space == "sig"


def test_feddes_rejects_old_flat_graphroute_settings():
    with pytest.raises(Exception, match="base_lr"):
        FedDESConfig(base_lr=0.01)


def test_feddes_rejects_graphroute_owned_pool_models_and_split_train():
    with pytest.raises(Exception, match="selected by the RigFL experiment"):
        FedDESConfig(graphroute={"base": {"models": ["mlp32"]}})
    with pytest.raises(Exception, match="oof_stacking"):
        FedDESConfig(graphroute={"base": {"split_mode": "split_train"}})


def test_pool_fp_tracks_templates_separately_from_readable_names():
    a = _ScaledLinear(1.0)
    b = _ScaledLinear(2.0)
    b.load_state_dict(a.state_dict())

    cfg = FedDESConfig(cache_dir="")
    first = FedDES(cfg, [a], 3, model_ids=["same-readable-name"])
    second = FedDES(cfg, [b], 3, model_ids=["same-readable-name"])

    assert first._pool_fp() != second._pool_fp()


def test_validation_fraction_reaches_the_pool_identity():
    from rigfl.experiment.registry import build_algorithm, config_class

    factories = [lambda: nn.Linear(4, 3) for _ in range(3)]
    cfg = config_class("feddes")(cache_dir="pool_cache")
    partition_id = "partition-fingerprint"
    common = dict(
        dataset="cifar10", partition_scheme="dirichlet",
        partition_id=partition_id, num_classes=3,
        model_family="image_heterogeneous_3",
    )
    a = build_algorithm("feddes", resolved_experiment(
        **common, validation_fraction=0.2), cfg,
                     base_pool=factories)
    b = build_algorithm("feddes", resolved_experiment(
        **common, validation_fraction=0.4), cfg,
                     base_pool=factories)

    assert a.data_id == b.data_id == f"cifar10-{partition_id}"
    assert a._pool_fp() != b._pool_fp()


def test_generated_partition_identity_reaches_feddes_cache():
    from rigfl.experiment.registry import build_algorithm, config_class

    exp = resolved_experiment(
        dataset="cifar10",
        partition_id="partition-fingerprint",
        partition_scheme="dirichlet",
        num_clients=3,
        num_classes=3,
        model_family="image_heterogeneous_3",
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
        m = _feddes(tmp)
        calls = {"n": 0}

        def fake_train(tr, va, dev, client_id):
            calls["n"] += 1
            return [f() for f in m.base_factories], torch.zeros(1, 2, 3)

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


def test_reused_pool_carries_its_training_measurement(tmp_path):
    model = _feddes(str(tmp_path))
    model._train = lambda tr, va, dev, cid: (
        [factory() for factory in model.base_factories], torch.zeros(1, 2, 3))
    device = torch.device("cpu")

    created = ResourceMonitor(device)
    model._train_or_load_pool(None, None, device, 0, created)
    resource_path = (tmp_path / DATA_ID / f"pool_{model._pool_fp()}" /
                     "clients" / "client_0" / "pool_resources.json")
    assert resource_path.exists()
    assert created.to_dict()["cache_reuse"]["base_pools"] == "none"

    reused = ResourceMonitor(device)
    model._train_or_load_pool(None, None, device, 0, reused)
    resources = reused.to_dict()
    assert resources["cache_reuse"]["base_pools"] == "complete"
    assert resources["reused"][0]["name"] == "base_pool/client_0"
    assert resources["attributed_training"]["wall_seconds"] is not None


def test_pool_measurement_includes_cache_publication(tmp_path, monkeypatch):
    from graphroute import pool_cache

    model = _feddes(str(tmp_path))
    timeline = {"now": 0.0}

    def train(tr, va, dev, cid):
        timeline["now"] = 2.0
        return [factory() for factory in model.base_factories], torch.zeros(1, 2, 3)

    original_save = pool_cache.save_pool

    def save(*args, **kwargs):
        artifact = original_save(*args, **kwargs)
        timeline["now"] = 5.0
        return artifact

    model._train = train
    monkeypatch.setattr(pool_cache, "save_pool", save)
    monitor = ResourceMonitor(
        torch.device("cpu"), clock=lambda: timeline["now"])
    model._train_or_load_pool(None, None, torch.device("cpu"), 0, monitor)

    resource_path = (tmp_path / DATA_ID / f"pool_{model._pool_fp()}" /
                     "clients" / "client_0" / "pool_resources.json")
    saved = load_cached_measurement(
        resource_path, fingerprint=model._pool_fp())
    assert saved["wall_seconds"] == 5.0


def test_reused_pool_outputs_carry_their_measurement(tmp_path):
    from graphroute.pool_cache import in_memory_pool

    dataset = TensorDataset(torch.randn(6, 4), torch.arange(6) % 3)
    loader = DataLoader(dataset, batch_size=3)
    pool = in_memory_pool(
        [nn.Linear(4, 3)], model_ids=["linear"],
        fingerprint_value="pool-a",
    ).for_data(output_directory=tmp_path)
    device = torch.device("cpu")

    created = ResourceMonitor(device)
    _MeasuredPoolOutputs(pool, created).cached_outputs(
        "train_logits", loader, device, task="classification")
    assert created.to_dict()["cache_reuse"]["pool_outputs"] == "none"
    assert (tmp_path / "train_logits.resources.json").exists()

    reused = ResourceMonitor(device)
    _MeasuredPoolOutputs(pool, reused).cached_outputs(
        "train_logits", loader, device, task="classification")
    resources = reused.to_dict()
    assert resources["cache_reuse"]["pool_outputs"] == "complete"
    assert resources["reused"][0]["name"].endswith("/train_logits")


def test_feddes_uses_rigfls_official_validation_split():
    train = TensorDataset(torch.randn(12, 4), torch.randint(0, 3, (12,)))
    validation = TensorDataset(torch.randn(5, 4), torch.randint(0, 3, (5,)))
    train_loader = DataLoader(train, batch_size=4)
    validation_loader = DataLoader(validation, batch_size=5)
    algorithm = FedDES(
        FedDESConfig(
            graphroute={"graph": {"pool_calibrate": False}}, cache_dir=""
        ),
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
            graphroute={
                "base": {"epochs": 1, "oof_folds": 2},
                "graph": {"k": 2, "pool_calibrate": False},
                "gnn": {"arch": "mlp", "epochs": 1, "patience": 1,
                        "hidden_dim": 8},
            },
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
    from rigfl.experiment.config import result_filename, run_fingerprint
    from rigfl.experiment.registry import config_class

    exp = resolved_experiment(rounds=2)
    Cfg = config_class("feddes")
    a = Cfg(cache_dir="/scratch/a").model_dump()
    b = Cfg(cache_dir="/scratch/b").model_dump()
    other = Cfg(
        cache_dir="/scratch/a", graphroute={"graph": {"k": 9}}
    ).model_dump()

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
