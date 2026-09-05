"""FedDES -- Federated Diverse Ensemble Selection (Mueller & Street).

FedDES uses RigFL's peer-to-peer one-shot lifecycle:
    prepare                  each client trains its local classifier pool.
    one_shot_communication   the local pools are shared with every client once.
    local_computation        each client independently builds its graph and
                             trains its complete GNN meta-learner.

There is no iterative local-training/server-aggregation loop. Each GNN retains
the epoch selected by its own validation split, and RigFL evaluates those final
per-client models once.

Per-client GraphRoute state is stored in ``ctx.client_state``. GraphRoute remains
an optional dependency for users running other algorithms.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Literal

import torch
from torch.utils.data import DataLoader, Dataset

from pydantic import Field

from rigfl.core.config import AlgorithmConfig
from rigfl.core.interfaces import Algorithm, LocalSelection, OneShotContext
from rigfl.prediction import Predictions
from rigfl.data.builder import _collate    # multi-input-safe collate (works for single-input too)

FEDDES_PREPROCESSING_KEY = "rigfl-feddes-multitensor-collation-v1"


class FedDESConfig(AlgorithmConfig):
    # FedDES trains base classifiers + a GNN meta-learner; it does not use the
    # local_epochs/lr settings used by algorithms with a client training loop.
    gnn_arch: Literal["gat", "graph_gps", "mlp"] = "gat"
    base_lr: float = Field(5e-4, gt=0)
    base_epochs: int = Field(100, ge=1)
    graph_k: int = Field(5, ge=1)
    hidden_dim: int = Field(128, ge=1)
    gnn_epochs: int = Field(500, ge=1)
    gnn_patience: int = Field(50, ge=1)
    calibrate: bool = True
    # OOF stacking produces training meta-labels from models that did not see the
    # corresponding rows. In-sample mode is cheaper but measures training fit.
    base_split_mode: Literal["oof_stacking", "in_sample"] = "oof_stacking"
    base_oof_folds: int = Field(3, ge=2)
    cache_dir: str = "pool_cache"     # reuse trained base pools across graph/GNN sweeps ("" disables)


class FedDES(Algorithm):
    def __init__(self, config: FedDESConfig,
                 base_factories: list[torch.nn.Module | Callable[[], torch.nn.Module]],
                 num_classes: int, *, data_id: str | None = None,
                 model_ids: list[str] | None = None, seed: int = 0,
                 validation_fraction: float = 0.2):
        super().__init__(config)
        sources = list(base_factories)
        # Configuration resolves to actual model templates. A few low-level test
        # integrations still supply constructors, so materialize those once and
        # use the resulting templates for both identity and isolated training.
        self.base_models = tuple(
            source if isinstance(source, torch.nn.Module) else source()
            for source in sources
        )
        self.base_factories = [
            lambda template=model: copy.deepcopy(template)
            for model in self.base_models
        ]
        self.num_classes = num_classes
        self.gnn_arch = config.gnn_arch
        self.base_lr, self.base_epochs = config.base_lr, config.base_epochs
        self.graph_k, self.hidden_dim = config.graph_k, config.hidden_dim
        self.calibrate = config.calibrate
        self.base_split_mode = config.base_split_mode
        self.base_oof_folds = config.base_oof_folds
        self.cache_dir = config.cache_dir or None
        self.data_id, self.seed = data_id, seed
        self.validation_fraction = validation_fraction
        self.model_ids = model_ids
        if self.cache_dir and not self.data_id:
            raise ValueError("FedDES pool reuse requires a stable data_id.")
        if self.cache_dir and not self.model_ids:
            raise ValueError("FedDES pool reuse requires stable model_ids.")
        if self.model_ids and len(self.model_ids) != len(self.base_models):
            raise ValueError("model_ids must name every base model in order.")
        if self.model_ids is None:
            self.model_ids = [f"model_{i}" for i in range(len(base_factories))]
        self.gnn_epochs, self.gnn_patience = config.gnn_epochs, config.gnn_patience

    @classmethod
    def from_config(cls, config, *, experiment, base_pool=None,
                    model_input_spec=None, **resources):
        from rigfl.core.adapters import LearnedProjection
        from rigfl.models.registry import (instantiate_models,
                                           resolve_model_architectures)

        data_id = f"{experiment.dataset}-{experiment.partition_id}"

        input_kind = model_input_spec["input_kind"] if model_input_spec else experiment.input_kind
        model_ids = resolve_model_architectures(
            architecture_family=experiment.model_architecture_family,
            architectures=experiment.model_architectures,
            input_kind=input_kind,
        )
        if base_pool is None:
            if model_input_spec is None:
                if experiment.input_kind == "temporal":
                    raise ValueError(
                        "FedDES temporal models require model_input_spec with "
                        "n_ts, n_static, and seq_len."
                    )
                model_input_spec = {
                    "input_kind": "image", "shape": (3, 32, 32)
                }
            base_pool = instantiate_models(
                model_ids,
                num_classes=experiment.num_classes,
                input_spec=model_input_spec,
                shared_dim=experiment.shared_dim,
                adapter=lambda native, shared: LearnedProjection(native, shared),
            )
        return cls(
            config,
            base_pool,
            experiment.num_classes,
            data_id=data_id,
            model_ids=model_ids,
            seed=experiment.seed,
            validation_fraction=experiment.validation_fraction,
        )

    def prepare(self, model, train_loader, ctx: OneShotContext):
        """Train one client's local base pool and return its outgoing payload."""
        del model
        if ctx.validation_loader is None:
            raise ValueError("FedDES requires each client's official validation split.")
        st = ctx.client_state
        st["train_dataset"] = train_loader.dataset
        st["validation_dataset"] = ctx.validation_loader.dataset
        st["local_pool"] = self._train_or_load_pool(
            st["train_dataset"], st["validation_dataset"],
            ctx.device, ctx.client_id)
        return st["local_pool"]

    def one_shot_communication(self, outgoing: list):
        """Share the ordered union of all local pools with every client once."""
        shared_pool = self._union_pools(outgoing)
        return [shared_pool for _ in outgoing]

    # ── decision space: pool probabilities [N,M*C] + per-classifier hard preds [N,M] ──
    def _splice_oof(self, tr_logits, pool, st, client_id):
        """Replace this client's own columns with their out-of-fold logits.

        The global pool is every client's classifiers concatenated, and only the
        slice this client contributed was fit on tr_ds -- the others already
        predict it out-of-sample, since the rows are not their data. Explicit
        client/model IDs locate the local columns in the ordered global pool.
        """
        local_pool = st.get("local_pool")
        oof = None if local_pool is None else local_pool.load_oof()
        if oof is None:
            return tr_logits
        prefix = f"client_{client_id}/"
        cols = [j for j, model_id in enumerate(pool.model_ids)
                if model_id.startswith(prefix)]
        if len(cols) != oof.shape[1] or oof.shape[0] != tr_logits.shape[0]:
            print(f"[FedDES][warn] cannot place OOF logits (matched {len(cols)} of "
                  f"{oof.shape[1]} own classifiers, {oof.shape[0]} vs "
                  f"{tr_logits.shape[0]} rows); using in-sample meta-labels.")
            return tr_logits
        tr_logits[:, cols, :] = oof.to(tr_logits.device, tr_logits.dtype)
        return tr_logits

    # ── base-pool artifact reuse (train once; reuse across graph/GNN sweeps) ──
    def _pool_fp(self) -> str:
        """Fingerprint the ordered local pool and its complete training policy."""
        from graphroute.pool_cache import fingerprint_model, fingerprint_pool
        from rigfl.experiment.env import _package
        template_fingerprints = [
            fingerprint_model(model) for model in self.base_models
        ]
        return fingerprint_pool(
            model_ids=self.model_ids,
            model_fingerprints=template_fingerprints,
            base_config={
                "task": "classification", "num_classes": self.num_classes,
                "split_mode": self.base_split_mode,
                "oof_folds": self.base_oof_folds,
                "lr": self.base_lr, "epochs": self.base_epochs,
                "batch_size": 64, "patience": 20, "optimizer": "Adam",
                "weight_decay": 5e-4, "weighted_by_class": True,
                "es_metric": "val_loss", "inner_val_ratio": 0.2,
                "client_validation_fraction": self.validation_fraction,
                "seed_policy": "experiment_seed_plus_client_id",
                "preprocessing": FEDDES_PREPROCESSING_KEY,
            },
            seed=self.seed,
            code_identity={"graphroute": _package("graphroute"),
                           "rigfl": _package("rigfl")})

    def _train(self, tr_ds, va_ds, device, client_id):
        """Return the trained models and optional out-of-fold logits."""
        from graphroute.run import seed_everything
        seed_everything(self.seed + int(client_id))
        if self.base_split_mode == "oof_stacking":
            from graphroute.pool import train_pool_oof
            models, oof_logits, _ = train_pool_oof(
                self.base_factories, tr_ds, va_ds, device,
                n_folds=self.base_oof_folds, num_classes=self.num_classes,
                lr=self.base_lr, max_epochs=self.base_epochs,
                seed=self.seed + int(client_id), collate_fn=_collate)
            return models, oof_logits          # [N_tr, M_local, C], row i unseen by its predictor
        from graphroute.pool import train_pool
        return train_pool(self.base_factories, tr_ds, va_ds, device,
                          num_classes=self.num_classes, lr=self.base_lr, max_epochs=self.base_epochs,
                          collate_fn=_collate), None   # keep multi-input (ts, static) as a MultiTensor

    def _train_or_load_pool(self, tr_ds, va_ds, device, client_id):
        """Load or train this client's pool for reuse across graph/GNN sweeps."""
        def train():
            return self._train(tr_ds, va_ds, device, client_id)

        if not self.cache_dir:
            from graphroute.pool_cache import in_memory_pool
            models, oof = train()
            artifact = in_memory_pool(
                models, model_ids=self.model_ids,
                fingerprint_value=self._pool_fp())
            artifact.oof_logits = oof
            return artifact
        from pathlib import Path

        from graphroute.pool_cache import cached_pool
        fp = self._pool_fp()
        print(f"[FedDES] base pool {fp} (client {client_id})")
        directory = (Path(self.cache_dir) / self.data_id / f"pool_{fp}"
                     / "clients" / f"client_{client_id}")
        return cached_pool(
            directory, self.base_factories, train,
            fingerprint_value=fp, model_ids=self.model_ids,
            require_oof=self.base_split_mode == "oof_stacking",
            data_id=self.data_id)

    def _graphroute_config(self, client_id, device):
        from graphroute.config import GraphRouteConfig
        return GraphRouteConfig(
            task="classification", loss_target="meta_labels",
            dataset=self.data_id or "federated-client",
            num_classes=self.num_classes, seed=self.seed + int(client_id),
            device=device.type,
            base={"split_mode": ("oof_stacking" if self.base_split_mode == "oof_stacking"
                                  else "split_train")},
            graph={"k": self.graph_k, "pool_calibrate": self.calibrate},
            gnn={"arch": self.gnn_arch, "hidden_dim": self.hidden_dim,
                 "epochs": self.gnn_epochs, "patience": self.gnn_patience,
                 "es_metric": "val_acc",
                 "ens_combination_mode": "hard_weighted_voting",
                 "voting_weight_space": "sig"})

    # ── post-communication local computation: train one complete local GNN ──
    def local_computation(self, model, pool, loader, ctx: OneShotContext):
        del model, loader
        device, st = ctx.device, ctx.client_state
        if pool is None:
            raise RuntimeError("FedDES communication did not provide a classifier pool.")
        train_dataset = st["train_dataset"]
        validation_dataset = st["validation_dataset"]
        tr_loader = DataLoader(train_dataset, 256, shuffle=False, collate_fn=_collate)
        output_dir = (pool.directory / "outputs" / f"client_{ctx.client_id}"
                      if pool.directory is not None else None)
        client_pool = pool.for_data(output_directory=output_dir)
        train_logits = client_pool.cached_outputs(
            self._output_name("train"),
            tr_loader, device, task="classification",
            transform=lambda value: self._splice_oof(
                value, pool, st, ctx.client_id))
        client_pool = client_pool.for_data(
            output_directory=output_dir, training_outputs=train_logits)

        from graphroute.run import fit_graphroute
        st["graphroute_model"] = fit_graphroute(
            self._graphroute_config(ctx.client_id, device), train_dataset,
            validation_set=validation_dataset, pool=client_pool,
            collate_fn=_collate)
        training = st["graphroute_model"].history
        return LocalSelection(
            selected_step=int(training["best_epoch"]),
            metric="accuracy",
            validation_value=float(training["best_metric"]),
        )

    @staticmethod
    def _output_name(split: str) -> str:
        return f"{split}_logits"

    def _union_pools(self, uploads: list):
        """Combine the prepared client pools in deterministic client/model order."""
        from graphroute.pool_cache import PoolArtifact, fingerprint
        model_ids, factories, paths, models = [], [], [], []
        can_hold_models = all(artifact.models is not None for artifact in uploads)
        for cid, artifact in enumerate(uploads):
            model_ids.extend(f"client_{cid}/{name}" for name in artifact.model_ids)
            factories.extend(artifact.model_factories)
            paths.extend(artifact.model_paths)
            if can_hold_models:
                models.extend(artifact.models)
        root = (None if not self.cache_dir else
                Path(self.cache_dir) / self.data_id / f"pool_{self._pool_fp()}")
        shared_fingerprint = fingerprint({"ordered_members": model_ids,
                                          "local_pool": self._pool_fp()})
        if root is not None:
            _write_manifest(root, shared_fingerprint, model_ids, self.data_id)
        return PoolArtifact(
            fingerprint=shared_fingerprint,
            model_ids=tuple(model_ids), model_factories=tuple(factories),
            model_paths=tuple(paths), directory=root,
            output_directory=None, models=models if can_hold_models else None)

    # prediction: connect the query batch into the train graph, run the GNN, ensemble-select
    def predict(self, client, x, shared) -> Predictions:
        st = client.state
        if "graphroute_model" not in st:
            raise RuntimeError(
                "FedDES cannot predict before its local GNN has been trained.")

        predicted = st["graphroute_model"].predict(
            _BatchDataset(x), split="batch", cache_outputs=False)
        return Predictions.from_probabilities(
            predicted["probabilities"], labels=predicted["predictions"])

# ── small helpers ────────────────────────────────────────────────────────────
class _BatchDataset(Dataset):
    """Wrap a query batch as a (labelless) Dataset so it goes through the pool.
    Handles single-input ``x`` (a tensor) and multi-input ``x`` (a MultiTensor /
    tuple of tensors, e.g. eICU's ``(ts, static)``) -- indexing samples, not fields."""
    def __init__(self, x):
        self.x = x
        self.multi = isinstance(x, tuple)          # MultiTensor is a tuple subclass
    def __len__(self):
        return len(self.x[0]) if self.multi else len(self.x)
    def __getitem__(self, i):
        return (tuple(f[i] for f in self.x) if self.multi else self.x[i]), 0


def _write_manifest(directory: Path, pool_fingerprint: str,
                    model_ids: list[str], data_id: str) -> None:
    """Record the deterministic order of the federated pool."""
    import json
    import os

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    value = {"schema_version": 2, "data_id": data_id,
             "fingerprint": pool_fingerprint, "model_ids": model_ids}
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"FedDES pool manifest does not match {path}")
        return
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
