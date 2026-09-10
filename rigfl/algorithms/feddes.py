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
from typing import Callable

import torch
from graphroute.config import GraphRouteConfig, GraphRouteSettings
from pydantic import Field, field_validator, model_validator
from torch.utils.data import DataLoader, Dataset

from rigfl.core.config import AlgorithmConfig
from rigfl.core.interfaces import Algorithm, LocalSelection, OneShotContext
from rigfl.data.builder import _collate
from rigfl.eval.resources import (
    load_cached_measurement,
    payload_bytes,
    write_cached_measurement,
)
from rigfl.experiment.paths import deep_merge
from rigfl.prediction import Predictions

FEDDES_PREPROCESSING_KEY = "rigfl-feddes-multitensor-collation-v1"
_OOF_INNER_VAL_RATIO = 0.2


def _default_graphroute_settings() -> GraphRouteSettings:
    return GraphRouteSettings(
        base={"oof_folds": 3, "epochs": 100, "batch_size": 64},
        gnn={
            "epochs": 500,
            "patience": 50,
            "es_metric": "val_acc",
            "ens_combination_mode": "hard_weighted_voting",
            "voting_weight_space": "sig",
        },
    )


class FedDESConfig(AlgorithmConfig):
    graphroute: GraphRouteSettings = Field(
        default_factory=_default_graphroute_settings
    )
    cache_dir: str = "pool_cache"

    @field_validator("graphroute", mode="before")
    @classmethod
    def _merge_graphroute_defaults(cls, value):
        if isinstance(value, GraphRouteSettings):
            return value
        if not isinstance(value, dict):
            return value
        return deep_merge(_default_graphroute_settings().model_dump(), value)

    @model_validator(mode="after")
    def _validate_feddes_settings(self):
        if self.graphroute.base.models is not None:
            raise ValueError(
                "FedDES base models are selected by the RigFL experiment model settings."
            )
        if self.graphroute.base.split_mode != "oof_stacking":
            raise ValueError("FedDES requires graphroute.base.split_mode='oof_stacking'.")
        return self


class FedDES(Algorithm):
    def __init__(self, config: FedDESConfig,
                 base_factories: list[torch.nn.Module | Callable[[], torch.nn.Module]],
                 num_classes: int, *, data_id: str | None = None,
                 model_ids: list[str] | None = None, seed: int = 0,
                 validation_fraction: float = 0.2,
                 feature_extractor: Callable | None = None):
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
        self.graphroute_settings = config.graphroute
        self.cache_dir = config.cache_dir or None
        self.data_id, self.seed = data_id, seed
        self.validation_fraction = validation_fraction
        self.feature_extractor = feature_extractor
        self.model_ids = model_ids
        if self.cache_dir and not self.data_id:
            raise ValueError("FedDES pool reuse requires a stable data_id.")
        if self.cache_dir and not self.model_ids:
            raise ValueError("FedDES pool reuse requires stable model_ids.")
        if self.model_ids and len(self.model_ids) != len(self.base_models):
            raise ValueError("model_ids must name every base model in order.")
        if self.model_ids is None:
            self.model_ids = [f"model_{i}" for i in range(len(base_factories))]

    @classmethod
    def from_config(cls, config, *, experiment, base_pool=None,
                    model_input_spec=None, **resources):
        from rigfl.data.features import graphroute_feature_extractor
        from rigfl.models.registry import instantiate_native_models

        data_id = f"{experiment.dataset}-{experiment.partition_id}"

        model_ids = experiment.resolved_models
        if base_pool is None:
            if model_input_spec is None:
                model_input_spec = {
                    "input_kind": "image", "shape": (3, 32, 32)
                }
            base_pool = instantiate_native_models(
                model_ids,
                num_classes=experiment.num_classes,
                input_spec=model_input_spec,
            )
        return cls(
            config,
            base_pool,
            experiment.num_classes,
            data_id=data_id,
            model_ids=model_ids,
            seed=experiment.seed,
            validation_fraction=experiment.validation_fraction,
            feature_extractor=graphroute_feature_extractor(
                config.graphroute.graph, model_input_spec
            ),
        )

    def prepare(self, model, train_loader, ctx: OneShotContext):
        """Train one client's local base pool and return its outgoing payload."""
        del model
        if ctx.validation_loader is None:
            raise ValueError("FedDES requires each client's official validation split.")
        st = ctx.client_state
        st["train_dataset"] = train_loader.dataset
        st["validation_dataset"] = ctx.validation_loader.dataset
        call = self._train_or_load_pool
        if ctx.resource_monitor is None:
            st["local_pool"] = call(
                st["train_dataset"], st["validation_dataset"],
                ctx.device, ctx.client_id)
        else:
            st["local_pool"] = call(
                st["train_dataset"], st["validation_dataset"],
                ctx.device, ctx.client_id, ctx.resource_monitor)
        return st["local_pool"]

    def communication_payload_bytes(self, payload, *, kind: str) -> int:
        return sum(payload_bytes(model) for model in self.base_models)

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
            raise RuntimeError("FedDES requires out-of-fold logits for its local pool.")
        prefix = f"client_{client_id}/"
        cols = [j for j, model_id in enumerate(pool.model_ids)
                if model_id.startswith(prefix)]
        if len(cols) != oof.shape[1] or oof.shape[0] != tr_logits.shape[0]:
            raise RuntimeError(
                "FedDES cannot align the local out-of-fold logits with the "
                f"shared pool (matched {len(cols)} of {oof.shape[1]} classifiers; "
                f"{oof.shape[0]} OOF rows for {tr_logits.shape[0]} training rows)."
            )
        tr_logits[:, cols, :] = oof.to(tr_logits.device, tr_logits.dtype)
        return tr_logits

    # ── base-pool artifact reuse (train once; reuse across graph/GNN sweeps) ──
    def _pool_fp(self) -> str:
        """Fingerprint the ordered local pool and its complete training policy."""
        from graphroute.pool_cache import fingerprint_model, fingerprint_pool
        from rigfl.experiment.env import _package
        base = self.graphroute_settings.base
        template_fingerprints = [
            fingerprint_model(model) for model in self.base_models
        ]
        return fingerprint_pool(
            model_ids=self.model_ids,
            model_fingerprints=template_fingerprints,
            base_config={
                "task": "classification", "num_classes": self.num_classes,
                **base.model_dump(exclude={"models"}),
                "inner_val_ratio": _OOF_INNER_VAL_RATIO,
                "client_validation_fraction": self.validation_fraction,
                "seed_policy": "experiment_seed_plus_client_id",
                "preprocessing": FEDDES_PREPROCESSING_KEY,
            },
            seed=self.seed,
            code_identity={"graphroute": _package("graphroute"),
                           "rigfl": _package("rigfl")})

    def _train(self, tr_ds, va_ds, device, client_id):
        """Return the trained models and out-of-fold logits."""
        from graphroute.pool import train_pool_oof
        from graphroute.run import seed_everything

        seed_everything(self.seed + int(client_id))
        base = self.graphroute_settings.base
        models, oof_logits, _ = train_pool_oof(
            self.base_factories, tr_ds, va_ds, device,
            n_folds=base.oof_folds,
            inner_val_ratio=_OOF_INNER_VAL_RATIO,
            batch_size=base.batch_size,
            max_epochs=base.epochs,
            patience=base.es_patience,
            lr=base.lr,
            optimizer_name=base.optimizer,
            weight_decay=base.weight_decay,
            task="classification",
            num_classes=self.num_classes,
            weighted_by_class=base.weighted_by_class,
            es_metric=base.es_metric,
            seed=self.seed + int(client_id),
            collate_fn=_collate,
        )
        return models, oof_logits

    def _train_or_load_pool(self, tr_ds, va_ds, device, client_id, monitor=None):
        """Load or train this client's pool for reuse across graph/GNN sweeps."""
        if not self.cache_dir:
            if monitor is not None:
                monitor.record_cache("base_pools", "disabled")
            from graphroute.pool_cache import in_memory_pool
            models, oof = self._train(tr_ds, va_ds, device, client_id)
            artifact = in_memory_pool(
                models, model_ids=self.model_ids,
                fingerprint_value=self._pool_fp())
            artifact.oof_logits = oof
            return artifact
        from pathlib import Path

        from filelock import FileLock
        from graphroute.pool_cache import cached_pool
        fp = self._pool_fp()
        print(f"[FedDES] base pool {fp} (client {client_id})")
        directory = (Path(self.cache_dir) / self.data_id / f"pool_{fp}"
                     / "clients" / f"client_{client_id}")
        resource_path = directory / "pool_resources.json"
        directory.mkdir(parents=True, exist_ok=True)
        built = False

        def train():
            nonlocal built
            built = True
            return self._train(tr_ds, va_ds, device, client_id)

        with FileLock(directory / ".resources.lock"):
            if monitor is None:
                artifact = cached_pool(
                    directory, self.base_factories, train,
                    fingerprint_value=fp, model_ids=self.model_ids,
                    require_oof=True,
                    data_id=self.data_id)
            else:
                with monitor.capture() as access:
                    artifact = cached_pool(
                        directory, self.base_factories, train,
                        fingerprint_value=fp, model_ids=self.model_ids,
                        require_oof=True,
                        data_id=self.data_id)
                monitor.record_cache("base_pools", "miss" if built else "hit")
                if built:
                    write_cached_measurement(
                        resource_path, fingerprint=fp,
                        measurement=monitor.artifact_measurement(access))
                else:
                    saved = load_cached_measurement(resource_path, fingerprint=fp)
                    monitor.add_reused(
                        f"base_pool/client_{client_id}", saved, access=access)
        return artifact

    def _graphroute_config(self, client_id, device):
        return GraphRouteConfig(
            **self.graphroute_settings.model_dump(),
            task="classification",
            dataset=self.data_id or "federated-client",
            num_classes=self.num_classes, seed=self.seed + int(client_id),
            device=device.type,
        )

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
        if ctx.resource_monitor is not None:
            client_pool = _MeasuredPoolOutputs(
                client_pool, ctx.resource_monitor)
        train_logits = client_pool.cached_outputs(
            self._output_name("train"),
            tr_loader, device, task="classification",
            transform=lambda value: self._splice_oof(
                value, pool, st, ctx.client_id))
        client_pool = client_pool.for_data(
            output_directory=output_dir, training_outputs=train_logits)
        if ctx.resource_monitor is not None:
            client_pool = _MeasuredPoolOutputs(
                client_pool, ctx.resource_monitor)

        from graphroute.run import fit_graphroute
        st["graphroute_model"] = fit_graphroute(
            self._graphroute_config(ctx.client_id, device), train_dataset,
            validation_set=validation_dataset, pool=client_pool,
            collate_fn=_collate, feature_extractor=self.feature_extractor)
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


class _MeasuredPoolOutputs:
    def __init__(self, pool, monitor):
        self._pool = pool
        self._monitor = monitor

    def __len__(self):
        return len(self._pool)

    def __getattr__(self, name):
        return getattr(self._pool, name)

    def cached_outputs(self, name, loader, device, *, task, transform=None):
        directory = self._pool.output_directory
        if directory is None:
            self._monitor.record_cache("pool_outputs", "disabled")
            return self._pool.cached_outputs(
                name, loader, device, task=task, transform=transform)

        from filelock import FileLock

        output_path = Path(directory) / f"{name}.pt"
        resource_path = output_path.with_suffix(".resources.json")
        fingerprint = f"{self._pool.fingerprint}:{name}"
        with FileLock(f"{output_path}.resources.lock"):
            hit = output_path.exists()
            with self._monitor.capture() as delta:
                value = self._pool.cached_outputs(
                    name, loader, device, task=task, transform=transform)
            self._monitor.record_cache("pool_outputs", "hit" if hit else "miss")
            if hit:
                saved = load_cached_measurement(
                    resource_path, fingerprint=fingerprint)
                self._monitor.add_reused(
                    f"pool_output/{Path(directory).name}/{name}", saved,
                    access=delta)
            else:
                write_cached_measurement(
                    resource_path, fingerprint=fingerprint,
                    measurement=self._monitor.artifact_measurement(delta))
        return value

# ── small helpers ────────────────────────────────────────────────────────────
class _BatchDataset(Dataset):
    """Wrap a query batch as a (labelless) Dataset so it goes through the pool.
    Handles both tensor and tuple inputs by indexing samples rather than fields."""
    def __init__(self, x):
        self.x = x
        self.multi = isinstance(x, tuple)
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
