"""Experiment-level configuration and run identity."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


def fingerprint(config: dict) -> str:
    """Short stable hash of a resolved config -> run identity (unique filenames, dedup)."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]


def hashable(v):
    """A dict value usable in a set or as part of a key."""
    return tuple(v) if isinstance(v, list) else v


def normalize_early_stopping(es: dict | None) -> dict:
    """Normalize inactive stopping policies to ``{"enabled": False}``."""
    es = es or {}
    if not es.get("enabled"):
        return {"enabled": False}
    return {k: hashable(v) for k, v in es.items() if v is not None}


#: What enabling early stopping means when no metric is named.
DEFAULT_METRIC = "loss"


class EarlyStoppingConfig(BaseModel):
    """Round-level early stopping, independent of result selection.

    Stopping decides when to stop spending compute; collection-time selection
    decides which completed round gets reported. Stopping is disabled by default
    and uses predictive validation loss when enabled unless overridden.
    """
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # Early stopping is validation-only.
    split: Literal["validation"] = "validation"
    # The validator resolves an unset metric to loss only when stopping is enabled.
    metric: Optional[str] = None
    direction: Optional[Literal["maximize", "minimize"]] = None   # None -> the metric's own
    aggregation: Literal["mean", "weighted_mean"] = "mean"
    patience: int = Field(10, ge=1)
    min_delta: float = Field(0.0, ge=0)

    @model_validator(mode="after")
    def _resolve_when_enabled(self):
        """Validate the control metric and record its resolved direction."""
        if not self.enabled:
            return self
        from rigfl.eval.metrics import direction_of, require_computable
        object.__setattr__(self, "metric",
                           require_computable(self.metric or DEFAULT_METRIC))
        if not self.direction:
            object.__setattr__(self, "direction", direction_of(self.metric))
        return self


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = "cifar10"
    dataset_config: str = "configs/datasets.yaml"
    data_dir: str = "data"
    alpha: float = Field(0.1, gt=0)                 # Dirichlet concentration
    num_clients: int = Field(20, ge=1)
    num_classes: int = Field(10, ge=2)
    rounds: int = Field(100, ge=1)
    seed: int = Field(0, ge=0)
    shared_dim: int = Field(512, ge=1)              # common representation width for the adapters
    model_architecture_family: Optional[str] = None
    model_architectures: Optional[list[str]] = None
    train_per_client: int = Field(2000, ge=1)
    test_per_client: int = Field(500, ge=1)
    val_frac: float = Field(0.2, gt=0, lt=1)
    batch: int = Field(32, ge=1)
    eval_gap: int = Field(1, ge=1)
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    out_dir: str = "results"
    quiet: bool = True
    wandb: bool = False                             # log to Weights & Biases (needs rigfl[wandb])
    wandb_project: str = "rigfl"

    # ``generated`` is resolved internally after loading the partition selected by
    # ``dataset``. ``natural`` retains the existing external-partition path.
    #: When to stop early. Off by default: every round is recorded either way.
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)

    scheme: Literal["dirichlet", "generated", "natural"] = "dirichlet"
    partition: Optional[str] = None                 # BioSilo partition id (natural scheme)
    data_root: Optional[str] = None                 # BIOSILO_DATA_ROOT override (natural scheme)

    @model_validator(mode="after")
    def _validate_data_and_models(self):
        """Validate natural-partition identity and client-model selection."""
        if self.scheme == "natural" and not (self.partition or "").strip():
            raise ValueError(
                "scheme='natural' requires an explicit partition id: a natural "
                "run is identified by the partition it loaded, and leaving it "
                "unset records the run under 'None' whatever BioSilo resolves. "
                "Set experiment.partition to the generated partition's id.")
        if (self.model_architecture_family is not None
                and self.model_architectures is not None):
            raise ValueError(
                "Set model_architecture_family or model_architectures, not both.")
        return self


# ── Scheme-aware run identity ────────────────────────────────────────────────
# Generated runs are identified by the generated partition fingerprint. Natural
# runs retain their external partition identity. Derived or inapplicable fields
# are excluded from the broader experiment fingerprint.
_NATURAL_IRRELEVANT = ("alpha", "num_clients", "num_classes", "train_per_client", "test_per_client")
_GENERATED_IRRELEVANT = (*_NATURAL_IRRELEVANT, "val_frac", "data_root")
_DIRICHLET_IRRELEVANT = ("partition", "data_root")
# Execution and output settings do not define the experimental condition.
_ENV_IRRELEVANT = (
    "device", "out_dir", "quiet", "wandb", "wandb_project",
    "dataset_config", "data_dir",
)
# Cache location does not define an algorithm configuration.
_ALGORITHM_ENV_IRRELEVANT = ("cache_dir",)


def algorithm_identity(algorithm_dump: dict) -> dict:
    """An algorithm's config, minus values that only locate stored artifacts."""
    return {k: v for k, v in algorithm_dump.items()
            if k not in _ALGORITHM_ENV_IRRELEVANT}


def run_fingerprint(exp: "ExperimentConfig", algorithm_dump: dict) -> str:
    """Run identity, ignoring fields irrelevant to the partition scheme + run env."""
    e = exp.model_dump()
    scheme = e.get("scheme")
    irrelevant = (_GENERATED_IRRELEVANT if scheme == "generated" else
                  _NATURAL_IRRELEVANT if scheme == "natural" else
                  _DIRICHLET_IRRELEVANT)
    for k in irrelevant:
        e.pop(k, None)
    for k in _ENV_IRRELEVANT:
        e.pop(k, None)
    # Disabled stopping governs nothing, so its other settings must not mint a
    # second identity for the same run.
    e["early_stopping"] = normalize_early_stopping(e.get("early_stopping"))
    return fingerprint({"experiment": e,
                        "algorithm": algorithm_identity(algorithm_dump)})


def result_filename(exp: "ExperimentConfig", algorithm: str, fp: str) -> str:
    """Result filename containing the resolved data-partition identity."""
    if exp.scheme in {"generated", "natural"}:
        return f"{exp.dataset}_{exp.partition}_{algorithm}_seed{exp.seed}_{fp}.json"
    return f"{exp.dataset}_a{exp.alpha}_{algorithm}_seed{exp.seed}_{fp}.json"
