"""Experiment-level configuration and run identity."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Annotated, Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)


def fingerprint(config: dict) -> str:
    """Short stable hash of a resolved config -> run identity (unique filenames, dedup)."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]


def hashable(v):
    """A dict value usable in a set or as part of a key."""
    if isinstance(v, dict):
        return tuple(sorted((k, hashable(value)) for k, value in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(hashable(value) for value in v)
    return v


_DATA_FIELDS = (
    "data_backend",
    "partition_scheme",
    "num_clients",
    "num_classes",
    "validation_fraction",
    "input_kind",
    "input_spec",
)


def result_data_configuration(record: dict) -> dict:
    """Return the seed-independent data configuration recorded by a run."""
    experiment = record.get("config", {}).get("experiment", {})
    dataset = experiment.get("dataset")

    partition = record.get("partition", {})
    if "generated" in partition:
        settings = deepcopy(partition["generated"].get("settings", {}))
        partition_settings = settings.get("partition", {})
        partition_settings.pop("partition_seed", None)
        partition_settings.pop("split_seed", None)
        source = {"backend": "flower", "settings": settings}
    elif "biosilo" in partition:
        settings = deepcopy(partition["biosilo"].get("settings", {}))
        settings.pop("seed", None)
        source = {
            "backend": "biosilo",
            "dataset": partition["biosilo"].get("dataset"),
            "settings": settings,
        }
    else:
        source = {
            "backend": experiment.get("data_backend"),
            "partition_scheme": experiment.get("partition_scheme"),
            "partition_id": experiment.get("partition_id"),
        }

    return {
        "dataset": str(dataset) if dataset is not None else None,
        "experiment": {field: experiment.get(field) for field in _DATA_FIELDS},
        "source": source,
    }


def result_data_configuration_id(record: dict) -> str:
    """Return the stable identity of a run's seed-independent data configuration."""
    return fingerprint(result_data_configuration(record))


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
    rounds: int = Field(100, ge=1)
    seed: int = Field(0, ge=0)
    partition_seed: int | None = Field(None, ge=0)
    split_seed: int | None = Field(None, ge=0)
    shared_dim: int = Field(512, ge=1)  # width for algorithms that align representations
    model: str = "fedavg_cnn"
    model_family: Optional[str] = None
    batch: int = Field(32, ge=1)
    eval_gap: int = Field(1, ge=1)
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    out_dir: str = "results"
    quiet: bool = True
    wandb: bool = False                             # log to Weights & Biases (needs rigfl[wandb])
    wandb_project: str = "rigfl"
    estimate_flops: bool = False

    #: When to stop early. Off by default: every round is recorded either way.
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)


Seed = Annotated[StrictInt, Field(ge=0)]


class ReplicateCondition(BaseModel):
    """One paired data-partition, validation-split, and training condition."""

    model_config = ConfigDict(extra="forbid")

    partition_seed: Seed
    split_seed: Seed
    experiment_seed: Seed


class SamplerConfig(BaseModel):
    """An Optuna sampler and its constructor arguments."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    class_path: str = Field(
        alias="class",
        min_length=1,
    )
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("class_path")
    @classmethod
    def _strip_class_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a Python class path")
        return value.strip()


class CategoricalDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["categorical"]
    values: Annotated[list[Any], Field(min_length=1)]


class IntegerDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["int"]
    low: StrictInt
    high: StrictInt
    step: Annotated[StrictInt, Field(ge=1)] = 1
    log: StrictBool = False

    @model_validator(mode="after")
    def _validate_range(self):
        if self.low > self.high:
            raise ValueError("needs integer low <= high")
        if self.log and "step" in self.model_fields_set:
            raise ValueError("cannot set both log: true and step")
        return self


class FloatDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["float"]
    low: StrictInt | StrictFloat
    high: StrictInt | StrictFloat
    step: Annotated[StrictInt | StrictFloat, Field(gt=0)] | None = None
    log: StrictBool = False

    @model_validator(mode="after")
    def _validate_range(self):
        if not self.low < self.high:
            raise ValueError("needs numeric low < high")
        if self.log and self.step is not None:
            raise ValueError("cannot set both log: true and step")
        return self


SearchDistribution = Annotated[
    CategoricalDistribution | IntegerDistribution | FloatDistribution,
    Field(discriminator="type"),
]


class BaseConfiguration(RootModel[dict[str, Any]]):
    """Fixed experiment and algorithm settings shared by generated runs."""

    @model_validator(mode="after")
    def _validate_sections(self):
        for name in ("experiment", "algorithm"):
            value = self.root.get(name)
            if value is not None and not isinstance(value, dict):
                raise ValueError(f"'{name}' must be a mapping")
        return self


class SweepConfiguration(
    RootModel[dict[str, list[Any] | tuple[Any, ...] | str]]
):
    """Cartesian sweep axes accepted from YAML or command-line flags."""

    @model_validator(mode="after")
    def _validate_axes(self):
        empty = [path for path, values in self.root.items() if not values]
        if empty:
            raise ValueError(
                "sweep axes must contain at least one value: " + ", ".join(empty)
            )
        return self


class AlgorithmConfiguration(RootModel[dict[str, Any]]):
    """Settings validated against the selected algorithm at execution time."""


class RunFileConfig(BaseModel):
    """The configuration file for one RigFL experiment."""

    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    algorithm: AlgorithmConfiguration = Field(
        default_factory=lambda: AlgorithmConfiguration({})
    )


class IntensificationConfig(BaseModel):
    """Evaluation of leading search candidates on additional conditions."""

    model_config = ConfigDict(extra="forbid")

    top_k: Annotated[StrictInt, Field(ge=2)] = 5
    replicates: Annotated[list[ReplicateCondition], Field(min_length=1)]
    practical_threshold: Annotated[StrictInt | StrictFloat, Field(gt=0)]
    tail_fraction: Annotated[StrictInt | StrictFloat, Field(gt=0, le=1)] = 0.10
    prefer: Literal["validation", "communication", "flops", "time"] = "validation"

    @model_validator(mode="after")
    def _validate_replicates(self):
        conditions = [condition.model_dump() for condition in self.replicates]
        identities = {tuple(condition.values()) for condition in conditions}
        if len(identities) != len(conditions):
            raise ValueError("replicates contains a duplicate")
        seeds = [condition.experiment_seed for condition in self.replicates]
        if len(set(seeds)) != len(seeds):
            raise ValueError("replicates must have distinct experiment_seed values")
        self.practical_threshold = float(self.practical_threshold)
        self.tail_fraction = float(self.tail_fraction)
        return self

    def to_dict(self) -> dict:
        return self.model_dump()


class TuningConfig(BaseModel):
    """Validation-based Optuna search settings."""

    model_config = ConfigDict(extra="forbid")

    search_space: Annotated[dict[str, SearchDistribution], Field(min_length=1)]
    trials: Annotated[StrictInt, Field(ge=1)] = 100
    sampler: SamplerConfig = Field(
        default_factory=lambda: SamplerConfig.model_validate(
            {"class": "optuna.samplers.TPESampler", "options": {"seed": 0}}
        )
    )
    metric: str = "accuracy"
    selection_view: Literal["global", "per-client"] = "global"
    selection_aggregation: Literal["mean", "weighted_mean"] = "mean"
    tie_break: Literal["earliest", "latest"] = "earliest"
    intensification: IntensificationConfig | None = None


class ExperimentFileConfig(BaseModel):
    """A RigFL sweep or Optuna-study file."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    algorithms: list[str] | str | None = None
    base: BaseConfiguration = Field(default_factory=lambda: BaseConfiguration({}))
    sweep: SweepConfiguration | None = None
    replicates: list[ReplicateCondition] | None = None
    tuning: TuningConfig | None = None

    @model_validator(mode="after")
    def _validate_replicates(self):
        if self.replicates is None:
            return self
        if not self.replicates:
            raise ValueError("replicates must be a nonempty list")
        identities = {
            tuple(condition.model_dump().values()) for condition in self.replicates
        }
        if len(identities) != len(self.replicates):
            raise ValueError("replicates contains a duplicate seed condition")
        seeds = [condition.experiment_seed for condition in self.replicates]
        if len(set(seeds)) != len(seeds):
            raise ValueError(
                "replicates must use a distinct experiment_seed for each condition"
            )
        return self

class ResolvedExperimentConfig(ExperimentConfig):
    """An experiment plus facts read from its selected dataset partition.

    The added fields are recorded in completed results and come from the
    resolved partition rather than user-authored experiment configuration.
    """

    data_backend: Literal["flower", "biosilo"]
    partition_id: str
    partition_scheme: str | None
    num_clients: int = Field(ge=1)
    num_classes: int = Field(ge=2)
    validation_fraction: float = Field(gt=0, lt=1)
    input_kind: str
    input_spec: dict[str, Any]
    resolved_models: list[str] = Field(min_length=1)


# ── Run identity ─────────────────────────────────────────────────────────────
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


def run_fingerprint(
    exp: "ResolvedExperimentConfig",
    algorithm_dump: dict,
    *,
    ignored_experiment_fields: tuple[str, ...] = (),
) -> str:
    """Run identity from resolved experiment and algorithm configurations."""
    if not isinstance(exp, ResolvedExperimentConfig):
        raise TypeError("run identity requires a resolved dataset partition")
    e = exp.model_dump()
    for k in _ENV_IRRELEVANT:
        e.pop(k, None)
    for k in ignored_experiment_fields:
        e.pop(k, None)
    for k in ("partition_seed", "split_seed"):
        if e.get(k) is None:
            e.pop(k, None)
    e.pop("model", None)
    e.pop("model_family", None)
    # Disabled stopping governs nothing, so its other settings must not mint a
    # second identity for the same run.
    e["early_stopping"] = normalize_early_stopping(e.get("early_stopping"))
    if not e.get("estimate_flops"):
        e.pop("estimate_flops", None)
    return fingerprint({"experiment": e,
                        "algorithm": algorithm_identity(algorithm_dump)})


def result_filename(exp: "ResolvedExperimentConfig", algorithm: str, fp: str) -> str:
    """Result filename containing the resolved data-partition identity."""
    return f"{exp.dataset}_{exp.partition_id}_{algorithm}_{fp}_seed{exp.seed}.json"
