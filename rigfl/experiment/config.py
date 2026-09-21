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

    enabled: bool = Field(
        False,
        description=(
            "Stop iterative training when validation performance stops improving; "
            "one-shot runners use algorithm-specific stopping settings."
        ),
    )
    # Early stopping is validation-only.
    split: Literal["validation"] = Field(
        "validation", description="Data split used for early stopping."
    )
    # The validator resolves an unset metric to loss only when stopping is enabled.
    metric: Optional[str] = Field(
        None, description="Validation metric to monitor; defaults to loss when enabled."
    )
    direction: Optional[Literal["maximize", "minimize"]] = Field(
        None,
        description="Improvement direction; inferred from the metric when omitted.",
    )
    aggregation: Literal["mean", "weighted_mean"] = Field(
        "mean", description="How client validation values are combined."
    )
    patience: int = Field(
        10, ge=1, description="Evaluations without improvement before stopping."
    )
    min_delta: float = Field(
        0.0, ge=0, description="Smallest change counted as an improvement."
    )

    @model_validator(mode="after")
    def _resolve_when_enabled(self):
        """Validate the control metric and record its resolved direction."""
        if not self.enabled:
            return self
        from rigfl.eval.metrics import direction_of, require_computable

        object.__setattr__(
            self, "metric", require_computable(self.metric or DEFAULT_METRIC)
        )
        if not self.direction:
            object.__setattr__(self, "direction", direction_of(self.metric))
        return self


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(
        "cifar10", description="Dataset name from the dataset configuration file."
    )
    dataset_config: str = Field(
        "configs/datasets.yaml", description="Path to the dataset configuration file."
    )
    data_dir: str = Field(
        "data", description="Directory containing generated client partitions."
    )
    rounds: int = Field(
        100,
        ge=1,
        description="Number of communication rounds for iterative runners.",
    )
    seed: int = Field(
        0, ge=0, description="Seed for training and model initialization."
    )
    partition_seed: int | None = Field(
        None, ge=0, description="Override for the dataset partition seed."
    )
    split_seed: int | None = Field(
        None, ge=0, description="Override for the client validation-split seed."
    )
    shared_dim: int = Field(
        512,
        ge=1,
        description="Shared representation width used by compatible algorithms.",
    )
    model: str = Field(
        "fedavg_cnn", description="Model architecture selected for the experiment."
    )
    model_family: Optional[str] = Field(
        None,
        description="Model family selected for algorithms that support different client architectures.",
    )
    batch: int = Field(32, ge=1, description="Client training batch size.")
    eval_gap: int = Field(
        1,
        ge=1,
        description="Evaluate every N communication rounds for iterative runners.",
    )
    device: Literal["auto", "cpu", "mps", "cuda"] = Field(
        "auto", description="Device used for training and evaluation."
    )
    out_dir: str = Field(
        "results", description="Root directory for experiment results."
    )
    quiet: bool = Field(True, description="Suppress per-round progress output.")
    wandb: bool = Field(False, description="Log training progress to Weights & Biases.")
    wandb_project: str = Field("rigfl", description="Weights & Biases project name.")
    estimate_flops: bool = Field(
        False, description="Estimate executed PyTorch operations."
    )

    #: When to stop early. Off by default: every round is recorded either way.
    early_stopping: EarlyStoppingConfig = Field(
        default_factory=EarlyStoppingConfig,
        description=(
            "Optional validation-based stopping policy for iterative runners; "
            "one-shot runners use their algorithm-specific stopping settings."
        ),
    )


Seed = Annotated[StrictInt, Field(ge=0)]


class ReplicateCondition(BaseModel):
    """One paired data-partition, validation-split, and training condition."""

    model_config = ConfigDict(extra="forbid")

    partition_seed: Seed = Field(description="Seed used to assign data to clients.")
    split_seed: Seed = Field(
        description="Seed used to create client validation splits."
    )
    experiment_seed: Seed = Field(
        description="Seed used for training and model initialization."
    )


def replicate_conditions_from_count(count: int, *, start: int = 0) -> list[dict]:
    """Expand a replicate count into matched seed conditions."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("a replicate count must be a positive integer")
    return [
        {"partition_seed": seed, "split_seed": seed, "experiment_seed": seed}
        for seed in range(start, start + count)
    ]


class SamplerConfig(BaseModel):
    """An Optuna sampler and its constructor arguments."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    class_path: str = Field(
        alias="class",
        min_length=1,
        description="Import path of the Optuna sampler class.",
    )
    options: dict[str, Any] = Field(
        default_factory=dict, description="Arguments passed to the sampler constructor."
    )

    @field_validator("class_path")
    @classmethod
    def _strip_class_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a Python class path")
        return value.strip()


class CategoricalDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["categorical"]
    values: Annotated[list[Any], Field(min_length=1, description="Candidate values.")]


class IntegerDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["int"]
    low: StrictInt = Field(description="Inclusive lower bound.")
    high: StrictInt = Field(description="Inclusive upper bound.")
    step: Annotated[
        StrictInt, Field(ge=1, description="Spacing between candidate values.")
    ] = 1
    log: StrictBool = Field(False, description="Sample on a logarithmic scale.")

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
    low: StrictInt | StrictFloat = Field(description="Inclusive lower bound.")
    high: StrictInt | StrictFloat = Field(description="Inclusive upper bound.")
    step: Annotated[StrictInt | StrictFloat, Field(gt=0)] | None = Field(
        None, description="Spacing between candidate values."
    )
    log: StrictBool = Field(False, description="Sample on a logarithmic scale.")

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


class SweepConfiguration(RootModel[dict[str, list[Any] | tuple[Any, ...] | str]]):
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

    experiment: ExperimentConfig = Field(
        default_factory=ExperimentConfig, description="Experiment-wide settings."
    )
    algorithm: AlgorithmConfiguration = Field(
        default_factory=lambda: AlgorithmConfiguration({}),
        description="Settings for the selected algorithm.",
    )


class IntensificationConfig(BaseModel):
    """Evaluation of leading search candidates on additional conditions."""

    model_config = ConfigDict(extra="forbid")

    top_k: Annotated[
        StrictInt,
        Field(
            ge=2, description="Leading candidates evaluated on additional replicates."
        ),
    ] = 5
    replicates: Annotated[
        int | list[ReplicateCondition],
        Field(
            description=(
                "Additional replicate conditions, or a count continuing the "
                "top-level replicate seeds."
            )
        ),
    ]
    practical_threshold: Annotated[
        StrictInt | StrictFloat,
        Field(
            gt=0, description="Largest difference treated as practically equivalent."
        ),
    ]
    tail_fraction: Annotated[
        StrictInt | StrictFloat,
        Field(gt=0, le=1, description="Client tail used for worst-tail gain."),
    ] = 0.10
    prefer: Literal["validation", "communication", "flops", "time"] = Field(
        "validation",
        description="Criterion used among practically equivalent candidates.",
    )
    ranking: Literal["pooled", "intensification"] = Field(
        "pooled",
        description=(
            "Replicates a shortlisted candidate is ranked on: the screening and "
            "intensification replicates together, or intensification only."
        ),
    )

    @model_validator(mode="after")
    def _validate_replicates(self):
        if isinstance(self.replicates, int):
            replicate_conditions_from_count(self.replicates)
            return self
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

    search_space: Annotated[
        dict[str, SearchDistribution],
        Field(
            min_length=1, description="Parameters and distributions explored by Optuna."
        ),
    ]
    trials: Annotated[
        StrictInt, Field(ge=1, description="Target total number of trials.")
    ] = 100
    sampler: SamplerConfig = Field(
        default_factory=lambda: SamplerConfig.model_validate(
            {"class": "optuna.samplers.TPESampler", "options": {"seed": 0}}
        ),
        description="Optuna sampler and constructor options.",
    )
    metric: str = Field(
        "accuracy", description="Validation metric optimized by the study."
    )
    selection_view: Literal["global", "per-client"] = Field(
        "global",
        description="Whether reporting rounds are selected jointly or per client.",
    )
    selection_aggregation: Literal["mean", "weighted_mean"] = Field(
        "mean", description="How client validation values are combined."
    )
    tie_break: Literal["earliest", "latest"] = Field(
        "earliest", description="Round chosen when validation values tie."
    )
    intensification: IntensificationConfig | None = Field(
        None, description="Optional evaluation of leading candidates on new replicates."
    )


class ExperimentFileConfig(BaseModel):
    """A RigFL sweep or Optuna-study file."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(
        None, description="Name used for the study output directory."
    )
    algorithms: list[str] | str | None = Field(
        None, description="Algorithms included in the sweep or study."
    )
    base: BaseConfiguration = Field(
        default_factory=lambda: BaseConfiguration({}),
        description="Settings shared by every generated run.",
    )
    sweep: SweepConfiguration | None = Field(
        None, description="Cartesian axes for an ordinary sweep."
    )
    replicates: list[ReplicateCondition] | None = Field(
        None,
        description=(
            "Paired data and training seed conditions, or a count expanding to "
            "that many conditions with matched seeds from zero."
        ),
    )

    @field_validator("replicates", mode="before")
    @classmethod
    def _expand_replicate_count(cls, value):
        if isinstance(value, int) and not isinstance(value, bool):
            return replicate_conditions_from_count(value)
        return value
    tuning: TuningConfig | None = Field(None, description="Optuna study settings.")

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
    "device",
    "out_dir",
    "quiet",
    "wandb",
    "wandb_project",
    "dataset_config",
    "data_dir",
)
# Cache location does not define an algorithm configuration.
_ALGORITHM_ENV_IRRELEVANT = ("cache_dir",)


def algorithm_identity(algorithm_dump: dict) -> dict:
    """An algorithm's config, minus values that only locate stored artifacts."""
    return {
        k: v for k, v in algorithm_dump.items() if k not in _ALGORITHM_ENV_IRRELEVANT
    }


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
    return fingerprint(
        {"experiment": e, "algorithm": algorithm_identity(algorithm_dump)}
    )


def result_filename(exp: "ResolvedExperimentConfig", algorithm: str, fp: str) -> str:
    """Result filename containing the resolved data-partition identity."""
    return f"{exp.dataset}_{exp.partition_id}_{algorithm}_{fp}_seed{exp.seed}.json"
