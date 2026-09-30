"""Experiment-level configuration and run identity."""

from __future__ import annotations

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

from rigfl.experiment.identity import fingerprint
from rigfl.experiment.identity import (
    normalize_early_stopping as normalize_early_stopping,
)

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


#: What enabling early stopping means when no metric is named.
DEFAULT_METRIC = "loss"


class EarlyStoppingConfig(BaseModel):
    """Round-level early stopping; separate from reporting-time round selection."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        False,
        description=(
            "Stop training when validation performance stops improving."
        ),
    )
    metric: Optional[str] = Field(
        None, description="Validation metric to monitor; defaults to loss when enabled."
    )
    aggregation: Literal["uniform", "sample_count"] = Field(
        "uniform", description="How client validation values are combined."
    )
    patience: int = Field(
        10, ge=1, description="Evaluations without improvement before stopping."
    )
    min_delta: float = Field(
        0.0, ge=0, description="Smallest change counted as an improvement."
    )

    @model_validator(mode="after")
    def _resolve_when_enabled(self):
        """Validate the control metric."""
        if not self.enabled:
            return self
        from rigfl.eval.metrics import canonical

        object.__setattr__(self, "metric", canonical(self.metric or DEFAULT_METRIC))
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
    training_seed: int = Field(
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
        description=(
            "Representation width used by algorithms that exchange "
            "representations (FedProto, FedGH, pFedMoE, FedTGP); ignored by "
            "all others."
        ),
    )
    model_arch: str = Field(
        "fedavg_cnn",
        description=(
            "Architecture name given to every client, or a model-family name "
            "assigning the family's architectures to clients. Algorithms that "
            "require identical client architectures accept only an "
            "architecture name."
        ),
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
    results_root: str = Field(
        "results", description="Base directory for runs, reports, and study files."
    )
    quiet: bool = Field(True, description="Suppress per-round progress output.")
    wandb: bool = Field(False, description="Log training progress to Weights & Biases.")
    wandb_project: str = Field("rigfl", description="Weights & Biases project name.")
    estimate_flops: bool = Field(
        False, description="Estimate FLOPs for supported PyTorch operators."
    )

    early_stopping: EarlyStoppingConfig = Field(
        default_factory=EarlyStoppingConfig,
        description=(
            "Optional validation-based stopping policy."
        ),
    )


def experiment_fields(values: dict) -> dict:
    """The entries of ``values`` that are user-settable experiment fields."""
    return {k: v for k, v in values.items() if k in ExperimentConfig.model_fields}


Seed = Annotated[StrictInt, Field(ge=0)]


class ReplicateCondition(BaseModel):
    """One paired data-partition, validation-split, and training condition."""

    model_config = ConfigDict(extra="forbid")

    partition_seed: Seed = Field(description="Seed used to assign data to clients.")
    split_seed: Seed = Field(
        description="Seed used to create client validation splits."
    )
    training_seed: Seed = Field(
        description="Seed used for training and model initialization."
    )


def replicate_conditions_from_count(count: int, *, start: int = 0) -> list[dict]:
    """Expand a replicate count into matched seed conditions."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("a replicate count must be a positive integer")
    return [
        {"partition_seed": seed, "split_seed": seed, "training_seed": seed}
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
    """Fixed settings shared by generated runs."""


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
        "accuracy",
        description=(
            "Validation metric optimized by the study: accuracy, balanced_accuracy, "
            "macro_f1, auroc, or auprc."
        ),
    )
    round_selection: Literal["shared", "client-specific"] = Field(
        "shared",
        description="Whether reporting uses one shared round or client-specific rounds.",
    )
    client_weighting: Literal["uniform", "sample_count"] = Field(
        "uniform", description="How clients contribute to reported metrics."
    )
class ExperimentFileConfig(BaseModel):
    """A RigFL sweep or Optuna-study file."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(
        None, description="Name used for the study output directory."
    )
    algorithm: list[str] | str | None = Field(
        None, description="Algorithm values included in the sweep or study."
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
        seeds = [condition.training_seed for condition in self.replicates]
        if len(set(seeds)) != len(seeds):
            raise ValueError(
                "replicates must use a distinct training_seed for each condition"
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

    @model_validator(mode="after")
    def _early_stopping_metric_fits_task(self):
        if self.early_stopping.enabled:
            from rigfl.eval.metrics import require_task_metric

            require_task_metric(self.early_stopping.metric, self.num_classes)
        return self


def result_filename(exp: "ResolvedExperimentConfig", algorithm: str, fp: str) -> str:
    """Result filename containing the resolved data-partition identity."""
    return f"{exp.dataset}_{exp.partition_id}_{algorithm}_{fp}_seed{exp.training_seed}.json"
