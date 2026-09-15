# Experiment configuration reference

These options configure individual runs, sweeps, and tuning studies.

## Experiment settings

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `experiment.dataset` | string | `cifar10` | — | Dataset name from the dataset configuration file. |
| `experiment.dataset_config` | string | `configs/datasets.yaml` | — | Path to the dataset configuration file. |
| `experiment.data_dir` | string | `data` | — | Directory containing generated client partitions. |
| `experiment.rounds` | integer | `100` | ≥ 1 | Number of communication rounds. |
| `experiment.seed` | integer | `0` | ≥ 0 | Seed for training and model initialization. |
| `experiment.partition_seed` | integer \| null | `null` | ≥ 0 | Override for the dataset partition seed. |
| `experiment.split_seed` | integer \| null | `null` | ≥ 0 | Override for the client validation-split seed. |
| `experiment.shared_dim` | integer | `512` | ≥ 1 | Shared representation width used by compatible algorithms. |
| `experiment.model` | string | `fedavg_cnn` | — | Model architecture selected for the experiment. |
| `experiment.model_family` | string \| null | `null` | — | Model family selected for algorithms that support different client architectures. |
| `experiment.batch` | integer | `32` | ≥ 1 | Client training batch size. |
| `experiment.eval_gap` | integer | `1` | ≥ 1 | Evaluate every N communication rounds. |
| `experiment.device` | string | `auto` | `auto`, `cpu`, `mps`, `cuda` | Device used for training and evaluation. |
| `experiment.out_dir` | string | `results` | — | Root directory for experiment results. |
| `experiment.quiet` | boolean | `true` | — | Suppress per-round progress output. |
| `experiment.wandb` | boolean | `false` | — | Log training progress to Weights & Biases. |
| `experiment.wandb_project` | string | `rigfl` | — | Weights & Biases project name. |
| `experiment.estimate_flops` | boolean | `false` | — | Estimate executed PyTorch operations. |
| `experiment.early_stopping.enabled` | boolean | `false` | — | Stop training when validation performance stops improving. |
| `experiment.early_stopping.split` | string | `validation` | `validation` | Data split used for early stopping. |
| `experiment.early_stopping.metric` | string \| null | `null` | — | Validation metric to monitor; defaults to loss when enabled. |
| `experiment.early_stopping.direction` | string \| null | `null` | `maximize`, `minimize` | Improvement direction; inferred from the metric when omitted. |
| `experiment.early_stopping.aggregation` | string | `mean` | `mean`, `weighted_mean` | How client validation values are combined. |
| `experiment.early_stopping.patience` | integer | `10` | ≥ 1 | Evaluations without improvement before stopping. |
| `experiment.early_stopping.min_delta` | number | `0.0` | ≥ 0 | Smallest change counted as an improvement. |

## Sweep and study files

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `name` | string \| null | `null` | — | Name used for the study output directory. |
| `algorithms` | list[string] \| string \| null | `null` | — | Algorithms included in the sweep or study. |
| `base` | mapping | `{}` | — | Fixed experiment and algorithm settings shared by generated runs. |
| `sweep` | mapping \| null | `null` | — | Cartesian axes for an ordinary sweep. |
| `replicates` | list[replicate settings] \| null | `null` | — | Paired data and training seed conditions. |
| `tuning` | tuning settings \| null | `null` | — | Optuna study settings. |

## Replicate conditions

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `replicates[].partition_seed` | integer | required | ≥ 0 | Seed used to assign data to clients. |
| `replicates[].split_seed` | integer | required | ≥ 0 | Seed used to create client validation splits. |
| `replicates[].experiment_seed` | integer | required | ≥ 0 | Seed used for training and model initialization. |

Replicate entries must be unique and use distinct `experiment_seed` values. Replicate seed fields cannot also be sweep axes.

## Tuning settings

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `tuning.search_space` | mapping[string, value] | required | — | Parameters and distributions explored by Optuna. |
| `tuning.trials` | integer | `100` | ≥ 1 | Target total number of trials. |
| `tuning.metric` | string | `accuracy` | — | Validation metric optimized by the study. |
| `tuning.selection_view` | string | `global` | `global`, `per-client` | Whether reporting rounds are selected jointly or per client. |
| `tuning.selection_aggregation` | string | `mean` | `mean`, `weighted_mean` | How client validation values are combined. |
| `tuning.tie_break` | string | `earliest` | `earliest`, `latest` | Round chosen when validation values tie. |

## Sampler settings

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `tuning.sampler.class` | string | `optuna.samplers.TPESampler` | non-empty | Import path of the Optuna sampler class. |
| `tuning.sampler.options` | mapping | `{"seed":0}` | — | Arguments passed to the sampler constructor. |

## Intensification settings

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `tuning.intensification.top_k` | integer | `5` | ≥ 2 | Leading candidates evaluated on additional replicates. |
| `tuning.intensification.replicates` | list[replicate settings] | required | non-empty | Additional replicate conditions. |
| `tuning.intensification.practical_threshold` | number | required | > 0 | Largest difference treated as practically equivalent. |
| `tuning.intensification.tail_fraction` | number | `0.1` | > 0; ≤ 1 | Client tail used for worst-tail gain. |
| `tuning.intensification.prefer` | string | `validation` | `validation`, `communication`, `flops`, `time` | Criterion used among practically equivalent candidates. |

## Categorical search parameters

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `tuning.search_space.<name>.type` | string | required | `categorical` | — |
| `tuning.search_space.<name>.values` | list[value] | required | non-empty | Candidate values. |

## Integer search parameters

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `tuning.search_space.<name>.type` | string | required | `int` | — |
| `tuning.search_space.<name>.low` | integer | required | — | Inclusive lower bound. |
| `tuning.search_space.<name>.high` | integer | required | — | Inclusive upper bound. |
| `tuning.search_space.<name>.step` | integer | `1` | ≥ 1 | Spacing between candidate values. |
| `tuning.search_space.<name>.log` | boolean | `false` | — | Sample on a logarithmic scale. |

## Floating-point search parameters

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `tuning.search_space.<name>.type` | string | required | `float` | — |
| `tuning.search_space.<name>.low` | number | required | — | Inclusive lower bound. |
| `tuning.search_space.<name>.high` | number | required | — | Inclusive upper bound. |
| `tuning.search_space.<name>.step` | number \| null | `null` | > 0 | Spacing between candidate values. |
| `tuning.search_space.<name>.log` | boolean | `false` | — | Sample on a logarithmic scale. |

`log: true` cannot be combined with `step`. Integer ranges allow equal bounds; floating-point ranges require `low < high`.
