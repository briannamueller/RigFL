# Experiment configuration reference

These options configure individual runs, sweeps, and tuning studies.

## Experiment settings

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `dataset` | string | `cifar10` | — | Dataset name from the dataset configuration file. |
| `dataset_config` | string | `configs/datasets.yaml` | — | Path to the dataset configuration file. |
| `data_dir` | string | `data` | — | Directory containing generated client partitions. |
| `rounds` | integer | `100` | ≥ 1 | Number of communication rounds for iterative runners. |
| `training_seed` | integer | `0` | ≥ 0 | Seed for training and model initialization. |
| `partition_seed` | integer \| null | `null` | ≥ 0 | Override for the dataset partition seed. |
| `split_seed` | integer \| null | `null` | ≥ 0 | Override for the client validation-split seed. |
| `shared_dim` | integer | `512` | ≥ 1 | Representation width used by algorithms that exchange representations (FedProto, FedGH, pFedMoE, FedTGP); ignored by all others. |
| `model_arch` | string | `fedavg_cnn` | — | Architecture name given to every client, or a model-family name assigning the family's architectures to clients. Algorithms that require identical client architectures accept only an architecture name. |
| `batch` | integer | `32` | ≥ 1 | Client training batch size. |
| `eval_gap` | integer | `1` | ≥ 1 | Evaluate every N communication rounds for iterative runners. |
| `device` | string | `auto` | `auto`, `cpu`, `mps`, `cuda` | Device used for training and evaluation. |
| `results_root` | string | `results` | — | Base directory for runs, reports, and study files. |
| `quiet` | boolean | `true` | — | Suppress per-round progress output. |
| `wandb` | boolean | `false` | — | Log training progress to Weights & Biases. |
| `wandb_project` | string | `rigfl` | — | Weights & Biases project name. |
| `estimate_flops` | boolean | `false` | — | Estimate FLOPs for supported PyTorch operators. |
| `early_stopping.enabled` | boolean | `false` | — | Stop training when validation performance stops improving. |
| `early_stopping.metric` | string \| null | `null` | — | Validation metric to monitor; defaults to loss when enabled. |
| `early_stopping.aggregation` | string | `uniform` | `uniform`, `sample_count` | How client validation values are combined. |
| `early_stopping.patience` | integer | `10` | ≥ 1 | Evaluations without improvement before stopping. |
| `early_stopping.min_delta` | number | `0.0` | ≥ 0 | Smallest change counted as an improvement. |

## Sweep and study files

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `name` | string \| null | `null` | — | Name used for the study output directory. |
| `algorithm` | list[string] \| string \| null | `null` | — | Algorithm values included in the sweep or study. |
| `base` | mapping | `{}` | — | Fixed settings shared by generated runs. |
| `sweep` | mapping \| null | `null` | — | Cartesian axes for an ordinary sweep. |
| `replicates` | list[replicate settings] \| null | `null` | — | Paired data and training seed conditions, or a count expanding to that many conditions with matched seeds from zero. |
| `tuning` | tuning settings \| null | `null` | — | Optuna study settings. |

## Replicate conditions

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `replicates[].partition_seed` | integer | required | ≥ 0 | Seed used to assign data to clients. |
| `replicates[].split_seed` | integer | required | ≥ 0 | Seed used to create client validation splits. |
| `replicates[].training_seed` | integer | required | ≥ 0 | Seed used for training and model initialization. |

Replicate entries must be unique and use distinct `training_seed` values. Replicate seed fields cannot also be sweep axes.

## Tuning settings

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `tuning.search_space` | mapping[string, value] | required | — | Parameters and distributions explored by Optuna. |
| `tuning.trials` | integer | `100` | ≥ 1 | Target total number of trials. |
| `tuning.metric` | string | `accuracy` | — | Validation metric optimized by the study: accuracy, balanced_accuracy, macro_f1, auroc, or auprc. |
| `tuning.round_selection` | string | `shared` | `shared`, `client-specific` | Whether reporting uses one shared round or client-specific rounds. |
| `tuning.client_weighting` | string | `uniform` | `uniform`, `sample_count` | How clients contribute to reported metrics. |

## Sampler settings

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `tuning.sampler.class` | string | `optuna.samplers.TPESampler` | non-empty | Import path of the Optuna sampler class. |
| `tuning.sampler.options` | mapping | `{"seed":0}` | — | Arguments passed to the sampler constructor. |

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
