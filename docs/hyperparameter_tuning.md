# Hyperparameter tuning

RigFL uses [Optuna](https://optuna.org/) to search complete hyperparameter
configurations and rank them using validation performance across matched
replicates. Test results are recorded during each experiment, but they are not
consulted when selecting a configuration.

Install the optional dependency with:

```bash
pip install "rigfl[hpo] @ git+https://github.com/briannamueller/RigFL.git"
```

## Configure a study

An Optuna study uses the same `base` experiment configuration as a RigFL sweep,
but it must specify exactly one algorithm. Parameters to optimize are declared
under `tuning.search_space` rather than `sweep`. The starter study,
`configs/experiments/mnist_hpo.yaml`, tunes FedProx's `lr` and `mu` at
demonstration scale:

```yaml
name: mnist_hpo
algorithm: fedprox

base:
  dataset: mnist
  model_arch: fedavg_mnist_cnn
  rounds: 2
  eval_gap: 1
  device: cpu
  results_root: results
  local_epochs: 1

replicates:
  - {partition_seed: 0, split_seed: 0, training_seed: 0}
  - {partition_seed: 0, split_seed: 0, training_seed: 1}

tuning:
  trials: 4
  sampler:
    class: optuna.samplers.TPESampler
    options:
      seed: 17
  metric: accuracy
  search_space:
    lr:
      type: float
      low: 0.001
      high: 0.1
      log: true
    mu:
      type: float
      low: 0.001
      high: 0.1
      log: true
```

For a real search, increase `rounds`, `trials`, and the number of replicates.

`replicates` defines the matched partition, split, and training-seed conditions
used for every trial. A positive integer is shorthand for conditions using the
same seeds from `0` through `N - 1`; write the list explicitly, as above, to hold
any seed component fixed.

Each trial is one joint assignment of all parameters in `search_space`. RigFL
runs that assignment under every listed replicate and averages its validation
score. Using the same replicate conditions for every trial makes the candidate
comparisons paired.

Search parameters use the same setting names as run and sweep files. Dataset
identity, replicate seeds, output locations, and execution settings cannot be
optimized.

The [search-parameter reference](reference/experiment.md#categorical-search-parameters)
lists the fields and constraints for categorical, integer, and floating-point
distributions.

## Run or resume the study

Run the study with:

```bash
rigfl hpo configs/experiments/mnist_hpo.yaml
```

The study name determines its directory under the selected results root, so
this study writes to `results/mnist_hpo/` (see
[`results_root`](configuration.md#individual-runs)). Running the same command
again resumes its Optuna study. Use `--trials N` to change the target total
number of trials without editing the YAML.

Completed experiments are stored in `<results-root>/runs/`. If an identical
experiment was already completed by an individual run, sweep, or earlier
Optuna trial, RigFL reuses that result instead of running it again.

## Choose a sampler

`TPESampler` is the default and is appropriate for adaptive searches. Sampler
constructor options are passed under `options`:

```yaml
sampler:
  class: optuna.samplers.TPESampler
  options:
    seed: 17
    n_startup_trials: 20
```

`RandomSampler` uses the same structure. Any importable class that inherits
from Optuna's `BaseSampler` can be named, provided its constructor can be
configured using YAML values and any additional package it requires is
installed. RigFL currently supports single-objective studies only; Optuna
features that require a multi-objective study are not supported.

For an exhaustive grid search, use `GridSampler` and categorical values:

```yaml
tuning:
  metric: accuracy
  sampler:
    class: optuna.samplers.GridSampler
    options:
      seed: 17
  search_space:
    lr:
      type: categorical
      values: [0.001, 0.01, 0.1]
    local_epochs:
      type: categorical
      values: [1, 3, 5]
```

RigFL derives the grid and number of trials from `search_space`; do not specify
`trials` or repeat the grid under `sampler.options`.

## Selection settings

The default selection settings are:

```yaml
tuning:
  metric: accuracy
  round_selection: shared
  client_weighting: uniform
```

For each run, RigFL uses validation history to select the reporting round. It
then averages the resulting validation score across the study's replicates and
ranks the candidate configurations, highest score first.

These settings have the same meaning as in reporting; see the
[results guide](results.md#summarize-completed-runs).

## Study files

The study's files are stored in `<results-root>/<study-name>/`, such as
`results/mnist_hpo/`:

```text
<study-name>/
├── optuna.db
├── study.json
├── ranking.json
└── selected.yaml  # written only for a unique winner
```

`optuna.db` stores the resumable Optuna study. `study.json` records the fixed
settings, search space, replicate conditions, and trial outcomes.
`ranking.json` records the candidates' validation results and ranks them across
replicates.

When one candidate has the best aggregate validation score, `selected.yaml`
contains that configuration and the study's replicate conditions in the format
accepted by `rigfl sweep`. The completed runs already contain their test
results; edit the replicate conditions only when extending the selected
configuration's evaluation.

When multiple candidates tie for the best aggregate validation score,
`ranking.json` records them at the same rank and `selected.yaml` is not written.
