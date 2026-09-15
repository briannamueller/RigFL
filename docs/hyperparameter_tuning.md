# Hyperparameter tuning

RigFL uses [Optuna](https://optuna.org/) to search complete hyperparameter
configurations and rank them using validation performance across matched
replicates. Test results are recorded during each experiment, but they are not
consulted when selecting a configuration.

Install the optional dependency with:

```bash
pip install "rigfl[hpo]"
```

## Configure a study

An Optuna study uses the same `base` experiment configuration as a RigFL sweep,
but it must specify exactly one algorithm. Parameters to optimize are declared
under `tuning.search_space` rather than `sweep`.

```yaml
name: cifar10_fedprox_optuna
algorithms: [fedprox]

base:
  experiment:
    dataset: cifar10
    model: fedavg_cnn
    rounds: 100
    eval_gap: 5

replicates:
  - {partition_seed: 0, split_seed: 0, experiment_seed: 0}
  - {partition_seed: 1, split_seed: 1, experiment_seed: 1}
  - {partition_seed: 2, split_seed: 2, experiment_seed: 2}

tuning:
  trials: 100
  metric: accuracy
  sampler:
    class: optuna.samplers.TPESampler
    options:
      seed: 17
  search_space:
    algorithm.lr:
      type: float
      low: 0.0001
      high: 0.1
      log: true
    algorithm.local_epochs:
      type: int
      low: 1
      high: 5
    algorithm.mu:
      type: float
      low: 0.0001
      high: 1.0
      log: true
    experiment.batch:
      type: categorical
      values: [32, 64, 128]
```

Each trial is one joint assignment of all parameters in `search_space`. RigFL
runs that assignment under every listed replicate and averages its validation
score. Using the same replicate conditions for every trial makes the candidate
comparisons paired.

Search paths beginning with `algorithm.` refer to fields in the selected
algorithm's configuration. Paths beginning with `experiment.` refer to experiment
fields. Dataset identity, replicate seeds, output locations, and execution
settings cannot be optimized.

The available distributions are:

```yaml
categorical_parameter:
  type: categorical
  values: [value_a, value_b]

integer_parameter:
  type: int
  low: 1
  high: 10
  step: 1

continuous_parameter:
  type: float
  low: 0.0001
  high: 0.1
  log: true
```

`step` and `log: true` cannot be combined for integer or floating-point
distributions.

## Run or resume the study

Run the study with:

```bash
python -m rigfl.experiment.optimize \
  --config experiments/cifar10_optuna.yaml
```

The study name determines its directory under `results/`. Running the same
command again resumes its Optuna study. Use `--trials N` to change the target
total number of trials without editing the YAML.

Completed experiments are stored in the shared `results/runs/` directory. If an
identical experiment was already completed by an individual run, sweep, or
earlier Optuna trial, RigFL reuses that result instead of running it again.

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
    algorithm.lr:
      type: categorical
      values: [0.001, 0.01, 0.1]
    algorithm.local_epochs:
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
  selection_view: global
  selection_aggregation: mean
  tie_break: earliest
```

For each run, RigFL uses validation history to select the reporting round. It
then averages the resulting validation score across the study's replicates and
ranks the candidate configurations. Metric direction is determined by the
metric: accuracy-like metrics are maximized, while loss is minimized.

`selection_view: global` selects one reporting round from validation performance
aggregated across clients. `per-client` selects a reporting round separately for
each client. `selection_aggregation` controls whether clients contribute equally
(`mean`) or according to their validation sample counts (`weighted_mean`).

## Optional intensification

A large search may evaluate many configurations on a modest number of
replicates. Intensification evaluates the leading candidates on additional,
previously unused replicate conditions before making the final selection. It is
optional and does not run automatically.

Add an intensification plan to the tuning configuration:

```yaml
tuning:
  # metric, sampler, and search_space omitted here
  intensification:
    top_k: 5
    replicates:
      - {partition_seed: 10, split_seed: 10, experiment_seed: 10}
      - {partition_seed: 11, split_seed: 11, experiment_seed: 11}
      - {partition_seed: 12, split_seed: 12, experiment_seed: 12}
    practical_threshold: 0.005
    prefer: validation
```

After the initial study, RigFL writes the additional tasks to:

```text
results/<study-name>/intensification/grid.jsonl
```

Run those tasks using the same grid execution command used for ordinary sweeps:

```bash
python -m rigfl.experiment.launch \
  --grid results/<study-name>/intensification/grid.jsonl \
  --grid-task 1
```

Each grid index is an independent task and can be submitted as a scheduler
array. For a small local study, the intensification command can run all missing
tasks directly:

```bash
python -m rigfl.experiment.intensification \
  --ranking results/<study-name>/ranking.json \
  --run
```

If the grid was executed separately, omit `--run` after all tasks finish:

```bash
python -m rigfl.experiment.intensification \
  --ranking results/<study-name>/ranking.json
```

The leading candidate is determined from validation performance on the new
replicates. `practical_threshold` defines how close another candidate must be to
the leader to be treated as practically equivalent. `prefer` may remain
`validation`, or may select the lowest communication, FLOPs, or runtime among
the candidates found to be practically equivalent. Runtime can be compared only
when its recorded hardware and software information matches.

## Study files

A study without intensification produces:

```text
results/
├── runs/                         # completed experiments shared by all workflows
└── <study-name>/
    ├── optuna.db                 # resumable Optuna study
    ├── study.json                # study definition, trials, and result references
    ├── ranking.json              # validation ranking for every candidate
    ├── selection.json            # selected configuration and supporting evidence
    └── selected.yaml             # runnable form of the selected configuration
```

When intensification is configured, `ranking.json` and
`intensification/grid.jsonl` are written first. Completing intensification adds:

```text
results/<study-name>/
├── intensification/
│   ├── grid.jsonl
│   └── evaluation.json
├── selection.json
└── selected.yaml
```

`selected.yaml` records the complete selected configuration and its replicate
conditions. It can be passed to `rigfl.experiment.launch`, but rerunning it is
not required to obtain test results: the completed runs referenced by
`selection.json` already contain them.
