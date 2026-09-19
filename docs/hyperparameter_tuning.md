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

replicates: 3

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

A replicate count expands to that many conditions with matched seeds from zero,
so `replicates: 3` is the `{partition_seed, split_seed, experiment_seed}` triples
for 0, 1, and 2. Write the list out to hold a seed constant.

Each trial is one joint assignment of all parameters in `search_space`. RigFL
runs that assignment under every listed replicate and averages its validation
score. Using the same replicate conditions for every trial makes the candidate
comparisons paired.

Search paths beginning with `algorithm.` refer to fields in the selected
algorithm's configuration. Paths beginning with `experiment.` refer to experiment
fields. Dataset identity, replicate seeds, output locations, and execution
settings cannot be optimized.

The [search-parameter reference](reference/experiment.md#categorical-search-parameters)
lists the fields and constraints for categorical, integer, and floating-point
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
    replicates: 3
    practical_threshold: 0.005
    prefer: validation
    ranking: pooled
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

An intensification count continues the screening seeds rather than restarting,
so three screening replicates followed by `replicates: 3` here uses seeds 3, 4,
and 5. That satisfies the rule that intensification must not reuse a screening
data-seed pair; write the list out to choose the seeds yourself.

`ranking` sets which replicates the shortlisted candidates are ranked on.
`pooled`, the default, ranks each candidate on its screening and intensification
replicates together. `intensification` ranks on the new replicates alone, which
keeps shortlisting and ranking on separate data conditions at the cost of half
the evidence. Pooled ranking reads the screening results again, so they must
still be in `results/runs/`.

`practical_threshold` defines how close another candidate must be to
the leader to be treated as practically equivalent. `prefer` may remain
`validation`, or may select the lowest communication, FLOPs, or runtime among
the candidates found to be practically equivalent. Runtime can be compared only
when its recorded hardware and software information matches.

## Study files

Completed experiments are stored in `results/runs`. The study's other files are
under `results/<study-name>/`:

```text
<study-name>/
├── optuna.db
├── study.json
├── ranking.json
├── selection.json
├── selected.yaml
└── intensification/       # when configured
    ├── grid.jsonl
    └── evaluation.json
```

When intensification is configured, `selection.json` and `selected.yaml` appear
after its tasks finish. `selected.yaml` can be passed to
`rigfl.experiment.launch`, but the selected runs already contain test results.
