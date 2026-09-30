# Configuration

RigFL uses YAML configuration files for datasets, runs, sweeps, and tuning studies.

## Individual runs

See the
[experiment reference](reference/experiment.md) and
[algorithm reference](reference/algorithms.md) for available settings.

The complete starter file is
[`configs/experiments/mnist_run.yaml`](../rigfl/templates/configs/experiments/mnist_run.yaml).

Use `--set` to override YAML settings for one run:

```bash
rigfl run configs/experiments/mnist_run.yaml \
  --algorithm fedprox \
  --set rounds=50 mu=0.1
```

`--algorithm` is optional when the YAML already defines `algorithm`. When
both are present, the command-line value takes precedence.

`results_root` defaults to `results` and contains completed runs, reports, and
study files in separate subdirectories. The run, sweep, HPO, report,
indexed-task, and seed-sensitivity commands accept `--results-root PATH`; for
commands with a YAML or saved task setting, the command-line value takes
precedence.

### Early stopping

Early stopping is disabled by default. You can enable and configure an early stopping policy under `early_stopping`:

```yaml
early_stopping:
  enabled: true
  metric: loss
  aggregation: uniform
  patience: 10
  min_delta: 0.0
```

## Sweeps and replicates

Sweep files place shared settings under `base` and Cartesian axes under `sweep`.

`replicates` pairs partition, split, and training seeds. When `replicates` is
set, `partition_seed`, `split_seed`, and `training_seed` cannot also be sweep axes. See
[`configs/experiments/mnist_sweep.yaml`](../rigfl/templates/configs/experiments/mnist_sweep.yaml) for a
complete example.

### External schedulers

Generate a frozen snapshot of the sweep for submission to an external scheduler:

```bash
rigfl sweep configs/experiments/mnist_sweep.yaml --snapshot
```

RigFL writes the snapshot under `results/mnist_sweep/snapshots/` and prints
its path and task-index range. RigFL task indexes begin at 1.

Configure the scheduler array using the printed range. In the submission
script, pass each job's array index to `rigfl task`:

```bash
rigfl task SNAPSHOT_GRID TASK_INDEX
```

Replace `SNAPSHOT_GRID` with the snapshot's `grid.jsonl` path and `TASK_INDEX`
with the scheduler's array-index variable.

## Tuning

Tuning files use the same `base` and `replicates` structure as sweeps. Search
parameters belong under `tuning.search_space`. See the
[hyperparameter-tuning guide](hyperparameter_tuning.md).

## Resolved settings

Completed result files include the resolved experiment and algorithm
configuration.
