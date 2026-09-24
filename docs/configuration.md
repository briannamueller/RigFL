# Configuration

RigFL uses YAML configuration files for datasets, runs, sweeps, and tuning studies.

## Individual runs

Run files contain `experiment` and `algorithm` sections. See the
[experiment reference](reference/experiment.md) and
[algorithm reference](reference/algorithms.md) for available settings.

The complete starter file is
[`configs/experiments/cifar10_run.yaml`](../configs/experiments/cifar10_run.yaml).

Use `--set` to override YAML settings for one run. Paths use the `experiment.`
or `algorithm.` YAML section name:

```bash
rigfl run configs/experiments/cifar10_run.yaml \
  --algorithm fedprox \
  --set experiment.rounds=50 algorithm.mu=0.1
```

## Sweeps and replicates

Sweep files place shared settings under `base` and Cartesian axes under `sweep`.
Sweep paths begin with `experiment.` or `algorithm.`.

`replicates` pairs partition, split, and experiment seeds. Its seed fields
cannot also appear under `sweep`. See
[`configs/experiments/cifar10_sweep.yaml`](../configs/experiments/cifar10_sweep.yaml) for a
complete example.

## Tuning

Tuning files use the same `base` and `replicates` structure as sweeps. Search
parameters belong under `tuning.search_space`. See the
[hyperparameter-tuning guide](hyperparameter_tuning.md).

## Resolved settings

Completed result files include the resolved experiment and algorithm
configuration. Fields such as the partition ID, input specification, and
resolved model list are recorded results and are not written in run YAML.
