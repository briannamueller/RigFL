# RigFL

RigFL is a modular framework for rigorous federated learning experimentation.
Algorithm-specific behavior is isolated behind a common interface, so that algorithms use the same orchestration, evaluation, configuration, and reporting
machinery.

## Key Features

- **Stable experiment and partition identity.** RigFL derives two separate
  fingerprints: one identifying a result from its distinct experiment
  configuration, the other identifying a partitioned dataset from its data
  configuration. An experiment whose result already exists is not
  rerun—expanding or changing a sweep will only execute new combinations.

- **Client-centered performance reporting.** Evaluation metrics that reveal
  whether the benefits of collaboration are broadly shared across clients,
  exposing disparities and uneven benefits that commonly reported averages
  obscure. See [Client-centered metrics](#client-centered-metrics).

- **Traceable result files.** Each completed experiment produces a result file
  containing its full evaluation history, resolved configuration, Git commit and
  uncommitted-change status, software versions, and client data-partition
  information.

- **Documented fidelity to the source papers.** Algorithms follow their published
  specifications; where a paper leaves a detail unspecified or its released code
  diverges from the text, the resolution is recorded in
  [DEVIATIONS.md](https://github.com/briannamueller/RigFL/blob/main/DEVIATIONS.md).

- **Optional W&B tracking.** Weights & Biases can be enabled to log experiment
  settings and validation performance during training.

- **Support for model-heterogeneous algorithms.** RigFL supports algorithms
  designed for clients with different model architectures. Registered model
  families define which architectures are assigned across clients.


## Algorithms

| Algorithm | `--algorithm` |
|---|---|
| [FedAvg](https://proceedings.mlr.press/v54/mcmahan17a.html) | `fedavg` |
| [FedProx](https://arxiv.org/abs/1812.06127) | `fedprox` |
| [FedProto](https://ojs.aaai.org/index.php/AAAI/article/view/20819) | `fedproto` |
| [FedGH](https://arxiv.org/abs/2303.13137) | `fedgh` |
| [LG-FedAvg](https://arxiv.org/abs/2001.01523) | `lgfedavg` |
| [FML](https://arxiv.org/abs/2006.16765) | `fml` |
| [FedKD](https://www.nature.com/articles/s41467-022-29763-x) | `fedkd` |
| [FedTGP](https://ojs.aaai.org/index.php/AAAI/article/view/29617) | `fedtgp` |
| [FedDES](https://arxiv.org/abs/2603.28006) | `feddes` |

Local training (`local`) and Global Ensemble (`global`) are available as
reference baselines.


## Quickstart

RigFL requires Python 3.10–3.12.

```bash
pip install rigfl
rigfl init my-rigfl-project
cd my-rigfl-project

python -m rigfl.experiment.run \
  --algorithm fedavg \
  --config configs/experiments/cifar10_run.yaml

python -m rigfl.experiment.collect
```

`rigfl init` sets up a starter project with example dataset and experiment
configurations. Dataset configurations are stored in `configs/datasets.yaml`
(see [Data partitions](#data-partitions)), while run and sweep configurations
are stored under `configs/experiments/` (see
[Configure and run experiments](#configure-and-run-experiments)). The results
summary produced by `collect` is explained in
[Evaluation and reporting](#evaluation-and-reporting).

The sections below walk through the same CIFAR-10 example in more detail.

## Data partitions

A data partition is the complete set of client datasets produced from one dataset configuration.

Each dataset is defined by an entry in
[`configs/datasets.yaml`](https://github.com/briannamueller/RigFL/blob/main/configs/datasets.yaml)
that specifies a backend, source, and partitioning strategy.
Flower is the default backend, which sources datasets from Hugging Face and partitions them using standard FL partitioning schemes that control the type and degree of client heterogeneity.

```yaml
datasets:
  cifar10:
    backend: flower
    source_dataset: uoft-cs/cifar10
    data_transform: cifar10
    partition:
      scheme: dirichlet
      num_clients: 5
      alpha: 0.5
```

Generate client datasets:

```bash
python -m rigfl.data.generate --dataset cifar10
```

RigFL derives a stable fingerprint from the dataset's configuration, which is used in the storage path:

```text
data/cifar10/partition_<fingerprint>/
├── manifest.json
└── clients/
    ├── client_0/
    │   ├── train.pt
    │   ├── validation.pt
    │   └── test.pt
    └── ...
```

For Flower-backed datasets, pre-generation is optional; RigFL generates a missing partition when an experiment first requires it. Run the generation command separately when you want
to prepare or inspect the partition in advance.

[`configs/datasets.yaml`](https://github.com/briannamueller/RigFL/blob/main/configs/datasets.yaml) ships with starting points for MNIST, Fashion-MNIST,
CIFAR-10, CIFAR-100, Tiny ImageNet, FEMNIST, PaySim fraud-detection, and
phishing URL detection. Add a new dataset by creating another entry. See the
[data guide](https://github.com/briannamueller/RigFL/blob/main/docs/data.md)
for preparing client datasets and the [data reference](https://github.com/briannamueller/RigFL/blob/main/docs/reference/data.md)
for available settings.

Naturally partitioned biomedical datasets are available through the optional
BioSilo backend, installed with `pip install "rigfl[biosilo]"`. BioSilo
partitions are not automatically generated when an experiment runs, so run the generation
command above first. The data guide covers BioSilo's configuration
and dataset-specific dependencies.


## Configure and run experiments

Configure the experiment in
[`configs/experiments/cifar10_run.yaml`](https://github.com/briannamueller/RigFL/blob/main/configs/experiments/cifar10_run.yaml).

The YAML has two sections. Entries under `experiment` define the overarching
configuration for the execution of RigFL’s shared workflow. Entries under
`algorithm` specify how individual algorithms operate; each entry may be
supported by one or several algorithms.

```yaml
experiment:
  dataset: cifar10
  model: fedavg_cnn
  rounds: 2
  seed: 0
  eval_gap: 1
  device: cpu
  out_dir: results

algorithm:
  local_epochs: 1
  lr: 0.01
```

`experiment.dataset` names an entry in `configs/datasets.yaml`. RigFL derives the partition
fingerprint from that entry and loads the matching partition. By default, RigFL reads dataset configurations from `configs/datasets.yaml` and stores
partitions under `data/`. Set `experiment.dataset_config` and
`experiment.data_dir` to override these paths.

Set `experiment.model` to choose a model architecture. For algorithms that
support different client architectures, set `experiment.model_family` to choose
a model family. See the [model reference](https://github.com/briannamueller/RigFL/blob/main/docs/reference/models.md) for the
available architectures and families.

See the [configuration guide](https://github.com/briannamueller/RigFL/blob/main/docs/configuration.md)
for working with run and sweep files. The
[experiment reference](https://github.com/briannamueller/RigFL/blob/main/docs/reference/experiment.md)
and [algorithm reference](https://github.com/briannamueller/RigFL/blob/main/docs/reference/algorithms.md)
document all supported settings, including their defaults and allowed values.

Run the experiment with:

```bash
python -m rigfl.experiment.run \
  --algorithm fedavg \
  --config configs/experiments/cifar10_run.yaml
```

This trains FedAvg for two communication rounds and writes the result under
`results/runs`. RigFL records accuracy, balanced accuracy, macro F1, and predictive log loss for
each client at every evaluation round, together with the corresponding sample
counts.



### Early Stopping

Early stopping controls when training ends and is separate from the reporting
round chosen during evaluation. It is disabled by default. You can enable and configure an early stopping policy under `experiment.early_stopping`:

```yaml
experiment:
  early_stopping:
    enabled: true
    metric: loss
    aggregation: mean
    patience: 10
    min_delta: 0.0
```



## Sweeps

A set of experiments can be defined declaratively in a sweep file. RigFL expands
the sweep into independent tasks that can run sequentially or in parallel, and
skips tasks whose results already exist, so sweeps can be resumed or extended
incrementally.

The following sweep defined in `configs/experiments/cifar10_sweep.yaml` runs Local and FedAvg at
two learning rates across three seed replicates (12 runs):

```yaml
name: cifar10_sweep
algorithms: [local, fedavg]

base:
  experiment:
    dataset: cifar10
    model: fedavg_cnn
    rounds: 100

replicates:
  - {partition_seed: 0, split_seed: 0, experiment_seed: 0}
  - {partition_seed: 1, split_seed: 1, experiment_seed: 1}
  - {partition_seed: 2, split_seed: 2, experiment_seed: 2}

sweep:
  algorithm.lr: [0.01, 0.03]
```

`base` specifies the experiment and algorithm configuration shared by all runs,
and each entry in `replicates` defines the partition, validation-split, and
training seeds for one repetition.

Generate the sweep grid with:

```bash
python -m rigfl.experiment.launch --config configs/experiments/cifar10_sweep.yaml
```

This writes `results/cifar10_sweep/grid.jsonl`, where each line is one task: a
complete experiment configuration. Tasks run independently by their 1-based
index, so the grid can be executed sequentially in a shell loop or in parallel as
an array job on any scheduler:

```bash
python -m rigfl.experiment.launch \
  --grid results/cifar10_sweep/grid.jsonl \
  --grid-task 1
```

RigFL also supports hyperparameter tuning with Optuna. See the
[hyperparameter-tuning guide](https://github.com/briannamueller/RigFL/blob/main/docs/hyperparameter_tuning.md).

## Evaluation and reporting

Summarize results with:

```bash
python -m rigfl.experiment.collect
```

Collection uses validation history to choose the reporting round. The same history can be summarized in two ways:

- `global` chooses one round from the validation score aggregated across clients;
- `per-client` chooses each client's best validation round.

By default, `rigfl.experiment.collect` reports test performance at the round with the highest mean validation accuracy across clients. Use
`--selection-metric`, `--selection-view`, and `--selection-aggregation` to
change how the reporting round is chosen.

Collection reads completed runs under `results/runs`. Runs with the same
settings apart from their replicate seeds are summarized together: the row
reports mean validation and test performance, with a 95% confidence interval
when the replicates have distinct experiment seeds.

When matching Local runs are available, `collect` also prints a client-level
performance analysis below the run summary.

### Client-level performance analysis

Aggregate performance metrics can signal that collaborative learning improves
upon local training on average, even though collaboration worsens performance at
some individual clients. RigFL provides evaluation metrics that surface unevenly
distributed benefits.

- **Negative-transfer rate:** the fraction of matched client-and-seed pairs that
  perform worse than Local by more than the selected threshold. The benefit rate
  counts pairs that improve by more than the threshold.

- **Negative-transfer magnitude:** the average performance loss among the pairs
  whose loss relative to Local exceeds the selected threshold.

- **Negative-transfer burden:** that loss averaged across all pairs, assigning
  zero to those that do not cross the threshold.

- **Worst-tail gain:** the average difference from Local among the lowest-gaining
  fraction of pairs.

These metrics compare each algorithm against Local, so they are reported only
when the collected results include Local runs on the same data, model, and
training schedule, with matching replicate seeds.

### Resource measurements

Completed runs record training communication, operation timing, and the hardware
used for timing. Communication measures the algorithm payloads exchanged during
training; transport and serialization overhead are not included.

FLOP estimation is optional because profiling adds runtime overhead. Enable it
with `experiment.estimate_flops: true`.
FLOP estimation requires PyTorch 2.1 or newer and records training and inference
separately.

Add resource measurements to the report with:

```bash
python -m rigfl.experiment.collect --include-resources
```

See the [results guide](https://github.com/briannamueller/RigFL/blob/main/docs/results.md)
for detailed workflows for summarizing runs, comparing experiments, and
examining sensitivity to different sources of randomness.

## Adding an algorithm

See [Extending RigFL](https://github.com/briannamueller/RigFL/blob/main/docs/extensibility.md)
for the algorithm interface, registration, communication measurements, and
testing requirements.

## Experiment tracking with Weights & Biases

Install the optional W&B dependency with:

```bash
pip install "rigfl[wandb]"
```

Enable tracking by setting `wandb: true` under `experiment` in the YAML
configuration file.

## Development and testing

Clone the repository and install RigFL in editable mode with its testing
dependency, then run the test suite:

```bash
git clone https://github.com/briannamueller/RigFL.git
cd RigFL
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest -q
```

## License

MIT. See [LICENSE](https://github.com/briannamueller/RigFL/blob/main/LICENSE).
