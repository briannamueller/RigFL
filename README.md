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
  obscure. See [Client-level performance analysis](#client-level-performance-analysis).

- **Traceable result files.** Each completed experiment produces a result file
  containing its full evaluation history, resolved configuration, Git commit and
  uncommitted-change status, software versions, and client data-partition
  information.

- **Documented algorithm fidelity.** Each implementation is checked against its
  paper and, where available, released reference code. Ambiguities, deliberate
  deviations, and implementation choices that could affect reproducibility are
  recorded in
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
| [FedCAC](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Bold_but_Cautious_Unlocking_the_Potential_of_Personalized_Federated_Learning_ICCV_2023_paper.html) | `fedcac` |
| [FedAPA](https://www.ijcai.org/proceedings/2025/0692.pdf) | `fedapa` |
| [FedAPEN](https://doi.org/10.1145/3580305.3599344) | `fedapen` |
| [FedAMP](https://ojs.aaai.org/index.php/AAAI/article/view/16960) | `fedamp` |
| [APPLE](https://www.ijcai.org/proceedings/2022/0301) | `apple` |
| [FedPAC](https://openreview.net/forum?id=SXZr8aDKia) | `fedpac` |
| [FedProto](https://ojs.aaai.org/index.php/AAAI/article/view/20819) | `fedproto` |
| [FedGH](https://arxiv.org/abs/2303.13137) | `fedgh` |
| [FML](https://arxiv.org/abs/2006.16765) | `fml` |
| [FedTGP](https://ojs.aaai.org/index.php/AAAI/article/view/29617) | `fedtgp` |
| [pFedMoE](https://arxiv.org/abs/2402.01350) | `pfedmoe` |

Local training (`local`) and Global Ensemble (`global`) are available as
reference baselines.


## Quickstart

RigFL requires Python 3.10–3.12.

```bash
pip install rigfl
rigfl init my-rigfl-project
cd my-rigfl-project

rigfl data generate --dataset cifar10
rigfl run configs/experiments/cifar10_run.yaml --algorithm fedavg

rigfl report \
  --config configs/reporting.yaml \
  --filter main_results \
  --save
```

`rigfl init` sets up a starter project with example dataset and experiment
configurations. Dataset configurations are stored in `configs/datasets.yaml`
(see [Data partitions](#data-partitions)), while run and sweep configurations
are stored under `configs/experiments/` (see
[Configure and run experiments](#configure-and-run-experiments)). The results
summary produced by the final command contains aggregate performance, as well
as client-level metrics that quantify negative transfer when matching Local
baseline results are available. Reporting is explained in
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
rigfl data generate --dataset cifar10
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
  partition_seed: 0
  split_seed: 0
  seed: 0
  shared_dim: 128
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
rigfl run configs/experiments/cifar10_run.yaml --algorithm fedavg
```

This trains FedAvg for two communication rounds and writes the result under
`results/runs`. RigFL records accuracy, balanced accuracy, macro F1, predictive
log loss, AUROC, and AUPRC for each client at every evaluation round, together
with the corresponding sample counts.



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
the sweep into independent tasks that can run sequentially. Tasks whose results
already exist are skipped, so sweeps can be resumed or extended incrementally.

The demonstration sweep in `configs/experiments/cifar10_sweep.yaml` runs Local
and FedAvg under the same three replicate conditions (six runs):

```yaml
name: cifar10_sweep
algorithms: [local, fedavg]

base:
  experiment:
    dataset: cifar10
    model: fedavg_cnn
    rounds: 2
    shared_dim: 128
    eval_gap: 1
    device: cpu
    out_dir: results
  algorithm:
    local_epochs: 1
    lr: 0.01

replicates:
  - {partition_seed: 0, split_seed: 0, experiment_seed: 0}
  - {partition_seed: 0, split_seed: 0, experiment_seed: 1}
  - {partition_seed: 0, split_seed: 0, experiment_seed: 2}
```

`base` specifies the experiment and algorithm configuration shared by all six
runs. Each entry in `replicates` defines a partition, validation-split, and
training-seed combination used once for each algorithm.

Run the sweep's experiments sequentially on the current machine:

```bash
rigfl sweep configs/experiments/cifar10_sweep.yaml --execute
```

RigFL writes the sweep grid to `results/cifar10_sweep/grid.jsonl`, where each
line corresponds to one experiment configuration.

To prepare the grid for execution through an external scheduler without running
the experiments locally, omit `--execute` and follow the
[scheduler workflow in the configuration guide](https://github.com/briannamueller/RigFL/blob/main/docs/configuration.md).

RigFL also supports hyperparameter tuning with Optuna. See the
[hyperparameter-tuning guide](https://github.com/briannamueller/RigFL/blob/main/docs/hyperparameter_tuning.md).

## Evaluation and reporting

`configs/reporting.yaml` defines named sets of completed runs and how RigFL
summarizes them. The included `main_results` set selects the Local and FedAvg
results from the CIFAR-10 example.

Summarize and save those results with:

```bash
rigfl report \
  --config configs/reporting.yaml \
  --filter main_results \
  --save
```

RigFL uses validation results to select the reporting round and reports the
corresponding test performance across replicates. The command prints the report
and saves its aggregate table to `results/reports/main_results.csv`. Add
`--per-client` to also save the selected result for every client and replicate
to `results/reports/main_results_per_client.csv`.

### Client-level performance analysis

Aggregate performance metrics can signal that collaborative learning improves
upon local training on average, even though collaboration worsens performance at
some individual clients. This failure mode in federated learning is referred to
as negative transfer. RigFL provides evaluation metrics that surface unevenly
distributed benefits, such as when negative transfer at some clients is
cancelled out by larger improvements at others.

These include benefit rate, negative-transfer rate, negative-transfer
magnitude, negative-transfer burden, and worst-tail gain. The analysis pairs the
same client under the same partition, split, and training-seed conditions.
Detailed definitions and interpretation are in the
[results guide](https://github.com/briannamueller/RigFL/blob/main/docs/results.md).

### Resource measurements

Completed runs record training communication, operation timing, and the hardware
used for timing. Communication totals measure algorithm payload bytes rather
than end-to-end network traffic.

FLOP estimation is optional because profiling adds runtime overhead. Enable it
with `experiment.estimate_flops: true`.
FLOP estimation requires PyTorch 2.1 or newer and records training and inference
separately.

Add resource measurements to the report with:

```bash
rigfl report \
  --config configs/reporting.yaml \
  --filter main_results \
  --resources \
  --save
```

See the [results guide](https://github.com/briannamueller/RigFL/blob/main/docs/results.md)
for detailed workflows for summarizing runs, interpreting client-level analyses,
and examining sensitivity to different sources of randomness.

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
ruff check .
pytest -q
```

## License

MIT. See [LICENSE](https://github.com/briannamueller/RigFL/blob/main/LICENSE).
