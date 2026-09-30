# RigFL

RigFL is a modular framework for rigorous federated learning experimentation.
Algorithm-specific behavior is isolated behind a common interface, so that
algorithms use the same orchestration, evaluation, configuration, and reporting
machinery.

## Key Features

- **Stable experiment and partition identity.** Run results and generated
  partitions have separate configuration-derived fingerprints. Existing results
  are reused, so extending a sweep executes only new combinations.

- **Client-centered performance reporting.** Evaluation metrics that reveal
  whether the benefits of collaboration are broadly shared across clients,
  exposing performance disparities and instances of negative transfer that
  commonly reported averages mask. See
  [Client-level gain analysis](#client-level-gain-analysis).

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

| Algorithm | Configuration value |
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

Local training (`local`) and Global Ensemble (`global_ensemble`, an ensemble of the clients' locally trained models) are available as reference baselines.


## Quickstart

RigFL requires Python 3.10–3.12.

```bash
pip install "git+https://github.com/briannamueller/RigFL.git"
rigfl init my-rigfl-project
cd my-rigfl-project

rigfl data generate --dataset mnist
rigfl run configs/experiments/mnist_run.yaml
rigfl sweep configs/experiments/mnist_sweep.yaml --execute

rigfl report --save
```

The starter experiments take a few minutes to run on CPU.

`rigfl init` sets up a starter project with example dataset and experiment
configurations. Dataset configurations are stored in `configs/datasets.yaml`
(see [Data partitions](#data-partitions)), while run and sweep configurations
are stored under `configs/experiments/` (see
[Configure and run experiments](#configure-and-run-experiments)). The report
produced by the final command contains aggregate performance, as well
as client-level metrics that quantify negative transfer when matching Local
baseline results are available. Reporting is explained in
[Reporting results](#reporting-results).

The sections below walk through the same MNIST example in more detail.

## Data partitions

A data partition is the complete set of client datasets produced from one dataset configuration.

Each dataset is defined by an entry in
[`configs/datasets.yaml`](https://github.com/briannamueller/RigFL/blob/main/rigfl/templates/configs/datasets.yaml)
that specifies a backend, source, and partitioning strategy.
Flower is the default backend, which sources datasets from Hugging Face and partitions them using standard FL partitioning schemes that control the type and degree of client heterogeneity.

```yaml
datasets:
  mnist:
    backend: flower
    source_dataset: ylecun/mnist
    data_transform: mnist
    partition:
      scheme: dirichlet
      num_clients: 5
      alpha: 0.5
```

Generate client datasets:

```bash
rigfl data generate --dataset mnist
```

RigFL derives a stable fingerprint from the dataset's configuration, which is used in the storage path:

```text
data/mnist/partition_<fingerprint>/
├── manifest.json
└── clients/
    ├── client_0/
    │   ├── train.pt
    │   └── test.pt
    └── ...
```

[`configs/datasets.yaml`](https://github.com/briannamueller/RigFL/blob/main/rigfl/templates/configs/datasets.yaml) ships with starting points for MNIST, Fashion-MNIST,
CIFAR-10, CIFAR-100, Tiny ImageNet, FEMNIST, PaySim fraud-detection, and
phishing URL detection. Add a new dataset by creating another entry. See the
[data guide](https://github.com/briannamueller/RigFL/blob/main/docs/data.md)
for preparing client datasets and the [data reference](https://github.com/briannamueller/RigFL/blob/main/docs/reference/data.md)
for available settings.

Naturally partitioned biomedical datasets are available through the optional
BioSilo backend, installed with
`pip install "rigfl[biosilo] @ git+https://github.com/briannamueller/RigFL.git"`.
The data guide covers BioSilo's configuration and dataset-specific dependencies.


## Configure and run experiments

Configure the experiment in
[`configs/experiments/mnist_run.yaml`](https://github.com/briannamueller/RigFL/blob/main/rigfl/templates/configs/experiments/mnist_run.yaml).

Some settings define the overarching configuration for RigFL's shared
experiment workflow, while others control how individual algorithms operate.
In this example, `local_epochs` and `lr` are algorithm-specific settings and the
rest are experiment settings shared across algorithms.

```yaml
algorithm: fedavg
dataset: mnist
model_arch: fedavg_mnist_cnn
rounds: 20
partition_seed: 0
split_seed: 0
training_seed: 0
eval_gap: 1
device: cpu
results_root: results
local_epochs: 1
lr: 0.05
```

`dataset` names an entry in the dataset configuration file
(`configs/datasets.yaml` by default). RigFL derives the partition fingerprint
from that entry and loads the matching partition from the data directory
(`data/` by default). Set `dataset_config` and `data_dir` to use a different
file or directory. A run's `partition_seed` and `split_seed` override the values
in its dataset entry.

`model_arch` can be a single model architecture or a model family (a defined
set of architectures assigned round robin across clients). See the [model reference](https://github.com/briannamueller/RigFL/blob/main/docs/reference/models.md) for the
available architectures and families.

See the [configuration guide](https://github.com/briannamueller/RigFL/blob/main/docs/configuration.md)
for working with run and sweep files. The
[experiment reference](https://github.com/briannamueller/RigFL/blob/main/docs/reference/experiment.md)
and [algorithm reference](https://github.com/briannamueller/RigFL/blob/main/docs/reference/algorithms.md)
document all supported settings, including their defaults and allowed values.

Run the experiment with:

```bash
rigfl run configs/experiments/mnist_run.yaml
```

This trains FedAvg for twenty communication rounds and writes the result under
`results/runs` (see
[`results_root`](https://github.com/briannamueller/RigFL/blob/main/docs/configuration.md#individual-runs)).
RigFL records accuracy, balanced accuracy, macro F1, predictive log loss, AUROC,
and AUPRC for each client at every evaluation round, together with the
corresponding sample counts. Early stopping is disabled by default; see [early stopping](https://github.com/briannamueller/RigFL/blob/main/docs/configuration.md#early-stopping).

## Sweeps

A set of experiments can be defined declaratively in a sweep file. RigFL expands
the sweep into independent tasks that can run sequentially. Tasks whose results
already exist are skipped, so sweeps can be resumed or extended incrementally.

The starter sweep in `configs/experiments/mnist_sweep.yaml` runs Local,
FedAvg, FedGH, and FedProto under the same three replicate conditions. Its
`model_arch` list contains a single architecture and a model family; since
FedAvg does not support heterogeneous client models, it runs only with the
single architecture (21 runs):

```yaml
name: mnist_sweep

base:
  dataset: mnist
  rounds: 20
  eval_gap: 1
  device: cpu
  results_root: results
  local_epochs: 1
  lr: 0.05

sweep:
  algorithm: [local, fedavg, fedgh, fedproto]
  model_arch: [fedavg_mnist_cnn, mnist_heterogeneous_3]

replicates:
  - {partition_seed: 0, split_seed: 0, training_seed: 0}
  - {partition_seed: 0, split_seed: 0, training_seed: 1}
  - {partition_seed: 0, split_seed: 0, training_seed: 2}
```

`base` specifies the settings shared by all 21
runs. Each entry in `replicates` defines a partition, validation-split, and
training-seed combination used once for each configuration.

Run the sweep's experiments sequentially on the current machine:

```bash
rigfl sweep configs/experiments/mnist_sweep.yaml --execute
```

RigFL writes the sweep grid to `results/mnist_sweep/grid.jsonl`, where each
line corresponds to one run.

To prepare the grid for execution through an external scheduler without running
the experiments locally, use `--snapshot` instead of `--execute` and follow the
[scheduler workflow in the configuration guide](https://github.com/briannamueller/RigFL/blob/main/docs/configuration.md#external-schedulers).

RigFL also supports hyperparameter tuning with Optuna. See the
[hyperparameter-tuning guide](https://github.com/briannamueller/RigFL/blob/main/docs/hyperparameter_tuning.md).

## Reporting results

Generate a report summarizing completed runs with:

```bash
rigfl report
```

The report is printed to the terminal and has two parts: aggregate performance
for each configuration, and, when matching Local results are available, a
client-level gain analysis. To also save the tables as CSV files in
`results/reports/`, add `--save`.

By default, `rigfl report` reads completed runs from `results/runs` and reports
test performance at the training round with the highest mean validation
accuracy across clients. If your runs are stored under a different results
root, pass `--results-root PATH`; runs are then read from `PATH/runs` and saved
tables are written to `PATH/reports`. To use a different reporting protocol,
change the default settings in `configs/reporting.yaml`. See the
[results guide](https://github.com/briannamueller/RigFL/blob/main/docs/results.md)
for further guidance.

### Aggregate performance

Each row is one configuration. Validation and test scores are averaged across
clients, then reported as a mean and 95% confidence interval across replicates.
The `p10` and `bottom-10%` columns show the 10th percentile and the mean of the
lowest-scoring 10% of clients.

### Client-level gain analysis

Aggregate performance metrics can signal that collaborative learning improves
upon local training on average, even though collaboration worsens performance at
some individual clients. This failure mode in federated learning is referred to
as negative transfer. RigFL provides evaluation metrics that surface unevenly
distributed benefits, such as when negative transfer at some clients is
cancelled out by larger improvements at others.

These include benefit rate, negative-transfer rate, negative-transfer
magnitude, negative-transfer burden, and worst-tail gain. They are reported
when matching Local results are available. See the
[results guide](https://github.com/briannamueller/RigFL/blob/main/docs/results.md)
for details.

### Resource measurements

Completed runs record communication volume, computational cost, and the hardware
used for each experiment. To include them in the report, add `--resources`.

## Adding an algorithm

See [Extending RigFL](https://github.com/briannamueller/RigFL/blob/main/docs/extensibility.md)
for the algorithm interface, registration, communication measurements, and
testing requirements.

## Experiment tracking with Weights & Biases

Install the optional W&B dependency with:

```bash
pip install "rigfl[wandb] @ git+https://github.com/briannamueller/RigFL.git"
```

Enable tracking by setting `wandb: true` in the experiment configuration file.

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
