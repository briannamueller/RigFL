# RigFL

RigFL is a modular framework for rigorous federated learning experimentation.
Algorithm-specific behavior is isolated behind a common interface, so that algorithms use the same orchestration, evaluation, configuration, and reporting
machinery.

## Key Features

- **Stable experiment and partition identity.** RigFL derives two separate
  fingerprints: one identifying a result from its distinct experiment
  configuration, the other identifying a partitioned dataset from its data
  configuration. A change to either produces a new identity, so earlier generated
  results and partitions are never overwritten. An experiment whose result
  already exists is not rerun—expanding or changing a sweep will only execute new
  combinations.

- **Joint hyperparameter tuning across multiple seeds.** Evaluate complete
  hyperparameter configurations with Optuna grid or adaptive search.

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
  --config experiments/cifar10_run.yaml

python -m rigfl.experiment.collect
```

`rigfl init` creates a starter project with two configuration files. Datasets are configured in `configs/datasets.yaml` (see [Data partitions](#data-partitions)),
while experiments are configured in YAML files such as
`experiments/cifar10_run.yaml` (see
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
[data configuration guide](https://github.com/briannamueller/RigFL/blob/main/rigfl/data/README.md)
for the available settings and guidance for datasets with multiple
configurations, nonstandard splits, or ambiguous input and target columns.

Naturally partitioned biomedical datasets are available through the optional
BioSilo backend, installed with `pip install "rigfl[biosilo]"`. BioSilo
partitions are not automatically generated when an experiment runs, so run the generation
command above first. The data configuration guide covers BioSilo's configuration
and dataset-specific dependencies.


## Configure and run experiments

Configure the experiment in
[`experiments/cifar10_run.yaml`](https://github.com/briannamueller/RigFL/blob/main/experiments/cifar10_run.yaml).

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

`dataset` names an entry in configs/datasets.yaml. RigFL derives the partition
fingerprint from that entry and loads the matching partition. By default, RigFL reads dataset configurations from `configs/datasets.yaml` and stores
partitions under `data/`. Set `experiment.dataset_config` and
`experiment.data_dir` to override these paths.

Specify the model architecture with `model`. For algorithms that support heterogeneous
architectures, optionally specify a `model_family`. See the
[model configuration guide](https://github.com/briannamueller/RigFL/blob/main/rigfl/models/README.md)
for the available architectures and model families.


Run the experiment with:

```bash
python -m rigfl.experiment.run \
  --algorithm fedavg \
  --config experiments/cifar10_run.yaml
```

This trains FedAvg for two communication rounds and writes the result under
`results/runs`. RigFL records accuracy, balanced accuracy, macro F1, and predictive log loss for
each client at every evaluation round, together with the corresponding sample
counts. Each result file contains the resolved experiment and algorithm configurations and
the complete per-client evaluation history.



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

The following sweep defined in `experiments/cifar10_sweep.yaml` runs Local and FedAvg at
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
python -m rigfl.experiment.launch --config experiments/cifar10_sweep.yaml
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
### Variance studies

To measure how much results depend on each source of randomness, vary the seeds
as sweep axes instead of replicates. Experiment settings take an `exp.` prefix on
sweep axes:

```yaml
sweep:
  exp.partition_seed: [0, 1, 2]
  exp.split_seed: [0, 1, 2]
  exp.seed: [0, 1, 2]
```

This runs all 27 seed combinations for each configuration. Analyze them with the
variance command rather than `collect`, which would treat the crossed runs as
independent replicates:

```bash
python -m rigfl.experiment.variance \
  --grid results/<sweep-name>/grid.jsonl \
  --out-json results/variance.json
```


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

Collection summarizes every result under `results/runs` in a table with one row
per configuration, showing validation and test performance at the reporting
round, aggregated across clients. When a configuration has been run under
multiple replicates, as in a sweep, its row reports the mean across replicates
with a 95% confidence interval.

### Client-centered metrics

Aggregate performance metrics can signal that collaborative learning improves
upon local training on average, even though collaboration worsens performance at
some individual clients. RigFL provides evaluation metrics that surface unevenly
distributed benefits.

- **Negative-transfer rate:** the fraction of matched client-and-seed pairs that
  perform worse than Local by more than the selected threshold. Benefit and
  neutral rates are reported alongside it.

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
for an experiment with `experiment.estimate_flops: true` or `--estimate-flops`.
FLOP estimation requires PyTorch 2.1 or newer and records training and inference
separately.

Add resource measurements to the report with:

```bash
python -m rigfl.experiment.collect --include-resources
```

## Adding an algorithm

Extend RigFL by adding a module under `rigfl/algorithms/` containing:

- A configuration class that inherits from `AlgorithmConfig`.
- An algorithm class that inherits from `Algorithm`.

The algorithm class must define four operations:

1. `init_globals()` initializes the shared state, which represents the
   information the server maintains and distributes to clients at the start of
   each round. The shared state may take the form of a global model, model
   parameters, prototypes, a classifier head, or another algorithm-specific
   structure.
2. `local_train(...)` is called once per client per round. It receives the client
   and shared state, performs the client-side computation, and returns the
   client's upload, which represents the information the client sends to the
   server. The upload may have the same form as the shared state, be a different
   structure entirely, or carry additional information required for server-side
   computation.
3. `aggregate(...)` receives all client uploads, performs the server-side
   computation, and returns the shared state for the next round. This may involve
   averaging parameters, combining prototypes, or training a server-side
   component.
4. `predict(...)` performs inference for the supplied inputs and returns a
   `Predictions` object.

Tensor and encoded-byte payloads are measured automatically for communication
reporting. Override `communication_payload_bytes(...)` if the algorithm exchanges
another payload representation.

Declare all of the relevant arguments for the algorithm in its configuration
class.

```python
from rigfl.core import Algorithm, Predictions
from rigfl.core.config import AlgorithmConfig


class NewAlgorithmConfig(AlgorithmConfig):
    local_epochs: int = 1
    lr: float = 0.01
    # ...additional arguments


class NewAlgorithm(Algorithm):
    def init_globals(self):
        ...

    def local_train(self, client, shared_state):
        ...

    def aggregate(self, client_uploads, shared_state):
        ...

    def predict(self, client, x, shared_state) -> Predictions:
        ...
```

In `local_train(...)` and `predict(...)`, `client` refers to the
`Client` instance being processed. The client's local model and training data loader are accessed
through `client.model` and `client.train_loader`, respectively. `client.state` is
a dictionary that can carry any additional client-specific information that must
persist across rounds.

Access the arguments defined in the algorithm’s configuration class through self.config, such as self.config.lr.

Register both classes in `rigfl/experiment/registry.py`:

```python
REGISTRY = {
    "local": AlgorithmSpec(Local, LocalConfig),
    "fedavg": AlgorithmSpec(FedAvg, FedAvgConfig),
    # ...other algorithms
    "new_algorithm": AlgorithmSpec(NewAlgorithm, NewAlgorithmConfig),
}
```

> **Runner note:** `AlgorithmSpec` uses the `iterative` runner by default. If an
> algorithm genuinely cannot be expressed as repeated local training followed
> by aggregation, define a different runner and matching operation protocol
> instead of changing the meaning of the standard operations. FedDES is one
> such exception and uses `p2p_one_shot`.

## Experiment tracking with Weights & Biases

Install the optional W&B dependency with:

```bash
pip install "rigfl[wandb]"
```

Enable tracking by setting `wandb: true` under `experiment` in the YAML
configuration file, or pass `--wandb` when running experiments from the command
line.

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
