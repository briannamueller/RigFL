# RigFL

RigFL is a modular framework for rigorous federated learning experimentation.
Algorithm-specific behavior is isolated behind a common interface, so that algorithms use the same orchestration, evaluation, configuration, and reporting
machinery.

- [Key Features](#key-features)
- [Installation](#installation)
- [Example workflow](#example-workflow)
- [Algorithms](#algorithms)
- [Sweeps and tuning](#sweeps-and-tuning)
- [Adding an algorithm](#adding-an-algorithm)
- [Experiment tracking with Weights & Biases](#experiment-tracking-with-weights--biases)
- [Development and testing](#development-and-testing)

## Key Features

- **Stable experiment and partition identity.** RigFL derives two separate
  fingerprints: one identifying a result from its distinct experiment
  configuration, the other identifying a partitioned dataset from its data
  configuration. A change to either produces a new identity, so earlier generated
  results and partitions are never overwritten. An experiment whose result
  already exists is not rerun—expanding or changing a sweep will only execute new
  combinations.

- **Support for model-heterogeneous algorithms.** RigFL supports algorithms
  designed for clients with different model architectures. The architecture
  selection can be configured as a named family or an explicit ordered list.

- **Joint hyperparameter tuning across multiple seeds.** Support for evaluating
  combinations of hyperparameters across several random seeds.

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

## Installation

```bash
pip install rigfl
```

RigFL requires Python 3.10–3.12.

## Example workflow

The following CIFAR-10 example walks you through generating client data
partitions, running an experiment, and reporting the results.

### Generate client data partitions

First define the data source and data partitioning configuration in
[`configs/datasets.yaml`](https://github.com/briannamueller/RigFL/blob/main/configs/datasets.yaml):

```yaml
datasets:
  cifar10:
    backend: flower
    source_dataset: uoft-cs/cifar10
    partition:
      scheme: dirichlet
      num_clients: 3
      alpha: 0.5
```

Generate client datasets by running:

```bash
python -m rigfl.data.generate --dataset cifar10
```

RigFL passes the settings to Flower, derives a stable fingerprint from the data
partitioning configuration, and saves the generated files under:

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

Running the command again with the same data configuration reuses the existing
partition. Changing a partitioning entry produces a different fingerprint and a
separate directory instead of replacing the previous partition.

To add another dataset, create another entry in
[`configs/datasets.yaml`](https://github.com/briannamueller/RigFL/blob/main/configs/datasets.yaml).
See the
[data configuration guide](https://github.com/briannamueller/RigFL/blob/main/rigfl/data/README.md)
for the available settings and guidance for datasets with multiple
configurations, nonstandard splits, or ambiguous input and target columns.

### Define and run the experiment

YAML files define experiment configurations:
[`experiments/cifar10_run.yaml`](https://github.com/briannamueller/RigFL/blob/main/experiments/cifar10_run.yaml):

```yaml
experiment:
  dataset: cifar10
  model_architectures: [fedavg_cnn]
  rounds: 2
  seed: 0
  shared_dim: 128
  eval_gap: 1
  device: cpu
  out_dir: results/cifar10_run

algorithm:
  local_epochs: 1
  lr: 0.01
```

The YAML has two sections. Entries under `experiment` define the overarching
configuration for the execution of RigFL’s shared workflow. Entries under
`algorithm` specify how individual algorithms operate. An algorithm entry may be
supported by one or several algorithms. In a multi-algorithm sweep, each entry is
applied only to algorithms that support it.

Run the experiment with:

```bash
python -m rigfl.experiment.run \
  --algorithm fedavg \
  --config experiments/cifar10_run.yaml
```

This trains FedAvg for two communication rounds and writes results to
`results/cifar10_run`.

### Report results

Summarize the results with:

```bash
python -m rigfl.experiment.collect \
  --results-dir results/cifar10_run
```

### Client-centered metrics

Aggregate performance metrics can signal that collaborative learning improves
upon local training on average, even though collaboration worsens performance at
some individual clients. RigFL provides evaluation metrics that surface unevenly
distributed benefits.

- **Win rate:** the fraction of matched client-and-seed pairs in which an algorithm
  results in improved performance over the Local baseline.

- **Performance among the worst-served clients:** reports the average performance
  of the lowest-scoring 10% of clients and the 10th-percentile score, which marks
  the lower tail of the client-performance distribution.

- **Standard deviation:** the spread in performance across clients.

## Algorithms

Implemented algorithms:

- [FedAvg](https://proceedings.mlr.press/v54/mcmahan17a.html)
- [FedProx](https://arxiv.org/abs/1812.06127)
- [FedProto](https://ojs.aaai.org/index.php/AAAI/article/view/20819)
- [FedGH](https://arxiv.org/abs/2303.13137)
- [LG-FedAvg](https://arxiv.org/abs/2001.01523)
- [FML](https://arxiv.org/abs/2006.16765)
- [FedKD](https://www.nature.com/articles/s41467-022-29763-x)
- [FedTGP](https://ojs.aaai.org/index.php/AAAI/article/view/29617)
- [FedDES](https://arxiv.org/abs/2603.28006)

Local training and Global Ensemble are available as reference baselines.

Algorithm-specific departures from the original papers are documented in
[DEVIATIONS.md](https://github.com/briannamueller/RigFL/blob/main/DEVIATIONS.md).

## Sweeps and tuning

[`experiments/cifar10_tune.yaml`](https://github.com/briannamueller/RigFL/blob/main/experiments/cifar10_tune.yaml)
provides a multi-algorithm, multi-seed tuning example. A sweep expands the values
defined along each axis. Algorithm entries are applied only to algorithms that
support them, so options belonging to different algorithms are not unnecessarily
cross-multiplied.

Expand the sweep and print its cluster submission command with:

```bash
python -m rigfl.experiment.launch \
  --config experiments/cifar10_tune.yaml \
  --queue <queue>
```

Each complete hyperparameter combination is treated as one candidate, with its
seeds aggregated as replicates. Rank the completed candidates and write runnable
selected configurations with:

```bash
python -m rigfl.experiment.collect \
  --results-dir results/cifar10_tune \
  --selection-metric accuracy \
  --selection-view both \
  --rank \
  --select-out results/cifar10_tune_selected
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
