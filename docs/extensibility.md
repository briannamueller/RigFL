# Extensibility

## Adding an algorithm

To add a new algorithm, you must edit RigFL's source code. Clone the GitHub
repository and install it with `pip install -e .` (see
[Development and testing](../README.md#development-and-testing)).

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
   server.
3. `aggregate(...)` receives all client uploads, performs the server-side
   computation, and returns the shared state for the next round.
4. `predict(...)` performs inference for the supplied inputs and returns a
   `Predictions` object.

Tensor, encoded-byte, module, dataclass, mapping, and sequence payloads are
measured automatically for communication reporting. Override
`communication_payload_bytes(...)` for other payload representations.

Declare all of the relevant arguments for the algorithm in its configuration
class.

```python
from pydantic import Field

from rigfl.core import Algorithm, Predictions
from rigfl.core.config import AlgorithmConfig


class NewAlgorithmConfig(AlgorithmConfig):
    local_epochs: int = Field(1, ge=1, description="Client training epochs per round.")
    lr: float = Field(0.01, gt=0, description="Client optimizer learning rate.")
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

Access the arguments defined in the algorithm's configuration class through `self.config`, such as `self.config.lr`.

Register both classes in `rigfl/experiment/registry.py`:

```python
REGISTRY = {
    "local": AlgorithmSpec(Local, LocalConfig),
    "fedavg": AlgorithmSpec(FedAvg, FedAvgConfig),
    # ...other algorithms
    "new_algorithm": AlgorithmSpec(NewAlgorithm, NewAlgorithmConfig),
}
```

`AlgorithmSpec` uses the `iterative` runner by default; algorithms that don't fit
repeated local training and aggregation can register a different runner.

By default, each client model's head sits directly on its backbone's native
output. If the algorithm exchanges or mixes representations across clients, set
`representation_adapter="projection"` (a learned linear map) or `"pool"`
(average pooling) on its `AlgorithmSpec`; client representations are then mapped
to the experiment's `shared_dim`.

### Return predictions

Use `Predictions.from_logits(...)` for model scores or
`Predictions.from_probabilities(...)` for normalized class probabilities. If
the algorithm only produces class labels, use `Predictions.labels_only(...)`;
probability-based metrics such as log loss will then be unavailable.

### Additional configuration and registration options

Use applicable Pydantic `Field` constraints to reject invalid configuration
values, and give every user-facing field a `description` for the generated
[algorithm reference](reference/algorithms.md).
See the [Pydantic `Field` API](https://docs.pydantic.dev/latest/api/fields/#pydantic.fields.Field)
for the full set of field options. Unknown fields are rejected.

The inherited `from_config` constructor works when the algorithm needs only
its validated settings. If it also needs a model template or another resource,
override `from_config`. See [`FedAvg`](../rigfl/algorithms/fedavg.py) for an
example of the complete interface.

Pass `supports_model_heterogeneity=False` to `AlgorithmSpec` if every client must
have the same architecture. Pass `requires_client_model=False` if the algorithm
does not use RigFL's standard client models.

### Check the integration

Add focused tests for the algorithm's scientific calculations, paper-defined
updates, or reproducibility invariants. The shared parametrized end-to-end test
automatically covers construction and standard-runner integration for every
registered algorithm. Then update the algorithm reference and run the tests:

```bash
python scripts/generate_config_reference.py
pytest -q
```

## Algorithm plugins

An algorithm can also be installed as a plugin: a separate package that
RigFL loads at startup. Use this when the algorithm belongs to another
project, such as a paper's code, and you want to run it with a pinned RigFL
release instead of a modified copy of RigFL.

Write the same classes as above, then declare an `AlgorithmSpec` under the
`rigfl.algorithms` entry point in the package's `pyproject.toml`:

```toml
[project.entry-points."rigfl.algorithms"]
my_algorithm = "my_paper.algorithm:SPEC"
```

After `pip install -e .` in that package, every `rigfl` command accepts
`my_algorithm` like a built-in algorithm. Each run records the plugin's
version and git commit; list other dependencies to record in
`AlgorithmSpec(provenance_packages=...)`.

List settings the algorithm doesn't use in `ignored_experiment_fields`. They're
recorded as null and left out of the run fingerprint. The report includes the
algorithm in every condition that matches its other settings, and a report
filter on an ignored setting keeps it. Saved CSVs list each configuration once,
with ignored settings left empty.

An algorithm that doesn't train in rounds can pass its own `runner` to
`AlgorithmSpec`. RigFL calls it as
`runner(algorithm, clients, num_rounds, device, num_classes, **options)`, and it
must return `{"evaluation_history": ...}` built with
`rigfl.core.append_evaluation`.
