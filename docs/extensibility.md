# Extensibility

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

Tensor, encoded-byte, module, dataclass, mapping, and sequence payloads are
measured automatically for communication reporting. Override
`communication_payload_bytes(...)` for other payload representations.

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
> instead of changing the meaning of the standard operations.

### Return predictions

Use `Predictions.from_logits(...)` for model scores or
`Predictions.from_probabilities(...)` for normalized class probabilities. If
the algorithm only produces class labels, use `Predictions.labels_only(...)`;
probability-based metrics such as log loss will then be unavailable.

### Additional configuration and registration options

Give user-facing configuration fields a description so they appear in the
[algorithm reference](reference/algorithms.md). Unknown fields are rejected.

The inherited `from_config` constructor works when the algorithm needs only
its validated settings. If it also needs a model template or another resource,
override `from_config`. See [`FedAvg`](../rigfl/algorithms/fedavg.py) for an
example of the complete interface.

Set `supports_model_heterogeneity=False` if every client must have the same
architecture. Set `requires_client_model=False` if the algorithm does not use
RigFL's standard client models.

Add the name to `ALL_ALGORITHMS` for `--algorithm all`. Add it to `BASELINES`
only if it should also run under `--algorithm baselines`.

### Check the integration

Add tests for configuration validation, training updates, predictions, and
the runner contract. Then update the algorithm reference and run the tests:

```bash
python scripts/generate_config_reference.py
pytest -q
```
