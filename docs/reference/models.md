# Model reference

## Architectures

| Architecture | Input type |
|---|---|
| `lenet5` | `image` |
| `fedavg_mnist_cnn` | `image` |
| `small_cnn` | `image` |
| `fedavg_cnn` | `image` |
| `cifar_resnet18` | `image` |
| `cifar_mobilenet_v2` | `image` |
| `tabular_linear` | `numeric` |
| `tabular_mlp` | `numeric` |
| `tabular_residual_mlp` | `numeric` |
| `phishing_byte_cnn` | `token_sequence` |
| `temporal_gru` | `temporal` |
| `temporal_cnn` | `temporal` |
| `temporal_lstm` | `temporal` |

## Model families

| Family | Architectures |
|---|---|
| `mnist_heterogeneous_3` | `lenet5`, `fedavg_mnist_cnn`, `small_cnn` |
| `image_heterogeneous_3` | `fedavg_cnn`, `cifar_resnet18`, `cifar_mobilenet_v2` |
| `tabular_heterogeneous_3` | `tabular_linear`, `tabular_mlp`, `tabular_residual_mlp` |
| `phishing_byte_cnn` | `phishing_byte_cnn` |
| `temporal_heterogeneous_3` | `temporal_gru`, `temporal_cnn`, `temporal_lstm` |

For a model family, RigFL assigns architectures to clients in the listed order,
round-robin by client ID. The order is therefore part of the resolved experiment
identity. Algorithms that require homogeneous models use the single architecture
selected by `experiment.model` instead.
