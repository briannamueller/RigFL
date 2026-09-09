# Models

Every experiment has a `model`, which is used by algorithms that require the
same architecture on every client:

```yaml
experiment:
  model: fedavg_cnn
```

Algorithms that support model heterogeneity can instead use a registered
`model_family`:

```yaml
experiment:
  model: fedavg_cnn
  model_family: image_heterogeneous_3
```

When `model_family` is omitted, every algorithm uses `model`. When it is
specified, model-heterogeneous algorithms use the complete family, while
homogeneous algorithms continue to use `model`.

## Available architectures

| Intended data | Architectures |
|---|---|
| 28×28 images, including MNIST, Fashion-MNIST, and FEMNIST | `lenet5`, `fedavg_mnist_cnn`, `small_cnn` |
| CIFAR-10, CIFAR-100, and Tiny ImageNet | `fedavg_cnn`, `cifar_resnet18`, `cifar_mobilenet_v2` |
| Fixed-width numeric data, including PaySim | `tabular_linear`, `tabular_mlp`, `tabular_residual_mlp` |
| Phishing URL byte sequences | `phishing_byte_cnn` |
| Paired time-series and static data from BioSilo | `temporal_gru`, `temporal_cnn`, `temporal_lstm` |

## Model families

The order shown here is the order used when assigning models to clients.

| Family | Architectures |
|---|---|
| `mnist_heterogeneous_3` | `lenet5`, `fedavg_mnist_cnn`, `small_cnn` |
| `image_heterogeneous_3` | `fedavg_cnn`, `cifar_resnet18`, `cifar_mobilenet_v2` |
| `tabular_heterogeneous_3` | `tabular_linear`, `tabular_mlp`, `tabular_residual_mlp` |
| `phishing_byte_cnn` | `phishing_byte_cnn` |
| `temporal_heterogeneous_3` | `temporal_gru`, `temporal_cnn`, `temporal_lstm` |

Adding another registered architecture does not change an existing family.

## Auxiliary models

FML and FedKD use an additional homogeneous model. Set it with `aux_model` in
the algorithm configuration:

```yaml
algorithm:
  aux_model: cifar_resnet18
```

If `aux_model` is omitted, these algorithms use the first architecture in
`model_family`, or `model` when no family is specified.
