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
| `cifar_cnn_37_74_148` | `image` |
| `cifar_cnn_43_86_172` | `image` |
| `cifar_cnn_49_98_196` | `image` |
| `cifar_cnn_55_110_220` | `image` |
| `cifar_cnn_61_122_244` | `image` |
| `cifar_cnn_64_128_256` | `image` |
| `cifar_cnn_67_134_268` | `image` |
| `cifar_cnn_73_146_292` | `image` |
| `cifar_cnn_79_158_316` | `image` |
| `cifar_cnn_85_170_340` | `image` |
| `cifar_cnn_91_182_364` | `image` |
| `fedgh_cnn_1` | `image` |
| `fedgh_cnn_2` | `image` |
| `fedgh_cnn_3` | `image` |
| `fedgh_cnn_4` | `image` |
| `fedgh_cnn_5` | `image` |
| `mobilenet_v2` | `image` |
| `convmixer_256d8` | `image` |
| `tabular_linear` | `numeric` |
| `tabular_logreg` | `numeric` |
| `tabular_mlp` | `numeric` |
| `tabular_mlp_64` | `numeric` |
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
| `cifar_width_heterogeneous_10` | `cifar_cnn_37_74_148`, `cifar_cnn_43_86_172`, `cifar_cnn_49_98_196`, `cifar_cnn_55_110_220`, `cifar_cnn_61_122_244`, `cifar_cnn_67_134_268`, `cifar_cnn_73_146_292`, `cifar_cnn_79_158_316`, `cifar_cnn_85_170_340`, `cifar_cnn_91_182_364` |
| `fedgh_heterogeneous_5` | `fedgh_cnn_1`, `fedgh_cnn_2`, `fedgh_cnn_3`, `fedgh_cnn_4`, `fedgh_cnn_5` |
| `brats_heterogeneous_2` | `convmixer_256d8`, `mobilenet_v2` |
| `tabular_heterogeneous_3` | `tabular_linear`, `tabular_mlp`, `tabular_residual_mlp` |
| `scrnaseq_heterogeneous_2` | `tabular_logreg`, `tabular_mlp_64` |
| `temporal_heterogeneous_3` | `temporal_gru`, `temporal_cnn`, `temporal_lstm` |

For a model family, RigFL assigns architectures to clients in the listed order, round-robin by client ID. The order is therefore part of the resolved experiment identity. `model_arch` accepts an architecture name or a family name; algorithms that require identical client architectures accept only an architecture name and are skipped for family values in sweeps.
