# Algorithm configuration reference

## Capabilities

| Algorithm | Heterogeneous models |
|---|---|
| `local` | yes |
| `fedavg` | no |
| `fedcac` | no |
| `fedprox` | no |
| `fedamp` | no |
| `apple` | no |
| `fedpac` | no |
| `global` | yes |
| `fedproto` | yes |
| `fedgh` | yes |
| `lgfedavg` | yes |
| `fml` | yes |
| `fedkd` | yes |
| `fedtgp` | yes |

## `local`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `algorithm.local_epochs` | integer | `1` | ≥ 1 | Client training epochs per round. |
| `algorithm.lr` | number | `0.01` | > 0 | Client optimizer learning rate. |

## `fedavg`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `algorithm.local_epochs` | integer | `1` | ≥ 1 | Client training epochs per round. |
| `algorithm.lr` | number | `0.01` | > 0 | Client optimizer learning rate. |

## `fedcac`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `algorithm.local_epochs` | integer | `5` | ≥ 1 | Client training epochs per round. |
| `algorithm.lr` | number | `0.1` | > 0 | Client optimizer learning rate. |
| `algorithm.tau` | number | `0.5` | ≥ 0; ≤ 1 | Fraction of parameters selected as critical within each tensor. |
| `algorithm.beta` | integer | `100` | ≥ 1 | Round at which critical-parameter collaboration ends. |

## `fedprox`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `algorithm.local_epochs` | integer | `1` | ≥ 1 | Client training epochs per round. |
| `algorithm.lr` | number | `0.01` | > 0 | Client optimizer learning rate. |
| `algorithm.mu` | number | `0.01` | ≥ 0 | Weight of the proximal penalty. |

## `fedamp`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `algorithm.local_epochs` | integer | `1` | ≥ 1 | Client training epochs per round. |
| `algorithm.lr` | number | `0.01` | > 0 | Client optimizer learning rate. |
| `algorithm.lamda` | number | `1.0` | ≥ 0 | Weight of the proximal personalized-model objective. |
| `algorithm.alpha` | number | `1.0` | > 0 | Step size used to construct personalized cloud models. |
| `algorithm.sigma` | number | `1.0` | > 0 | Scale of the negative-exponential attention function. |

## `apple`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `algorithm.local_epochs` | integer | `5` | ≥ 1 | Client training epochs per round. |
| `algorithm.core_lr` | number | `0.01` | > 0 | Learning rate for core models. |
| `algorithm.relationship_lr` | number | `0.001` | > 0 | Learning rate for directed-relationship vectors. |
| `algorithm.momentum` | number | `0.9` | ≥ 0 | SGD momentum for local core training. |
| `algorithm.lr_decay` | number | `1.0` | > 0; ≤ 1 | Per-round multiplier for both learning rates. |
| `algorithm.mu` | number | `0.1` | ≥ 0 | Weight of the directed-relationship proximal term. |
| `algorithm.regularization_fraction` | number | `0.1` | > 0; ≤ 1 | Fraction of rounds during which relationship regularization decays. |
| `algorithm.scheduler` | string | `exponential` | `cosine`, `exponential` | Decay shape for relationship regularization. |

## `fedpac`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `algorithm.feature_epochs` | integer | `1` | ≥ 1 | Feature-extractor training epochs per round. |
| `algorithm.feature_lr` | number | `0.01` | > 0 | Feature-extractor optimizer learning rate. |
| `algorithm.head_lr` | number | `0.1` | > 0 | Classifier learning rate for its one local epoch. |
| `algorithm.lamda` | number | `1.0` | ≥ 0 | Weight of global feature-centroid alignment. |
| `algorithm.momentum` | number | `0.5` | ≥ 0 | Momentum for local SGD. |
| `algorithm.weight_decay` | number | `0.0005` | ≥ 0 | Weight decay for local SGD. |
| `algorithm.qp_weight_threshold` | number | `0.001` | ≥ 0 | Classifier weights at or below this value are set to zero. |
| `algorithm.qp_eigen_threshold` | number | `0.01` | > 0 | Eigenvalue threshold used when repairing a non-PSD QP matrix. |
| `algorithm.qp_max_iterations` | integer | `5000` | ≥ 1 | Maximum projected-gradient iterations per classifier QP. |
| `algorithm.qp_tolerance` | number | `1e-10` | > 0 | Convergence tolerance for classifier QP solves. |

## `global`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `algorithm.local_epochs` | integer | `1` | ≥ 1 | Client training epochs per round. |
| `algorithm.lr` | number | `0.01` | > 0 | Client optimizer learning rate. |

## `fedproto`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `algorithm.local_epochs` | integer | `1` | ≥ 1 | Client training epochs per round. |
| `algorithm.lr` | number | `0.01` | > 0 | Client optimizer learning rate. |
| `algorithm.lamda` | number | `0.1` | ≥ 0 | Weight of the prototype-alignment loss. |

## `fedgh`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `algorithm.local_epochs` | integer | `1` | ≥ 1 | Client training epochs per round. |
| `algorithm.lr` | number | `0.01` | > 0 | Client optimizer learning rate. |
| `algorithm.server_epochs` | integer | `1` | ≥ 1 | Server header-training epochs per round. |
| `algorithm.server_lr` | number | `0.01` | > 0 | Server optimizer learning rate. |

## `lgfedavg`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `algorithm.local_epochs` | integer | `1` | ≥ 1 | Client training epochs per round. |
| `algorithm.lr` | number | `0.1` | > 0 | Client optimizer learning rate. |

## `fml`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `algorithm.local_epochs` | integer | `1` | ≥ 1 | Client training epochs per round. |
| `algorithm.lr` | number | `0.01` | > 0 | Client optimizer learning rate. |
| `algorithm.alpha` | number | `0.5` | ≥ 0; ≤ 1 | Private-model classification weight. |
| `algorithm.beta` | number | `0.5` | ≥ 0; ≤ 1 | Meme-model classification weight. |
| `algorithm.aux_model` | string \| null | `null` | — | Architecture used for the shared meme model. |

## `fedkd`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `algorithm.local_epochs` | integer | `1` | ≥ 1 | Client training epochs per round. |
| `algorithm.lr` | number | `0.01` | > 0 | Client optimizer learning rate. |
| `algorithm.t_start` | number | `0.95` | ≥ 0; ≤ 1 | Initial SVD energy-retention threshold. |
| `algorithm.t_end` | number | `0.98` | ≥ 0; ≤ 1 | Final SVD energy-retention threshold. |
| `algorithm.aux_model` | string \| null | `null` | — | Architecture used for the shared mentee model. |

## `fedtgp`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `algorithm.local_epochs` | integer | `1` | ≥ 1 | Client training epochs per round. |
| `algorithm.lr` | number | `0.01` | > 0 | Client optimizer learning rate. |
| `algorithm.lamda` | number | `0.1` | ≥ 0 | Weight of the client prototype loss. |
| `algorithm.server_epochs` | integer | `1` | ≥ 1 | Server prototype-training epochs per round. |
| `algorithm.server_lr` | number | `0.01` | > 0 | Server optimizer learning rate. |
| `algorithm.margin_cap` | number | `100.0` | > 0 | Maximum adaptive contrastive margin. |
