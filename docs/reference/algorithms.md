# Algorithm configuration reference

## Capabilities

| Algorithm | Heterogeneous models | Representation adapter |
|---|---|---|
| `local` | yes | none |
| `fedproto` | yes | projection |
| `fedgh` | yes | projection |
| `fml` | yes | none |
| `fedtgp` | yes | pool |
| `fedavg` | no | none |
| `fedprox` | no | none |
| `fedcac` | no | none |
| `fedapa` | no | none |
| `fedapen` | yes | none |
| `fedamp` | no | none |
| `apple` | no | none |
| `fedpac` | no | none |
| `pfedmoe` | yes | projection |
| `global_ensemble` | yes | none |

## `local`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `local_epochs` | integer | `1` | ≥ 1 | Client training epochs per round. |
| `lr` | number | `0.01` | > 0 | Client optimizer learning rate. |

## `fedproto`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `local_epochs` | integer | `1` | ≥ 1 | Client training epochs per round. |
| `lr` | number | `0.01` | > 0 | Client optimizer learning rate. |
| `lamda` | number | `0.1` | ≥ 0 | Weight of the prototype-alignment loss. |

## `fedgh`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `local_epochs` | integer | `1` | ≥ 1 | Client training epochs per round. |
| `lr` | number | `0.01` | > 0 | Client optimizer learning rate. |
| `server_epochs` | integer | `1` | ≥ 1 | Server header-training epochs per round. |
| `server_lr` | number | `0.01` | > 0 | Server optimizer learning rate. |

## `fml`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `local_epochs` | integer | `1` | ≥ 1 | Client training epochs per round. |
| `lr` | number | `0.01` | > 0 | Client optimizer learning rate. |
| `alpha` | number | `0.5` | ≥ 0; ≤ 1 | Private-model classification weight. |
| `beta` | number | `0.5` | ≥ 0; ≤ 1 | Meme-model classification weight. |
| `aux_model_arch` | string \| null | `null` | — | Architecture used for the shared meme model. Defaults to model_arch, or its family's first member. |

## `fedtgp`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `local_epochs` | integer | `1` | ≥ 1 | Client training epochs per round. |
| `lr` | number | `0.01` | > 0 | Client optimizer learning rate. |
| `lamda` | number | `0.1` | ≥ 0 | Weight of the client prototype loss. |
| `server_epochs` | integer | `100` | ≥ 1 | Server prototype-training epochs per round. |
| `server_lr` | number | `0.01` | > 0 | Server optimizer learning rate. |
| `margin_cap` | number | `100.0` | > 0 | Maximum adaptive contrastive margin. |

## `fedavg`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `local_epochs` | integer | `1` | ≥ 1 | Client training epochs per round. |
| `lr` | number | `0.01` | > 0 | Client optimizer learning rate. |

## `fedprox`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `local_epochs` | integer | `1` | ≥ 1 | Client training epochs per round. |
| `lr` | number | `0.01` | > 0 | Client optimizer learning rate. |
| `mu` | number | `0.01` | ≥ 0 | Weight of the proximal penalty. |

## `fedcac`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `local_epochs` | integer | `5` | ≥ 1 | Client training epochs per round. |
| `lr` | number | `0.1` | > 0 | Client optimizer learning rate. |
| `tau` | number | `0.5` | ≥ 0; ≤ 1 | Fraction of parameters selected as critical within each tensor. |
| `beta` | integer | `100` | ≥ 1 | Round at which critical-parameter collaboration ends. |

## `fedapa`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `local_epochs` | integer | `5` | ≥ 1 | Client training epochs per round. |
| `lr` | number | `0.01` | > 0 | Client optimizer learning rate. |
| `aggregation_lr` | number | `0.01` | > 0 | Server learning rate for client aggregation-weight vectors. |
| `mu` | number | `0.5` | > 0; ≤ 1 | Self-weight assigned before each aggregation-vector normalization. |

## `fedapen`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `local_epochs` | integer | `5` | ≥ 1 | Client ensemble-training epochs per round. |
| `lr` | number | `0.01` | > 0 | Private- and shared-model learning rate. |
| `adaptation_fraction` | number | `0.05` | > 0; < 1 | Fraction of client training samples reserved to learn the ensemble weight. |
| `adaptation_epochs` | integer | `10` | ≥ 1 | Ensemble-weight training epochs per round. |
| `adaptation_lr` | number | `0.001` | > 0 | Learning rate for the client ensemble weight. |
| `initial_lambda` | number | `0.5` | ≥ 0; ≤ 1 | Initial weight assigned to the private model's probabilities. |
| `aux_model_arch` | string \| null | `null` | — | Architecture used for the complete homogeneous shared model. Defaults to model_arch, or its family's first member. |

## `fedamp`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `local_epochs` | integer | `1` | ≥ 1 | Client training epochs per round. |
| `lr` | number | `0.01` | > 0 | Client optimizer learning rate. |
| `self_weight` | number | `0.5` | ≥ 0; ≤ 1 | Weight of the client's own model in its personalized cloud. |
| `sigma` | number | `10.0` | > 0 | Distance scale for normalized peer attention. |
| `beta` | number | `10000.0` | > 0 | Denominator of the squared proximal penalty. |

## `apple`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `local_epochs` | integer | `5` | ≥ 1 | Client training epochs per round. |
| `core_lr` | number | `0.01` | > 0 | Learning rate for core models. |
| `relationship_lr` | number | `0.001` | > 0 | Learning rate for directed-relationship vectors. |
| `momentum` | number | `0.9` | ≥ 0 | SGD momentum for local core training. |
| `lr_decay` | number | `1.0` | > 0; ≤ 1 | Per-round multiplier for both learning rates. |
| `mu` | number | `0.1` | ≥ 0 | Weight of the directed-relationship proximal term. |
| `regularization_fraction` | number | `0.1` | > 0; ≤ 1 | Fraction of rounds during which relationship regularization decays. |
| `scheduler` | string | `exponential` | `cosine`, `exponential` | Decay shape for relationship regularization. |

## `fedpac`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `feature_epochs` | integer | `1` | ≥ 1 | Feature-extractor training epochs per round. |
| `feature_lr` | number | `0.01` | > 0 | Feature-extractor optimizer learning rate. |
| `head_lr` | number | `0.1` | > 0 | Classifier learning rate for its one local epoch. |
| `lamda` | number | `1.0` | ≥ 0 | Weight of global feature-centroid alignment. |
| `momentum` | number | `0.5` | ≥ 0 | Momentum for local SGD. |
| `weight_decay` | number | `0.0005` | ≥ 0 | Weight decay for local SGD. |
| `qp_weight_threshold` | number | `0.001` | ≥ 0 | Classifier weights at or below this value are set to zero. |
| `qp_eigen_threshold` | number | `0.01` | > 0 | Eigenvalue threshold used when repairing a non-PSD QP matrix. |
| `qp_max_iterations` | integer | `5000` | ≥ 1 | Maximum projected-gradient iterations per classifier QP. |
| `qp_tolerance` | number | `1e-10` | > 0 | Convergence tolerance for classifier QP solves. |

## `pfedmoe`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `local_epochs` | integer | `1` | ≥ 1 | Client training epochs per round. |
| `lr` | number | `0.01` | > 0 | Private-model learning rate. |
| `proxy_lr` | number | `0.01` | > 0 | Proxy-extractor learning rate. |
| `gate_lr` | number | `0.01` | > 0 | Private gating-network learning rate. |
| `gate_hidden_dim` | integer | `32` | ≥ 1 | Width of the private two-layer gating network. |
| `aux_model_arch` | string \| null | `null` | — | Architecture used for the shared homogeneous proxy extractor. Defaults to model_arch, or its family's first member. |

## `global_ensemble`

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `local_epochs` | integer | `1` | ≥ 1 | Client training epochs per round. |
| `lr` | number | `0.01` | > 0 | Client optimizer learning rate. |
