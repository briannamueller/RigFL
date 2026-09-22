# Algorithm configuration reference

## Capabilities

| Algorithm | Heterogeneous models |
|---|---|
| `local` | yes |
| `fedavg` | no |
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
| `feddes` | yes |

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

## `feddes`

`base.models` is set from the experiment's model selection and cannot be configured here. `base_models_per_client` chooses whether each client trains the full selected family or its standard client-ID-assigned model. `base.split_mode` must be `oof_stacking`. Other settings under `algorithm.graphroute` follow the [GraphRoute configuration guide](https://github.com/briannamueller/GraphRoute#configuration).

FedDES additionally provides `local_embedding` as a GraphRoute node or edge feature source. It concatenates representations from the current client's locally trained models in their stable pool order; with `base_models_per_client: assigned`, it is simply that client's one model representation. `graph.embedding_normalization: per_model_l2` normalizes each local model representation before concatenation. Unlike GraphRoute's built-in `embedding_concat`, `local_embedding` does not use models communicated by other clients.

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `algorithm.graphroute` | mapping | see below | — | Settings passed to GraphRoute. |
| `algorithm.base_models_per_client` | string | `assigned` | `all`, `assigned` | Base models trained by each client: every model in the experiment family, or the single model assigned by client ID. |
| `algorithm.cache_dir` | string | `pool_cache` | — | Directory used to reuse trained base pools; empty disables caching. |

### GraphRoute defaults changed by RigFL

| Setting | GraphRoute default | RigFL default |
|---|---|---|
| `algorithm.graphroute.base.oof_folds` | `5` | `3` |
| `algorithm.graphroute.base.epochs` | `300` | `100` |
| `algorithm.graphroute.base.batch_size` | `10` | `64` |
| `algorithm.graphroute.gnn.epochs` | `300` | `500` |
| `algorithm.graphroute.gnn.patience` | `20` | `50` |
| `algorithm.graphroute.gnn.es_metric` | `val_loss` | `val_acc` |
| `algorithm.graphroute.gnn.ens_combination_mode` | `soft_weighted_voting` | `hard_weighted_voting` |
| `algorithm.graphroute.gnn.voting_weight_space` | `null` | `sig` |
