# Algorithm configuration reference

## Capabilities

| Algorithm | Heterogeneous models |
|---|---|
| `local` | yes |
| `fedavg` | no |
| `fedprox` | no |
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
