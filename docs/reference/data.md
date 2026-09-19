# Data configuration reference

## Flower datasets

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `backend` | string | `flower` | `flower` | — |
| `source_dataset` | string | required | — | Hugging Face dataset identifier. |
| `source_subset` | string \| null | `null` | — | Optional Hugging Face dataset subset. |
| `source_revision` | string \| null | `null` | — | Optional Hugging Face dataset revision. |
| `source_splits` | named source splits \| merged source splits \| null | `null` | — | Source splits used to build client data. |
| `client_split` | client split settings \| null | `null` | — | Client split fractions used after merging source splits. |
| `input_column` | string \| null | `null` | — | Source column containing model inputs. |
| `target_column` | string \| null | `null` | — | Source column containing prediction targets. |
| `task` | string | `auto` | `auto`, `classification` | Use source metadata or explicitly treat the target as class labels. |
| `data_transform` | string | `auto` | non-empty | Registered transform applied to source examples. |
| `partition` | partition settings | `dirichlet` | — | Client partitioning strategy and limits. |

### Named source splits

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `source_splits.train` | string | required | — | Source split used for client training data. |
| `source_splits.test` | string | required | — | Source split used for client test data. |
| `source_splits.validation` | string \| null | `null` | — | Optional source split used for client validation data. |

### Merged source splits

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `source_splits.merge_splits` | list[string] | required | non-empty | Source splits combined before client partitioning. |

### Client splits

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `client_split.validation_fraction` | number | required | > 0; < 1 | Fraction of each client partition reserved for validation. |
| `client_split.test_fraction` | number | required | > 0; < 1 | Fraction of each client partition reserved for testing. |
| `client_split.stratify` | boolean | required | — | Preserve target proportions when creating client splits. |

`client_split` and `source_splits.merge_splits` must be used together. Merged split names must be unique, and the validation and test fractions must sum to less than 1.

### Common partition settings

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `partition.partition_seed` | integer | `0` | ≥ 0 | Seed used to assign data to clients. |
| `partition.split_seed` | integer | `0` | ≥ 0 | Seed used to create client validation splits. |
| `partition.shuffle` | boolean | `true` | — | Shuffle source rows before partitioning. |
| `partition.train_per_client` | integer \| null | `2000` | ≥ 1 | Maximum training samples saved per client; null keeps all samples. |
| `partition.test_per_client` | integer \| null | `500` | ≥ 1 | Maximum test samples saved per client; null keeps all samples. |
| `partition.val_frac` | number | `0.2` | > 0; < 1 | Fraction of client training data reserved for validation. |

### `continuous` partitioning

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `partition.scheme` | string | `continuous` | `continuous` | — |
| `partition.num_clients` | integer | `5` | ≥ 1 | Number of client partitions. |
| `partition.partition_by` | string | required | — | Column used to order samples. |
| `partition.strictness` | number | required | ≥ 0; ≤ 1 | Strength of the continuous ordering constraint. |

### `dirichlet` partitioning

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `partition.scheme` | string | `dirichlet` | `dirichlet` | — |
| `partition.num_clients` | integer | `5` | ≥ 1 | Number of client partitions. |
| `partition.partition_by` | string \| null | `null` | — | Column whose values are distributed across clients. |
| `partition.alpha` | number \| list of positive numbers | `0.1` | > 0 | Dirichlet concentration for client distributions. |
| `partition.min_partition_size` | integer | `10` | ≥ 1 | Minimum samples assigned to each client. |
| `partition.self_balancing` | boolean | `false` | — | Reduce assignments to clients that already exceed the average size. |

A list-valued `alpha` must contain one value per client.

### `distribution` partitioning

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `partition.scheme` | string | `distribution` | `distribution` | — |
| `partition.distribution_array` | list[list[number]] | required | — | Requested label distribution for each client. |
| `partition.num_clients` | integer | `5` | ≥ 1 | Number of client partitions. |
| `partition.num_unique_labels_per_partition` | integer | required | ≥ 1 | Labels represented in each client partition. |
| `partition.partition_by` | string \| null | `null` | — | Column whose values are distributed across clients. |
| `partition.preassigned_num_samples_per_label` | integer | required | ≥ 0 | Samples reserved per label before distribution. |
| `partition.rescale` | boolean | `true` | — | Rescale requested distributions to available samples. |

### `exponential` partitioning

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `partition.scheme` | string | `exponential` | `exponential` | — |
| `partition.num_clients` | integer | `5` | ≥ 1 | Number of client partitions. |

### `grouped_natural_id` partitioning

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `partition.scheme` | string | `grouped_natural_id` | `grouped_natural_id` | — |
| `partition.partition_by` | string | required | — | Column containing natural client identifiers. |
| `partition.group_size` | integer | required | ≥ 1 | Natural identifiers grouped into each client. |
| `partition.mode` | string | `allow-smaller` | `allow-smaller`, `allow-bigger`, `drop-reminder`, `strict` | Handling of a final incomplete group. |
| `partition.sort_unique_ids` | boolean | `true` | — | Sort natural identifiers before grouping. |
| `partition.client_limit` | integer \| null | `null` | ≥ 1 | Maximum number of generated clients. |

### `iid` partitioning

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `partition.scheme` | string | `iid` | `iid` | — |
| `partition.num_clients` | integer | `5` | ≥ 1 | Number of client partitions. |

### `inner_dirichlet` partitioning

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `partition.scheme` | string | `inner_dirichlet` | `inner_dirichlet` | — |
| `partition.partition_sizes` | list of positive integers | required | non-empty | Requested size of each client partition. |
| `partition.partition_by` | string \| null | `null` | — | Column whose values are distributed across clients. |
| `partition.alpha` | number \| list of positive numbers | `0.1` | > 0 | Dirichlet concentration within each partition. |

### `linear` partitioning

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `partition.scheme` | string | `linear` | `linear` | — |
| `partition.num_clients` | integer | `5` | ≥ 1 | Number of client partitions. |

### `natural_id` partitioning

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `partition.scheme` | string | `natural_id` | `natural_id` | — |
| `partition.partition_by` | string | required | — | Column containing natural client identifiers. |
| `partition.client_limit` | integer \| null | `null` | ≥ 1 | Maximum number of generated clients. |

### `pathological` partitioning

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `partition.scheme` | string | `pathological` | `pathological` | — |
| `partition.num_clients` | integer | `5` | ≥ 1 | Number of client partitions. |
| `partition.partition_by` | string \| null | `null` | — | Column whose values are distributed across clients. |
| `partition.num_classes_per_partition` | integer | required | ≥ 1 | Classes assigned to each client. |
| `partition.class_assignment_mode` | string | `random` | `random`, `deterministic`, `first-deterministic` | How classes are assigned to clients. |

### `shard` partitioning

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `partition.scheme` | string | `shard` | `shard` | — |
| `partition.num_clients` | integer | `5` | ≥ 1 | Number of client partitions. |
| `partition.partition_by` | string \| null | `null` | — | Column used to form label-sorted shards. |
| `partition.num_shards_per_partition` | integer \| null | `null` | ≥ 1 | Shards assigned to each client. |
| `partition.shard_size` | integer \| null | `null` | ≥ 1 | Number of samples in each shard. |
| `partition.keep_incomplete_shard` | boolean | `false` | — | Retain a final shard smaller than the requested size. |

Set at least one of `num_shards_per_partition` or `shard_size`.

### `size` partitioning

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `partition.scheme` | string | `size` | `size` | — |
| `partition.partition_sizes` | list of positive integers | required | non-empty | Requested size of each client partition. |

### `square` partitioning

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `partition.scheme` | string | `square` | `square` | — |
| `partition.num_clients` | integer | `5` | ≥ 1 | Number of client partitions. |

## BioSilo datasets

| Setting | Type | Default | Allowed | Description |
|---|---|---|---|---|
| `backend` | string | `biosilo` | `biosilo` | — |
| `source_dataset` | string | required | non-empty | BioSilo dataset name. |
| `data_root` | string \| null | `null` | — | Optional BioSilo storage root. |
| `parameters` | mapping | `{}` | — | Parameters passed to the BioSilo dataset. |
| `validation_fraction` | number | `0.2` | > 0; < 1 | Fraction of client training data reserved for validation. |
| `split_seed` | integer | `0` | ≥ 0 | Seed used to create client validation splits. |

`parameters` cannot set `dataset`, `params`, `root`, `overwrite`, or `version`; RigFL supplies those arguments.
