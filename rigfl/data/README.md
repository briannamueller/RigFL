# Data configuration

RigFL uses the dataset specifications in configs/datasets.yaml to generate reusable client partitions. Each specification declares the partitioning backend, the dataset source, the partitioning settings, and any required data transformations.

## Flower sources

For the Flower backend, set `source_dataset` to the Hugging Face Hub dataset id. When a source dataset offers more than one subset or version,  RigFL loads the Hubs's default unless you specify a different one with `source_subset`. The Hugging Face dataset page lists the available options

```yaml
datasets:
  my_images:
    backend: flower
    source_dataset: organization/dataset-name
    source_subset: subset-name
    partition:
      scheme: dirichlet
      num_clients: 20
      alpha: 0.5
      partition_seed: 0
```

RigFL reads the source dataset’s metadata from the Hugging Face Hub before invoking Flower’s partitioning functionality. It normally infers:

- `train` and `test` as the source splits;
- the input and target columns from the source's supervised-task metadata or an
  unambiguous feature schema;
- whether the target represents classification or regression;
- class names and the number of classes for classification data.


### Nonstandard split names

If the source dataset uses names other than `train` and `test` for its training or test splits, specify the names of the splits RigFL should use for training and testing.


```yaml
    source_splits:
      train: training
      validation: validation
      test: holdout
```

Specifying a validation split under source_splits is optional. If left unspecified, RigFL creates each client’s validation set from its training partition using partition.val_frac. If the expected or specified split names are not available for the dataset, an error message lists the split names retrieved from the dataset’s Hugging Face metadata.



### Ambiguous input or target columns

RigFL identifies the input and target columns automatically for most datasets. Specify input_column or target_column only when RigFL cannot determine these columns unambiguously from the metadata—for example, when a dataset contains both fine and coarse labels, or both an image and a segmentation mask:

```yaml
    input_column: image
    target_column: fine_label
```

the error message lists the columns retrieved from the dataset’s Hugging Face metadata.

## Partition settings

Set `partition.scheme` to select a horizontal Flower partitioner. Fields marked
as required in the table must be included in the dataset specification.

| `scheme` | Flower partitioner | Scheme-specific fields |
| --- | --- | --- |
| `continuous` | `ContinuousPartitioner` | `num_clients`; required: `partition_by`, `strictness` |
| `dirichlet` | `DirichletPartitioner` | `num_clients`, `partition_by`, `alpha`, `min_partition_size`, `self_balancing` |
| `distribution` | `DistributionPartitioner` | `num_clients`, `partition_by`, `rescale`; required: `distribution_array`, `num_unique_labels_per_partition`, `preassigned_num_samples_per_label` |
| `exponential` | `ExponentialPartitioner` | `num_clients` |
| `grouped_natural_id` | `GroupedNaturalIdPartitioner` | `mode`, `sort_unique_ids`; required: `partition_by`, `group_size` |
| `iid` | `IidPartitioner` | `num_clients` |
| `inner_dirichlet` | `InnerDirichletPartitioner` | `partition_by`, `alpha`; required: `partition_sizes` |
| `linear` | `LinearPartitioner` | `num_clients` |
| `natural_id` | `NaturalIdPartitioner` | required: `partition_by` |
| `pathological` | `PathologicalPartitioner` | `num_clients`, `partition_by`, `class_assignment_mode`; required: `num_classes_per_partition` |
| `shard` | `ShardPartitioner` | `num_clients`, `partition_by`, `keep_incomplete_shard`; at least one required: `num_shards_per_partition`, `shard_size` |
| `size` | `SizePartitioner` | required: `partition_sizes` |
| `square` | `SquarePartitioner` | `num_clients` |

For label-based partitioners, `partition_by` defaults to the configured or
inferred target column. Set it explicitly when a different column should govern
the partitioning. `num_clients` sets the number of client partitions for schemes
that accept it. `size` and `inner_dirichlet` derive the number of clients from
the length of `partition_sizes`; `natural_id` and `grouped_natural_id` derive it
from the IDs present in the dataset.

Every scheme also accepts `partition_seed` and `shuffle`. The partition seed
controls data shuffling and every random operation exposed by the selected
partitioner. It is separate from the experiment seed used for model training.
RigFL applies the same partition configuration to the training, validation, and
test splits and requires each split to produce the same number of clients. For
the natural-ID schemes, the IDs assigned to each client must also match across
the splits.

`train_per_client`, `validation_per_client`, and `test_per_client` optionally cap
the number of saved samples per client. Set a limit to `null` to retain every
sample assigned by Flower. `val_frac` is used only when RigFL must derive a
validation split from training data.

Image inputs may define `preprocessing.mean` and `preprocessing.std` for channel
normalization. Numeric and image inputs are otherwise converted to PyTorch
tensors, and the resolved data description is stored in `manifest.json`.

## Task support

RigFL currently supports classification experiments. Regression requires a
compatible partitioning scheme, prediction losses, and evaluation metrics and is
not yet supported.
