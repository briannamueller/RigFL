# Data configuration

RigFL uses the dataset specifications in configs/datasets.yaml to generate reusable client partitions. Each specification declares the partitioning backend, the dataset source, the partitioning settings, and any required data transformations.

The included configuration provides starting points for MNIST, Fashion-MNIST,
CIFAR-10, CIFAR-100, Tiny ImageNet, FEMNIST, PaySim fraud detection, and phishing
URL detection. The sample limits keep initial runs small; set the per-client
limits to `null` to retain all samples assigned to each client.

## BioSilo sources

The BioSilo backend generates naturally partitioned biomedical datasets through
BioSilo and loads them directly without repartitioning or copying their data.

```yaml
datasets:
  tcga_cancer_type:
    backend: biosilo
    source_dataset: TCGA
    data_root: data
    parameters:
      task: cancer_type
      source: xena
      cohorts: [LUAD, LUSC, BRCA, COAD]
      train_ratio: 0.8
      seed: 1
    validation_fraction: 0.2
```

Install the backend with:

```bash
pip install "rigfl[biosilo]"
```

Some BioSilo datasets require additional dependencies or source-data
preparation; follow the relevant guide in
[BioSilo](https://github.com/briannamueller/BioSilo#datasets). Then use RigFL's
normal data generation command:

```bash
python -m rigfl.data.generate --dataset tcga_cancer_type
```

Entries under `parameters` are passed to BioSilo for the selected
`source_dataset`. BioSilo derives the partition identity from those parameters,
and running the command again reuses the same completed partition. RigFL uses
the same configuration to locate that exact partition when an experiment starts,
so a partition ID does not need to be copied into the YAML. `data_root` is
optional; when omitted, the command's `--data-dir` value is used.

BioSilo supplies each client's training and test data, and RigFL derives the
validation set from the training data. When BioSilo supplies subject or patient
group IDs, whole groups are assigned to either training or validation so the
same individual cannot appear in both.

Single-vector BioSilo datasets use RigFL's numeric model architectures, and
image partitions use its image architectures. Paired time-series and static
inputs use `temporal_gru`, `temporal_cnn`, or `temporal_lstm`, available together
as `temporal_heterogeneous_3`.

## Flower sources

For the Flower backend, set `source_dataset` to the Hugging Face Hub dataset id. When a source dataset offers more than one subset or version, RigFL loads the Hub's default unless you specify a different one with `source_subset`. The Hugging Face dataset page lists the available options.

```yaml
datasets:
  my_images:
    backend: flower
    source_dataset: organization/dataset-name
    source_subset: subset-name
    source_revision: revision-hash
    data_transform: image
    partition:
      scheme: dirichlet
      num_clients: 5
      alpha: 0.5
      partition_seed: 0
```

`source_revision` is optional. Set it to a Hugging Face commit hash when the
generated partition should remain tied to a particular dataset revision.

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

### Defining client train, validation, and test fractions

The default behavior above preserves the dataset's published training and test
splits. Flower partitions those splits separately, and `partition.val_frac`
creates validation data from each client's training partition when needed.

To define all three client splits yourself, list the dataset splits to combine
under `merge_splits` and specify the validation and test fractions:

```yaml
    source_splits:
      merge_splits: [train, test]
    client_split:
      validation_fraction: 0.1
      test_fraction: 0.2
      stratify: true
```

RigFL merges the listed splits, applies the Flower partitioner once, and then
divides each client partition. The fractions refer to the client's complete
partition; the remaining 70% in this example becomes training data. Set
`stratify` to `true` to preserve class proportions when creating the three
client splits.

The per-client sample limits are applied after this division. Set
`train_per_client`, `validation_per_client`, or `test_per_client` to `null` to
retain every assigned sample in that split.

The ordered `merge_splits` list and all `client_split` fields are included in
the partition fingerprint. Changing any of them creates a separate partition
directory.



### Ambiguous input or target columns

RigFL identifies the input and target columns automatically for most datasets. Specify input_column or target_column only when RigFL cannot determine these columns unambiguously from the metadata—for example, when a dataset contains both fine and coarse labels, or both an image and a segmentation mask:

```yaml
    input_column: image
    target_column: fine_label
```

the error message lists the columns retrieved from the dataset’s Hugging Face metadata.

### Data transforms

`data_transform` selects the conversion from the source dataset to model-ready
PyTorch tensors. The included datasets use named transforms such as `cifar10`,
`femnist`, `paysim_fraud`, and `phishing_urls`. The `image` and `numeric`
transforms support custom datasets that need only direct tensor conversion;
`auto` infers between those two forms from the input column. The phishing URL
transform converts each URL into a padded sequence of byte-token indices.
The phishing dataset contains potentially malicious URL strings; RigFL encodes
them without visiting or interacting with the URLs.

Dataset-specific transforms are registered under `rigfl/data/transforms/`. A
transform can define the expected columns, task, tensor conversion, and any
statistics that must be fitted from the source training split. The PaySim transform
follows Flower Labs' FedFinFraud preprocessing and records its source and transform
version in the generated partition manifest.

## Partition settings

Set `partition.scheme` to select a horizontal Flower partitioner. Fields marked
as required in the table must be included in the dataset specification.

| `scheme` | Flower partitioner | Scheme-specific fields |
| --- | --- | --- |
| `continuous` | `ContinuousPartitioner` | `num_clients`; required: `partition_by`, `strictness` |
| `dirichlet` | `DirichletPartitioner` | `num_clients`, `partition_by`, `alpha`, `min_partition_size`, `self_balancing` |
| `distribution` | `DistributionPartitioner` | `num_clients`, `partition_by`, `rescale`; required: `distribution_array`, `num_unique_labels_per_partition`, `preassigned_num_samples_per_label` |
| `exponential` | `ExponentialPartitioner` | `num_clients` |
| `grouped_natural_id` | `GroupedNaturalIdPartitioner` | `mode`, `sort_unique_ids`, `client_limit`; required: `partition_by`, `group_size` |
| `iid` | `IidPartitioner` | `num_clients` |
| `inner_dirichlet` | `InnerDirichletPartitioner` | `partition_by`, `alpha`; required: `partition_sizes` |
| `linear` | `LinearPartitioner` | `num_clients` |
| `natural_id` | `NaturalIdPartitioner` | `client_limit`; required: `partition_by` |
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

For the natural-ID schemes, `client_limit` selects a deterministic subset of
the available clients. The partition manifest records the original natural ID
or IDs represented by each saved RigFL client.

Every scheme also accepts `partition_seed` and `shuffle`. The partition seed
controls data shuffling and every random operation exposed by the selected
partitioner. It is separate from the experiment seed used for model training.
When the dataset's published splits are preserved, RigFL applies the same
partition configuration to the training, validation, and test splits and
requires each split to produce the same number of clients. For the natural-ID
schemes, the IDs assigned to each client must also match across the splits.

`train_per_client`, `validation_per_client`, and `test_per_client` optionally cap
the number of saved samples per client. Set a limit to `null` to retain every
sample assigned by Flower. `val_frac` is used only when RigFL derives validation
data from a separately partitioned training split; `client_split` supplies the
fractions when `merge_splits` is used.

Every data configuration field contributes to the partition fingerprint,
including split handling, the data transform, client limits, and `source_revision`.
The transform's implementation version and registered parameters also contribute
to the fingerprint. Statistics fitted during generation are stored in the manifest.
The fingerprint also includes RigFL's partition-pipeline version so a future
change to tensor-generation behavior cannot silently reuse an incompatible
partition.

## Task support

RigFL currently supports classification experiments. Regression requires a
compatible partitioning scheme, prediction losses, and evaluation metrics and is
not yet supported.
