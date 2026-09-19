# Data

RigFL builds reusable client datasets from named entries in
[`configs/datasets.yaml`](../configs/datasets.yaml). Each entry identifies a
source and describes how that source becomes client training, validation, and
test data.

See the [data reference](reference/data.md) for every field, default, and
constraint.

## Flower datasets

The Flower backend loads a source from the Hugging Face Hub and partitions it
across clients. Missing Flower partitions are generated automatically when an
experiment starts.

RigFL normally preserves the source's training and test splits. A published
validation split is stored as-is; otherwise a partition stores training and test
data only, and validation is carved from each client's training data when the
clients are built. `split_seed` and the validation fraction therefore select a
different validation split without generating another partition.

When the published splits do not match the desired experiment, they can be
merged before partitioning. RigFL then divides every client partition into new
training and test subsets. The data reference documents the required
`source_splits` and `client_split` fields.

### Add a Hugging Face dataset

Add a named entry to `configs/datasets.yaml`. For a classification dataset
with numeric features, nonnegative integer labels, and `train` and `test`
splits, it can look like this:

```yaml
datasets:
  my_dataset:
    backend: flower
    source_dataset: organization/dataset-name
    input_column: features
    target_column: label
    task: classification
    data_transform: numeric
    partition:
      scheme: iid
      num_clients: 5
```

Set `source_revision` to a Hub commit if you need to keep using the same source
version when the Hub dataset changes.

RigFL may infer `input_column` and `target_column` from the dataset's metadata;
set them explicitly if it cannot. `data_transform: auto` converts a single
image or numeric input column. Choose `image` or `numeric` to specify which
conversion to use.

If the source does not provide `train` and `test` splits, map its split names
with `source_splits`. To combine source splits and divide each client partition
into new training, validation, and test data, use `source_splits.merge_splits`
together with `client_split`. See the [data reference](reference/data.md) for
those structures and the available partitioning schemes.

Then generate the partition to check the source, columns, and partition
settings:

```bash
python -m rigfl.data.generate --dataset my_dataset
```

Choose an experiment model that accepts the dataset's input type; the
[model reference](reference/models.md) lists the input type of each architecture.

### Add a data transform

Use a custom transform when the source needs preprocessing beyond the built-in
image and numeric conversions, such as encoding text or combining columns.
Define it under `rigfl/data/transforms/`, then add a named `DataTransform` to
`DATA_TRANSFORMS` in [`rigfl/data/transforms/__init__.py`](../rigfl/data/transforms/__init__.py).
Set that name as `data_transform` in the dataset entry.

For a one-column conversion, see
[`encode_urls`](../rigfl/data/transforms/phishing.py). For preprocessing that
uses training data and combines several columns, see
[`prepare_paysim`](../rigfl/data/transforms/paysim.py). In a `DataTransform`,
`convert` receives one input column's values and returns a tensor; `prepare`
returns a transformed Hugging Face dataset and any fitted parameters. Increase
the transform's `version` when its output changes so RigFL does not reuse an
older partition.

## BioSilo datasets

The BioSilo backend uses naturally partitioned biomedical datasets without
repartitioning or copying their source data. Install it with:

```bash
pip install "rigfl[biosilo]"
```

BioSilo datasets must be generated before an experiment starts. Settings under
`parameters` are passed to the selected BioSilo dataset, while RigFL creates a
validation split from each client's training data. When subject or patient
identifiers are available, whole groups remain together during that split.

Some BioSilo datasets require additional dependencies or source preparation.
Follow the relevant dataset instructions in
[BioSilo](https://github.com/briannamueller/BioSilo#datasets).

## Seeds

The partition seed controls which samples belong to each client. The split seed
controls validation data derived from client training data. Both are separate
from the experiment seed used for model training.

For BioSilo, generate any requested data-seed variant before launching the
experiment. Flower can generate a missing variant automatically.

## Task support

RigFL currently supports classification datasets.
