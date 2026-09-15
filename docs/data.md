# Data

RigFL builds reusable client datasets from named entries in
[`configs/datasets.yaml`](../configs/datasets.yaml). Each entry identifies a
source and describes how that source becomes client training, validation, and
test data.

See the [data reference](reference/data.md) for every field, default, and
constraint.

## Workflow

Generate a dataset before running experiments when you want to inspect it or
prepare it on another machine:

```bash
python -m rigfl.data.generate --dataset cifar10
```

RigFL fingerprints the complete data configuration and stores each distinct
partition separately. Changing the source, split handling, transformation, or
partitioning settings therefore creates a new reusable partition instead of
overwriting an existing one.

## Flower datasets

The Flower backend loads a source from the Hugging Face Hub and partitions it
across clients. Missing Flower partitions are generated automatically when an
experiment starts.

RigFL normally preserves the source's training and test splits. A published
validation split can also be used directly; otherwise, validation data is
derived from each client's training partition.

When the published splits do not match the desired experiment, they can be
merged before partitioning. RigFL then divides every client partition into new
training, validation, and test subsets. The data reference documents the
required `source_splits` and `client_split` fields.

The partitioning scheme controls how samples are distributed among clients.
Choose a scheme based on the kind of heterogeneity the experiment should
represent, then use the scheme-specific section of the data reference.

Data transforms convert source examples into model-ready tensors. Included
datasets select registered transforms in `configs/datasets.yaml`; custom
datasets can use a general transform or add a dataset-specific one under
`rigfl/data/transforms/`.

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
