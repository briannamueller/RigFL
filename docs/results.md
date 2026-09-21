# Results

## Summarize runs

Summarize completed runs under `results/runs` with:

```bash
python -m rigfl.experiment.collect
```

Runs that differ only in their replicate seeds are summarized in one row. Rows
with the same dataset setup and reporting view are ordered by validation
performance; differing algorithm settings appear in their labels.

### Choose which round to report

For each run, RigFL selects a reporting round using validation performance.
Specify `--selection-metric`, `--selection-view`, and
`--selection-aggregation` to change the defaults for how the reporting round
is chosen.

```bash
python -m rigfl.experiment.collect \
  --selection-metric balanced_accuracy \
  --selection-view per-client
```

| Flag | What it controls | Values |
| --- | --- | --- |
| `--selection-metric` | Validation metric used to select the round | `accuracy` (default), `balanced_accuracy`, `macro_f1`, `loss` |
| `--selection-view` | Whether to report one round for all clients or a separate round for each | `global` (default), `per-client`, `both` |
| `--selection-aggregation` | How client scores are combined for global selection | `mean` (default), `weighted_mean` |

### Client-level performance analysis

When matching Local runs are available, `collect` also prints a client-level
performance analysis below the run summary.

Negative transfer is a failure mode in federated learning in which
collaborative training leaves a client with worse performance on its local
distribution than training alone would have. Aggregate results can mask these
losses.

The analysis compares each client's test result with its Local result from the
same run conditions.
`--performance-margin` sets the minimum difference in performance from the
Local baseline to be considered as a performance improvement or decline when
computing the benefit rate or negative transfer rate. The default is zero.
For accuracy, `0.01` means one percentage point.

- **Benefit rate** measures how often collaboration helps: the share of
  client-level federated vs local performance comparisons where the performance
  improvement under collaborative learning is greater than the margin.
- **Negative-transfer rate (NTR)** is the share of comparisons where a client's
  federated performance is worse than Local by more than the margin.
- **Negative-transfer magnitude (NTM)** is the average decline among
  comparisons with negative transfer.
- **Negative-transfer burden (NTB)** is the total decline among comparisons
  with negative transfer, divided by the number of comparisons.
- **Worst-10% gain** is the average difference from Local among the
  lowest-gaining 10% of comparisons by default.

Performance differences that do not exceed the margin in either direction are
neutral.

For example, suppose Local and FedAvg both use partition and split seeds 0,
with matching experiment seeds 1 and 2. With a performance margin of `0.01`:

| Experiment seed | Client | Local accuracy | FedAvg accuracy | Counts toward |
| --- | --- | ---: | ---: | --- |
| 1 | A | 0.70 | 0.74 | Benefit rate |
| 1 | B | 0.80 | 0.78 | NTR |
| 2 | A | 0.72 | 0.725 | -- |
| 2 | B | 0.77 | 0.77 | -- |

Of the four FedAvg client results, one counts toward each rate. The benefit
rate and NTR are therefore both 25%.

If these are the only completed runs, this is the output from
`python -m rigfl.experiment.collect --performance-margin 0.01`:

```text
### selection-view: global  (metric=accuracy, split=validation, direction=maximize, aggregation=mean, tie_break=earliest)
| algorithm | selection | val accuracy | test accuracy | p10 | bottom-10% | seeds | runs |
|---|---|---|---|---|---|---:|---:|
| fedavg | global | 0.754 ± 0.079 | 0.754 ± 0.079 | 0.737 | 0.732 | 2 | 2 |
| local | global | 0.748 ± 0.032 | 0.748 ± 0.032 | 0.718 | 0.710 | 2 | 2 |

Completed seeds (partition, split, experiment):
- fedavg: (0, 0, 1), (0, 0, 2)
- local: (0, 0, 1), (0, 0, 2)

### Client-level performance analysis
| algorithm | selection | pairs | benefit rate | NTR | NTM | NTB | worst-10% gain |
|---|---|---:|---:|---:|---:|---:|---:|
| fedavg | global | 4 | 25.0% [0.0%, 75.0%] | 25.0% [0.0%, 75.0%] | 0.020 [0.020, 0.020] | 0.005 [0.000, 0.015] | -0.020 [-0.020, 0.005] |

Performance margin: 0.01 metric units. Intervals resample replicate conditions and clients; they require more than one of each.
```

`--negative-transfer-profile` shows NTR at several margins.
`--negative-transfer-tail` changes the fraction used for worst-tail gain.

### Filter runs

To report only runs from a particular sweep, specify its grid file:

```bash
python -m rigfl.experiment.collect \
  --grid results/cifar10_sweep/grid.jsonl
```

Use `--dataset` to filter by dataset. You can combine it with `--grid`. With
`--grid`, the runs column shows completed/expected runs, and the report lists
completed and missing seed combinations for each row.

For an exact past SGE submission, use its fixed grid at
`results/<sweep-name>/submissions/<submission>/grid.jsonl` instead of the
replaceable working grid.

### Save the report

```bash
python -m rigfl.experiment.collect \
  --out results/summary.md \
  --out-json results/summary.json
```

### Resource measurements

Add communication, training time, and available FLOP estimates with
`--include-resources`. FLOP estimates require runs made with
`experiment.estimate_flops: true`.

## Compare algorithms or settings

Use `compare` to evaluate alternatives such as FedAvg versus Local or two
FedAvg learning rates. Both alternatives must have been run on the same
dataset, with the same combinations of partition, split, and experiment
seeds. RigFL pairs the same client's test results from those runs. Keep every
setting other than the seeds fixed for each alternative, and use a distinct
experiment seed for every repetition.

The required `--practical-threshold` is the smallest performance difference
you consider meaningful. For accuracy, `0.01` means one percentage point.

If `results` contains just two alternatives, run:

```bash
python -m rigfl.experiment.compare \
  --results-dir results \
  --practical-threshold 0.01
```

The comparison uses accuracy by default. Set `--metric` to compare a different
recorded metric. Use `--selection-view` or `--selection-aggregation` to change
how the reporting round is chosen.

If the directory contains more than two alternatives, specify what to
compare:

```bash
python -m rigfl.experiment.compare \
  --results-dir results \
  --all-pairs \
  --practical-threshold 0.01
```

`--all-pairs` compares every pair. To limit comparisons, use `--reference NAME`
for comparisons with one reference or `--contrast LEFT:RIGHT` for a specific
pair. When an algorithm has multiple settings in the results, RigFL adds a
short identifier to each label; use those full labels with `--reference` or
`--contrast`.

The comparison table reports the mean test-performance difference, a 95%
confidence interval when available, and a practical conclusion. Positive
differences favor the first name, even for metrics such as loss where lower is
better. With at least six matched seed repetitions, it also reports a paired-test
p-value. When several comparisons are requested, their p-values receive a Holm
adjustment.

Available resource differences appear in a separate table.

With the commands above, RigFL writes the full comparison to
`results/comparison.json`. Use `--out` for a Markdown copy of the table or
`--out-json` to choose the JSON path.

For tuned experiments, `compare` uses the selected runs listed in
`selection.json` and excludes the search trials. Finish the
[selection step](hyperparameter_tuning.md) before comparing tuned results.

## Examine seed sensitivity

To see how results change with data partitioning, validation splitting, and
training randomness, sweep all three seeds together:

```yaml
sweep:
  experiment.partition_seed: [0, 1, 2]
  experiment.split_seed: [0, 1, 2]
  experiment.seed: [0, 1, 2]
```

Run every combination, then analyze its saved grid:

```bash
python -m rigfl.experiment.variance \
  --grid results/<sweep-name>/grid.jsonl \
  --out results/variance.md \
  --out-json results/variance.json
```

The analysis requires at least two values for each seed and every combination
of those values. It ranks seed sources by how much their mean validation
scores vary. This is a descriptive check, not a significance test or an
estimate of how much variance each source causes.
