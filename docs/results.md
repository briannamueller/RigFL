# Results

## Summarize completed runs

The default results root is `results`, so completed runs are read from
`results/runs` with:

```bash
rigfl report
```
When more than one replicate exists for a given experiment configuration, results are summarized across replicates by mean performance and a 95% confidence interval. The output also includes `pooled_client_sd`, which summarizes the variation in performance across clients. With only one replicate available, it is the standard deviation of client test performance.

The report groups results by experiment condition and method. Within each condition/method block, configurations are sorted by validation score, averaged across replicates. Use `--best` to keep only each method's configuration with the best validation score.

By default, test performance is reported at the training round with the highest mean validation accuracy across clients. To use a different reporting protocol, change the default settings in `configs/reporting.yaml`:

| Setting | What it controls | Options |
| --- | --- | --- |
| `metric` | The validation metric used to select the reporting round and the corresponding test metric reported. | `accuracy` (default), `balanced_accuracy`, `macro_f1`, `auroc`, `auprc` |
| `round_selection` | Whether to select one reporting round based on aggregate validation performance across clients or a separate round for each client based on their local validation performance.| `shared` (default), `client-specific` |
| `client_weighting` | How client scores are averaged, for round selection and for the reported validation and test values. | `uniform` (default), `sample_count` |

When validation values tie, RigFL selects the earliest round.

RigFL records `accuracy`, `balanced_accuracy`, `macro_f1`, `auroc`, `auprc`, and `loss` at every evaluation round. `loss` is used only for [early stopping](configuration.md#early-stopping). AUROC and AUPRC are available for binary tasks, where label 1 is the positive class. AUPRC is computed as average precision. A client's value is reported as unavailable, not as 0, when its split lacks a positive or a negative example.

## Client-level gain analysis

Aggregate performance can indicate that collaborative learning improves upon
Local training on average even when collaboration worsens performance for some
clients. This failure mode is negative transfer. When matching Local results are
available, `rigfl report` includes a client-level gain analysis relative to
Local.

RigFL pairs each federated result with Local only when their shared experiment
settings agree, including the resolved data configuration, generated partition,
partition seed, split seed, training seed, model and training schedule, client
identity, sample count, and client-model assignment. If corresponding results
cannot be paired, the report identifies why the analysis is unavailable rather
than substituting unmatched results. Within each experiment condition, each
method's best validation configuration is compared with Local's best validation
configuration; the report lists which configurations were used.

A client's gain is its federated result minus its Local result, so a positive
gain favors the federated algorithm. A zero gain is neutral.

The analysis reports:

- **Mean gain:** the average client gain within each replicate, summarized
  across replicates.
- **Pooled client-difference SD:** descriptive variation among the client-level
  gains within replicates. It is not uncertainty in the overall mean gain.
- **Benefit rate:** the fraction of clients with a positive gain, calculated
  within each replicate and then summarized across replicates.
- **Negative-transfer rate (NTR):** the fraction of clients with a negative
  gain, calculated within each replicate and then summarized across replicates.
- **Negative-transfer magnitude (NTM):** the average loss among harmed clients.
  It is conditional on harm and is reported with harmed-client and
  affected-replicate counts rather than an ordinary all-replicate confidence
  interval.
- **Negative-transfer burden (NTB):** the total harm within a replicate divided
  by all clients in that replicate, assigning zero harm to clients that did not
  experience negative transfer.
- **Worst-tail gain:** the average gain among the lowest-gaining 10% of clients
  within each replicate.

Except for the conditional NTM, each quantity is calculated within every
replicate and then reported with its across-replicate mean, sample standard
deviation, and 95% t-confidence interval when independent replicates are
available.

When the report is saved, this analysis is written to
`<stem>_client_level_gain_analysis.csv` next to the aggregate table, such as
`results/reports/model_heterogeneous_client_level_gain_analysis.csv` with
`--filter model_heterogeneous`.

## Filters

Subset results by defining a filter in `configs/reporting.yaml` and providing the filter name to `rigfl report`.

For example, this filter selects FedAvg runs with a learning rate of either `0.1` or `0.001`:

```yaml
filters:
  my_filter:
    algorithm: fedavg
    lr: [0.1, 0.001]
```

```bash
rigfl report --filter my_filter
```

## Command-line options

| Option | Effect |
| --- | --- |
| `--results-root PATH` | Uses `PATH` as the base directory containing `runs/` and `reports/`. Defaults to `results`. |
| `--config PATH` | Uses the reporting configuration at `PATH` instead of the default `configs/reporting.yaml`. |
| `--filter NAME` | Specifies a filter to apply. `NAME` must match a filter defined in the reporting configuration YAML. |
| `--grid PATH` | Reports only the runs of this sweep grid (`grid.jsonl`). |
| `--study PATH` | Reports only the runs of this HPO study (study directory or `study.json`). |
| `--best` | Shows only each method's configuration with the best validation score in each condition. |
| `--save [PATH]` | Saves report tables as CSV files. Omit `PATH` to use the `reports/` directory under the selected results root. |
| `--per-client` | When paired with `--save`, additionally saves a table of individual client results for every run. |
| `--resources` | Adds resource measurements to the terminal output and, with `--save`, saves them as a separate table. |

Without `PATH`, the aggregate table is saved as `reports/<filter>.csv`, or
`reports/summary.csv` without `--filter`. For example,
`--save paper_results/summary.csv` writes the aggregate table to that path.
Additional tables use the same filename stem.

## Resource measurements

Completed runs record algorithm communication payloads and operation timing.
Communication totals measure algorithm payload bytes rather than end-to-end
network traffic, so transport and serialization overhead are not included.

Add resource results to a report with `--resources`:

```bash
rigfl report --filter model_heterogeneous --resources --save
```

The terminal report and `results/reports/model_heterogeneous_resources.csv` include
communication, algorithm time, and available FLOP estimates. Timing is
reported as comparable only when the recorded hardware signatures support that
comparison.

FLOP estimation is disabled by default because profiling adds runtime overhead.
Enable it for new runs with:

```yaml
estimate_flops: true
```

FLOP estimation requires PyTorch 2.1 or newer and records algorithm and
evaluation operations separately. RigFL uses PyTorch's `FlopCounterMode` and
counts a multiply-add as two FLOPs. Estimates include only operators supported
by that counter.

## Examine crossed-seed sensitivity

Ordinary replicate confidence intervals do not require varying all three seed
components. Use the separate crossed-seed analysis only when you want to
describe sensitivity to partitioning, validation splitting, and training
randomness individually.

Define a sweep containing every combination of at least two values for each
seed:

```yaml
sweep:
  partition_seed: [0, 1, 2]
  split_seed: [0, 1, 2]
  training_seed: [0, 1, 2]
```

Run every experiment configuration, then analyze the completed grid:

```bash
rigfl seed-sensitivity \
  --grid results/<sweep-name>/grid.jsonl \
  --out results/seed_sensitivity.md \
  --out-json results/seed_sensitivity.json
```

For each seed source, RigFL reports the validation-score mean at every seed
value and the marginal spread between its largest and smallest mean. It also
records the actual partition, split, and training seed values included in the
design.

These marginal spreads are descriptive diagnostics. They are not variance
component estimates, significance tests, or causal attributions, and they do
not measure interactions between seed sources. The analysis rejects incomplete
crossed designs rather than interpreting missing combinations.
