#!/bin/bash
# Generic SGE array runner for a sweep grid produced by `rigfl.experiment.launch`.
# Task N runs the Nth configuration in the grid file.
#
#   python -m rigfl.experiment.launch --name my_sweep --algorithms baselines --seeds 0-2
#   python -m rigfl.experiment.launch --grid results/my_sweep/grid.jsonl \
#       --queue <gpu-queue> --submit
#   python -m rigfl.experiment.collect --results-dir results/runs
#
# `--submit` gives this script a fixed copy of the grid. Manual qsub commands
# should likewise use an immutable copy rather than the replaceable working grid.
#
#$ -S /bin/bash
#$ -N rigfl_grid
#$ -cwd
#$ -o /dev/null                   # SGE's own stdout/stderr -> /dev/null (never fails to open);
#$ -e /dev/null                   # the script self-redirects to the sweep's logs/ dir below.
### #$ -pe smp 4                  # EDIT: CPU cores per task, if desired

set -euo pipefail
GRID="${1:?usage: qsub ... scripts/run_grid.sh <submission>/grid.jsonl <results-root>}"
RESULTS_ROOT="${2:?usage: qsub ... scripts/run_grid.sh <submission>/grid.jsonl <results-root>}"
SWEEP_DIR="$(dirname "$GRID")"
mkdir -p "$SWEEP_DIR/logs"

# ── EDIT: activate your Python environment (venv shown; adapt for conda) ──────
source .venv/bin/activate
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1            # read the prefetched dataset cache; don't hit the network
export OMP_NUM_THREADS="${NSLOTS:-1}"
# If your experiment enables W&B and the compute nodes have no outbound network,
# log offline here and `wandb sync <sweep>/wandb/offline-run-*` from a login node:
# export WANDB_MODE=offline

# Redirect task output after creating the sweep log directory.
exec > "$SWEEP_DIR/logs/task_${SGE_TASK_ID}.out" 2>&1
echo "task $SGE_TASK_ID  ($(hostname), $(date))  grid=$GRID"
python -m rigfl.experiment.launch \
  --grid-task "$SGE_TASK_ID" \
  --grid "$GRID" \
  --results-root "$RESULTS_ROOT"
