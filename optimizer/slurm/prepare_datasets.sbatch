#!/usr/bin/env bash
# Prepares datasets with scripts/datasets/prepare_datasets.py on one whole node (base, query
# splits, ids/qrels, and the exact ground truth, which needs the cores). Files already in
# $DAISY_DATA_ROOT are kept, never changed, so a stopped job can be submitted again.
#
#   sbatch optimizer/slurm/prepare_datasets.sbatch SIFT1M HOTPOTQA5M      # from the repository root
#   sbatch optimizer/slurm/prepare_datasets.sbatch --skip-gt SEISMIC100M  # base and queries only
#
# Download first in a small job (optimizer/slurm/download_datasets.sbatch) and make this one wait for
# it (--dependency=afterok:<id>): a whole node then only computes.
#SBATCH --job-name=prepare-datasets
#SBATCH --partition=ncpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --output=optimizer/results/stella/logs/%x-%j.out
set -euo pipefail
export DAISY_REPO=${DAISY_REPO:-$SLURM_SUBMIT_DIR}
source "$DAISY_REPO/optimizer/slurm/config.sh"
print_job_header "$@"
"$DAISY_PYTHON" -m pip list --format=freeze 2>/dev/null | grep -iE '^(numpy|faiss-cpu|torch|transformers|pyarrow|huggingface.hub)=='
export HF_HOME="$DAISY_RAW_DIR/hf/cache"
export HF_HUB_DISABLE_XET=1 # Xet downloads fail from this cluster
numactl --interleave=all "$DAISY_PYTHON" -u "$DAISY_REPO/scripts/datasets/prepare_datasets.py" \
    --data-root "$DAISY_DATA_ROOT" --raw-dir "$DAISY_RAW_DIR" --series-dir "$DAISY_SERIES_DIR" "$@"
echo "Finished $(date --iso-8601=seconds)"
