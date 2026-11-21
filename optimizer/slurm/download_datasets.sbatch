#!/usr/bin/env bash
# Downloads the raw files of datasets (prepare_datasets.py --download-only) on two cores of
# a shared node, ahead of their optimizer/slurm/prepare_datasets.sbatch jobs. Each attempt continues
# where the last stopped; after the last one the job fails, so jobs waiting for it with
# --dependency=afterok do not start.
#
#   D=$(sbatch --parsable optimizer/slurm/download_datasets.sbatch HOTPOTQA5M FEVER5M)
#   sbatch --dependency=afterok:$D optimizer/slurm/prepare_datasets.sbatch HOTPOTQA5M
#SBATCH --job-name=download-datasets
#SBATCH --partition=ncpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=optimizer/results/stella/logs/%x-%j.out
set -uo pipefail
export DAISY_REPO=${DAISY_REPO:-$SLURM_SUBMIT_DIR}
source "$DAISY_REPO/optimizer/slurm/config.sh"
print_job_header "$@"
export HF_HOME="$DAISY_RAW_DIR/hf/cache"
export HF_HUB_DISABLE_XET=1 # Xet downloads fail from this cluster
for attempt in 1 2 3 4 5; do
    if "$DAISY_PYTHON" -u "$DAISY_REPO/scripts/datasets/prepare_datasets.py" --download-only \
        --data-root "$DAISY_DATA_ROOT" --raw-dir "$DAISY_RAW_DIR" --series-dir "$DAISY_SERIES_DIR" "$@"; then
        echo "Finished $(date --iso-8601=seconds)"
        exit 0
    fi
    echo "Attempt $attempt stopped $(date --iso-8601=seconds)"
    sleep 60
done
exit 1
