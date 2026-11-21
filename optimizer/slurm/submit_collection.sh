#!/usr/bin/env bash
# Submits the knob-optimizer collection: one job per dataset (optimizer/slurm/collect_dataset.sbatch),
# each on its own whole node with all algorithms; each job chains its own follow-up rounds
# until its dataset's grid is done. Run from anywhere:
#
#   optimizer/slurm/submit_collection.sh                      # all 19 datasets
#   optimizer/slurm/submit_collection.sh SIFT1M GIST1M        # some of them
#   AFTER=<jobid> optimizer/slurm/submit_collection.sh        # start after a job (e.g. the build) succeeds
#   DRY=1 optimizer/slurm/submit_collection.sh                # print the sbatch commands only
#
# Partitions: the 256-dimensional 100M series (SEISMIC100M, ASTRO100M, RANDOM100M) go to
# ncpum (505 GB nodes): Odyssey holds a second ~102 GB copy of their base next to the
# collector's. Everything else goes to ncpu (247 GB nodes). Override with PARTITION=...
set -euo pipefail
export DAISY_REPO=${DAISY_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
source "$DAISY_REPO/optimizer/slurm/config.sh"

ALL=(SIFT1M GLOVE1M GIST1M HOTPOTQA5M FEVER5M MIRACLJA6M MSMARCO8M DEEP10M DEEP50M DEEP100M
     MIRACLRU9M UCASELAW7M T2I10M T2I50M T2I100M SEISMIC100M ASTRO100M SALD100M RANDOM100M)
DATASETS=("$@")
(( ${#DATASETS[@]} )) || DATASETS=("${ALL[@]}")

partition_for() {
    if [[ -n "${PARTITION:-}" ]]; then echo "$PARTITION"; return; fi
    case "$1" in
        SEISMIC100M|ASTRO100M|RANDOM100M) echo ncpum ;;
        *) echo ncpu ;;
    esac
}

BIN="$DAISY_BUILD_DIR/lib/optimizer/optimizer_collector"
if [[ ! -x "$BIN" && -z "${AFTER:-}" ]]; then
    echo "$BIN is missing: sbatch optimizer/slurm/build_optimizer.sbatch first (or pass AFTER=<build job id>)" >&2
    exit 2
fi
mkdir -p "$DAISY_RESULTS_DIR" "$DAISY_LOG_DIR" "$DAISY_WORK_DIR"

for D in "${DATASETS[@]}"; do
    if [[ ! " ${ALL[*]} " == *" $D "* ]]; then
        echo "unknown dataset $D; choose from: ${ALL[*]}" >&2
        exit 2
    fi
    cmd=(sbatch --parsable --job-name="collect-$D" --partition="$(partition_for "$D")"
         --output="$DAISY_LOG_DIR/%x-%j.out" --export=ALL,DAISY_REPO="$DAISY_REPO",DAISY_ROUND=1)
    [[ -n "${AFTER:-}" ]] && cmd+=(--dependency=afterok:"$AFTER")
    cmd+=("$DAISY_REPO/optimizer/slurm/collect_dataset.sbatch" "$D")
    if [[ -n "${DRY:-}" ]]; then
        echo "${cmd[*]}"
    else
        echo "$D: job $("${cmd[@]}") on $(partition_for "$D")"
    fi
done
echo "CSVs: $DAISY_RESULTS_DIR/samples_<DATASET>.csv, logs: $DAISY_LOG_DIR/collect-<DATASET>-<job>.out"
