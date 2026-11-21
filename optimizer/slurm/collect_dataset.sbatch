#!/usr/bin/env bash
# Collects the knob-optimizer samples of ONE dataset on one whole node: every algorithm of
# the build (MESSI, FreSh, SOFA, Odyssey) over its whole knob grid, into
# $DAISY_RESULTS_DIR/samples_<DATASET>.csv.
#
# A dataset's grid can take longer than a job may run (2 days). The collector runs with
# --resume 1 (configurations already in the CSV are skipped), and half an hour before the
# time limit the job submits its next round, which starts when this one ends and continues
# where it stopped. Rounds stop when the grid is done, when a round added no rows, or after
# DAISY_MAX_ROUNDS rounds. At the time limit Slurm stops the collector; the configuration
# it was running is lost and runs again in the next round.
#
# Usually submitted by optimizer/slurm/submit_collection.sh; by hand, from the repository root:
#   sbatch --job-name=collect-SIFT1M optimizer/slurm/collect_dataset.sbatch SIFT1M
#
# The node must be exclusive: the collector's thread grid goes up to DAISY_MAX_THREADS
# (128: 128, 96, 64, 32, 16 threads), all of a node's cores.
#SBATCH --job-name=collect
#SBATCH --partition=ncpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=optimizer/results/stella/logs/%x-%j.out
set -uo pipefail
DATASET=${1:?name the dataset, e.g. SIFT1M}
export DAISY_REPO=${DAISY_REPO:-$SLURM_SUBMIT_DIR}
source "$DAISY_REPO/optimizer/slurm/config.sh"
ROUND=${DAISY_ROUND:-1}
OUT="$DAISY_RESULTS_DIR/samples_${DATASET}.csv"
BIN="$DAISY_BUILD_DIR/lib/optimizer/optimizer_collector"
mkdir -p "$DAISY_RESULTS_DIR" "$DAISY_WORK_DIR" "$DAISY_LOG_DIR"
load_modules # the MPI libraries Odyssey was linked with
print_job_header "$DATASET, round $ROUND of at most $DAISY_MAX_ROUNDS"
if [[ ! -x "$BIN" ]]; then
    echo "$BIN is missing: run optimizer/slurm/build_optimizer.sbatch first" >&2
    exit 2
fi

rows_at_start=$(wc -l < "$OUT" 2>/dev/null || echo 0)

# Half an hour before the time limit: submit the next round, unless this one is the last
# or added nothing (a configuration that never finishes would otherwise loop forever).
next_round() {
    local rows_now
    rows_now=$(wc -l < "$OUT" 2>/dev/null || echo 0)
    if (( ROUND >= DAISY_MAX_ROUNDS )); then
        echo "Round $ROUND is the last (DAISY_MAX_ROUNDS=$DAISY_MAX_ROUNDS); not resubmitting"
    elif (( rows_now <= rows_at_start )); then
        echo "This round added no rows; not resubmitting"
    else
        local job
        job=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" \
            --job-name="$SLURM_JOB_NAME" --partition="$SLURM_JOB_PARTITION" \
            --output="$DAISY_LOG_DIR/%x-%j.out" \
            --export=ALL,DAISY_REPO="$DAISY_REPO",DAISY_ROUND=$((ROUND + 1)) \
            "$DAISY_REPO/optimizer/slurm/collect_dataset.sbatch" "$DATASET")
        echo "Submitted round $((ROUND + 1)) as job $job ($(date --iso-8601=seconds))"
    fi
}
trap next_round USR1

# Odyssey starts MPI as a singleton inside the collector. With Slurm's variables in its
# environment, Open MPI (built without Slurm's PMI) takes the process for an srun-launched
# task and aborts in MPI_Init, which kills the whole collector. So the collector runs without
# them (this script keeps them, for next_round).
unset_slurm=()
for var in $(env | grep -oE '^(SLURM|SLURMD|PMI|PMIX)_[A-Za-z0-9_]*'); do
    unset_slurm+=(-u "$var")
done
# shellcheck disable=SC2086
env "${unset_slurm[@]}" "$BIN" --datasets "$DATASET" --data-root "$DAISY_DATA_ROOT" --out "$OUT" \
    --work-dir "$DAISY_WORK_DIR" --max-threads "$DAISY_MAX_THREADS" --resume 1 $DAISY_COLLECTOR_ARGS &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
    wait "$pid" # returns early when USR1 arrives; keep waiting for the collector
done
wait "$pid"
rc=$?
echo "Collector exited with $rc; $(($(wc -l < "$OUT") - 1)) rows in $OUT. Finished $(date --iso-8601=seconds)"
exit $rc
