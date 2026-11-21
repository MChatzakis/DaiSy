# Settings shared by the Slurm scripts of this folder, for the stella cluster. Every
# value can be overridden by exporting it before sbatch / the submit scripts.
# Sourced by the scripts; DAISY_REPO must be set (the scripts set it).

# Where things are
: "${DAISY_DATA_ROOT:=/gpfs/scratch/chatzakis/endor/processed}"     # prepared datasets (shared with Endor-DR-Bench: existing files are never changed)
: "${DAISY_RAW_DIR:=/gpfs/scratch/chatzakis/daisy-data/raw}"       # downloads of prepare_datasets.py
: "${DAISY_SERIES_DIR:=/gpfs/scratch/chatzakis}"                   # the data-series .bin files (and randomDatasets/)
: "${DAISY_RESULTS_DIR:=$DAISY_REPO/optimizer/results/stella}"    # collector CSVs, one per dataset (samples_<DATASET>.csv)
: "${DAISY_LOG_DIR:=$DAISY_RESULTS_DIR/logs}"                      # Slurm output of every job
: "${DAISY_WORK_DIR:=/gpfs/scratch/chatzakis/daisy-data/optimizer-work}" # Odyssey's raw copies of the bases (~745 GB for all; scratch, not the home quota)
: "${DAISY_BUILD_DIR:=$DAISY_REPO/build-stella}"                   # where build_optimizer.sbatch builds
: "${DAISY_PYTHON:=/gpfs/users/chatzakis/envs/daisy-data/bin/python}" # numpy 2.2.6, faiss-cpu 1.9.0, torch, transformers

# Toolchain: these modules (SOFA needs FFTW, Odyssey MPI; the openmpi module brings GCC 10.3,
# which build_optimizer.sbatch compiles with, so the run-time libstdc++ matches).
: "${DAISY_MODULES:=cmake/3.29.1 fftw/3.3.9 openmpi/gcc-10.3.0/4.1.1}"
: "${DAISY_FFTW_ROOT:=/softs/fftw/3.3.9}"

# Collection
: "${DAISY_MAX_THREADS:=128}"   # largest thread value of the grid: 128, 96, 64, 32, 16 on every node
                                # (some CPU nodes have SMT on and show 256 hardware threads)
: "${DAISY_COLLECTOR_ARGS:=}"   # extra collector options, e.g. "--queries 100 --k 1,10,100 --budget-s 600"
: "${DAISY_MAX_ROUNDS:=8}"      # 2-day jobs a dataset may chain before giving up

load_modules() {
    if command -v module >/dev/null 2>&1 || [ -f /usr/share/Modules/init/bash ]; then
        command -v module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
        module purge >/dev/null 2>&1 || true
        # shellcheck disable=SC2086
        module load $DAISY_MODULES
    fi
}

# What a job ran with: node, sources, toolchain.
print_job_header() {
    echo "Job ${SLURM_JOB_ID:-none} (${SLURM_JOB_NAME:-}) on $(hostname), $(nproc) cores, started $(date --iso-8601=seconds): $*"
    lscpu | grep -E "^(Model name|Thread\(s\) per core|NUMA node\(s\))" | tr -s ' ' | paste -sd ';'
    git -C "$DAISY_REPO" rev-parse HEAD
    git -C "$DAISY_REPO" status --short
}
