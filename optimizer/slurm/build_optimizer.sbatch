#!/usr/bin/env bash
# Builds the knob-optimizer collector (lib/optimizer/optimizer_collector) with MESSI, FreSh,
# SOFA (FFTW) and Odyssey (MPI) into $DAISY_BUILD_DIR. It runs on a compute node because
# the build uses -march=native: the binary must be compiled for the nodes that run it (all
# of stella's CPU nodes are AMD EPYC 7H12). SING needs CUDA and is not built here.
#
#   sbatch optimizer/slurm/build_optimizer.sbatch          # from the repository root
#
#SBATCH --job-name=build-optimizer
#SBATCH --partition=ncpushort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=optimizer/results/stella/logs/%x-%j.out
set -euo pipefail
export DAISY_REPO=${DAISY_REPO:-$SLURM_SUBMIT_DIR}
source "$DAISY_REPO/optimizer/slurm/config.sh"
load_modules
print_job_header
mpicc --version | head -1
# The compiler of the loaded modules (the openmpi module brings its GCC), not the system's
# /usr/bin/c++: the jobs run with the modules' libstdc++, which must be at least as new as
# the compiler's (a build with the system GCC 11 fails at run time: GLIBCXX_3.4.29 not found).
CC=$(command -v gcc)
CXX=$(command -v g++)
"$CXX" --version | head -1
if [[ -f "$DAISY_BUILD_DIR/CMakeCache.txt" ]] && ! grep -q "CMAKE_CXX_COMPILER:.*=$CXX\$" "$DAISY_BUILD_DIR/CMakeCache.txt"; then
    echo "$DAISY_BUILD_DIR was configured with another compiler; starting it over"
    rm -rf "$DAISY_BUILD_DIR"
fi

cmake -S "$DAISY_REPO" -B "$DAISY_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_PREFIX_PATH="$DAISY_FFTW_ROOT" \
    -DBUILD_OPTIMIZER=ON -DBUILD_SOFA=ON -DBUILD_ODYSSEY=ON -DBUILD_SING=OFF \
    -DBUILD_TESTS=OFF -DBUILD_DEMO=OFF -DBUILD_BENCHMARK=OFF -DBUILD_PYTHON=OFF
cmake --build "$DAISY_BUILD_DIR" --target optimizer_collector -j "${SLURM_CPUS_PER_TASK:-8}"

BIN="$DAISY_BUILD_DIR/lib/optimizer/optimizer_collector"
grep -E "SOFA_FFTW_ENABLED|ODYSSEY_MPI" -o -m 2 "$DAISY_BUILD_DIR/lib/optimizer/CMakeFiles/optimizer_collector.dir/flags.make" || true
echo "Built $BIN"
echo "Finished $(date --iso-8601=seconds)"
