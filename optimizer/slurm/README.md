# Slurm scripts (stella)

Jobs for the stella cluster: preparing the benchmark datasets and collecting the
knob-optimizer samples, one dataset per node. Paths, modules and the Python environment
are in [`config.sh`](config.sh); export any of its variables to override them.

| Script | What it does | Resources |
|---|---|---|
| `build_optimizer.sbatch` | Builds `optimizer_collector` (MESSI, FreSh, SOFA with FFTW, Odyssey with MPI) into `$DAISY_BUILD_DIR` (`build-stella/`) | 32 cores, `ncpushort`, under a minute |
| `submit_collection.sh` | Submits one `collect_dataset.sbatch` job per dataset | none (runs on the login node) |
| `collect_dataset.sbatch` | Runs the whole grid of every algorithm on one dataset, into `samples_<DATASET>.csv`; chains 2-day rounds | one whole node |
| `download_datasets.sbatch` | `prepare_datasets.py --download-only` | 2 cores, shared node |
| `prepare_datasets.sbatch` | `prepare_datasets.py` (base, splits, ground truth) | one whole node |

Submit the `.sbatch` files from the repository root (they find the repository through the
submit directory); `submit_collection.sh` can be run from anywhere.

## Collecting optimizer samples

```bash
cd /gpfs/users/chatzakis/DaiSy
B=$(sbatch --parsable optimizer/slurm/build_optimizer.sbatch)   # once, and after every code change
AFTER=$B optimizer/slurm/submit_collection.sh                   # all 19 datasets, once the build succeeded
optimizer/slurm/submit_collection.sh SIFT1M GIST1M              # or some datasets (with a build already there)
DRY=1 optimizer/slurm/submit_collection.sh                      # only print the sbatch commands
```

- **One dataset per node.** Each job runs every algorithm of the build over its whole grid
  (810 configurations per dataset on stella: see
  [docs/optimizer-notes/collector.md](../../docs/optimizer-notes/collector.md#knobs)) on one
  exclusive node. Exclusive matters: the thread grid uses all the node's cores (128, 96,
  64, 32, 16 threads on stella's EPYC 7H12 nodes, fixed by `DAISY_MAX_THREADS=128`).
- **The CPU nodes are not all alike**, although Slurm reports them so: the same 2× EPYC 7H12,
  but some have SMT on (256 hardware threads, e.g. ncpu034) and the NUMA layout differs
  (8 NUMA nodes on ncpu027, 2 on ncpu030 and ncpu034). The thread grid is capped at 128 so
  every node runs the same configurations; the CSV's machine columns (`host`,
  `cpu_threads`, `numa_nodes`) record which kind of node answered, and the job log prints
  it. The rounds of one dataset can land on different kinds of nodes.
- **Partitions.** SEISMIC100M, ASTRO100M and RANDOM100M run on `ncpum` (505 GB): Odyssey keeps
  a second ~102 GB copy of their base. All other datasets run on `ncpu` (247 GB). Set
  `PARTITION=...` to force one.
- **Rounds.** Jobs may run 2 days; a large dataset's grid takes longer. Each job runs the
  collector with `--resume 1` and, 30 minutes before its time limit, submits the next round
  of the same dataset (`--dependency=afterany`), which skips every configuration already in
  the CSV. At the time limit Slurm stops the collector, which never stops in the middle of
  writing rows; the configuration it was running is lost and runs again in the next round.
  Rounds stop when the grid is done, when a round added no rows, or after
  `DAISY_MAX_ROUNDS` (8) rounds.
- **Outputs** go to `$DAISY_RESULTS_DIR`, [`optimizer/results/stella/`](../results/stella):
  `samples_<DATASET>.csv` (one per dataset, same header, so they concatenate) and
  `logs/collect-<DATASET>-<job>.out` (one line per configuration; the build and dataset
  jobs log there too). Odyssey's raw copies of the bases, about 745 GB for all datasets and
  reused by later rounds, go to `$DAISY_WORK_DIR` on scratch
  (`/gpfs/scratch/chatzakis/daisy-data/optimizer-work`).
- **Progress:** `squeue -u $USER`, `grep -c ": ok " $DAISY_LOG_DIR/collect-SIFT1M-*.out`, or
  `wc -l $DAISY_RESULTS_DIR/samples_*.csv`. A dataset is done when its log says
  `nothing to run`, or when its last round ends without submitting another.
- **Restarting by hand** (e.g. after changing the code): resubmit the dataset; the round
  skips what the CSV has. A new grid value only adds configurations; changing the CSV
  header (new columns) needs a new `$DAISY_RESULTS_DIR`.
- **Extra collector options** go through `DAISY_COLLECTOR_ARGS`, e.g.
  `DAISY_COLLECTOR_ARGS="--budget-s 300" optimizer/slurm/submit_collection.sh`. Options must be the same in
  every round of a dataset, or `--resume` sees different configurations.
- **Odyssey and Slurm.** Odyssey starts MPI as a singleton inside the collector. Open MPI on
  stella is built without Slurm's PMI, so with Slurm's variables in its environment it takes
  the collector for an `srun` task and aborts in `MPI_Init`, killing the whole run. The job
  therefore starts the collector with every `SLURM_*` / `PMI*` variable removed. Never run
  the collector under `srun` or `mpirun`.
- **SING** needs CUDA and a GPU node; it is not in this build. On a GPU node the thread grid
  would be the GPU node's own (32 or 48 cores: 32, 24, 16, 8, 4).

## Preparing datasets

All 19 datasets are already in `$DAISY_DATA_ROOT` (`/gpfs/scratch/chatzakis/endor/processed`,
which the Endor-DR-Bench project also uses; the script never changes an existing file). To
rebuild or add one ([docs/optimizer-notes/datasets.md](../../docs/optimizer-notes/datasets.md)):

```bash
D=$(sbatch --parsable optimizer/slurm/download_datasets.sbatch HOTPOTQA5M)
sbatch --dependency=afterok:$D optimizer/slurm/prepare_datasets.sbatch HOTPOTQA5M
sbatch optimizer/slurm/prepare_datasets.sbatch SEISMIC100M          # series: base and queries only, no download
```

The optimizer needs only each dataset's base and query files; pass `--skip-gt` to skip the
ground truth.
