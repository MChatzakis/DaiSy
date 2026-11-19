# Knob-optimizer sample collector

`optimizer_collector` produces the training and analysis data for the knob optimizer.
It sweeps a grid of knob values per algorithm and writes one CSV row per answered query:
the server it ran on, the configuration that answered, how long it took, the shape of the
index that configuration built, and how much search work the query did.

Source: [`lib/optimizer/optimizer_collector.cpp`](../../lib/optimizer/optimizer_collector.cpp).
The grid, the dataset list and the defaults below all live at the top of that file.

Status (2026-11-19): MESSI, FreSh, SOFA, single-node Odyssey and, on servers with a GPU,
SING; L2 squared only.

## Build

```bash
cmake .. -DBUILD_OPTIMIZER=ON -DBUILD_SOFA=ON -DBUILD_ODYSSEY=ON   # SOFA needs FFTW3, Odyssey needs MPI
# on a server with an NVIDIA GPU and the CUDA toolkit, also add -DBUILD_SING=ON
make optimizer_collector                                          # -> build/lib/optimizer/optimizer_collector
```

The build uses `-march=native`, so build on the server you collect on.
SOFA and Odyssey are skipped (with a log line) when the build lacks their dependency.
Run the binary directly, never under `mpirun`: Odyssey runs as a single-node MPI singleton.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--datasets` | `all` | Comma-separated names from the dataset table, or `all` |
| `--algos` | `messi,fresh,sofa,odyssey,sing` | Algorithms to run (unavailable ones are skipped with a log line) |
| `--k` | `1,10,100` | k values; every query is answered once per k |
| `--queries` | `100` | Queries per dataset (see the dataset table for where they come from) |
| `--max-base` | `0` (all) | Index only the first N base vectors |
| `--budget-s` | `600` | Time cap per (configuration, k): once its queries have used this many seconds, the remaining queries of that k are skipped. `0` = no cap |
| `--out` | `optimizer_samples.csv` | Output CSV; created, or appended to if it has the same header |
| `--work-dir` | `optimizer_work` | Raw float32 copies of fvecs bases, which Odyssey builds from |
| `--vector-root` | `/mnt/hddhelp/mchatzakis/datasets/processed` | Root of the fvecs datasets |
| `--series-root` | `/mnt/hddhelp/mchatzakis/datasets/similarity-search-datasets` | Root of the raw data-series files |
| `--dry-run` | `0` | `1` prints the plan (sizes, memory, configurations left) without building or writing anything |

Every option takes a value. Bad arguments and a header mismatch exit with code 2.

## Running a full collection

1. Check the plan:
   ```bash
   cd build
   ./lib/optimizer/optimizer_collector --dry-run 1 --out samples_erato.csv
   ```
   On the lab machine (erato) it lists 14 datasets × 228 configurations = 3,192
   configurations (up to 957,600 rows). SING is skipped there: erato has no GPU and the
   build has no CUDA.
2. Start it in the background:
   ```bash
   nohup ./lib/optimizer/optimizer_collector --out samples_erato.csv > collector_erato.log 2>&1 &
   ```
   Before Odyssey's first configuration on each fvecs dataset, the collector writes a
   raw float32 copy of the base into `--work-dir` (default `optimizer_work`; about 316 GB
   for all eleven vector datasets). Later configurations reuse it. The series datasets
   are already raw files and need no copy.
3. Follow it with `tail -f collector_erato.log`. Every configuration prints one line;
   `grep -c ": ok " collector_erato.log` counts finished ones.
4. There is no resume: a rerun starts from the first configuration again. If a run
   stops, the CSV keeps every configuration that finished; restart with a new `--out`, or
   with `--datasets` / `--algos` limited to what is missing.

Expected duration on erato, in run order. "Measured" query times come from the exactness
test's timings (48 threads, Gaussian breakpoints and library-default knobs, 100 queries,
averaged over the four algorithms and k); "scaled" ones are guesses from a dataset of the
same dimension, scaled by size:

| Dataset | Query time | Per configuration | 171 configurations (previous grid) |
|---|---|---|---|
| SIFT1M | ~0.16 s (measured) | ~1 min | ~2.5 h |
| GLOVE1M | ~0.15 s (scaled from SIFT1M) | ~1 min | ~2.5 h |
| GIST1M | ~0.08 s (measured) | ~0.5 min | ~1.5 h |
| HOTPOTQA5M | ~0.36 s (measured) | ~2 min | ~5.5 h |
| FEVER5M | ~0.37 s (scaled from HOTPOTQA5M) | ~2 min | ~5.5 h |
| MIRACLJA6M | ~0.48 s (scaled from HOTPOTQA5M) | ~2.5 min | ~7 h |
| MSMARCO8M | ~0.70 s (measured) | ~4 min | ~10.5 h |
| DEEP100M | ~0.95 s (measured) | ~5 min | ~15 h |
| MIRACLRU9M | ~0.75 s (scaled from MSMARCO8M) | ~4 min | ~11.5 h |
| UCASELAW7M | ~0.57 s (measured) | ~3 min | ~9 h |
| T2I100M | ~2.4 s (scaled from T2I10M's 0.24 s) | ~14 min | ~39 h |
| SEISMIC100M | ~2.1 s (measured, MESSI only) | ~11 min | ~31 h |
| ASTRO100M | assumed equal to SEISMIC100M | ~11 min | ~31 h |
| SALD100M | assumed half of SEISMIC100M's (half as many floats) | ~6 min | ~15 h |

The table was computed for the previous grid (171 configurations per dataset, threads
12, 24 and 48): about 7 days if fewer threads cost nothing. The current grid has 228
configurations per dataset (threads 32, 24, 16, 8 on erato), which scales that to about
9.5 days, and SALD100M adds about 0.8 more. Fewer threads are slower on the full
datasets, and 8 threads is the slowest setting in the grid, so plan for roughly 12 to 14
days. These are rough extrapolations.
Gaussian-breakpoint configurations on the vector datasets are the slow end (they prune
almost nothing); `--budget-s` caps any configuration at 3 × 600 s of queries.
To shorten the run, lower `--queries`, or run subsets with `--datasets` / `--algos`
into the same `--out`.

### On another server

- Build there (the binary is tuned to the CPU it is built on).
- Point `--vector-root` / `--series-root` at that server's copies of the datasets.
  Datasets whose files are missing are skipped.
- The machine columns are filled in automatically, and the `threads` grid adapts to the
  server (P, 3P/4, P/2, P/4, where P is the largest power of two not above its hardware
  threads).
- SING runs only if the collector was built with `-DBUILD_SING=ON` (CUDA) and the server
  has an NVIDIA GPU; otherwise the log says why it was skipped.
- Use one `--out` per server (for example `samples_<host>.csv`) and concatenate the
  files afterwards: every file has the same header, and the `host` column tells the
  rows apart.

## What a run does

For each dataset, then each algorithm, then each grid configuration:

1. Build the index once, with all cores, and time the `buildIndex` call.
2. Walk the built tree once and record its shape (not timed).
3. For each k: answer every query alone (`searchIndex` with one query) and time each
   call, until the queries run out or the `--budget-s` cap for this k is reached.
4. Append all rows of the configuration to the CSV in one write and sync it to disk.

A configuration that throws writes nothing; the log shows the error. Rows per configuration = number of k values × number of queries (300 with
the defaults), fewer when the budget stopped a k early.

Before loading a dataset, the collector checks that the base (plus a rough index
estimate) fits in 90% of the memory currently available (`MemAvailable`); after loading
it does the same for Odyssey's own copy of the base. Whatever does not fit is skipped
with a log line that suggests `--max-base`.

## Datasets

| Name | Format | Base | Queries |
|---|---|---|---|
| SIFT1M | fvecs, dim 128 | `SIFT1M/base.1M.fvecs` | first N of `query.10K.fvecs` |
| GLOVE1M | fvecs, dim 100 | `GLOVE1M/base.1183514.fvecs` | first N of `query.10K.fvecs` |
| GIST1M | fvecs, dim 960 | `GIST1M/base.1M.fvecs` | first N of `query.1K.fvecs` |
| HOTPOTQA5M | fvecs, dim 1024 | `HOTPOTQA5M/base.5233329.fvecs` | first N of `query.7405.fvecs` |
| FEVER5M | fvecs, dim 1024 | `FEVER5M/base.5416568.fvecs` | first N of `query.6666.fvecs` |
| MIRACLJA6M | fvecs, dim 1024 (MIRACL Japanese) | `MIRACLJA6M/base.6953614.fvecs` | first N of `query.860.fvecs` |
| MSMARCO8M | fvecs, dim 1024 | `MSMARCO8M/base.8841823.fvecs` | first N of `query.6980.fvecs` |
| DEEP100M | fvecs, dim 96 | `DEEP100M/base.100M.fvecs` | first N of `query.10K.fvecs` |
| MIRACLRU9M | fvecs, dim 1024 (MIRACL Russian) | `MIRACLRU9M/base.9543918.fvecs` | first N of `query.1252.fvecs` |
| UCASELAW7M | fvecs, dim 1536 | `UCASELAW7M/base.7M.fvecs` | first N of `query.1K.fvecs` |
| T2I100M | fvecs, dim 200 | `T2I100M/base.100M.fvecs` | first N of `query.10K.fvecs` |
| SEISMIC100M | raw float32, length 256, z-normalized (110.0M records) | `data_size100M_seismic_len256_znorm.bin` | last N records of the file, held out of the base |
| ASTRO100M | raw float32, length 256, z-normalized; the base is the file's first 100M of its 277.4M records (all of them need about 302 GB, which does not fit) | `data_size270M_astronomy_len256_znorm.bin` | last N records of the file, held out of the base |
| SALD100M | raw float32, length 128, z-normalized; brain-MRI voxel time series, built by [`scripts/datasets/make_sald.py`](../../scripts/datasets/make_sald.py) (see below); the base is the file's first 100M of its 237.3M records | `data_size200M_sald_len128_znorm.bin` | first N of `queries_size10K_sald_len128_znorm.bin`, windows of a subject that is not in the base |

N is `--queries`, capped by the size of the query file. The fvecs vector datasets are not
z-normalized, which matters for the Gaussian-breakpoint settings below.
All datasets are searched with squared L2 on the raw vectors, including GLOVE1M and
T2I100M, which are usually searched by inner product (their folders only have
inner-product ground truth); the collector needs no ground truth.

### SALD100M

SALD is not distributed as data series: it is the Southwest University Adult Lifespan
Dataset, 494 subjects with one 8-minute resting-state fMRI scan each (Siemens Trio 3 T,
TR 2 s, 64 × 64 × 32 voxels × 242 time points), licensed CC BY-NC (cite Wei et al. 2017,
bioRxiv 177279). The data-series papers (MESSI, SING, Odyssey) use "200M series of
length 128, 100 GB" derived from it, but do not publish the file or the procedure.
`scripts/datasets/` rebuilds a dataset of that size:

```bash
scripts/datasets/download_sald.sh /mnt/hddhelp/mchatzakis/datasets/raw/SALD     # 23 GB, anonymous S3
python3 -I scripts/datasets/make_sald.py /mnt/hddhelp/mchatzakis/datasets/raw/SALD \
    /mnt/hddhelp/mchatzakis/datasets/similarity-search-datasets
```

- Each brain voxel's time course (voxel mean above 10% of the scan's highest voxel mean,
  about 32,000 voxels per subject on average) is cut into windows of 128 points with
  stride 8, 15 per voxel, and each window is z-normalized. The archives hold 493
  resting-state scans; the 2026-11-19 build wrote 237,294,685 series of 492 subjects
  (121.5 GB) in 17 minutes. The collector indexes the file's first 100M series (51.2 GB,
  `max_n`, about the first 207 subjects), so it fits on servers with less memory; the
  benchmark's SALD has 200M. To set it up on another server, copying the first 51.2 GB of
  the base file is enough: `head -c 51200000000 data_size200M_sald_len128_znorm.bin`, keeping
  the file name, plus the query file.
- The first subject read is held out and only provides the 10,000 queries
  (`queries_size10K_sald_len128_znorm.bin`). Consecutive windows of a voxel share 120 of
  128 points, so a query taken from a base subject would find near-copies of itself.
- `data_size200M_sald_len128_znorm.txt` next to the files records the parameters, the
  counts and the held-out subject.
- The result matches the papers' SALD in source, length and size, not necessarily
  series by series. The overlapping windows make it denser in near-duplicates than
  Seismic or Astro.

## Algorithms

| Algorithm | Needs | Builds from | Notes |
|---|---|---|---|
| MESSI | — | in-memory base (no copy) | |
| FreSh | — | in-memory base (no copy) | |
| SOFA | `BUILD_SOFA` (FFTW3) | in-memory base (no copy) | Learns its own bins, so breakpoint settings do not apply |
| Odyssey | `BUILD_ODYSSEY` (MPI) | raw float32 file, read into its own copy | Single node. For fvecs datasets the base is first written to `--work-dir` (a file with the right name and size is reused) |
| SING | `BUILD_SING` (CUDA) and an NVIDIA GPU | in-memory base; SAX words and a bitmap (about 20 bytes per series) on the GPU | Skipped when that does not fit in 90% of the GPU's free memory. Not validated for exactness, and never run on erato (no GPU); see "Caveats" |

## Knobs

One CSV column per knob, always in this order. A row has `NaN` in the knob columns its
algorithm does not have.

| Column | Algorithms | Grid values | What it controls |
|---|---|---|---|
| `threads` | all | P, 3P/4, P/2, P/4, where P is the largest power of two not above the server's hardware threads (erato, 48 threads: 32, 24, 16, 8) | Threads that build the index and that answer one query (`setIndexWorkers` and `setSearchWorkers`; Odyssey's `index_threads`, `query_threads` and `search_workers`; SING's `setIndexWorkers` and `setNumThreads`) |
| `leaf_size` | all | MESSI/FreSh: 500, 1000, 2000, 5000. SOFA: 5000, 20000. Odyssey: 500, 1000, 2000 | Maximum series per leaf before it splits. The leaf buffer size (`initial_lbl_size`) is set to the same value |
| `sax_cardinality` | MESSI, FreSh | 4, 8 | Bits per iSAX segment at full resolution (4 bits = 16 symbols, 8 bits = 256) |
| `equidepth_breakpoints` | MESSI, FreSh | 0, 1 | 0: Gaussian breakpoints, which assume z-normalized data. 1: equi-depth breakpoints learned from the data (`setNormalized(false)`) |
| `word_length` | SOFA | 8, 16 | Fourier coefficients per SFA word |
| `alphabet_size` | SOFA | 8, 64 | Symbols per coefficient (the bit cardinality is its log2) |
| `histogram_type` | SOFA | 1, 2 | How each coefficient's bins are learned: 1 equi-depth (equal counts), 2 equi-width (equal value range) |
| `is_norm` | SOFA | 0 | 1 drops the DC (first) Fourier coefficient, for z-normalized data. Fixed at 0 in the grid |
| `sample_size` | SOFA | 100000 | Series sampled to learn the bins. Fixed in the grid |
| `pq_th_div_factor` | Odyssey | 4, 16, 64 | Odyssey splits a query's root subtrees into `threads` batches and queues each batch's candidate leaves in priority queues of at most E / `pq_th_div_factor` leaves, where E is estimated from the approximate answer's k-th distance. A larger factor gives more, smaller queues |

Grid sizes (full Cartesian product, with 4 thread values):

| Algorithm | Configurations |
|---|---|
| MESSI | 4 × 4 × 2 × 2 = 64 |
| FreSh | 64 |
| SOFA | 4 × 2 × 2 × 2 × 2 = 64 |
| Odyssey | 4 × 3 × 3 = 36 |
| SING (GPU servers only) | 4 |
| Total per dataset | 228 without SING (68,400 rows with the defaults), 232 with it |

History: 98 configurations per dataset at first; 171 on 2026-11-10 (a third search-thread
value, and leaf_size 500 for MESSI and FreSh); 228 on 2026-11-19, when the thread knob
became one `threads` value for both build and search, with the four power-of-two-based
values. SING exposes only its thread counts, so its grid is the four thread values.
In the product the first knob of the grid varies slowest and the last fastest. The
progress log prints each configuration's knobs.

Settings that are not knobs:

- Build-only settings other than the thread count (read block length) are not columns.
- `paa_segments` stays 16 for MESSI, FreSh and Odyssey: their SIMD lower bound assumes 16 segments.
- Odyssey `leaf_size` is at most 2000 because Odyssey has no setter for its leaf buffer size.
- Odyssey always uses Gaussian breakpoints: it ignores `setNormalized`.
- MESSI and SOFA use 42 work queues per query and FreSh uses `threads` / 2. These
  are fixed in the library.
- Everything else (minimum leaf size, flush limit, `tight_bound` = 0, `min_checked_leaves` = -1)
  stays at the library default.

## Outputs

### CSV columns

Machine columns (the same on every row of a run; read from `/proc` and `/sys` at start-up):

| Column | Meaning |
|---|---|
| `host` | Host name |
| `cpu_model` | CPU model name (commas removed) |
| `cpu_sockets` | Sockets |
| `cpu_cores` | Physical cores, all sockets |
| `cpu_threads` | Hardware threads (logical CPUs) |
| `cpu_max_mhz` | Maximum CPU frequency, MHz (`NaN` if the kernel does not report it) |
| `l2_cache_kb` | L2 cache of one core, KB |
| `l3_cache_kb` | L3 cache, KB, summed over all L3 instances (all sockets) |
| `numa_nodes` | NUMA nodes |
| `memory_gb` | Total RAM, GiB |
| `simd` | Widest vector extension the CPU has: `avx512`, `avx2`, `avx`, `sse4.2` |
| `gpu_count` | NVIDIA GPUs (CUDA runtime when SING is built, otherwise `/proc/driver/nvidia`) |
| `gpu_model` | Model of the first GPU, `none` without one |
| `compiler` | Compiler that built the collector |

Run and configuration columns:

| Column | Unit | Meaning |
|---|---|---|
| `algorithm` | | `messi`, `fresh`, `sofa`, `odyssey` or `sing` |
| `dataset_name` | | Name from the dataset table |
| `dataset_size` | series | Number of series indexed (after `--max-base`) |
| `dataset_dimensionality` | | Series length / vector dimension |
| `k` | | Neighbours requested |
| `query_id` | | 0-based index of the query in this dataset's workload |
| knob columns | | See "Knobs" |

Measurements:

| Column | Unit | Meaning |
|---|---|---|
| `latency_ms` | ms | Wall-clock time of the single-query `searchIndex` call |
| `build_time_ms` | ms | Wall-clock time of the configuration's `buildIndex` call. Odyssey's includes reading its raw file |
| `n_roots` | count | Root subtrees of the built index |
| `n_leaves` | count | Leaves of the built index |
| `leaf_fill_mean` | series | `dataset_size / n_leaves` |
| `leaf_fill_max` | series | Series in the largest leaf |
| `largest_leaf_share` | 0–1 | `leaf_fill_max / dataset_size` |
| `tree_depth` | levels | Depth of the deepest leaf; 0 when every root is a leaf |
| `node_lb_calcs` | count | Lower bounds computed for tree nodes |
| `leaves_enqueued` | count | Leaves that passed node pruning in the exact phase and were queued |
| `leaves_visited` | count | Leaf scans, including the approximate seed leaf. A leaf scanned twice counts twice |
| `series_lb_calcs` | count | Lower bounds computed for single series inside scanned leaves |
| `distance_calcs` | count | Real distance computations started. Computations that stopped early (because they could no longer beat the k-th best) count too |
| `seed_kth_dist` | squared L2 | k-th best distance when the exact phase starts: the quality of the approximate answer. `inf` if it had fewer than k series |
| `final_kth_dist` | squared L2 | k-th distance of the returned answer |

`build_time_ms` and the six tree columns describe the configuration, so they repeat on
all its rows. The tree columns are measured right after the build (see "Caveats").

The six columns from `node_lb_calcs` to `seed_kth_dist` come from the library's
`getLastSearchStats()` ([`lib/utils/SearchStats.hpp`](../../lib/utils/SearchStats.hpp)).
They are `NaN` for an algorithm that does not collect them. `final_kth_dist` is always
filled, from the returned distances.

The counters cover every phase of a query:

1. **Approximate seed.** Descend to one leaf and compute the real distance to every series in it.
2. **Refinement.** Runs serially only when the seed found fewer than k series. It visits
   nodes in lower-bound order until k series are found.
3. **Exact phase.** Parallel. Compute node lower bounds and queue the surviving leaves.
   Then scan each queued leaf: compute each series' lower bound, and compute the real
   distance only if the bound does not exceed the current k-th best distance.

Odyssey also recomputes the k exact distances of its approximate answer before the exact
phase; they are part of `distance_calcs`.

Useful ratios:

- `leaves_enqueued / n_leaves`: share of the leaves that survived node pruning.
- `distance_calcs / dataset_size`: share of the data that needed a real distance. It can
  exceed 1, because the exact phase scans the seed leaf again.
- `series_lb_calcs − distance_calcs`: roughly the series removed by their lower bound.
  The seed leaf's distances have no per-series bound, so this undercounts by the seed size.
- `seed_kth_dist / final_kth_dist` (≥ 1): how far the approximate answer was from the exact one.

Row order: dataset → algorithm → configuration → k → query.

### Progress log (stdout)

```
machine: erato,Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz,2,24,48,2900,256,61440,2,377.8,avx2,0,none,gcc 11.4.0
threads grid: 32,24,16,8
== SIFT1M
   n=1000000 dim=128 queries=100, base 0.5 GB, available memory 294.3 GB
   messi: 64 configurations
   ...
   sing: not built (needs BUILD_SING and CUDA), skipped
   loading
   messi 1/64 [ threads=32 leaf_size=500 sax_cardinality=4 equidepth_breakpoints=0 ]: ok (build 0.6 s, total 50.3 s, 300 rows)
   ...
samples written to samples_erato.csv
```

A configuration stopped by the budget adds `k=100 stopped after 37/100 queries (budget)`.
A failed one prints `error (nothing written): <message>`. Odyssey adds its own lines
(`[Node 0] Worker ... inserted ... records`).

### Header check

The header is written only when `--out` is new or empty. An existing file is appended to
only if its first line matches the current header exactly; otherwise the collector exits
with code 2 without writing anything (files from before 2026-11-10 have an older header).
Appending runs every configuration again, so running the same command twice into one
file gives every row twice; nothing de-duplicates them.

## Caveats

- **SING is unvalidated.** It has never been built or run on erato (no CUDA, no GPU),
  so its rows have not been checked against the other algorithms. It also still uses the
  top-k insert that allows the same series twice (MESSI, FreSh, SOFA and Odyssey were
  fixed), so for k > 1 its answers may contain duplicates. Compare its `final_kth_dist`
  with the other algorithms' before using its rows. It collects no search counters (`NaN`).

- **Counters vary between runs.** With several search threads, the threads race on the
  best-so-far distance, so the same query's counts change from run to run. The final
  answer does not.
- **The tree changes during k = 100 queries.** When the seed leaf holds fewer than k
  series, the refinement of MESSI, FreSh and Odyssey splits the leaves it visits
  (adaptive splitting), and the index keeps the splits. The tree columns describe the
  tree right after the build. On 100,000 SIFT vectors with MESSI, 100 queries changed
  the leaf count by nothing at k = 1, by at most 0.2% at k = 10, and by +11% (equi-depth
  breakpoints) to +50% (Gaussian) at k = 100. Since each configuration runs k = 1, 10,
  100 in that order, the k = 1 and k = 10 rows see the built tree, and the k = 100 rows a
  progressively finer one. SOFA's refinement does not split.
- **Cold first query.** There is no warm-up query, so `query_id` 0 of each
  (configuration, k), and especially of k = 1 right after the build, is usually slower.
  Drop it in the analysis if it matters.
- **Budget-truncated configurations have fewer queries**, and only the first ones of the
  workload. The log lists them; in the CSV, count rows per (configuration, k).
- **FreSh rescans leaves.** Its helper threads can scan the same leaf again before it is
  marked done. That is real work and the counters include it, so FreSh's
  `leaves_visited` can exceed `leaves_enqueued` + 1.
- **Not counted:** the refinement's optional tight-bound pass (`tight_bound` = 1, off in every configuration).
- **Machine columns are static.** They do not capture frequency scaling, turbo state, or
  other load on the machine during the run.

## First observations (SIFT1M, first 100,000 vectors)

Search work, means over each group's grid configurations, 5 queries each, k = 10
(2026-11-08, previous grid):

| Group | latency (ms) | node LBs | leaves queued | leaves visited | series LBs | distances |
|---|---|---|---|---|---|---|
| MESSI, Gaussian breakpoints | 23.4 | 181 | 101 | 102 | 100,000 | 172,604 |
| MESSI, equi-depth breakpoints | 8.3 | 10,873 | 10,872 | 10,198 | 96,458 | 82,902 |
| FreSh, Gaussian breakpoints | 32.4 | 181 | 101 | 184 | 3,240,196 | 3,313,187 |
| FreSh, equi-depth breakpoints | 4.9 | 10,873 | 10,872 | 10,351 | 154,791 | 135,734 |
| SOFA | 20.3 | 17,685 | 10,805 | 10,035 | 95,564 | 88,354 |
| Odyssey (Gaussian breakpoints) | 32.1 | 235 | 122 | 123 | 100,000 | 170,821 |

Tree shape (2026-11-10, from the tree columns):

| Index | roots | leaves | largest leaf | depth |
|---|---|---|---|---|
| MESSI, Gaussian, sax_cardinality 4 (any leaf_size) | 49 | 97–100 | 92,880 (93%) | 48 |
| MESSI, Gaussian, sax_cardinality 8 | 49 | 161–163 | 88,399 (88%) | 112 |
| MESSI, equi-depth, leaf_size ≥ 2000 | 10,872 | 10,872 | 1,478 | 0 |
| MESSI, equi-depth, leaf_size 500 | 10,872 | 10,888–10,889 | 482–500 | 3–4 |
| SOFA, word_length 8 | 128 | 128–132 | 2,092–7,908 | 0–2 |
| SOFA, word_length 16 | 17,184–25,791 | same | 77–257 | 0 |
| Odyssey (Gaussian, library-default sax_cardinality 8) | 49 | 161–163 | 88,399 (88%) | 112 |

SIFT is not z-normalized, so with Gaussian breakpoints most values fall outside the
breakpoints: the tree splits to its maximum depth and still leaves about 90% of the
vectors in one leaf. Every query then lower-bounds every series, the seed leaf alone
costs about 90,000 distances, and FreSh's helpers rescan that leaf about once per thread.
With equi-depth breakpoints almost every root is already a leaf of about 9 series, so
`leaf_size` barely matters on this data, node pruning removes almost nothing, and the
series lower bounds remove only about 15% of the distances.

## Not collected yet

- Python access to the counters.
- A measured machine characteristic such as memory bandwidth; exact search is largely
  memory-bound, and the static machine columns do not capture it.
