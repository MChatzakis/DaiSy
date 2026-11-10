# Knob-optimizer sample collector

`optimizer_collector` produces the training and analysis data for the knob optimizer.
It sweeps a grid of knob values per algorithm and writes one CSV row per answered query:
the server it ran on, the configuration that answered, how long it took, the shape of the
index that configuration built, and how much search work the query did.

Source: [`lib/optimizer/optimizer_collector.cpp`](../../lib/optimizer/optimizer_collector.cpp).
The grid, the dataset list and the defaults below all live at the top of that file.

Status (2026-11-10): MESSI, FreSh, SOFA and single-node Odyssey, L2 squared only.

## Build

```bash
cmake .. -DBUILD_OPTIMIZER=ON -DBUILD_SOFA=ON -DBUILD_ODYSSEY=ON   # SOFA needs FFTW3, Odyssey needs MPI
make optimizer_collector                                          # -> build/lib/optimizer/optimizer_collector
```

The build uses `-march=native`, so build on the server you collect on.
SOFA and Odyssey are skipped (with a log line) when the build lacks their dependency.
Run the binary directly, never under `mpirun`: Odyssey runs as a single-node MPI singleton.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--datasets` | `all` | Comma-separated names from the dataset table, or `all` |
| `--algos` | `messi,fresh,sofa,odyssey` | Algorithms to run |
| `--k` | `1,10,100` | k values; every query is answered once per k |
| `--queries` | `100` | Queries per dataset (see the dataset table for where they come from) |
| `--max-base` | `0` (all) | Index only the first N base vectors |
| `--budget-s` | `600` | Time cap per (configuration, k): once its queries have used this many seconds, the remaining queries of that k are skipped. `0` = no cap |
| `--out` | `optimizer_samples.csv` | Output CSV; appended to, and used to resume |
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
   On the lab machine (erato) it lists 8 datasets × 171 configurations = 1,368
   configurations (up to 410,400 rows). Astro is skipped: it needs about 302 GB and
   294 GB was available.
2. Start it in the background. On erato, reuse the raw base copies that the
   exactness test already wrote (about 150 GB, same file names), so Odyssey does not
   need to write them again:
   ```bash
   nohup ./lib/optimizer/optimizer_collector --out samples_erato.csv \
       --work-dir validate_large_datasets_work/bin_cache > collector_erato.log 2>&1 &
   ```
3. Follow it with `tail -f collector_erato.log`. Every configuration prints one line;
   `grep -c ": ok " collector_erato.log` counts finished ones.
4. If it stops (crash, reboot, `kill`), run the same command again. Configurations
   already in the CSV are skipped; a dataset with nothing left to run is not even loaded.

Expected duration on erato, extrapolated from the exactness test's timings
(48 threads, Gaussian breakpoints and library-default knobs, 100 queries, averaged over
the four algorithms and k):

| Dataset | Query time | Per configuration | 171 configurations |
|---|---|---|---|
| SIFT1M | ~0.16 s | ~1 min | ~2.5 h |
| GIST1M | ~0.08 s | ~0.5 min | ~1.5 h |
| T2I10M | ~0.24 s | ~1.5 min | ~4 h |
| HOTPOTQA5M | ~0.36 s | ~2 min | ~5.5 h |
| UCASELAW7M | ~0.57 s | ~3 min | ~9 h |
| MSMARCOMXBAI8M | ~0.70 s | ~4 min | ~10.5 h |
| DEEP100M | ~0.95 s | ~5 min | ~15 h |
| Seismic | ~2.1 s (MESSI only) | ~11 min | ~31 h |

That is about 3.5 days if 12 and 24 threads are as fast as 48. On 100,000 SIFT vectors
they were, but on the full datasets fewer threads will be slower, so plan for 4 to 5 days.
Gaussian-breakpoint configurations on the vector datasets are the slow end (they prune
almost nothing); `--budget-s` caps any configuration at 3 × 600 s of queries.
To shorten the run, lower `--queries`, or run subsets with `--datasets` / `--algos`
into the same `--out`.

### On another server

- Build there (the binary is tuned to the CPU it is built on).
- Point `--vector-root` / `--series-root` at that server's copies of the datasets.
  Datasets whose files are missing are skipped.
- The machine columns are filled in automatically, and the `search_threads` grid adapts
  to the server (half its physical cores, all of them, all hardware threads).
- Use one `--out` per server (for example `samples_<host>.csv`) and concatenate the
  files afterwards: every file has the same header, and the `host` column tells the
  rows apart. Writing several servers into one shared file also works, because resume
  only skips configurations written by the same host.

## What a run does

For each dataset, then each algorithm, then each grid configuration not yet in the CSV:

1. Build the index once, with all cores, and time the `buildIndex` call.
2. Walk the built tree once and record its shape (not timed).
3. For each k: run one untimed warm-up query (query 0), then answer every query alone
   (`searchIndex` with one query) and time each call, until the queries run out or the
   `--budget-s` cap for this k is reached.
4. Append all rows of the configuration to the CSV in one write and sync it to disk.

A configuration that throws writes nothing; the log shows the error, and the next run
retries it. Rows per configuration = number of k values × number of queries (300 with
the defaults), fewer when the budget stopped a k early.

Before loading a dataset, the collector checks that the base (plus a rough index
estimate) fits in 90% of the memory currently available (`MemAvailable`); after loading
it does the same for Odyssey's own copy of the base. Whatever does not fit is skipped
with a log line that suggests `--max-base`.

## Datasets

| Name | Format | Base | Queries |
|---|---|---|---|
| SIFT1M | fvecs | `SIFT1M/base.1M.fvecs` | first N of `query.10K.fvecs` |
| GIST1M | fvecs | `GIST1M/base.1M.fvecs` | first N of `query.1K.fvecs` |
| T2I10M | fvecs | `T2I10M/base.10M.fvecs` | first N of `query.10K.fvecs` |
| HOTPOTQA5M | fvecs | `HOTPOTQA5M/base.5233329.fvecs` | first N of `query.7405.fvecs` |
| UCASELAW7M | fvecs | `UCASELAW7M/base.7M.fvecs` | first N of `query.1K.fvecs` |
| MSMARCOMXBAI8M | fvecs | `MSMARCOMXBAI8M/base.8841823.fvecs` | first N of `query.6980.fvecs` |
| DEEP100M | fvecs | `DEEP100M/base.100M.fvecs` | first N of `query.10K.fvecs` |
| Seismic | raw float32, length 256, z-normalized | `data_size100M_seismic_len256_znorm.bin` | last N records of the file, held out of the base |
| Astro | raw float32, length 256, z-normalized | `data_size270M_astronomy_len256_znorm.bin` | last N records of the file, held out of the base |

N is `--queries`, capped by the size of the query file. The fvecs vector datasets are not
z-normalized, which matters for the Gaussian-breakpoint settings below.

## Algorithms

| Algorithm | Needs | Builds from | Notes |
|---|---|---|---|
| MESSI | — | in-memory base (no copy) | |
| FreSh | — | in-memory base (no copy) | |
| SOFA | `BUILD_SOFA` (FFTW3) | in-memory base (no copy) | Learns its own bins, so breakpoint settings do not apply |
| Odyssey | `BUILD_ODYSSEY` (MPI) | raw float32 file, read into its own copy | Single node. For fvecs datasets the base is first written to `--work-dir` (a file with the right name and size is reused) |

## Knobs

One CSV column per knob, always in this order. A row has `NaN` in the knob columns its
algorithm does not have.

| Column | Algorithms | Grid values | What it controls |
|---|---|---|---|
| `search_threads` | all | half the physical cores, all physical cores, all hardware threads (erato: 12, 24, 48) | Threads that answer one query (`setSearchWorkers`; for Odyssey both `query_threads` and `search_workers`) |
| `leaf_size` | all | MESSI/FreSh: 500, 1000, 2000, 5000. SOFA: 5000, 20000. Odyssey: 500, 1000, 2000 | Maximum series per leaf before it splits. The leaf buffer size (`initial_lbl_size`) is set to the same value |
| `sax_cardinality` | MESSI, FreSh | 4, 8 | Bits per iSAX segment at full resolution (4 bits = 16 symbols, 8 bits = 256) |
| `equidepth_breakpoints` | MESSI, FreSh | 0, 1 | 0: Gaussian breakpoints, which assume z-normalized data. 1: equi-depth breakpoints learned from the data (`setNormalized(false)`) |
| `word_length` | SOFA | 8, 16 | Fourier coefficients per SFA word |
| `alphabet_size` | SOFA | 8, 64 | Symbols per coefficient (the bit cardinality is its log2) |
| `histogram_type` | SOFA | 1, 2 | How each coefficient's bins are learned: 1 equi-depth (equal counts), 2 equi-width (equal value range) |
| `is_norm` | SOFA | 0 | 1 drops the DC (first) Fourier coefficient, for z-normalized data. Fixed at 0 in the grid |
| `sample_size` | SOFA | 100000 | Series sampled to learn the bins. Fixed in the grid |
| `pq_th_div_factor` | Odyssey | 4, 16, 64 | Odyssey splits a query's root subtrees into `search_threads` batches and queues each batch's candidate leaves in priority queues of at most E / `pq_th_div_factor` leaves, where E is estimated from the approximate answer's k-th distance. A larger factor gives more, smaller queues |

Grid sizes (full Cartesian product, with 3 thread values):

| Algorithm | Configurations |
|---|---|
| MESSI | 3 × 4 × 2 × 2 = 48 |
| FreSh | 48 |
| SOFA | 3 × 2 × 2 × 2 × 2 = 48 |
| Odyssey | 3 × 3 × 3 = 27 |
| Total per dataset | 171 (51,300 rows with the defaults) |

The grid was extended on 2026-11-10 from 98 configurations: a third thread value (half
the physical cores) for every algorithm, and leaf_size 500 for MESSI and FreSh.
In the product the first knob of the grid varies slowest and the last fastest. The
progress log prints each configuration's knobs.

Settings that are not knobs:

- Index build always uses all hardware threads. Build-only settings are not columns.
- `paa_segments` stays 16 for MESSI, FreSh and Odyssey: their SIMD lower bound assumes 16 segments.
- Odyssey `leaf_size` is at most 2000 because Odyssey has no setter for its leaf buffer size.
- Odyssey always uses Gaussian breakpoints: it ignores `setNormalized`.
- MESSI and SOFA use 42 work queues per query and FreSh uses `search_threads` / 2. These
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
| `compiler` | Compiler that built the collector |

Run and configuration columns:

| Column | Unit | Meaning |
|---|---|---|
| `algorithm` | | `messi`, `fresh`, `sofa` or `odyssey` |
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
machine: erato,Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz,2,24,48,2900,256,61440,2,377.8,avx2,gcc 11.4.0
search_threads grid: 12,24,48
samples_erato.csv: 0 configurations already present (all hosts)
== SIFT1M
   n=1000000 dim=128 queries=100, base 0.5 GB, available memory 294.3 GB
   messi: 48 configurations, 0 already present, 48 to run
   ...
   loading
   messi 1/48 [ search_threads=12 leaf_size=500 sax_cardinality=4 equidepth_breakpoints=0 ]: ok (build 2.1 s, total 50.3 s, 300 rows)
   ...
samples written to samples_erato.csv
```

A configuration stopped by the budget adds `k=100 stopped after 37/100 queries (budget)`.
A failed one prints `error (nothing written): <message>`. Odyssey adds its own lines
(`[Node 0] Worker ... inserted ... records`).

### Header check and resume

The header is written only when `--out` is new or empty. An existing file is used only if
its first line matches the current header exactly; otherwise the collector exits with
code 2 without writing anything (files from before 2026-11-10 have an older header).
A partial last line, left by a run killed during a write, is removed with a warning.

A configuration counts as done when the CSV has any row with the same host, algorithm,
dataset, dataset size and knob values. Done configurations are skipped whatever `--k` and
`--queries` are now, so to collect other k values or more queries, use a new `--out`.

## Caveats

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
