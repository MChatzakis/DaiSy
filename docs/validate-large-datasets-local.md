# Large local datasets: exactness validation (`validate_large_datasets_local`)

`tests/validate_large_datasets_local.cpp` checks that the in-memory methods,
MESSI, FreSh, SOFA and Odyssey (single node), return exact top-k answers on the
large datasets stored on the lab machine. Hercules, ParIS and SING are not
included, to keep the run short. It is part of the test suite as the CTest entry
`validate_large_datasets_local`, with label `large_local`. It is also the first
tool of the knob optimizer: knobs are only worth tuning for algorithms that
pass. It is a standalone executable, not a gtest.

## Build and run under CTest

```bash
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON -DBUILD_ODYSSEY=ON -DBUILD_SOFA=ON
make -j validate_large_datasets_local

ctest -L large_local --output-on-failure   # only this test
ctest -LE large_local                      # everything else
```

A full run, with every dataset, k = 1, 10, 100 and 1000 queries, takes many
hours. Plain `ctest` includes it, so use `-LE large_local` for routine runs.
Two cache variables configure the CTest entry:

| Variable | Default | Meaning |
|---|---|---|
| `DAISY_LARGE_DATASETS_ARGS` | empty (the full run) | extra arguments, e.g. `"--datasets SIFT1M,Seismic --queries 100 --max-base 10000000"` |
| `DAISY_LARGE_DATASETS_TIMEOUT` | `0` (no limit) | CTest timeout in seconds |

The test is `RUN_SERIAL` because it uses every core and a lot of RAM. It exits
with code 77, which CTest reports as skipped, when no dataset is present
(for example on CI or a laptop). Its work directory is
`build/validate_large_datasets_work/`.

SOFA and Odyssey are compiled in only when FFTW3 and MPI are found. Otherwise
their runs are reported as `SKIP (not compiled ...)`. If the machine has the
FFTW runtime but no development package, point CMake at a header and the
`.so.3` explicitly:
`-DFFTW3_INCLUDE_DIR=<dir with fftw3.h> -DFFTW3f_LIBRARY=/usr/lib/x86_64-linux-gnu/libfftw3f.so.3`.

## Running the binary directly

```bash
./tests/validate_large_datasets_local --list                     # resolve datasets, no work
./tests/validate_large_datasets_local --datasets SIFT1M,GIST1M    # all algorithms, k=1,10,100
./tests/validate_large_datasets_local --datasets Seismic --max-base 10000000 --algos messi,odyssey
./tests/validate_large_datasets_local --set messi.paa_segments=8 --datasets T2I10M --algos messi
```

Never launch it under `mpirun`. Odyssey runs single node as an MPI singleton
inside a forked child.

Datasets: `SIFT1M GIST1M T2I10M HOTPOTQA5M UCASELAW7M MSMARCOMXBAI8M DEEP100M`
(fvecs, under `--vector-root`), and `Seismic Astro` (raw float32 series of
length 256, under `--series-root`). The data-series files ship without queries,
so the last `--queries` records of the file are held out as queries and the base
is read from the start of the file.

`--help` lists all options. The important ones are `--k`, `--queries` (default
1000), `--max-base`, `--threads`, `--timeout` (per algorithm run),
`--bp gaussian|equidepth` and `--set ALGO.KNOB=VALUE` (`--list-knobs` shows the
accepted knobs).

Exit status: 0 if every executed run is `PASS`, 1 if any run is not, 77 if
nothing could run.

## Reference answers

* If the dataset has an L2 ground-truth file and the full base is loaded, the
  file is used. A few queries (`--gt-verify`, default 10) are brute-forced
  first. If the file disagrees, it is rejected and brute force is used instead.
* Otherwise the tool runs an exact brute-force scan: a float32 candidate pass,
  then a double-precision re-rank. T2I10M has only inner-product ground truth,
  and Seismic and Astro have none. The result is cached under
  `<work-dir>/reference_cache/`, keyed by a fingerprint of the base and queries.

## Checks per (dataset, algorithm, k)

The tool recomputes the true squared L2 distance (in double precision) of every
returned id. It then checks:

| Check | Meaning | Status impact |
|---|---|---|
| exact | every returned id is within the reference k-th distance (`--rtol`, `--atol`); ties are accepted | FAIL if any query is not exact |
| duplicates | the same id returned twice for one query (padding) | FAIL |
| invalid / unwritten | id ≥ n, or output slot never written | FAIL |
| reported distance | returned `D` differs from true squared L2 (`--dist-rtol`, `--dist-atol`) | WARN |
| sorted | returned `D` ascending | WARN |
| input mutation | the library changed the caller's base or query buffer | WARN |
| reference suspect | returned set strictly closer than the reference | informational |

Id recall against the reference is reported but not used for pass/fail, since
it is tie-sensitive. For example, T2I10M contains duplicate vectors, so the
k = 1 recall can be 0 for a query that is still exact.

Crashes (signals), `exit()` calls inside the library, and timeouts are reported
as `CRASH` or `TIMEOUT`, together with the phase they happened in. Each run's
library output goes to `<work-dir>/logs/<dataset>__<algo>.log`. The last lines
are printed when a run fails. Results are appended to `<work-dir>/results.csv`.

## Resource notes

* The tool keeps the base in RAM and shares it with each child through
  copy-on-write. MESSI, FreSh and SOFA use the caller's buffer directly.
* Odyssey builds from a raw float32 file and keeps a private copy in RAM. For
  fvecs datasets the tool writes `<work-dir>/bin_cache/<name>_n<N>_d<dim>.f32`
  once and reuses it. Odyssey is skipped when there is not enough RAM for the
  second copy (Astro at full size). Use `--max-base`, or `--force` to run it
  anyway.
* `--bp equidepth` only affects MESSI and FreSh. SOFA and Odyssey ignore the
  breakpoint mode. With the
  default Gaussian breakpoints, all indexes are very slow on non-normalized
  vectors (about 5 queries/s on SIFT1M with 48 threads), and equi-depth makes
  MESSI and FreSh about 10× faster.
