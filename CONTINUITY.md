# CONTINUITY.md

This file captures the current shape of the DaiSy repository so future contributors and LLM agents can recover context quickly.
It is intentionally operational, not aspirational.

## Current Repo Snapshot

DaiSy is an alpha-stage exact similarity-search library with:

- C++ core implementations
- Optional Python bindings
- Tests, demos, and benchmarks as separate build layers
- Multiple algorithm families spanning in-memory, disk-based, distributed, GPU, and streaming cases

The repo is broad, but integration is still somewhat manual. Many additions require touching several disconnected files.

## High-Level Layout

- `lib/algos/`: algorithm implementations and the main extension point
- `lib/distance_computers/`: distance logic shared by algorithms
- `lib/isax/`: SAX/iSAX support used by several index-based algorithms
- `lib/ds_tree/`: tree structures used by `Hercules`
- `commons/`: shared loaders, params, test and benchmark helpers
- `pybinds/setup.cpp`: Python binding entry point
- `tests/`: C++ tests
- `benchmark/`: benchmark executables
- `demos/`: C++ and Python usage examples
- `docs/`: contributor and architecture docs
- `scripts/groundtruth/`: utilities to generate or validate ground-truth data

## Public And Internal Integration Points

When adding or changing an algorithm, the important wiring points are:

- Base interface: `lib/algos/SimilaritySearchAlgorithm.hpp`
- Source registration: `lib/algos/CMakeLists.txt`
- Public umbrella include: `lib/daisy.hpp`
- Python exposure: `pybinds/setup.cpp`
- Tests: `tests/CMakeLists.txt` plus new `tests/test_*.cpp`
- Benchmarks: `benchmark/CMakeLists.txt` plus new `benchmark/bm_*.cpp`
- Demos: `demos/CMakeLists.txt` plus new `demos/demo_*`

The project does not currently centralize algorithm metadata in one place.
Assume cross-file manual registration is required.

## Algorithm Surface Today

From the current source tree and public includes, the main algorithm set includes:

- `Bruteforce`
- `LbBruteforce`
- `Coconut`
- `Messi`
- `ParIS`
- `Sing`
- `Odyssey`
- `Sofa`
- `Hercules`
- `DumpyOS`
- `Fresh`

These algorithms do not all share the same dependency profile or feature set.

## Important Behavioral Conventions

- Top-k search is the baseline capability.
- Range search is modeled through `SearchConfig` and is not universally implemented.
- Streaming insert is implemented by `BruteForceSearch`, `LbBruteforce`, `Messi`, and `Coconut`;
  the base implementation still throws for algorithms that do not support it.
- Bruteforce streaming grows the owned in-memory database incrementally. LbBruteforce also
  computes SAX summaries incrementally, using the breakpoint set fixed by the initial build.
- MESSI uses those fixed breakpoints to route stable SAX/position records into its live iSAX
  tree, including new-root creation and leaf splitting. Callers must serialize inserts and
  queries; simultaneous update/search is not supported.
- `setNormalized(bool)` is a declaration about the input data, not a preprocessing step.
- Data can come from in-memory arrays or file-backed sources via `DataSource`.

If you extend behavior, keep these semantics stable unless you are intentionally redesigning the interface.

## Build System Reality

The root `CMakeLists.txt` drives feature toggles and dependency detection.

Important options:

- `BUILD_PYTHON`
- `BUILD_BENCHMARK`
- `BUILD_TESTS`
- `BUILD_DEMO`
- `BUILD_ODYSSEY`
- `BUILD_SING`
- `BUILD_SOFA`
- `BUILD_COCONUT`

Current optional dependency model:

- MPI gates `Odyssey`
- CUDA gates `Sing`
- FFTW3 gates `Sofa`
- Coconut can be fully toggled off
- External FAISS comparisons are configured only under `benchmark/` with
  `BUILD_BENCHMARK`, when its submodule and CMake >=3.24 are available. It is not
  a core algorithm, library dependency, Python binding, or library demo.

Backward-compatibility option aliases exist for some old names such as MPI and CUDA toggles.

## Known Rough Edges

These are worth checking before assuming the tree is fully uniform:

- `pybinds/setup.cpp` is monolithic and repetitive; new bindings usually require copy-adapt work rather than a clean abstraction.
- Several test entries for DTW are present but commented out in `tests/CMakeLists.txt`.
- Test and benchmark registration is manual and verbose.
- Optional algorithms are not uniformly guarded everywhere at the conceptual level, so verify includes and bindings carefully when adding new conditional code.
- The repo mixes “library architecture” docs with practical contributor docs; not every workflow detail is encoded in one canonical place.

## What To Check Before Adding A New Algorithm

Read these first:

- `docs/how-to-contribute.md`
- `README.md`
- `lib/algos/SimilaritySearchAlgorithm.hpp`
- A similar existing algorithm in `lib/algos/`
- `lib/algos/CMakeLists.txt`
- `pybinds/setup.cpp`
- `tests/CMakeLists.txt`
- `benchmark/CMakeLists.txt`
- `demos/CMakeLists.txt`

Then decide:

- Is the algorithm always buildable, or optional?
- Does it support top-k only, or also range?
- Does it support L2 squared, DTW, or both?
- Does it need new loaders, new shared utilities, or only local code?
- Should it be exposed to Python now, or intentionally remain C++-only?

## Expected Contribution Pattern

For a normal new algorithm contribution, expect to touch at least:

- `lib/algos/<Algo>.hpp`
- `lib/algos/<Algo>.cpp`
- `lib/algos/CMakeLists.txt`
- `lib/daisy.hpp` if it should be part of the public header surface
- `tests/test_<Algo>_*.cpp`
- `tests/CMakeLists.txt`
- `demos/demo_<Algo>_*.cpp`
- Python demo and binding files if supported from Python
- `README.md`

Benchmarks are also expected for performance-oriented algorithms.

## Documentation Expectations

The current repo already has useful support docs:

- `docs/how-to-contribute.md`
- `docs/benchmark-framework-guide.md`
- `docs/test-framework-guide.md`
- `docs/pybinds-guide.md`
- `docs/demos-guide.md`

Use them, but do not assume they fully replace source inspection.
The repository still relies heavily on pattern matching against existing implementations.

## Practical Advice For LLMs

- Do not edit only one layer and assume the job is done.
- Mirror existing algorithm naming exactly across files.
- Prefer the nearest existing algorithm as a template.
- Be explicit about unsupported features.
- Keep optional dependencies optional.
- If you add a build flag, document it immediately.
- If you notice a current inconsistency, record it in docs or fix it in the same change if safe.

## Coconut Query And Storage Behavior

Coconut top-k uses ordered query-key seeding followed by exact SAX-bound
refinement, with independent queries distributed across the requested workers.
Seeded records are excluded from the later scan to keep returned IDs distinct.
Top-k and range load candidate leaves into private byte buffers, handle partial
last leaves, and report missing/truncated reads after the parallel region.
Contiguous inputs are borrowed only during the initial build; completed leaves
own the raw vectors. Top-k and range support squared L2 only.
`tests/test_coconut_topk.py` covers query workers, streaming, ties, seed-directory
endpoints, oversized k, input lifetime, and failed reads; CMake registers it
only with `BUILD_PYTHON` and `BUILD_COCONUT` enabled.
The resident sorted directory replaces upstream Berkeley DB lookup, and the
insert buffer is still unmerged: this is not a complete Coconut-LSM port.

## When This File Should Be Updated

Update `CONTINUITY.md` when:

- A new algorithm is added or removed
- The expected contributor workflow changes
- A major build flag or optional dependency changes
- Python binding structure changes materially
- Tests, demos, or benchmarks gain a new required pattern
- A major known rough edge is fixed or a new one appears


## Coconut materialized disk streaming

Coconut uses a bounded raw memtable (default 1M records), external-sort bulk build,
immutable sorted disk runs and background binary compaction. Resident SAX remains
O(N); raw vectors persist to disk. Include waitForMerges() when timing completed
insertion. Temporary indexes have no reopen/WAL/recovery or concurrent caller
write/search support. Gaussian SAX only; inputs are never normalized internally.

Top-k uses per-run sortable-key seeds, safe SAX pruning and independent exact
heaps, sharing bounded candidate reads across configurable groups of up to 64
queries (default 32). Recent-window top-k filters stable insertion IDs. Python
bindings expose layout/storage/merge/query counters; BUILD_COCONUT remains optional.
Tests cover strict physical neighbors through flushes/merges, batch/scalar parity,
lookup-bound parity, ranges, corruption and API validation. See
`docs/coconut-disk-streaming.md` and `docs/coconut-paper-implementation.md`.

## Seismic/DEEP experiment runner

`scripts/experiments/run.py` runs process-isolated static range/breakpoint and
streaming comparisons on Seismic/DEEP, with a small-subset preflight and FAISS
validation. See its README for dataset paths, resource requirements and nohup.
The SIMD L2 early-abandon check must continue on equality with the bound;
MESSI range traversal also retains nodes whose lower bound equals the radius.
`tests/test_range_boundary.py` covers partial sums at the bound and radius zero.

MESSI exact L2/DTW leaf refinement uses `pqueue_bsf_insert_no_dup_position`
under its existing queue write lock; approximate seeds must not be inserted
again as separate neighbors. Shared approximate/refinement helpers preserve
each seed distance instead of replacing every entry with the kth distance.
`tests/test_messi_topk_unique.py` covers static/streaming, ties, both metrics,
both breakpoint modes and one/multiple workers. The experiment runner rejects
duplicate top-k IDs and writes live completion/failure state to `status.json`.

`run_breakpoints.py` adds a serial k=10/100/1000 sweep on the same datasets
with exact FAISS ground truth, per-call validation and automatic figure export.
Its latest-run pointer is separate from the paused streaming experiment.

The breakpoint runner accepts additional dataset configs and fbin memory maps.
`prepare_msmarco.py` prepares an unchanged first-100M prefix of MS MARCO Web
Search / SimANS (768D), distinct from MS MARCO's 8.8M passage corpus. Its public
source task is inner product, but these experiments recompute exact squared-L2
ground truth in disjoint FAISS blocks to avoid duplicating the 307.2 GB input.
The exporter can merge completed sweeps into up to three dataset panels;
original measurements and run pointers are preserved. Downloads support resume
with byte-range validation, ETags and durable chunk hashes. No normalization is
applied, and per-vector distribution statistics are recorded. The Oct 10 host
process check confirmed the original runner (447551) and Coconut worker
(921397) still suspended with SIGSTOP; no resume or restart was performed.

`run_range.py` queues a serial range-only comparison after a completed preparation/
breakpoint workflow. It reuses exact neighbor distances to calibrate shared
radii at ranks 1/10/100, validates every call and writes per-query hit counts.
FAISS uses its exhaustive `range_search_L2sqr` kernel directly on the contiguous
float32 input; no full database copy is needed. This is verified against
IndexFlatL2 on the 768D smoke dataset. The MS MARCO range queue has a separate
latest pointer and starts only after the prior workflow writes `SUCCESS`.

`plot_range.py` merges the completed original static range workers and MS MARCO
range run into `docs/figures/range/` (PGFPlots, PDF/PNG, CSV and provenance).
Threshold values are shown on the x-axis, with independent y-axis maxima and
proportional bar widths. Original Seismic numerical boundary discrepancies
are retained in the CSV and checked against the reported extra/missing counts.

Coconut top-k formerly ignored `num_threads` and reopened the same leaf file for
every surviving record, causing ~11.7h per 100-query round at 90M Seismic rows.
It now parallelizes independent queries and opens each candidate leaf once per
query; missing/truncated leaf reads throw after the parallel region. Regression
coverage in `test_coconut_topk.py` compares static/streaming results against FAISS
with multiple worker counts, partial leaves, k=1/10/1000 and failed leaf reads.
On unchanged 50K samples, 100 queries fell from ~40s to 0.30s (Seismic) and
~38s to 0.23s (DEEP); both 1M sample trajectories passed exact FAISS validation.
`run_coconut_streaming.py` snapshots its extension/runner and reuses original
FAISS/MESSI streaming measurements and references. New runs have a separate
`build/streaming-runs/latest-coconut.json` pointer; the original paused worker
is preserved and its interrupted timer is excluded.

The user clarified that Coconut is wanted for streaming top-k query time and
insertion time, not as an addition to range. `run_streaming.py` supports a
queued MS MARCO trajectory for FAISS/MESSI/Coconut using the same 90M→100M,
1M-batch, 100-query, k=10 workload. MS MARCO FAISS final storage is preallocated
inside the timed build to avoid a >RAM reallocation peak; the policy is recorded.
Coconut build now borrows `DataSource::rawPointer()` only until leaves are
written, avoiding a second 307.2 GB copy. A regression verifies that modifying
or releasing that input after build does not affect searches. Existing streaming
workers keep their isolated older binary snapshots. `plot_streaming.py` requires
complete trajectories and exports separate query/insertion figures with
independent panel maxima. The queued MS MARCO pointer is
`build/streaming-runs/latest-msmarco.json`.

An upstream source audit is recorded in `docs/coconut-upstream-audit.md`, pinned
to kon0925/coconut commit `0e7ae37777c6588bea0e03117db357161503cb93`. Its selected
2018 executables seed exact 1-NN through an ordered approximate search, then
scan SAX bounds; the full summary scan itself is not a missing upstream feature.
DaiSy lacks that seed and the upstream Berkeley DB B-tree/page access, and its
unmerged buffer is not Coconut-LSM. Isolated top-k seeding diagnostics passed
FAISS checks on all three dataset subsets but reduced disk-vector reads by only
3.78%/0.12%/0% at initial k=10 (Seismic/DEEP/MS MARCO). Sortable-SAX conversion
matched upstream on 320,000 words. These diagnostic binaries do not replace
the production module or the active workers' snapshots.
An additional isolated whole-leaf-read diagnostic passed twelve searches on
the same subsets and observed 3.16–4.89x lower initial k=10 query times than
per-vector stdio. These single-call, concurrent subset timings are not a
full-100M performance prediction; production query/storage code is unchanged.

The subsequent user-authorized upstream port adds ordered query-key top-k
seeding, distinct-seed exclusion during exact refinement, and shared whole-leaf
buffers for top-k/range. A binary search over the resident key directory
implements the ordered lookup; Berkeley DB itself is not imported. Regression
coverage includes directory endpoints, ties, k greater than a leaf/dataset,
empty initial builds followed by inserts, partial leaves and failed reads.
Top-k rejects unsupported DTW explicitly. The insertion buffer still has no
LSM flush/merge policy, so this is not a full Coconut-LSM implementation.

Verified port module SHA256:
`660e4838ba82e7bc458da1acbd2c5b8bc082a29fe62a01f54dcc0507deb65575`.
The library build, five Coconut regression tests, six existing streaming/MESSI/
boundary tests, twelve real-subset searches and a full three-algorithm 768D
streaming smoke trajectory passed. New nohup Seismic/DEEP Coconut run:
`build/streaming-runs/20261010-132645-coconut-upstream-port` (parent 2254972).
The Oct 10 earlier corrected run/queue are marked superseded and preserved;
original Oct 3 suspended processes are not resumed. The replacement MS MARCO
queue is `build/streaming-runs/20261010-132853-msmarco100M-upstream-port`
(parent 2255667). The first replacement queue failed during dependency startup;
the runner now tolerates a missing initial status file while the dependency PID
is alive, and rejects failed/superseded or exited dependencies explicitly.

## Streaming capacity experiment (2026-10-12)

`scripts/experiments/run_capacity_streaming.py` adds the user-requested initial
100M build and disjoint 10M batches toward 500M, with 100 top-10 queries per
stage. DEEP has 500M real vectors available. The original Seismic source has
about 110M rows; its finite continuation starts at the earlier preparation's
`source_rows_scanned`, preserving the initial 100M ordering. The selected MS
MARCO release has 101,070,374 rows in total, so only its missing tail is fetched.
No dataset is repeated or synthesized to reach 500M.
Workers are serial and use independent systemd user-service cgroups with
340 GiB memory budgets and zero swap; resource-limit exits are results and
do not stop other methods. The target is not preallocated, and transient
growth allocations remain part of the workload. Inputs use a file-backed
initial prefix and bounded batch loads. Exact missing references use bounded
block merging outside timing. Kernel peak RSS and sampled cgroup usage are
recorded; this kernel lacks `memory.peak`, so cgroup peaks are sampled estimates.
Snapshots preserve the production module and algorithm code without altering
the paper or earlier experiments. See the experiment README for limitations.
Validated nine small trajectories (fvecs, fbin, finite continuation) and bounded
reference fallback, plus a worker-local cgroup OOM. Launched with nohup:
`build/capacity-streaming-runs/20261012-020622-100m-to-500m`, controller PID
2770898; its independent pointer is `build/capacity-streaming-runs/latest.json`.
User lingering was enabled for ssideri so capped systemd user-service workers
continue after terminal logout. The launch began by preparing the file-backed
DEEP 100M prefix; Seismic and the true MS MARCO tail are queued after DEEP.

## MESSI exactness tests on PR #81 (2026-10-12)

PR #81 head `2606fea` restores adaptive MESSI's brute-force comparison and
removes the clustered regression's 99% recall allowance. Every expected
neighbor, unique IDs, and returned-vector distances are checked; integer
tie fixtures allow distinct equal-distance alternatives. Parameterized L2
checks cover Gaussian/adaptive modes, one/four workers, k=1/10/100/1000,
repeated static/streaming searches, and a second live adaptive index. All 13
focused CTest cases passed in an isolated Release build. Tests and their guide
were pushed to `fix/messi-topk-duplicate-results`; no algorithm changes were
made by this follow-up. New remote CI checks were in progress at verification.
A workflow-selection edit could not be pushed with the existing OAuth scopes
and was excluded; the existing Simple CI already runs the entire CTest suite.
The review worktree is `build/pr81-review`, local branch
`test/messi-exactness-pr81`; the primary checkout and production module remain
unchanged. Its small concurrent build/test load was recorded in the active
capacity run's `external-load.jsonl` for measurement provenance.

## Corrected selective range rerun (2026-10-12)

The active checkout remains `improvement/extending-streaming`; preserve its dirty work.
An isolated snapshot at `build/experiment-recheck` uses PR #81 commit
`a1bd906` plus the current Coconut port and experiment-only timing probes.
The ignored runner `scripts/experiments/recheck.py` launched a range-only
nohup rerun (PID 3340910), recorded in `build/experiment-rechecks/latest-range.json`.
It waits for the existing capacity run's FINISHED marker before measuring.
No new streaming rerun is scheduled. Scope: first 100M vectors, 100 queries,
FAISS and Gaussian/adaptive MESSI, shared radii calibrated at nearest-distance
percentiles 10/25/40. Per-query hit counts and empty-query fractions are required.
The 4K-vector preflight passed 90 range checks with zero set mismatches;
full-scale results are pending. Keep existing figures/manuscript unchanged
until validated results are reviewed. Details: `build/experiment-rechecks/audit.md`.

### Streaming stopped to prioritize range (2026-10-12)

The user explicitly stopped streaming. Capacity controller 2770898 and
MS MARCO Coconut service `daisy-capacity-20261012-020622-100m-to-500m-10`
were terminated; the service is inactive. Completed results are retained;
MS MARCO Coconut stopped during its initial build and has no validated stage.
The old queued range controller 3340910 was superseded. The new range-only
nohup controller is 3358609, without a capacity dependency.
Output: `/home/ssideri/DaiSy/build/experiment-rechecks/20261012-200822-selective-range-100m`.
`build/experiment-rechecks/latest-range.json` points to this active run.

### Selective range results completed (2026-10-13)

The unqueued range rerun completed successfully: all 12 jobs and 162
range validations passed the documented float32 boundary tolerance.
New plots are in `docs/figures/range-selective/` (PNG/PDF/SVG, PGFPlots, CSV,
provenance); older range figures and the manuscript remain unchanged.
Means include all three warmed repetitions, with sample SD error bars.
Seismic uses a clearly marked log y-axis; other panels are linear with
independent maxima. MESSI wins all Seismic and MS MARCO radii; DEEP only
at p10, while FAISS wins p25/p40. At DEEP p40 each MESSI mode returns
one additional hit within the numerical boundary tolerance (313043 vs
313042); all other result sets match, with no missing hits.

## RandWalk-only top-k and range run (2026-10-13)

The user authorized only RandWalk100M, raw and paired per-series z-normalized,
for MESSI Gaussian vs Adaptive top-k (k10/100/1000) and FAISS/MESSI range.
No other dataset is rerun. No streaming or Coconut job is scheduled.
Nohup controller PID 4022915 creates paired100M-by256 files, then runs
serial corrected-module workers (100queries,48threads,3warmed repetitions).
Query stream is independent; seed20261013, PCG64 SeedSequence children0/1.
Pointers: `build/experiment-rechecks/latest-randwalk.json`;
output `/home/ssideri/DaiSy/build/experiment-rechecks/20261013-114521-randwalk100m-topk-range`.
Generator and runner are ignored scripts under `scripts/experiments/`.
Preparation uses4GiB, reference8GiB, measured workers340GiB; swap disabled.
Small preflight passed24top-k and60range checks, exhaustive float64 reference
checks and deterministic/raw-z pairing checks. Details in run README and
`build/experiment-rechecks/randwalk-preflight.json`.

### RandWalk run completed (2026-10-14 verification)

The RandWalk-only run has SUCCESS and all eight jobs exited zero.
All 156 validation calls passed (48 top-k,108 range), with zero range
missing/extra boundary hits. In raw RandWalk, Adaptive top-k is20–36x
faster than Gaussian across k10/100/1000; Adaptive range is24–26x faster
than native FAISS. In z-normalized RandWalk, Gaussian and Adaptive are
close; Adaptive range is22–57x faster than native FAISS. Means retain
all three warmed calls. Details and SDs are in results-summary.json and
measurements.csv/json inside the run; no other dataset was rerun.

## Combined four-dataset LaTeX exports (2026-10-14)

`docs/figures/all-datasets-20261014/` contains compiled native PGFPlots
figures, PDF/PNG previews, CSV/JSON provenance and a zip. Selected panels:
DEEP,Seismic,RandWalkRaw,RandWalkZ; MS MARCO excluded per current selection.
Layout2x2, independent maxima, SD error bars, non-overlapping proportional
bars; Seismic range uses an explicit logarithmic axis. DEEP/Seismic top-k
values preserve the local Overleaf manuscript tables and are marked dagger
as legacy measurements requiring corrected-build revalidation. All range
values and RandWalk top-k come from corrected runs. Both snippets compiled
without overfull boxes or undefined references. No experiments were rerun
and no manuscript/older figures were changed. Generator:
`scripts/experiments/export_all_figures.py` (ignored experiment script).

## Corrected DEEP/Seismic top-k-only rerun (2026-10-14)

The user authorized rerunning the legacy DEEP/Seismic Gaussian-vs-Adaptive
panels with the corrected build. Only top-k at k10/100/1000 is scheduled:
100M identical stored vectors,100queries,48threads,3warmed repeats.
Nohup controller 961947; output `/home/ssideri/DaiSy/build/experiment-rechecks/20261014-100343-deep-seismic-topk-corrected`.
Pointer: `build/experiment-rechecks/latest-deep-seismic-topk.json`.
Six serial capped jobs, FAISS exhaustive references and two MESSI modes;
no range,streaming,Coconut,RandWalk or MS MARCO rerun. The corrected core
matches the validated RandWalk module. Preflight passed24top-k validations
and exhaustive float64 reference checks on4096vectors per dataset.
After SUCCESS the frozen exporter automatically produces updated four-panel
LaTeX/PDF/PNG/CSV/zip files without the legacy dagger in
`/home/ssideri/DaiSy/docs/figures/all-datasets-corrected-20261014-100343`; completed RandWalk and range results are reused.
Figure export status/failures are separate from measurement correctness.

### Figure layout changed to one row (2026-10-14)

The user requested all four datasets in one row in both LaTeX figures.
The reusable exporter and the active top-k run frozen exporter were updated
atomically; previous exporter versions and hashes are preserved in run
postprocess-history and metadata. Benchmark code,config and measurements
are unchanged. Each subfigure uses0.24linewidth and its own y maximum,
with a shared y label on the first panel and numeric ticks on every panel.
New temporary one-row previews in `docs/figures/all-datasets-one-row-20261014/`
compiled without layout/reference warnings; DEEP/Seismic top-k still show
legacy daggers until the active corrected rerun completes. The final automatic
export will use this one-row layout. Short preview compile load was recorded
in the active run external-load.jsonl.

## DEEP smaller range and visible hit counts (2026-10-14)

The user requested total hits on all range plots and explicit DEEP squared-L2
radii0.04/0.08/0.10. The exporter now shows each radius above the exact
independent-reference total hits over100queries; per-method hits, empty-query
counts and boundary differences remain in CSV/provenance. One-row layout
and per-panel maxima remain. The active top-k exporter snapshot was updated
with previous versions/hashes retained. Cached fonts and native previews
were verified; `docs/figures/all-datasets-hits-20261014/` currently still uses
the previous measured DEEP radii, pending the new run.
New DEEP-only nohup range controller 1042915: `/home/ssideri/DaiSy/build/experiment-rechecks/20261014-103135-deep-range-004-008-010`.
Pointer `build/experiment-rechecks/latest-deep-range.json`; waits for the
active corrected DEEP/Seismic top-k FINISHED marker before four serial jobs.
Same original DEEP first100M data and100queries,48threads,3warmed repeats;
no other dataset/top-k/streaming rerun. Correctness preflight used4096genuine
rows including known nearest-neighbor IDs for nonempty coverage, original
100queries;30range validations and three exhaustive float64 checks passed
with zero set differences. It is not performance evidence.
After SUCCESS, export merges this DEEP range with previous Seismic/RandWalk
range and corrected top-k, with explicit-radius policy and actual empty-query
counts in the caption. Final figure output: `/home/ssideri/DaiSy/docs/figures/all-datasets-deep-small-range-20261014-103135`.

### Corrected DEEP/Seismic top-k completed (2026-10-14)

All six top-k jobs completed,48validations passed, and automatic one-row
LaTeX/PDF/PNG exports succeeded without legacy daggers. Figures are in
`/home/ssideri/DaiSy/docs/figures/all-datasets-corrected-20261014-100343`.
The smaller DEEP range run is now active (FAISS timing phase at verification);
MESSI Gaussian/Adaptive are next. At radii0.04/0.08/0.10, independently
validated FAISS totals so far are12/739/2863hits and96/90/89empty queries.
Final smaller-radius range plots remain pending all three algorithm jobs.

### Smaller DEEP range and final figures completed (2026-10-14)

All four DEEP range jobs succeeded and54range validations passed with
zero missing/extra IDs, including numerical boundary differences. For
radii0.04/0.08/0.10 the exact total hits over100queries are12/739/2863;
empty-query counts96/90/89. Mean seconds FAISS36.68/32.61/35.95,
Gaussian5.10/15.61/21.37, Adaptive5.13/14.84/19.83.
Final automatic figure export succeeded in `/home/ssideri/DaiSy/docs/figures/all-datasets-deep-small-range-20261014-103135`: both figures
use one row of four datasets; all top-k panels are corrected, with no
legacy dagger; range ticks display radii and exact independent-reference
hits. CSV,provenance,compiled PDFs,cropped PNGs and latex-plots.zip exist.
Only DEEP range was rerun; prior Seismic/RandWalk range was reused.

## New DEEP streaming top-1 scale trajectory (2026-10-14)

The user requested a new streaming experiment: initial build100M, then50M
native insert batches through500M (nine stages), top-1 queries only. The
local original DEEP fvecs source has876,305,598complete96-dimensional
records plus an incomplete8-byte trailing record; only the first500M are
used and every header/value is validated. No download or synthetic extension
is needed. Official DEEP1B contains1B96-dimensional vectors.

New ignored runner `scripts/experiments/run_streaming_scale.py`, config
`scripts/experiments/config-streaming-deep-100m-500m-top1-50m.json`.
Nohup controller1386111 started successfully; live reference worker1386118
was verified with MemoryMax8GiB and MemorySwapMax0. Run directory
`build/streaming-scale-runs/20261014-131157-deep-100m-500m-top1-50m`;
pointer `build/streaming-scale-runs/latest-deep-top1.json`. Controller log
is the sibling `.nohup.log`; detailed progress is in each algorithm log,
JSONL, progress.json, memory.jsonl and outcome.json. status.json records
the current worker. At verification reference had processed9.5M rows.

Independent reference runs first, exhaustively scanning disjoint500K blocks
using direct squared-L2 FAISS with33candidate guard and physical float64
reranking, cumulatively merging all prefixes. It compares every initial100M
vector against the previously prepared contiguous initial file, records its
SHA256, and rejects any mismatch. Reference is outside measured timing.
Then serial FAISS, MESSI Gaussian, MESSI Adaptive, Coconut trajectories run
with100original queries, k1,48threads,340GiB cap and zero swap. FAISS native
BLAS defaults are retained, with native add/insertBatch APIs and no rebuild
or artificial reservation. Initial build, each insertion, loading outside
insert timing, first query after update and three warmed queries per stage
are recorded separately, with RAM and page-fault evidence. Every query
repetition is independently validated at its exact dataset prefix. Physical
ID/distance pairs and all neighbors are checked; only two float32 ULP
near-ties are allowed and counted, without percentage recall tolerance.
An algorithm OOM/failure is recorded and other trajectories continue;
reference failure prevents launching measured jobs.

Runner, helpers, corrected compiled module and source hashes are snapshotted
in the run directory. Module SHA matches the independently validated
RandWalk/static/range module from build/experiment-recheck; root production
module and all completed experiments/figures/paper remain untouched.
MESSI's Gaussian/adaptive modes use the same original vectors; adaptive
breakpoints are learned once at initial build. Coconut uses disk-backed
initial leaves plus an unmerged in-memory insertion buffer; this is not
an LSM compaction experiment.500M DEEP raw vectors require192GB; failure
at500M must not be presumed, and any native allocation peak is reported
separately from final vector size.

Preflight `build/streaming-scale-runs/preflight-20261014-top1-50m` completed
all five capped workers, eight inserts/nine prefixes4096through20480rows,
eight genuine queries, k1, two threads and first+one warmed query per stage.
72top-1 checks and nine independently exhaustive float64 prefix checks
passed, with zero neighbor gaps; maximum physical distance error6.69e-8.
It is correctness evidence, not performance evidence. Full run is still
active; do not report finished until SUCCESS/FINISHED and outcomes exist.

### Conditional DEEP1B extension queued (2026-10-14)

The user explicitly authorized proceeding to1B after confirming the500M
experiment is correct, and asked for duration. The active500M reference
completed in814.9seconds (~13.6minutes); FAISS was validated through250M
and executing300M at this turn. No correctness failures recorded so far.
Provisional estimates communicated:12--18hours total for the500M experiment,
then24--48hours for the1B trajectory, plus uncertain download time. These
are estimates based on current FAISS timings and previous native MESSI/
Coconut500M trajectories, not promises; Coconut insertion-buffer scans and
natural allocation/OOM limits can change the duration substantially.

New ignored `scripts/experiments/queue_streaming_billion.py` and
`config-streaming-billion-queued.json`. Nohup queue controller1398268 is
alive and detached; queue directory `build/streaming-scale-runs/deep-billion-queue-20261014`,
pointer `build/streaming-scale-runs/latest-deep-billion-queue.json`.
It waits for the active500M run's FINISHED, SUCCESS and complete status
with no failures. Failure/stopped/completed_with_failures holds the extension
without downloading or launching benchmarks. Download occurs only after
the dependency passes, to avoid interference with its timed jobs.

The official source
`https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin`
returned exact206byte ranges, header(1B,96),384000000008bytes and ETag
`"ed22edcea08640670973dde650b852b9-5723"`. Physical float32 vectors matched
the local original at rows0,499999999 and876305597. Download verifies
additional overlap/join rows and exact Content-Range/ETag; durable chunk
SHA256 journal supports resumption and rejects identity/data mismatches.
It preserves local originals and fetches850Mthrough1B as a separate raw
float32 tail (57.6GB). The850Mjoin aligns with50Mbatches, avoids a large
cross-file concatenation at900M, and includes26.3Malready-local overlap;
it is not a synthetic/repeated extension. Sources for the1B trajectory are
local first850M plus the official last150M, with IDs/order unchanged.

The future output is `build/streaming-scale-runs/deep-100m-1b-top1-50m-after-500m`:
same100Minitial build,50Mincrements,19stages,100top-1queries, first+three
warmed query repetitions,48threads, four algorithms,340GiB cap/swap0,
independent prefix truth and validation. Existing indexes are not persisted;
the new trajectory replays from100M to preserve comparable native growth
and Coconut disk-leaf/buffer proportions. This was explicitly explained to
the user. The corrected module/sources/provenance and all queue/runner/helper
code are frozen now, so later working-tree changes cannot silently change
the queued experiment. run_streaming_scale.py now accepts that frozen
module provenance instead of deriving Git state from a snapshot directory;
the active500M runner snapshot remains unchanged.

1B DEEP raw vectors alone need384GB, above the340GiB(365.1GB) experimental
cap; reaching1B in memory cannot be promised. Each trajectory will record
its last validated prefix and natural memory-limit failure, then allow other
algorithms to run. This is a memory-cap stress evaluation, not a claim that
all methods will finish1B or that indexed methods necessarily win.

Extension preflight `build/streaming-scale-runs/preflight-billion-queue-20261014`
downloaded3MiB genuine DEEP tail, joined it with a local prefix, and completed
all five capped workers across nine prefixes.72top-1 and nine exhaustive
float64 prefix checks passed, with zero neighbor gaps. A separate failed-
dependency fixture confirmed the queue holds without touching the network.
The brief two-thread preflight overlap is recorded in the active500M run's
external-load.jsonl. Full1B queue is currently waiting; tail download and
full1B measurements have not started.

### 500M trajectory finished; 1B queue held (checked2026-10-15)

The user asked whether it ran. The500M controller finished in11.676hours
(about11h40), status completed_with_failures and FINISHED exists; no SUCCESS.
FAISS, MESSI Adaptive and Coconut all independently validated through500M
and exited successfully. MESSI Gaussian validated through400M, then its
native400M->450M insertion was killed by the340GiB cgroup cap; systemd
Result=oom-kill, last event insert450M started. This is a memory-limit
outcome, not a detected wrong-neighbor result or proof that450M's final
steady-state index cannot fit. All136completed top-1 batch checks passed,
with zero float32 neighbor gaps/near-ties. Ground truth completed through500M.
Mean warmed wall seconds for100top-1queries at500M: FAISS267.967,
MESSI Adaptive399.286, Coconut474.364; Gaussian has no450/500M timings.
No new plots generated at this status request.

The conditional1B queue is state held, reason500M run did not pass all
algorithm checks; its event extension_held records dependency_state
completed_with_failures. The queue exited, did not download the full tail,
and did not launch the1B trajectory. Do not claim1B started or is currently
waiting in an active process. User was informed of the500M outcomes and
the automatically held1B extension; do not silently describe an OOM as
a correctness failure.

### Surviving methods resumed toward1B (2026-10-15)

The user explicitly instructed continuing the remaining approaches as far
as possible, naming MESSI Adaptive, FAISS and Coconut, accepting timeouts.
New nohup queue2031119:
`build/streaming-scale-runs/deep-billion-survivors-20261015`.
Pointer `build/streaming-scale-runs/latest-deep-billion-queue.json` now points
here and retains previous_queue. Previous held queue/500M outcomes are
preserved. New output after preparation/reference:
`build/streaming-scale-runs/deep-100m-1b-top1-50m-survivors-20261015`.
Config `scripts/experiments/config-streaming-billion-survivors.json`.
At launch state preparing_tail: selected three independently successful
500M outcomes were accepted, and the57.6GB verified official DEEP tail
download850M->1B is starting. Do not claim measured1B workers have started
until the trajectory output/status exists. Module and sources remain the
same immutable corrected500M snapshot; no production algorithm changes.

Queue eligibility now has an explicit continue_survivors policy. It requires
FINISHED and completed/partial-failure status; same original source/query,
dimension, initial file, initial size, batch, k, query count, threads,
repeats and memory cap; reference successful; every SELECTED algorithm
successful and independently validated through the original target500M.
Gaussian's already-recorded memory limit does not prevent these three
from running. Failures/incomplete prefixes of a selected method still
reject the dependency. The old strict policy remains the default.

Protocol remains initial100M,50Minserts through1B,100original top-1queries,
first query plus three warmed repeats per prefix,48threads,340GiB cap,
zero swap. Indexes were not persisted, so native100M->500M growth is replayed
before extending to preserve the same allocation/Coconut-buffer history.
Gaussian is not rerun. The user authorized stopping individual methods at
memory/time limits and letting others proceed. New watchdog limits:
2hours without an operation/progress completion,24hours per algorithm,
4hours total for independent reference. RuntimeMaxSec enforces per-worker
deadline; watchdog stops only its private systemd unit. outcome state
timeout is distinct from memory_limit, failed and complete, and retains the
last validated prefix. Existing scheduler continues subsequent methods
after a timeout or OOM; independent-reference failure still prevents
unverifiable measured queries. Thresholds were communicated to the user.

Preflight `build/streaming-scale-runs/preflight-billion-survivors-20261015`
passed54top-1 checks, nine exhaustive float64 prefix checks, zero neighbor
gaps, with genuine3MiB remote tail joined to local data and the frozen module.
Eligibility tests accepted the valid partial500M run and rejected a failed
selected method, incomplete selected prefix and strict-mode partial run.
Real systemd scheduler test in
`build/streaming-scale-runs/timeout-preflight-20261015` confirmed operation
watchdog timeout, RuntimeMaxSec timeout, preserved last validated prefix,
and successful execution of the next worker. No other datasets, completed
figures or paper files were changed.

### Current streaming plot preview exported (2026-10-15)

User requested a plot of available numbers. Reusable ignored exporter
`scripts/experiments/plot_streaming_scale.py` reads the finished100M->500M
run, verifies136query-batch validations and34completed algorithm/prefix
rows, and exports only validated points. Latest figure directory:
`docs/figures/streaming-scale-current-20261015`.
`streaming_preview.png/pdf/svg` has two panels: mean100top-1query minutes
over all three post-first-call repetitions, and seconds per50Minsertion.
Each panel has its own y maximum. Gaussian ends at400M; footer explicitly
records its memory-limit termination during400M->450M insertion at340GiB.
No missing/OOM point is replaced with zero and no failed insertion duration
is fabricated. Initial build is separate in CSV, not an insertion at100M.

FAISS300M warm repeats152.298/972.571/99.979seconds are all retained in
the408.282second mean; the figure explicitly flags this variability rather
than dropping the slow repeat. First calls are excluded from the displayed
warm mean but retained in `query_repetitions.csv` and `streaming.csv`.
`query_repeat_ranges.png/pdf` separately shows min/max observed ranges.
`provenance.json` includes config, core hash, all source-log/outcome hashes,
outcomes and timing/missing-data policy. Main plot was visually inspected;
an overlapping inline annotation was removed before the final export.
Prior initial draft is in `docs/figures/streaming-scale-500m-20261015`.
No experiments were rerun for the plot. New1B queue was still preparing
the downloaded tail at this request; its incomplete rerun was not mixed
into this figure.

### Survivor1B stress trajectory completed (checked2026-10-17)

User asked whether ready. Both queue and trajectory have FINISHED and
completed_with_failures, all requested methods attempted until their
natural memory limit. No timeout occurred. The57.6GB tail download and
independent reference through1B completed. Native algorithm outcomes under
the unchanged340GiB zero-swap cap: FAISS validated800M then OOM during
800M->850M insertion; MESSI Adaptive validated700M then OOM during
700M->750M insertion; Coconut validated500M then OOM during500M->550M
insertion. All three systemd results oom-kill. None reached1B. These are
allocation/cgroup-limit outcomes, not a claim about each final steady-state
index size; Coconut still uses the unmerged RAM insertion buffer.

All148completed top-1 batch checks passed: FAISS60, Adaptive52, Coconut36;
zero near-float32 ties and zero neighbor gaps. Total duration14.59hours
for reference+trajectories,15.47hours including download/preparation.
Structured summary saved in
`build/streaming-scale-runs/deep-100m-1b-top1-50m-survivors-20261015/summary.json`.
Previous500M figure remains unchanged; its numbers are a separate run and
must not silently be mixed with the new trajectory. No new plot was
requested/generated during this status turn.

## Critical correction: disk-based Coconut streaming (2026-10-17)

The user correctly objected that the intended Coconut experiment was DISK-BASED.
The earlier adapter had static disk leaves but all inserted raw vectors in an
unmerged RAM buffer. This was the wrong implementation/setup for the requested
experiment, not evidence that disk-based Coconut stops at500M. Acknowledged to
the user, corrected the code, and launched ONLY a replacement Coconut run.
FAISS/MESSI timings/outcomes stay valid; historical Coconut streaming curves/
memory limits must not be described as disk-based Coconut results.

Current authoritative pointer: `build/streaming-scale-runs/latest-coconut-disk-top1.json`.
New nohup controller2901898, run
`build/streaming-scale-runs/20261017-122803-coconut-disk-100m-1b-top1-50m`.
At verification its capped reference worker2901901 was revalidating/reusing
the prior independent1B prefix truth, with20Mof100M initial-prefix records
checked. Do not claim the native Coconut measurement has begun until its
progress file exists. The controller then runs Coconut only100M->1B,50M
inserts,100original top-1queries,48threads, first+three warmed calls per
prefix,340GiB cap/swap0,2hoperation and24hworker deadlines unchanged.
Config `scripts/experiments/config-coconut-disk-100m-1b.json`.
The latest-deep-billion-queue pointer still describes the completed comparison
but now explicitly links the Coconut replacement pointer/output. Check the
replacement for future status requests, not just the old completed RAM run.

Implemented a materialized Coconut-LSM adaptation in the existing Coconut
class (public name unchanged, not a claim of upstream/Berkeley DB parity):
bounded memtable, sortable-SAX disk runs, synchronous binary compaction,
streaming bounded initial sort/build, raw vectors in sorted disk files,
reclaimable mmap key directories, stable IDs, exact top-k/inclusive range,
per-query leaf buffers, fsync before old runs are retired. Default buffer
1Mrecords (~384MB raw at96D); full batches flush/merge before returning.
Partial memtables remain bounded and queryable. Layout changes after build
are rejected; input is never normalized and Gaussian SAX is explicit.
Native batch time includes sorting/flush/fsync/merge. This is temporary
POSIX storage without reopen/WAL/crash recovery/concurrent reads+writes/
temporal window support. Batch I/O failures may leave an accepted prefix.

New public/Python setters/getters: set/getBufferCapacity, set/getIndexDirectory,
getBufferedRecords,getDiskRecords,getDiskBytes,getRunCount,getFlushCount,
getMergeCount,getPeakBufferedRecords,flush; getDim/getNDatabase exposed.
Raw file builds supported. Python inputs forced contiguous and ranks/dimensions
validated, including strided queries/inserts. Optional Coconut bindings and
umbrella include are guarded by COCONUT_ENABLED. BUILD_COCONUT=OFF was compiled
as an isolated Python module and verified to import without Coconut while
Messi remains available. No new dependency or CMake source target introduced.

Root code modifications preserved all pre-existing dirty changes, and a backup
of the previous Coconut.cpp/hpp is `/tmp/daisy-coconut-before-disk-20261017`.
Isolated corrected build `build/coconut-disk-streaming` copies the validated
experiment-recheck base plus the new Coconut/binding/tests; its provenance.json
records base commit, actual source hashes, diff and core SHA. Root production
module is unchanged SHA660e4838ba82e7bc458da1acbd2c5b8bc082a29fe62a01f54dcc0507deb65575.
Do not rebuild/replace it during the active measurement. All run code/module/
source snapshots are immutable. Independent truth reuse verifies identical
dataset/query/metric/protocol paths, unchanged source mtimes, full initial100M
source-vs-initial equality and hash, all reused physical candidate distances,
and records copied reference-file hashes. No FAISS/MESSI performance rerun.

Runner `scripts/experiments/run_streaming_scale.py` now supports optional
coconut_disk_streaming/coconut_buffer_capacity and reference reuse; it checks
disk+buffer counts after every build/insertion, peak buffer<=capacity, and
actual run-file sizes vs reported disk bytes, emitting storage evidence.
Preflight `preflight-coconut-disk-20261017`:20,480genuine DEEP rows, nine
prefixes,512-record buffer,18exact query checks,40flushes/38merges and all
20,480records on disk at the final stage. All passed, zero neighbor gaps.
Stronger `preflight-coconut-disk-outofcore-20261017`:1Mgenuine96D DEEP vectors
(384,000,000raw bytes) under a256MiB(268,435,456byte)worker cgroup limit,
no swap;100Kinitial/100Kinserts,8queries,10prefixes,8192-record buffer.
All20top-1 checks and10storage audits passed; peak process RSS164,646,912bytes.
This establishes operation beyond RAM capacity on a correctness subset,
not a billion-scale performance claim. Both verification.json files exist.

New tests `tests/test_coconut_disk.py` cover exact float64 physical ranks
through flush/merge, all k through1000, stable IDs, odd key/vector sizes,
raw file build/cleanup, inclusive range, strided arrays, non-finite/shape
rejection, and corrupt metadata. Existing five Python Coconut tests pass.
New C++ integer-data test asserts exact IDs/distances after compactions.
Strict tests found the shared SIMD distance kernel reads in groups of8;
Coconut now handles a scalar tail locally for other dimensions without
changing the shared kernel or other algorithms.
Final Coconut CTest:14passed,9explicitly skipped,0failed (23discovered).
The legacy Astronomy input has two non-finite rows10321/40380, no invalid
queries. The nine Astronomy range configurations now explicitly detect/skip
that invalid fixture instead of silently filtering/reindexing data; finite
rejection is separately tested. All finite RandomWalk range cases passed.
Full log `build/coconut-disk-streaming/ctest-coconut-final.log`.
Enabled/disabled builds and streaming C++ demo target compiled. README,
architecture docs, demos, contributor notes and historical upstream audit
updated; details in `docs/coconut-disk-streaming.md`.

Original logs/figures remain for forensic provenance. The two recent RAM
streaming runs and two500Mfigure directories now carry
coconut-storage-correction.json and COCONUT_DISK_RESULT_INVALID.md, linking
the replacement. Their Coconut values must be excluded/relabelled in future
paper plots. Do not silently merge old/new Coconut storage policies. No
paper/Overleaf edits or Git push/PR were made in this correction turn.

### Corrected disk Coconut finished; 1B figure exported (2026-10-19)

The corrected disk-only run
`build/streaming-scale-runs/20261017-122803-coconut-disk-100m-1b-top1-50m`
has SUCCESS/FINISHED, complete status and a successful Coconut outcome through1B.
All76query-batch validations passed, zero near-float32 ties/neighbor gaps,
no timeout/OOM/swap. All19prefix storage audits show every record on disk,
zero buffered records at measured stages and peak memtable<=1M.
At1B the three post-first-call query repetitions took3815.064,3383.287,
3501.526seconds: mean59.444minutes per100top-1queries.
Final950M->1B insertion took4229.545seconds, including flush/compaction.

User requested a plot with MESSI, FAISS and Coconut until1B. New artifacts:
`docs/figures/streaming-1b-disk-20261019/streaming_preview.png/pdf/svg`,
separate query/insertion panels, self-contained `fig_streaming_1b.tex`, CSVs
including all repetitions and separate initial build times, README/provenance.
Ignored reusable exporter `scripts/experiments/plot_streaming_billion.py`
is also copied into the artifact directory. It verifies same data/config/query
hash,19byte-identical independent references, relevant unchanged MESSI/SAX
source hashes, storage evidence and188query validations over47complete prefixes.
One whole trajectory per method: FAISS and MESSI Adaptive from the finished
survivor1B run; Coconut exclusively from the corrected disk run. No legacy
RAM-insert Coconut values or mixed rerun prefixes. Gaussian is not in this
three-method figure. Curves end at700M for MESSI Adaptive and800M for FAISS;
footer records their340GiB memory-limit failures during the next50Minsert.
Both panels have linear axes and independent y maxima; query mean uses all
three repeats, with observed min/max shading. PNG visually inspected; PDF
exported. PGFPlots source generated; no local pdflatex is installed.
No experiments rerun or paper/Overleaf edits made to produce the figure.

### User requires original Coconut code (2026-10-19)

User objected to using a custom Coconut implementation instead of the authors'
original code. Acknowledged this was our implementation choice/error; the1B
numbers measure DaiSy's disk LSM adaptation only. All current1B figure exports
and exporter legend now explicitly say `DaiSy Coconut-LSM adaptation`, with
upstream_coconut_benchmark=false in provenance. Numeric logs remain unchanged.

Rechecked pristine `/tmp/daisy-coconut-upstream`, commit0e7ae37777c6588bea0e03117db357161503cb93.
Live GitHub API confirms onlymaster at that commit, one listed2018fork,
and no separately named Coconut-LSM repo in the author's public repo list.
The supplied code documents the2018 static build/query Berkeley DB programs;
their indexing deletes the old DB and their queries open it read-only. No
upstream LSM/live50Mbatch executable was found in this repo. This does NOT
establish that authors' streaming code is unavailable elsewhere. Active hardcoded
256D distance calls also prevent claiming untouched `coconut.c` as a96D baseline.
See updateddocs/coconut-upstream-audit.md. Future work should obtain/identify
the authors' actual streaming source or separately scope the original static
baseline, not silently write another streaming implementation. No new benchmark
was launched and no algorithm source was changed during this clarification.

### Paper-based Coconut-LSM-Full implemented and launched (2026-10-19)

User explicitly authorized implementing what the extended Coconut paper describes.
Paper URL https://nivdayan.github.io/Coconut_VLDBJ_CR.pdf, saved in the isolated
build, SHA52da8f081c167ee8c75c303d7764194f226c9caac4e366b5d916e30c16c870be.
`docs/coconut-paper-implementation.md` maps Algorithms3/4/6/7 and Section5.3
to code/tests and explicitly records engineering choices and unsupported variants.
This is a paper-based materialized LSM-Full, not the authors' source/binary or a
claim of equivalent performance. Nonmaterialized LSM, Trie, arbitrary timestamps,
windowed range and approximate-only APIs are not implemented under that label.

Root Coconut now bulk-loads sorted chunks with bounded fan-in64 external merging
(no insertion-style binary compactions during initial build); evenly packs leaves
and builds a resident128-way fence-key B+tree bottom-up. Packed SAX snapshots are
explicitly resident,16bytes/record at16segments; bound arrays need4bytes/record
of the largest searched run. Top-k executes queries sequentially, parallelizing
lower-bound calculations within each query as Algorithm7 specifies, then skip-
sequentially refines candidate raw leaves. This replaces across-query parallelism.
O(N) resident summaries are intentional; raw vectors remain on disk. In-flight
merges can temporarily retain old and new snapshots; counters describe published
runs, while RSS/cgroup captures actual/transient costs.

Background binary compaction defaults on, merges only adjacent temporal spans,
uses bounded pending-run backpressure, and atomically publishes fully synced
replacements. Immutable reader snapshots retain old files until reads finish.
Caller inserts/queries remain serialized. New `set/getAsyncMerges`,
`waitForMerges`, `getResidentSummaryBytes`, `getLastSearchStats`,
`searchIndexRecent(queries,k,window_size)` are exposed in C++/Python. Recent windows
use insertion counts and original IDs; wholly old runs are skipped (BTP), boundary
runs filtered. Stats include summaries, logical raw reads/bytes, searched/skipped
runs and approximate/bounds/refinement times. No crash recovery/WAL/reopen added.

New C++ and seven-method Python disk tests exercise exact IDs/distances through
sync/async compactions and live reader snapshots, ties, BTP windows, odd dimensions,
multi-level trees with>128^2 leaves and>64 initial sort chunks. Focused CTest:
14passed,9explicit legacy non-finite Astronomy fixture skips,0failed (23total).
Log `build/coconut-paper-20261019/ctest-coconut.log`. New optional
`bm_Coconut_Streaming` compiles/executes; insert timing includes compaction barrier
and uses wall time. C++ streaming demo compiled; Python demo/docs/architecture
updated. BUILD_COCONUT=OFF configured and exposes no Coconut targets. Standard
Threads::Threads dependency is guarded by BUILD_COCONUT; no external engine added.

Isolated source/module: `build/coconut-paper-20261019`, frozen core
SHA2044324add110b750007c13bd7a6d4de37b3ca14e5f4ad13be7a5d50270f9a8a,
full source hashes/diff/paper hash inprovenance.json. Root production core stays
SHA660e4838ba82e7bc458da1acbd2c5b8bc082a29fe62a01f54dcc0507deb65575.
Backup before this change: `build/coconut-paper-20261019-before/`.
Do not overwrite isolated or production modules while the full run is active.

DEEP out-of-core preflight
`build/streaming-scale-runs/preflight-coconut-paper-outofcore-20261019`
has SUCCESS/FINISHED and verification.json. Genuine100K->1M vectors,100Kinserts,
8queries,k1,2threads,8192memtable,256MiBcap/no-swap,first+one repeat:
20/20query batches and10storage audits passed, zero neighbor gaps, peak RSS
213475328bytes <256MiB, raw database384000000bytes >cap. Performance claims false.
At1M eight queries requested~3.066GB logical raw reads: this workload still has
weak pruning; correctness/paper alignment does not imply superiority to FAISS.

Reused prior authorization for the corrected Coconut-only100M->1B experiment.
New nohup controller3604591/output
`build/streaming-scale-runs/20261019-coconut-paper-100m-1b-top1-50m`.
Authoritative NEW pointer `build/streaming-scale-runs/latest-coconut-paper-top1.json`.
The old latest-coconut-disk-top1.json retains its old output and now links the
paper replacement pointer/output; use the NEW pointer for future status requests.
Same genuine DEEP,50Mbatches,100top-1queries,48threads,first+3repeats,1Mbuffer,
340GiB/no-swap,2hoperation and24hworker deadlines. Independent previous1Btruth
is reverified/reused, full initial100M equality/hash checked as before. Runner
now charges waitForMerges after build/insert INSIDE the timed operation and emits
query_diagnostics outside measured query time. Old run snapshots stay untouched.
Check progress before claiming native build/query has begun; launch starts with
reference revalidation. FAISS/MESSI are not rerun and prior values stay valid.
Old Coconut1B curves describe the previous adaptation, not this SIMS policy.
No paper/Overleaf changes, Git commits/pushes or PR made in this turn.

### Coconut optimized without changing its search/index logic (2026-10-20)

User authorized making Coconut disk-based and fast, explicitly preserving Coconut
logic. Keep ordered sortable SAX runs, tree-based approximate seeds, exact SAX
pruning, independent per-query heaps/stable IDs, Gaussian regions and binary LSM
updates. No FAISS/brute-force fallback or normalization was introduced.

Stopped only our slow paper-run controller3604591 and systemd worker3605829;
preserved logs/outcomes and wrote superseded.json/status state superseded. Its
last complete prefix is350M;26query batches validated through400M before stopping
the second warm repeat there. Do not report that run as completed1B or resume it.
New code/measurements are separate from all previous Coconut snapshots.

Top-k now shares candidate raw reads across groups of up to32queries (runtime
set/getQueryBatchSize,1..64). Independent heaps/bounds/seed exclusions and BTP
filters preserve exact answers. Bound arrays/masks are tiled, not run-sized;
one raw block is<=8MiB, maximum32768records, with sparse<=4KiB-gap coalescing and
dense candidate blocks coalesced sequentially. Only blocks with candidates are
read. Parallel refinement operates on independent queries sharing that block.
New internal CoconutBounds.hpp precomputes double interval-square contributions
per query; float summation order matches the existing SAX helper. A partial
nonnegative sum may prune only above the bound. Dedicated C++ parity/threshold
tests cover1..8bits,1/3/16/32/48/96segments and adjacent-float thresholds.

New stats: candidate/query pairs, unique candidate positions per group, distance
and seed evaluations, seed/raw records read, peak scratch arrays/LUTs. Existing
phase/bytes/runs stats retained. Runner fingerprints the new header and records
process read_bytes/write_bytes/rchar/wchar plus instantaneous RSS outside timing.
initial_input_released marks release of the borrowed initial source before queries.
Lifetime maxRSS can exceed worker memory.current limits because shared source
file pages may be charged elsewhere; do not equate that peak with index heap.

Isolated optimized build `build/coconut-optimized-20261020`; frozen core
SHAdeac91f8e53342ff6695a304b89a24dc3d0c7b0a278c173a458ac1e1c99beda5.
Source9hashes/core/diff inprovenance.json. Root production core remains
SHA660e4838ba82e7bc458da1acbd2c5b8bc082a29fe62a01f54dcc0507deb65575.
Backup before changes `build/coconut-optimized-20261020-before/`.
Do not rebuild/overwrite the frozen module while its full run is active.

Controlled DEEP calibration uses genuine prefix data and query rows100:200,
disjoint from final benchmark queries0:100. Same references,48threads and caps
within each comparison; all warm repeats retained. Ignored reusable orchestrator
scripts/experiments/tune_coconut_disk.py. Raw prototype/results retained separately.
Final1M/100query means: old scalar-query16segments11.558s; shared-read16segments
1.336s; shared-read96segments0.821s. An optimized single-query16segment control
was slower13.733s and remains in results. Source runs under tuning-1m-final,
old baseline fromtuning-1m/baseline; query/reference identities verified on reuse.
Final3M/8queries under1GiB cap (raw1.152GB) means: old14.553s, new16segments2.105s,
new96segments0.158s. All9out-of-core calibration checks passed. Ratios are subset
calibration results, not billion-scale speedup claims. comparison.json/.csv retain
all variants, counter data and source paths. Calibration performance_claims=false.

DEEP experiment explicitly selects96PAAsegments (length1 at96D) and query batch32.
Gaussian quantization/raw data/metric stay unchanged. This uses96GB resident SAX
at1B instead of16GB; live disk files584GB instead of424GB. Raw vectors still384GB
on disk. The library's general PAA default stays16. Full-run hyperparameters are
fixed at initial build, not changed between streaming stages.

Final preflight `build/streaming-scale-runs/preflight-coconut-optimized-4m-20261020`:
3M->4M genuine DEEP,100K inserts,8original queries,k1,48threads,1GiB/no-swap,
65536memtable,96segments,batch32,first+one warm repeat. SUCCESS/FINISHED,
22/22exact query batches and11storage audits pass, zero neighbor gaps. Raw1.536GB
exceeds cap; final3,983,040disk records plus16,960buffered. Maximum native-query
RSS727523328bytes (~694MiB), resident summaries382371840bytes. Lifetime peak
RSS2239864832includes shared initial input mapping; verification.json explains.
This preflight overlaps correctness/build work, so do not use its elapsed times
as performance evidence. Controlled calibration timings are separate.

Final focused CTest24cases:15passed,9explicit non-finite Astronomy fixture skips,
0failed;9Python disk methods included, plus existing5top-k tests and C++ tests.
Log ctest-coconut.log. Streaming benchmark built/executed with batch1/32 controls;
C++ demo compiled; Coconut-disabled configuration has no Coconut targets.
Docs/README/architecture/demos updated with memory, timing and algorithm mapping.

New nohup controller3856914:
`build/streaming-scale-runs/20261020-coconut-optimized-100m-1b-top1-50m`.
Authoritative pointer `build/streaming-scale-runs/latest-coconut-optimized-top1.json`.
latest-coconut-paper-top1.json and latest-coconut-disk-top1.json also point to this
current run, preserving previous pointers/outputs. Check this run for future
status, not the stopped Oct19 trajectory or completed old Oct17 adapter.
Same DEEP/query prefixes,100Minitial,50Minserts through1B,100top-1queries,
48threads,first+3warm calls,1Mmemtable,340GiB/no-swap,2hoperation/24hworker limits.
Config config-coconut-optimized-100m-1b.json; insertion includes waitForMerges;
independent prior1Btruth is reverified/reused. Reference revalidation completed;
native worker3858037 started the100Mbuild. Check current progress before claiming
later stages. No FAISS/MESSI rerun, paper edits, commits, pushes or PR.

### Completed optimized Coconut 1B trajectory and plot (2026-10-21)

The Oct20 optimized run above now has SUCCESS/FINISHED, complete outcomes and no
failures. All19prefixes100M..1B and76/76query batches passed independent exact
validation, with zero neighbor gaps and zero near-float32 ties. Native build
through final query took11.9455hours, finishing Oct21 at00:50Europe/Paris.
At1B the three warm100-query times are569.9833/611.9708/590.0603seconds;
mean590.6715seconds (9.8445minutes). Final50Minsert including merge barrier took
6185.3076seconds (103.0885minutes). No OOM/timeout. Do not report this run as active.

New figure bundle `docs/figures/streaming-1b-optimized-20261021` combines ONLY this
new Coconut trajectory with original survivor FAISS800M/MESSI Adaptive700M.
PNG/PDF/SVG preview, individual panels, self-contained PGFPlots, CSVs and complete
hash/config/reference provenance are exported by ignored plot_streaming_billion.py.
188query batches/47prefixes validated for export; source/query/reference hashes
and common protocol checked. Means retain all three warm repeats and bands show
their min/max, including the FAISS550M outlier. Query axis includes the entire
band. Initial build times exported separately; insertion includes compaction.
Coconut label is `Coconut (DaiSy, disk)`; footer records resident96segmentSAX and
query batch32. This is DaiSy's paper-based implementation, not an upstream binary
claim. Curves end at validated prefixes, with no fabricated points to1B for
FAISS/MESSI. No reruns, paper edits, commits, pushes or PR for this figure.

### Coconut-only PR prepared on current main (2026-10-21)

User requested a Coconut PR and an inventory of remaining changes. Refreshed
origin/main to d98008d (PR81 merged; PR80 already merged), created isolated
worktree `build/coconut-lsm-pr-20261021`, branch `fix/coconut-disk-streaming`.
Commit25834ea contains23reviewed Coconut implementation/integration/test/doc files
only. Shared bindings/README/test registration were merged selectively onto main;
no MESSI/SAX implementation, experiment files, figures, auth or dependency changes.
Three Coconut algorithm source hashes match the completed optimized1B run.
Root dirty improvement/extending-streaming checkout remains unchanged apart from
this continuity note. Do not copy its old SAX.cpp over main: that would revert
the adaptive encoding fixes already merged in PR81.

Published draft PR https://github.com/MChatzakis/DaiSy/pull/82; remote file list
verified, mergeable=true, isolated worktree clean. No merge performed.
Clean Release full build (Python/tests/demos/benchmarks), benchmark smoke, focused
Coconut CTest15passed/9explicit invalid-fixture skips/0failures, and separate
BUILD_COCONUT=OFF library/Python build/import/target-exclusion checks passed.
Disabled source/build lives in build/coconut-lsm-pr-off-20261021; main ON module
and frozen experiment modules were not overwritten.

Broader295caseCTest is still running in exec session67946, with180s/test and6jobs;
log `/tmp/daisy-coconut-lsm-pr-ctest.log`, build/review/Testing holds CTest results.
Several ParIS cases exceeded the local timeout. PR body explicitly reports those
timeouts and keeps draft pending broader results/CI; do not claim full-suite pass
or identify their cause without further evidence. Focused logs:
/tmp/daisy-coconut-lsm-pr-focused.log and build/off logs with same prefix.
PR description/payload and explicit file manifest are in root build/ and ignored.

Remaining inventory: FAISS streaming and main MESSI/adaptive fixes already merged;
two local MESSI regression scripts (unique seeded candidates/range boundary) both
pass against unchanged main MESSI in the new PR build. Extra local seed-distance
preservation changes in iSAXSearch.cpp still need separate review/reproduction,
not a wholesale replay of the stale root diff. Experiment/paper artifacts remain
separate; scripts/experiments is already ignored on current main.


### Original checkout synchronized with main (2026-10-21)

User requested bringing main into the original VSCode checkout. origin/main now
includes merged Coconut PR82 at c6cac82. Primary checkout switched to main and
fast-forwarded; the original branch tip is retained. All30previous local changes
were verified/copied into build/ backup and a safety stash before switching.
Backup path is recorded in build/latest-main-sync-backup.json; no stash was dropped.
Three-way restoration preserves main adaptive encoding, MESSI input-layout checks,
strict top-k comparisons and inclusive-range bounds. README/bindings and merged
Coconut implementation now match main. Remaining source edits are the local SIMD
partial-bound equality change and preservation of seed distances, plus a MESSI
comment; additional two Python tests/registrations and local docs/ignore changes
remain reviewable. No new algorithm change, commit, push or experiment restart.
The older30file inventory describes the pre-sync checkout; refresh it before use.


### FAISS removed from the library (2026-10-21)

User explicitly requires FAISS to be an external comparison baseline, never a
DaiSy library algorithm. Removed lib/algos/Faiss.cpp/.hpp, public umbrella/header
and Python binding exposure, BUILD_FAISS/FAISS_ENABLED wiring, wrapper demos and
wrapper-specific tests. Existing native faiss::IndexFlatL2 benchmark executables
and independent Python ground-truth checks remain. CPU FAISS dependency setup is
now confined to benchmark/CMakeLists.txt and only runs with BUILD_BENCHMARK.
README supported algorithms/options/streaming API and contributor/demo/architecture
notes reflect this boundary. Earlier FAISS-wrapper descriptions above are history.
Pending MESSI/SAX/test/doc changes were preserved. Backups/build validation for
this removal are in build/faiss-library-removal-20261021. No experiment rerun,
Git commit, push or PR is part of this removal step.


Removal validation: fresh isolated Release core/Python/tests/demos build passed
with no FAISS sources;21/21focused tests passed (streaming, MESSI exactness in both
breakpoint/worker modes, inclusive range, adaptive SAX, Python layouts, Coconut
physical neighbors/bounds/disk behavior). BUILD_BENCHMARK=ON still configures with
missing FAISS sources and skips only its comparison targets; Coconut benchmark
build/smoke passed. Actual FAISS comparison binaries were not built (submodule
absent; host CMake3.22). New module has no FaissFlat or libfaiss dependency.
Verified native module was atomically installed into root daisy/, with old binary
preserved in build/faiss-library-removal-20261021/runtime-before. Fresh root import
confirms FaissFlat absent and core algorithms present. This supersedes earlier
notes about the production core hash staying660e4838; historical frozen binaries
and run records were not modified. validation.json/runtime-update.json record
hashes, checks and logs. No commit, push or new PR yet.
