# Coconut upstream comparison (2026-10-10)

Reviewed the user-supplied repository, checked out without executing its build or
run scripts at `/tmp/daisy-coconut-upstream`:

- Repository: <https://github.com/kon0925/coconut/>
- Commit: `0e7ae37777c6588bea0e03117db357161503cb93` (2021-07-27).
- The README identifies this as the implementation of the **2018 PVLDB paper**.
- `build_unix/run.sh` invokes `coconut` and `coconut_plus`; the Makefile builds
  these from `examples/c/coconut.c` and `examples/c/coconut_plus.c`.
- `coconut_NEW.c` is another source file, but is not the executable selected by
  those Makefile targets. Results below refer to the selected implementations.

## Comparison before the port

The DaiSy column below describes the pre-port implementation reviewed on
2026-10-10. The changes now implemented are listed under "Implementation
follow-up"; the initial-query and leaf-access limitations in this table are
historical, while the Berkeley DB and LSM differences remain.

| Operation | Selected upstream executables | DaiSy Coconut before this port |
| --- | --- | --- |
| Initial search bound | Encode the query as sortable SAX; seek with `DB_SET_RANGE`; inspect neighboring records with `DB_PREV`/`DB_NEXT`; obtain an approximate nearest-neighbor distance before exact refinement. | Begins at sorted record zero with an empty top-k heap and effectively unbounded distance. The query's sortable key is not used to choose initial candidates. |
| Storage | Berkeley DB `DB_BTREE`, database pages, duplicate-key support and sorted/bulk insertion. `coconut` stores raw series in the database; `coconut_plus` stores original-file offsets. | A sorted record vector and separate raw leaf files, accessed through `fseek`/`fread` per surviving vector. No B-tree or explicit page cache. |
| Exact refinement | Compute all SAX lower bounds with 16 pthread workers **within one query**, then refine surviving series against the approximate bound. | Compute bounds and refine inline; parallelism is across queries. Repeated small stdio reads remain even after the earlier fix that opens each leaf only once per query. |
| Query result | One best distance: the reviewed search paths implement **1-NN**. | Top-k heap and IDs, including k=10 streaming experiments. A port must generalize the initial search to at least k distinct candidates and preserve unique IDs. |
| Streaming | The selected executables expose initial indexing/query modes. Their reviewed query paths have no live batch-insert/LSM trajectory. | Appends SAX records and raw vectors to an unmerged in-memory buffer. This is not a complete Coconut-LSM implementation. |

Source locations at the pinned commit:

- [Upstream README](https://github.com/kon0925/coconut/blob/0e7ae37777c6588bea0e03117db357161503cb93/README.md).
- [Query-key search and approximate seeding in coconut_plus.c](https://github.com/kon0925/coconut/blob/0e7ae37777c6588bea0e03117db357161503cb93/examples/c/coconut_plus.c#L1196).
- [Parallel lower bounds and exact refinement in coconut_plus.c](https://github.com/kon0925/coconut/blob/0e7ae37777c6588bea0e03117db357161503cb93/examples/c/coconut_plus.c#L1259).
- [B-tree setup in coconut_plus.c](https://github.com/kon0925/coconut/blob/0e7ae37777c6588bea0e03117db357161503cb93/examples/c/coconut_plus.c#L818).
- [Approximate search in coconut.c](https://github.com/kon0925/coconut/blob/0e7ae37777c6588bea0e03117db357161503cb93/examples/c/coconut.c#L1736).

**Correction to the earlier diagnosis:** scanning all SAX summaries is also part
of these upstream exact search paths. Its presence alone does not establish an
incorrect implementation of Coconut exact search. Missing approximate seeding,
different storage access, and incomplete update handling are the substantive
differences. Parallelizing over queries is a different execution policy, not
proof that it must be slower than upstream's within-query parallelism.

The [extended Coconut paper](https://arxiv.org/pdf/2006.11474) introduces
Coconut-LSM. Finding initial bulk loading in the supplied 2018 repository is not
evidence that its selected executables implement that later streaming variant.
The `coconut.c` query path also contains hard-coded 256-dimensional distance
calls; it cannot be used unchanged as our 96D/768D baseline.

## Isolated checks

Artifacts are in `build/coconut-diagnostics/upstream-audit/`. They do not replace
`daisy/_core`, alter existing measurements, or restart running experiments.

1. Extracted upstream `invSax` and compared it with the compiled DaiSy
   `sortable_sax_from_sax`: **320,000 matching keys**, across 1–8 bits and
   4/8/16/32 segments. The key transformation itself matches upstream.
2. Built an instrumented baseline module and a separate diagnostic module with
   approximate seeding adapted to top-k. The latter binary-searches the query's
   sortable key, refines a neighboring window of 1,000 records, and excludes
   those records from the later scan to prevent duplicate IDs.
3. Both modules were checked against FAISS on the unchanged first 1M Seismic
   and DEEP vectors, and first 100K MS MARCO vectors. Each uses the same eight
   queries, k=1/10, two workers, and a second stage after inserting 10% more
   vectors. All 24 measured searches across the two modules passed distance,
   unique-ID and returned-vector checks.

Initial-stage k=10 on-disk vector reads, summed over eight queries (seed reads
included):

| Dataset sample | Baseline | With seeding | Reduction |
| --- | ---: | ---: | ---: |
| Seismic 1M | 4,600,648 | 4,426,673 | 3.78% |
| DEEP 1M | 7,835,786 | 7,826,360 | 0.12% |
| MS MARCO 100K | 800,000 | 800,000 | 0.00% |

Seeding is a genuine missing step, but these samples show that **adding it alone
does not solve the performance problem**. Many vectors survive the 16-segment
SAX filter for these top-k queries, particularly the embedding datasets, leaving
per-vector stdio overhead exposed. These are diagnostic subsets, not full-100M
measurements or a speed claim for the published algorithm. Elapsed subset times
are saved for transparency; single calls made while the full job is running are
not a controlled timing comparison.

4. A third isolated module changes only the refinement access: load each
   candidate leaf into a private byte buffer once, then copy surviving vectors
   from that buffer. It retains the baseline's bounds, order and top-k logic;
   it does not add seeding or implement a B-tree/LSM. All twelve searches also
   passed the same FAISS/unique-ID/returned-vector checks. Initial-stage k=10
   timings for eight queries were:

| Dataset sample | Per-vector stdio baseline (s) | Whole-leaf diagnostic (s) | Observed ratio |
| --- | ---: | ---: | ---: |
| Seismic 1M | 7.714 | 2.076 | 3.72x |
| DEEP 1M | 7.543 | 1.542 | 4.89x |
| MS MARCO 100K | 1.587 | 0.503 | 3.16x |

This supports investigating I/O call overhead as a substantial bottleneck. It
does **not** establish the expected 100M speedup, account for sparse-candidate
disk-read amplification, or turn the prototype into published Coconut-LSM.
Raw results, per-query refinement counters and module/source hashes are saved
in `build/coconut-diagnostics/upstream-audit/comparison.json`; the preparation
and benchmark scripts and logs are in the same directory.

## Existing full run

The preserved run `build/streaming-runs/20261010-122735-coconut-fixed` is still
using its original isolated prototype binary. Its first Seismic 90M stage
returned exact validated results in 1,307.98 seconds for 100 queries; the 91M
stage took 361.30 seconds. The first 1M insertion took 3.13 seconds, and the
second took 4.00 seconds. Those insertion timings measure appending to the
buffer, with no LSM flush/merge work. The large difference between the first two
query rounds is consistent with a cache effect, but was not an isolated cache
experiment.

For a representative Coconut streaming comparison, implement and validate the
update/merge policy, restore query-key seeding, and address storage/page access.
For an in-memory comparison, specify and measure the residency policy too;
correct neighbors alone do not make the current disk-backed prototype a fair
in-memory performance baseline.

## Implementation follow-up

On the user's request, DaiSy now implements the ordered query-key search
before exact top-k refinement. It binary-searches the resident sorted key
directory, evaluates a neighboring window of at least k distinct records when
available, and excludes that window from the later exact scan. The returned
heap still includes exact refinement over all surviving unseeded records and
the live insertion buffer. Top-k and range share whole-leaf buffered reads,
including partial last leaves and explicit failed-read errors. The raw float
copy respects alignment even with unusual SAX key lengths.

This adapts the upstream `DB_SET_RANGE`/neighbor traversal behavior to DaiSy's
resident directory. It does not import the Berkeley DB storage engine or claim
to implement Coconut-LSM. The earlier audit/diagnostic measurements describe
the pre-port source snapshots; replacement experiment runs use separate
directories and capture their own extension and source snapshots.
