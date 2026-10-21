# Coconut disk streaming

The implementation was aligned further with the extended paper on 2026-10-19:
resident SAX, within-query parallel SIMS, bottom-up indexed runs, background merges
and recent-window queries. See the [algorithm-by-algorithm mapping](coconut-paper-implementation.md).
The earlier 1B run predates these changes and measures the previous adaptation.

The previous adapter built static leaves on disk but retained **all subsequent
raw insertions in RAM**. Its streaming timings omitted flush/merge costs and its
memory-limit outcomes must not be reported as Coconut disk-streaming limits.
The prior source and experiment snapshots remain available for provenance.

The corrected `Coconut` follows the materialized sorted-run design of
[Coconut-LSM](https://arxiv.org/abs/2006.11474), adapted to DaiSy's exact top-k and
inclusive range APIs. It is a new implementation of that storage/update policy,
not an import of the 2018 Berkeley DB executable or a claim of performance parity.

* A configurable memtable holds at most `setBufferCapacity(records)` raw vectors
  plus SAX summaries and sortable keys. Sorting adds one bounded permutation.
* Full tables are sorted and persisted as immutable runs with stable insertion
  IDs. The initial build also processes bounded chunks, including file sources.
* Each run has a fixed-width `[sortable key][SAX word][ID]` directory and a raw
  float32 data file in the corresponding order. Directories are reclaimable
  read-only file mappings; no full vector database or heap key directory is kept.
* Equal-level runs are merged with sequential bounded input/output buffers,
  using binary size tiers in a background worker. A new run is written, flushed and synced before its
  input runs are retired. Bulk build consolidates its sorted chunks into a run.
* Queries seed a shared bound from each run's sortable-key neighborhood, then
  refine every surviving SAX summary and inspect the remaining bounded memtable.
  Top-k uses resident SAX, exact query interval lookups and bounded shared
  candidate reads across independent queries. Returned IDs refer to the original stream.
* `insertBatch` includes automatic flushes and schedules asynchronous merges.
  Include `waitForMerges()` in completed-insert timing; `flush()` can persist an
  incomplete table. `getDiskRecords`, `getBufferedRecords`, `getDiskBytes`,
  `getRunCount`, `getFlushCount`, `getMergeCount`, and `getPeakBufferedRecords`
  provide evidence that inserted data are actually reaching disk.

There is no WAL, recoverable manifest, restart/reopen support, deletion or arbitrary-timestamp
window API. Writes and searches must not run concurrently. The temporary directory
is removed when its instance is destroyed. A failed batch can have accepted a
prefix before an I/O failure; inspect the count and treat the error as a failed
operation, not an atomic transaction. Invalid shapes/non-finite vectors are rejected.
Changing SAX/leaf/buffer/storage layout after building is rejected. Gaussian SAX
is supported; selecting adaptive breakpoints is explicitly rejected in C++.
No additional dependency is introduced; `BUILD_COCONUT=OFF` still excludes it.

For the corrected DEEP experiment retain the original 100M build, 50M external
insertion batches, original 100 top-1 queries, 48 threads and 340GiB/no-swap cap.
Use the same 1M-record memtable at all stages. Validate top-k against the independent
prefix reference and log disk/buffer counters after build and every insertion.
Do not mix the old prototype's timing curves with the corrected storage policy.
Only Coconut needs a replacement performance run; FAISS/MESSI results remain valid.

Validation includes strict float64 physical-neighbor checks after repeated flushes
and merges, stable IDs, inclusive ranges, partial buffers, file-backed builds,
strided Python inputs, corruption detection and optional-build exclusion. The
legacy Astronomy range fixture has non-finite rows10321and40380; those nine
parameterized cases explicitly skip the invalid fixture rather than filtering it
or claiming exact range accuracy on NaNs. Finite-input rejection has its own test.
