# Coconut-LSM-Full implementation against the paper

Source: Kondylakis, Dayan, Zoumpatianos and Palpanas,
[Coconut: Sortable Summarizations for Scalable Indexes over Static and Streaming Data Series](https://arxiv.org/abs/2006.11474),
[author PDF](https://nivdayan.github.io/Coconut_VLDBJ_CR.pdf).
The PDF used on 2026-10-19 is saved with the isolated build; its SHA256 is
`52da8f081c167ee8c75c303d7764194f226c9caac4e366b5d916e30c16c870be`.

This is a paper-based implementation of the **materialized Coconut-LSM-Full**
variant, using the existing public `Coconut` name. It is not the authors' binary,
a bit-for-bit reproduction of their storage engine, or a performance-parity claim.
The nonmaterialized variant, Coconut-Trie and the static Berkeley DB executable
are different implementations and are not silently substituted here.

## Traceability

| Paper operation | DaiSy implementation | Validation |
|---|---|---|
| Sortable SAX | Existing bit-interleaved sortable SAX; Gaussian regions, no implicit normalization | Prior upstream-key comparison; exact result tests |
| Bottom-up build, Algorithms 3/6 | Sort initial bounded chunks once, then bounded-fan-in external merge; no repeated insertion-style compactions during bulk loading | More than 64 initial chunks, file input, partial chunks |
| Balanced indexed run | Evenly packed adjacent leaves (at least half full), bottom-up 128-way fence-key B+tree; ordered keys/IDs and sorted raw files | More than 128² leaves, duplicate keys and both query-key endpoints |
| Approximate seed, Algorithm 4 | Tree traversal to insertion point, leaf-sized neighboring window in every eligible run, one global best-so-far heap | k=1/10/100/1000, ties, empty build, distinct stable IDs |
| Resident summaries, Algorithms 5/7 | Packed SAX snapshot in each immutable run, exactly PAA-segments bytes per record. Original SAX is retained rather than undoing the invertible key permutation for every bound | Resident bytes equal persisted-record count times segments |
| Parallel bounds, Algorithm 7 | OpenMP bounds across resident summaries and independent queries; bounded block arrays (original scalar-query schedule available with batch size 1) | One/four threads return exactly the same integer-data IDs/distances |
| Skip sequential refinement | Iterate run positions in order; skip prunable records, read candidate leaves into one reusable buffer, update shared query bound | Physical float64 ranks and returned-vector distances; read/phase counters |
| LSM updates, Section 4.4/Figure 7 | Bounded memtable, durable sorted flushes, asynchronous adjacent-run binary merges (size ratio 2), bounded pending runs/backpressure | Queries while worker replaces runs, barrier and file accounting, sync/async modes |
| Bounded temporal partitioning, Section 5.3 | Runs carry contiguous insertion-ID intervals. Skip wholly older runs; filter the boundary run by original IDs. `searchIndexRecent` accepts the count of latest records | Windows 1 through entire stream; old runs skipped; partial memtable; ties |

## Storage, memory and timing decisions

* The materialized variant rewrites raw vectors during compaction. Every published
  replacement is fully written and synced before its inputs are retired.
* Readers retain immutable shared run snapshots, so background replacement cannot
  remove their files. Caller-issued inserts and queries must still be serialized.
* Background merging defaults on; `setAsyncMerges(false)` selects blocking updates.
  `waitForMerges()` waits for all pending work and surfaces background I/O errors.
  Time `insertBatch` **and this barrier** for completed-batch comparisons. Enqueue
  latency alone is not a comparable insertion time. A partial table stays queryable
  in RAM; call `flush()` to persist it, then the barrier to finish compaction.
* Initial chunk merging has fan-in at most 64 to bound file descriptors and I/O
  buffers. Inputs beyond that limit require additional external-sort passes. This
  is an engineering limit rather than a promise of two passes at every scale.
* The RAM memtable remains bounded, but SIMS deliberately has O(N) resident SAX:
  16 GB of SAX at 1B records with 16 segments, plus bounded block-bound arrays and small tree directories. In-flight merges/query snapshots can
  temporarily retain both old and new summaries. `getResidentSummaryBytes()`
  reports the currently published runs; RSS/cgroup usage captures transients too.
* Keys and IDs are mmap-backed; raw leaf reads use `pread`. OS file cache counts
  toward the memory budget. Explicit resident summaries are never described as
  the whole raw database in RAM. `getDiskBytes()` counts live logical run files;
  concurrent compaction creates temporary additional files. Audit physical bytes
  only after `waitForMerges()` and after outstanding queries return.
* `getLastSearchStats()` gives last top-k call totals: resident summaries examined,
  logical leaf reads/bytes requested (not physical device counters), runs searched/
  skipped, and approximate/bounds/refinement phase wall times. Cold summary
  construction is part of build/merge, not hidden inside query timing.
* `setLeafSize()` controls the approximate radius and maximum records per leaf.
  Fanout 128, split files and POSIX I/O are documented implementation choices;
  the paper does not fix the source-level layout for reproductions.

## Bounded batch SIMS optimization (2026-10-20)

The current default groups at most 32 queries (`setQueryBatchSize(1..64)`). Each
query still traverses the Coconut run trees for its own seed and keeps its own
exact heap, SAX lower bounds and stable-ID tie order. Summaries are evaluated in
ordered blocks, using the current per-query bound. Qualifying positions across
the query group share reads; no raw block is read unless it contains a candidate.
Sparse positions are coalesced across gaps of at most 4 KiB. Dense candidates
share a contiguous read of at most 8 MiB. Refinement rechecks every query's bound
before a distance computation. This changes scheduling and I/O granularity, not
the sortable-SAX, LSM, pruning or exact-search logic. It is an engineering
optimization of paper-based SIMS, not the paper's original sequential-query
schedule. No unconditional exhaustive-search fallback is introduced.

Lower-bound arrays are now bounded by query count times block capacity, rather
than total run size. A block has at most 32,768 records and at most 8 MiB of raw
vectors (or one vector for dimensions exceeding that size). Scratch counters
cover raw/bounds/mask buffers; heaps and resident summaries are separate. Query
results from different groups never share heap state. Caller writes/reads remain
serialized; immutable run snapshots still protect background compactions.

The per-query interval-distance lookup (`CoconutBounds.hpp`) retains the reference
helper's double squared contributions and float accumulation order. A partial
nonnegative sum may stop only once its scaled bound exceeds the query bound;
otherwise it returns the same full bound. A dedicated C++ test checks exact float
equality against the existing SAX helper for 1-8 bits and 1/3/16/32/48/96 segments,
including equal and adjacent-float thresholds. This is a faster evaluation of the
same Coconut lower bound, not a learned/approximate bound. Dense candidate blocks
(at least half their positions qualify across the group) coalesce into one read;
sparse blocks retain the 4 KiB gap limit. The lookup tables are included in the
reported scratch-array memory.

New counters distinguish candidate/query pairs (`candidate_records`), unique
qualifying positions per group (`unique_candidate_records`), actual distance
computations, seed computations/records read, raw records read and peak scratch
array bytes. To evaluate block-read amplification, exclude seed reads and divide
raw records read by the unique qualifying positions. Logical read counters are
not physical device I/O measurements. PAA segment count is an ordinary Coconut
parameter; tuning it changes summaries and memory needs, never the input vectors
or the exact squared-L2 answers. Freeze that parameter in every measured config.

## API extensions and limits

Top-k generalizes the paper's 1-NN with a unique-ID heap and deterministic equal-
distance ID ordering. Inclusive range search remains an existing DaiSy extension.
Recency uses insertion order, not arbitrary user-provided wall-clock timestamps;
`searchIndexRecent(queries, k, window_size)` returns original stream IDs, with
existing max-float/ID-zero padding if fewer than k records qualify. It is a
Coconut-specific API; it does not change the common SearchConfig contract.
Windowed range/approximate-only APIs are not exposed. Raw input is never normalized;
the paper's z-normalized data and our unchanged DEEP embeddings are distinct
workloads, so performance claims must state the actual data preprocessing.

Indexes remain temporary POSIX files with no reopen, recovery/WAL, deletion, or
concurrent caller reads/writes. Failed inserts may accept a prefix; a failed
background merge is surfaced and the index must be rebuilt. Optional build guards
remain unchanged. Neither this implementation nor its exactness tests guarantees
that Coconut will outperform FAISS/MESSI on the chosen workload.

Performance evaluations must freeze binary/source and configuration, include the
compaction barrier in insertion timing, validate each prefix against independent
ground truth, and retain phase/read/memory diagnostics.
