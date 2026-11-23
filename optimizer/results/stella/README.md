# Collector results on stella

`samples_<DATASET>.csv.gz`: gzipped snapshots of the collector's CSVs (pandas reads them
directly: `pd.read_csv("samples_SIFT1M.csv.gz")`). The jobs write the uncompressed
`samples_<DATASET>.csv` next to them, which is not versioned; the snapshots are refreshed by
hand (each cut at its last complete line). Columns: [docs/optimizer-notes/collector.md](../../../docs/optimizer-notes/collector.md#csv-columns).

Grid: 810 configurations per dataset (MESSI 150, FreSh 150, SOFA 270, Odyssey 240),
k = 1, 10, 100, the first 100 queries of each dataset (all 100 for ASTRO100M and SALD100M),
threads 128, 96, 64, 32, 16, on stella's AMD EPYC 7H12 nodes (one node per dataset).

Checks on this snapshot: no duplicated (configuration, k, query) rows; every configuration of
every algorithm gives the same k-th distance per (k, query) (exact search); where L2 ground
truth exists, the k-th distances match it, except 2 queries of GIST1M (duplicates of base
vectors: distance 0, where Faiss's float32 ground truth has about 1e-6) and 2 of UCASELAW7M
(near-duplicates at about 5e-5, below Faiss's float32 precision); there the collector is right.

Snapshot of 2026-11-23 10:00:

| Dataset | Rows | Configurations | Per algorithm | State | Node |
|---|---|---|---|---|---|
| ASTRO100M | 138,300 | 461 / 810 | messi 150, fresh 150, sofa 161, odyssey 0 | partial | ncpum011 |
| DEEP100M | 162,300 | 541 / 810 | messi 150, fresh 150, sofa 241, odyssey 0 | partial | ncpu009 |
| DEEP10M | 243,000 | 810 / 810 | messi 150, fresh 150, sofa 270, odyssey 240 | complete | ncpu026 |
| DEEP50M | 243,000 | 810 / 810 | messi 150, fresh 150, sofa 270, odyssey 240 | complete | ncpu022 |
| FEVER5M | 243,000 | 810 / 810 | messi 150, fresh 150, sofa 270, odyssey 240 | complete | ncpu029 |
| GIST1M | 242,840 | 810 / 810 | messi 150, fresh 150, sofa 270, odyssey 240 | complete | ncpu037 |
| GLOVE1M | 243,000 | 810 / 810 | messi 150, fresh 150, sofa 270, odyssey 240 | complete | ncpu036 |
| HOTPOTQA5M | 243,000 | 810 / 810 | messi 150, fresh 150, sofa 270, odyssey 240 | complete | ncpu006 |
| MIRACLJA6M | 243,000 | 810 / 810 | messi 150, fresh 150, sofa 270, odyssey 240 | complete | ncpu008 |
| MIRACLRU9M | 200,700 | 669 / 810 | messi 150, fresh 150, sofa 270, odyssey 99 | partial | ncpu036 |
| MSMARCO8M | 243,000 | 810 / 810 | messi 150, fresh 150, sofa 270, odyssey 240 | complete | ncpu014 |
| RANDOM100M | 171,900 | 573 / 810 | messi 150, fresh 150, sofa 270, odyssey 3 | partial | ncpum014 |
| SALD100M | 171,900 | 573 / 810 | messi 150, fresh 150, sofa 270, odyssey 3 | partial | ncpu015 |
| SEISMIC100M | 83,700 | 279 / 810 | messi 150, fresh 129, sofa 0, odyssey 0 | partial | ncpum012 |
| SIFT1M | 243,000 | 810 / 810 | messi 150, fresh 150, sofa 270, odyssey 240 | complete | ncpu028 |
| T2I100M | 102,600 | 342 / 810 | messi 150, fresh 150, sofa 42, odyssey 0 | partial | ncpu007 |
| T2I10M | 243,000 | 810 / 810 | messi 150, fresh 150, sofa 270, odyssey 240 | complete | ncpu013 |
| T2I50M | 139,200 | 464 / 810 | messi 150, fresh 150, sofa 164, odyssey 0 | partial | ncpu040 |
| UCASELAW7M | 243,000 | 810 / 810 | messi 150, fresh 150, sofa 270, odyssey 240 | complete | ncpu039 |

GIST1M has 160 rows fewer than the others: 17 (configuration, k) pairs hit the 600-second
query budget. SALD100M and RANDOM100M stopped at Odyssey's 4th configuration (Odyssey's
per-batch priority-queue cap exited the process; fixed afterwards) and resume from there.
