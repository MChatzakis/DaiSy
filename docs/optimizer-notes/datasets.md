# Preparing the benchmark datasets on a server

`scripts/datasets/prepare_datasets.py` downloads and prepares DaiSy's 14 benchmark datasets
(the ones the knob-optimizer collector runs on, [collector.md](collector.md)) on any server,
with the same files as the lab server (erato). It replaces the two scripts of the
Endor-DR-Bench project that built erato's folders (`create_datasets.py` and
`create_datasets_2.py`), and keeps their rules, so the outputs are byte-identical.

## What a dataset folder holds

`DATA_ROOT/<NAME>/` (`.fvecs` / `.ivecs`: every row is an int32 dimension, then the
float32 / int32 values):

| File | Content |
|---|---|
| `base.<n>.fvecs` | the base vectors or series |
| `<split>.<n>.fvecs` | a query split: `query` (testing) for every dataset; `learn`, `validation` and for some `tuning` for the vector datasets. No query is in the base |
| `<l2.\|innprod.><split>.groundtruth.<n>.k5000.ivecs` | ids (0-based base rows) of the exact 5000 nearest neighbours of every query of the split |
| `<l2.\|innprod.><split>.groundtruth.<n>.k5000.fvecs` | their squared L2 distances (ascending) or inner products (descending) |
| `groundtruth_times.csv` | wall-clock seconds of every ground-truth step |

| Dataset | Base | Query splits | Ground truth |
|---|---|---|---|
| SIFT1M | `base.1M` × 128 | `learn.80K`, `tuning.1K`, `validation.1K`, `query.10K` | `l2.` |
| GLOVE1M | `base.1183514` × 100 | `learn.80K`, `tuning.1K`, `validation.1K`, `query.10K` | `innprod.` |
| GIST1M | `base.1M` × 960 | `learn.80K`, `tuning.1K`, `validation.1K`, `query.1K` | `l2.` |
| HOTPOTQA5M | `base.5233329` × 1024 | `learn.10K`, `validation.1K`, `query.7405` | `innprod.`, `l2.` |
| FEVER5M | `base.5416568` × 1024 | `learn.10K`, `validation.1K`, `query.6666` | `innprod.`, `l2.` |
| MIRACLJA6M | `base.6953614` × 1024 | `learn.4268`, `validation.1K`, `query.860` | `innprod.`, `l2.` |
| MSMARCO8M | `base.8841823` × 1024 | `learn.80K`, `tuning.1K`, `validation.1K`, `query.6980` | `innprod.`, `l2.` |
| DEEP100M | `base.100M` × 96 | `learn.80K`, `tuning.1K`, `validation.1K`, `query.10K` | `l2.` |
| MIRACLRU9M | `base.9543918` × 1024 | `learn.5312`, `validation.1K`, `query.1252` | `innprod.`, `l2.` |
| UCASELAW7M | `base.7M` × 1536 (7,414,023) | `learn.80K`, `tuning.1K`, `validation.1K`, `query.1K` | `l2.` |
| T2I100M | `base.100M` × 200 | `learn.80K`, `tuning.1K`, `validation.1K`, `query.10K` | `innprod.` |
| SEISMIC100M | `base.100M` × 256 | `query.10K` | `l2.` |
| ASTRO100M | `base.100M` × 256 | `query.10K` | `l2.` |
| SALD100M | `base.100M` × 128 | `query.10K` | `l2.` |

Every split has a ground-truth pair for each of the dataset's metrics. The data series have
no learning splits.

## Quick start on a new server

```bash
# 1. DaiSy (only needed for --gt-engine messi and for the collector)
mkdir -p build && cd build
cmake .. -DBUILD_OPTIMIZER=ON -DBUILD_SOFA=ON -DBUILD_ODYSSEY=ON   # add -DBUILD_SING=ON with CUDA and a GPU
make -j optimizer_collector daisy_groundtruth
cd ..

# 2. Python packages
pip install -r scripts/datasets/requirements.txt

# 3. Data (resumable: rerun the same command after an interruption)
export DAISY_DATA_ROOT=/data/daisy/processed DAISY_RAW_DIR=/data/daisy/raw
python3 scripts/datasets/prepare_datasets.py --list          # what is there already
python3 scripts/datasets/prepare_datasets.py                 # everything
python3 scripts/datasets/prepare_datasets.py SIFT1M GIST1M   # or some datasets
```

SEISMIC100M and ASTRO100M need their series file first (see "Series files"); without it
they are skipped, the run goes on with the other datasets, and it ends with an error that
lists them.

## Options

| Option | Environment | Default | Meaning |
|---|---|---|---|
| `DATASET ...` | | all | Datasets to prepare |
| `--data-root` | `DAISY_DATA_ROOT` | required | Where the prepared datasets go |
| `--raw-dir` | `DAISY_RAW_DIR` | `<data-root>/../raw` | Where downloads go, in the layout below |
| `--hf-dir` | `DAISY_HF_DIR` | `<raw-dir>/hf` | Hugging Face downloads and the BGE-M3 model |
| `--series-dir` | `DAISY_SERIES_DIR` | `<raw-dir>/similarity-search-datasets` | Folder of the series `.bin` files, under their usual names |
| `--series-file NAME=PATH` | | none | A series `.bin` file of another name or place, e.g. `ASTRO100M=/data/astro.bin` (repeatable) |
| `--series-url` | `DAISY_SERIES_URL` | none | URL prefix to download the non-public series files from |
| `--gt-k` | | 5000 | Neighbours per query in the ground truth |
| `--gt-splits` | | all | Comma-separated splits to compute ground truth for, e.g. `query` |
| `--gt-engine` | | `faiss` | `faiss` (exact flat index) or `messi` (`daisy_groundtruth`, DaiSy's MESSI) |
| `--gt-tool` | | `<repo>/build/lib/optimizer/daisy_groundtruth` | The MESSI binary, for `--gt-engine messi` |
| `--skip-gt` | | off | Base and query splits only |
| `--threads` | | all cores | Threads for the ground truth and the MIRACL encoder |
| `--list` | | | Show every dataset and what is already prepared |

A step whose output already exists with the right shape is skipped, downloads continue where
they stopped, and every output is written through a temporary file, so an interrupted run is
resumed by running it again.

## Sources

| Dataset | Source | Raw files under `--raw-dir` | Download |
|---|---|---|---|
| SIFT1M | BIGANN (SIFT1B) at dl.fbaipublicfiles.com: first 1M rows of `base.1B.u8bin`, first 1M of `learn.100M.u8bin`, `query.public.10K.u8bin` | `BIGANN/` | 0.26 GB (rows only) |
| GLOVE1M | Stanford GloVe: `glove.twitter.27B.zip` (base, ann-benchmarks' glove-100-angular "train" set), `glove.6B.zip` (queries) | `GloVe/misc/` | 2.4 GB |
| GIST1M | texmex `gist.tar.gz` (FTP, MD5 checked) | `GIST1M/gist/` | 2.7 GB |
| HOTPOTQA5M | Hugging Face `CohereLabs/beir-embed-english-v3`, `hotpotqa/` | `hf/` | 11.6 GB |
| FEVER5M | the same, `fever/` | `hf/` | 12.7 GB |
| MIRACLJA6M | Hugging Face `hanhainebula/bge-m3_miracl_2cr` `dense/ja.tar.gz`; topics and qrels from `miracl/miracl`, encoded with `BAAI/bge-m3` | `hf/` | 16.7 GB + 2.3 GB model |
| MSMARCO8M | `CohereLabs/beir-embed-english-v3`, `msmarco/` | `hf/` | 19.8 GB |
| DEEP100M | Yandex Deep1B: first 100M rows of `base.1B.fbin`, first 1M of `learn.350M.fbin`, `query.public.10K.fbin` | `DEEP1B/` | 38.8 GB (rows only) |
| MIRACLRU9M | as MIRACLJA6M, `dense/ru.tar.gz` | `hf/` | 22.9 GB + model |
| UCASELAW7M | big-ann-benchmarks caselaw embeddings (Azure) | `caselaw/` | 59.3 GB |
| T2I100M | Yandex Text-to-Image: first 100M rows of `base.1B.fbin`, first 1M of `query.learn.50M.fbin`, `query.public.100K.fbin` | `t2i/` | 81.2 GB (rows only) |
| SEISMIC100M | `data_size100M_seismic_len256_znorm.bin`, **not public** | `similarity-search-datasets/` (`--series-dir`) | copy (112.6 GB) |
| ASTRO100M | `data_size270M_astronomy_len256_znorm.bin`, **not public** (no online source found) | `similarity-search-datasets/` (`--series-dir`) | copy (284.1 GB) |
| SALD100M | [SALD](https://fcon_1000.projects.nitrc.org/indi/retro/sald.html) brain fMRI (INDI S3, CC BY-NC, cite Wei et al. 2017); see below | `SALD/` | 23 GB |

### Series files

The series files of SEISMIC100M and ASTRO100M come from the data-series benchmarks of
Palpanas' group and cannot be downloaded. They are `.bin` files of raw little-endian float32
records (256 values each, z-normalized, no header). On a server that has them:

- **Under their usual names, in one folder:** pass `--series-dir DIR` (or set
  `DAISY_SERIES_DIR`). The default is `<raw-dir>/similarity-search-datasets/`; erato's folder
  is `/mnt/hddhelp/mchatzakis/datasets/similarity-search-datasets/`.
- **Under other names:** pass `--series-file SEISMIC100M=PATH` and/or
  `--series-file ASTRO100M=PATH`.
- **Elsewhere on the web:** `--series-url PREFIX` downloads `PREFIX/<usual name>` into
  `--series-dir`.

The script reads the file in place; it does not copy or change it. A file shorter than
100,010,000 records (100M base + 10K queries), or whose size is not a whole number of
records, is rejected like a missing one: the dataset is skipped with a message. Seismic's
file has 109,999,999 records and Astro's 277,443,827. Any longer file works: only its first
100M finite records and its last 10K finite records are used (see "How the query splits are
made"). Only these two `.bin` files are read; the `queries_ctrl*` files next to them on
erato are not used.

Instead of the `.bin` files, you can copy the prepared `DATA_ROOT/SEISMIC100M/` and
`DATA_ROOT/ASTRO100M/` folders from a server that has them. The script then finds the base
and query files and skips straight to the ground truth. The collector reads only these
`.fvecs` files.

## How the query splits are made

The rules of the scripts that built erato's folders, so every server gets the same files:

- **SIFT1M, GLOVE1M, GIST1M, DEEP100M, UCASELAW7M, T2I100M** (create_datasets.py): from a learning
  file read by row. `learn` is its first 10K rows plus 70K more from row 30,000, `validation`
  the 1K rows from row 10,000, `tuning` 1K from row 20,000. Tuning and the 70K added learning
  rows skip every vector that is already in a split (the learning files repeat some
  vectors), the testing queries included. The learning files: BIGANN's (SIFT), Deep1B's
  and T2I's first 1M learning rows (SIFT's uint8 values as float32), GIST's `gist_learn.fvecs`, caselaw's single-vector query documents (with
  learn from row 0, validation 10,000, tuning 11,000, more learning from 13,000, and `query`
  = rows 12,000-12,999), and for GloVe the earlier learn and validation words followed by all
  other glove.6B words in seeded order (`docs/datasets/glove1m_query_rows.txt` lists the
  earlier rows). `query` is the official testing queries (T2I: the first 10K of 100K).
- **HOTPOTQA5M, FEVER5M, MSMARCO8M, MIRACLJA6M, MIRACLRU9M** (create_datasets_2.py): `query`
  is the judged testing queries (sorted ids of the test or dev qrels). The other splits are
  seeded samples (numpy `default_rng(42)`, each sorted) of the learning pool: the judged
  train queries for BEIR (85,000 / 109,810 / 502,939), and MIRACL's train, test-A and test-B
  topics (5,268 / 6,312), so MIRACL's `learn` is the whole pool minus `validation`. They are
  drawn in this order: `learn` (10K, or MIRACL's whole learn), `validation` (1K), then for
  MSMARCO8M `tuning` (1K) and 70K more learning queries, appended to `learn`
  (create_datasets.py's MSMARCO 80K rule). So MSMARCO8M's `learn.80K` starts with the
  10K-query `learn` of the other BEIR datasets' rule, and its `validation` is the same. BEIR's
  query vectors come with the dataset; MIRACL's topics are encoded with BGE-M3 (all topics, in
  file order, batches of 32).
- **SEISMIC100M, ASTRO100M:** `query` is the series file's last 10K finite records; the base is its
  first 100M finite records and stops before them. Seismic's file contains all-NaN records
  (probably constant series that z-normalization divided by zero); they are skipped.
- **SALD100M:** `query` is 10,000 windows of a held-out subject (see "SALD100M").

## Ground truth

The ground truth is computed with Faiss's exact flat index (`IndexFlatL2` or `IndexFlatIP`,
all threads), one index per metric reused for all splits, as erato's files were. Recomputed
on erato, it is byte-identical to erato's files (checked for GIST1M's L2 and GLOVE1M's
inner-product query ground truth).

`--gt-engine messi` uses `lib/optimizer/daisy_groundtruth` instead: DaiSy's MESSI with
equi-depth breakpoints, inner product through the exact reduction to L2 (each base vector gets
the extra coordinate sqrt(M² − |x|²), each query a 0). It checks every answer (ids, distances,
order) and the first queries of a run against brute force, and resumes a stopped run. It gives
the same distances as Faiss, but at k = 5000 it is much slower: 870 ms per query on SIFT1M
against Faiss's 13.7 ms, because a 5000-th neighbour leaves MESSI almost nothing to prune.

Faiss's times on erato (48 threads, every split, k = 5000): GLOVE1M 0.2 h, GIST1M 0.8 h,
UCASELAW7M 8.9 h, DEEP100M 13.2 h, T2I100M 24.7 h (most of it the 80K learning queries).
The other datasets were not timed.

## What was checked (2026-11-19)

- **Base files:** byte-identical to erato's: SIFT1M, GLOVE1M and GIST1M in full; the first
  1M rows of DEEP100M, T2I100M, UCASELAW7M, HOTPOTQA5M, SEISMIC100M, ASTRO100M and SALD100M,
  and the first 100K of MIRACLJA6M (streamed).
- **Query splits:** byte-identical to erato's files: every split of GLOVE1M, GIST1M,
  DEEP100M, UCASELAW7M and T2I100M; `learn.10K`, `validation.1K` and `query` of HOTPOTQA5M
  and FEVER5M; `learn.4268`, `validation.1K` and `query.860` of MIRACLJA6M. MIRACLRU9M's
  splits were not checked.
- **SIFT1M** (80K rule): `query.10K` is erato's; the first 10K rows of `learn.80K` are erato's
  `learn.10K`; `validation.1K` is the first 1K rows of erato's `validation.10K`. All 82K
  learning, tuning and validation vectors are distinct and none is a query.
- **MSMARCO8M** (80K rule): `learn.80K` and `tuning.1K` are byte-identical to the
  create_datasets.py output in erato's `MSMARCO8M/troubleshoot_80K/`; the first 10K rows of
  `learn.80K`, `validation.1K` and `query.6980` are erato's current files.
- **erato's folders:** SIFT1M and MSMARCO8M there still hold the earlier `learn.10K`
  (and SIFT1M's `validation.10K`) with their ground truth. The script does not make or
  delete these files; on erato it would add the 80K splits next to them.
- **MIRACL's encoding:** the 860 MIRACLJA6M testing queries re-encoded on erato were
  bit-identical; other hardware or library versions can change their last digits, so copy the
  MIRACL query files from erato where that matters.

## SALD100M

SALD is the Southwest University Adult Lifespan Dataset: 494 subjects with one 8-minute
resting-state fMRI scan each (Siemens Trio 3 T, TR 2 s, 64 × 64 × 32 voxels × 242 time
points). The data-series papers (MESSI, SING, Odyssey) use "200M series of length 128,
100 GB" derived from it, but publish neither the file nor the procedure. The script
rebuilds a dataset of that kind:

- It downloads the 10 archives (23 GB, anonymous S3) and reads the resting-state scans
  straight from them (nothing is extracted); they hold 493 scans.
- Each brain voxel's time course (voxel mean above 10% of the scan's highest voxel mean,
  about 32,000 voxels per subject) is cut into windows of 128 points with stride 8, 15 per
  voxel, and each window is z-normalized.
- The first subject read (sub-031274) is held out: 10,000 of its windows, drawn with seed 0,
  are the queries. Consecutive windows of a voxel share 120 of 128 points, so a query from a
  base subject would find near-copies of itself.
- The base is the first 100M windows of the other subjects (about the first 207); all of
  them give 237.3M. The benchmark's SALD has 200M; 100M was chosen so it fits on smaller
  servers.

The result matches the papers' SALD in source, length and size, not necessarily series by
series, and its overlapping windows make it denser in near-duplicates than Seismic or Astro.
