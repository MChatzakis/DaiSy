#!/usr/bin/env python3
"""Downloads and prepares the benchmark datasets of DaiSy on any server, as on the lab servers.

Every dataset goes to DATA_ROOT/<NAME>/ (in .fvecs/.ivecs files every row is an int32
dimension followed by the float32/int32 values):
    base.<n>.fvecs                                    base vectors (or series)
    <split>.<n>.fvecs                                 query splits, never part of the base:
                                                      query (testing), and for the vector
                                                      datasets also learn, validation and
                                                      (some) tuning
    <l2.|innprod.><split>.groundtruth.<n>.k<K>.ivecs  exact neighbours of every query of a split
    <l2.|innprod.><split>.groundtruth.<n>.k<K>.fvecs  their squared L2 distances (ascending) or
                                                      inner products (descending)
    groundtruth_times.csv                             wall-clock seconds of every ground-truth step
    base.<n>.ids.txt, <split>.<n>.ids.txt             (BEIR and MIRACL datasets) the document or
                                                      query id of every row
    query.<n>.qrels.tsv                               (BEIR and MIRACL datasets) the relevance
                                                      judgements of the testing queries
The files are the same as the lab servers' (erato, and stella's /gpfs/scratch/chatzakis/endor/
processed), built by the same rules: those of create_datasets.py (Endor's dataset_organizer.py)
and create_datasets_2.py. On stella, the base, split, id and qrels files of every dataset there
are byte for byte this script's, and so is the ground truth with Faiss 1.9.0 (queries are
searched in batches of 1,000, as there):

  dataset      source                                       splits (rows)                       metrics
  SIFT1M       BIGANN (SIFT1B): first 1M base rows          learn 80K, tuning 1K, validation 1K, L2
                                                            query 10K (public)
  GLOVE1M      GloVe: ann-benchmarks' glove-100 base;       learn 80K, tuning 1K,                inner product
               queries are glove.6B words                   validation 1K, query 10K
  GIST1M       texmex gist.tar.gz                           learn 80K, tuning 1K, validation 1K, L2
                                                            query 1K
  HOTPOTQA5M   BEIR + Cohere embed-english-v3               learn 10K, validation 1K (train),    inner product, L2
                                                            query 7,405 (test)
  FEVER5M      BEIR + Cohere embed-english-v3               learn 10K, validation 1K, query 6,666 inner product, L2
  MIRACLJA6M   MIRACL ja + BGE-M3 (queries encoded here)    learn 4,268, validation 1K,          inner product, L2
                                                            query 860 (dev)
  MSMARCO8M    BEIR + Cohere embed-english-v3               learn 80K, tuning 1K, validation 1K, inner product
                                                            query 6,980 (dev)
  DEEP10M      Yandex Deep1B: first 10M base rows           as DEEP100M                          L2
  DEEP50M      Yandex Deep1B: first 50M base rows           as DEEP100M                          L2
  DEEP100M     Yandex Deep1B: first 100M base rows          learn 80K, tuning 1K, validation 1K, L2
                                                            query 10K (public)
  MIRACLRU9M   MIRACL ru + BGE-M3                           learn 5,312, validation 1K,          inner product, L2
                                                            query 1,252
  UCASELAW7M   big-ann caselaw, single-vector documents     learn 80K, tuning 1K, validation 1K, L2
                                                            query 1K (all from the query file)
  T2I10M       Yandex Text-to-Image: first 10M base rows    as T2I100M                           inner product
  T2I50M       Yandex Text-to-Image: first 50M base rows    as T2I100M                           inner product
  T2I100M      Yandex Text-to-Image: first 100M base rows   learn 80K, tuning 1K, validation 1K, inner product
                                                            query 10K (of the public 100K)
  SEISMIC100M  seismic series file (not public)             query 10K (its queries file)         none
  ASTRO100M    astronomy series file (not public)           query 100 (its queries file)         none
  SALD100M     SALD series file (not public)                query 100 (its queries file)         none
  RANDOM100M   random-walk series file (not public)         query 1K (its queries file)          none
The data series have no learning splits and no ground truth: only base and query. Their base
is the first 100M finite records of the series file (Seismic's has all-NaN records, which are
skipped), and their queries are the whole queries file that comes with it.
learn/tuning/validation come from each dataset's own learning queries: by row
(create_datasets.py's rules, LEARN_FILE_STARTS) or as seeded samples of the judged train
queries (BEIR, MIRACL). The series datasets need their .bin files (raw little-endian float32
records, no header): under their usual names in --series-dir or one folder below it, or
anywhere with --series-file NAME=PATH and --series-queries NAME=PATH, or downloaded with
--series-url; or their prepared DATA_ROOT/<NAME>/ directory copied from another server.

Existing files are never replaced or changed: a file that is already there with the expected
shape is kept, one with another shape stops the run, and the times of a run that finds a
groundtruth_times.csv go to a new groundtruth_times.<date>.csv. So DATA_ROOT can be a folder
that other projects use.

The ground truth is computed with Faiss (exact flat index), as the lab server's files were;
--gt-engine messi uses DaiSy's MESSI instead (lib/optimizer/daisy_groundtruth), which is
exact too but much slower at k = 5000.

Raw downloads go to RAW_DIR. A step whose output is already there (checked by shape) is
skipped, and downloads resume, so a stopped run continues when started again.

Requirements: numpy, faiss-cpu; tqdm (optional); pyarrow and huggingface_hub for the Hugging
Face datasets; torch and transformers for MIRACL's queries (scripts/datasets/requirements.txt).

Usage:
  python3 scripts/datasets/prepare_datasets.py --data-root DIR --raw-dir DIR [DATASET ...]
  python3 scripts/datasets/prepare_datasets.py --data-root DIR --list
"""
import argparse
import glob
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # progress bars are optional
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else _NoBar()

    class _NoBar:
        def update(self, n):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YANDEX = "https://storage.yandexcloud.net/yandex-research/ann-datasets"
BIGANN = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann"
TEXMEX = "ftp://ftp.irisa.fr/local/texmex/corpus"
COHERE_REPO = "CohereLabs/beir-embed-english-v3"
BGE_M3_MIRACL_REPO = "hanhainebula/bge-m3_miracl_2cr"
MIRACL_REPO = "miracl/miracl"

SEED = 42
GT_PREFIXES = {"l2": "l2.", "ip": "innprod."}
# create_datasets.py's splits: learn 80K (its first 10K are the earlier learn.10K), tuning 1K,
# validation 1K, taken from a learning file starting at these rows; tuning and the 70K added
# learning rows skip every vector already in a split. LEARN_ROWS rows of a learning file are
# read (downloaded) for them.
ROW_SPLITS = {"learn": 80_000, "tuning": 1_000, "validation": 1_000}
LEARN_FILE_STARTS = {"learn": 0, "validation": 10_000, "tuning": 20_000, "more_learn": 30_000}
CASELAW_STARTS = {"learn": 0, "validation": 10_000, "tuning": 11_000, "query": 12_000, "more_learn": 13_000}
LEARN_ROWS = 1_000_000

# Deep1B and Text-to-Image (T2I-1B): each of their datasets is the first n base rows of the 1B
# file, with the same query splits (only the rows used are downloaded, download_rows). T2I's
# testing queries are the first 10K of its 100K public ones.
DEEP1B = dict(kind="fbin", raw="DEEP1B", url=f"{YANDEX}/DEEP", base="base.1B.fbin", query="query.public.10K.fbin",
              learn="learn.350M.fbin", layout="row_splits", splits=ROW_SPLITS, nq=10_000, d=96, metrics=["l2"])
T2I1B = dict(kind="fbin", raw="t2i", url=f"{YANDEX}/T2I", base="base.1B.fbin", query="query.public.100K.fbin",
             learn="query.learn.50M.fbin", layout="row_splits", splits=ROW_SPLITS, nq=10_000, d=200, metrics=["ip"])

# n: base vectors, label: the size in base.<label>.fvecs, nq: testing queries, d: dimension,
# splits: query splits besides "query" (count each), layout: how they are made, metrics:
# ground-truth metrics, raw: the dataset's folder under RAW_DIR.
DATASETS = {
    "SIFT1M": dict(kind="fbin", raw="BIGANN", url=BIGANN, base="base.1B.u8bin", query="query.public.10K.u8bin",
                   learn="learn.100M.u8bin", layout="row_splits", splits=ROW_SPLITS,
                   n=1_000_000, label="1M", nq=10_000, d=128, metrics=["l2"]),
    "GLOVE1M": dict(kind="glove", raw="GloVe/misc",
                    base_zip="glove.twitter.27B.zip", base_url="https://downloads.cs.stanford.edu/nlp/data/glove.twitter.27B.zip",
                    words_zip="glove.6B.zip", words_url="https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip",
                    layout="row_splits", splits=ROW_SPLITS,
                    n=1_183_514, label="1183514", nq=10_000, d=100, metrics=["ip"]),
    "GIST1M": dict(kind="texmex", raw="GIST1M", archive="gist.tar.gz", md5="31185e0f00854f74d27e8ad8d52628a9",
                   base="gist/gist_base.fvecs", query="gist/gist_query.fvecs", learn="gist/gist_learn.fvecs",
                   layout="row_splits", splits=ROW_SPLITS,
                   n=1_000_000, label="1M", nq=1_000, d=960, metrics=["l2"]),
    "HOTPOTQA5M": dict(kind="beir", beir="hotpotqa", qrels="test", learn_pool=85_000, layout="qrels",
                       splits={"learn": 10_000, "validation": 1_000},
                       n=5_233_329, label="5233329", nq=7_405, d=1024, metrics=["ip", "l2"]),
    "FEVER5M": dict(kind="beir", beir="fever", qrels="test", learn_pool=109_810, layout="qrels",
                    splits={"learn": 10_000, "validation": 1_000},
                    n=5_416_568, label="5416568", nq=6_666, d=1024, metrics=["ip", "l2"]),
    "MIRACLJA6M": dict(kind="miracl", lang="ja", learn_pool=5_268, layout="qrels",
                       splits={"learn": 4_268, "validation": 1_000},
                       n=6_953_614, label="6953614", nq=860, d=1024, metrics=["ip", "l2"]),
    # Inner product only, as create_datasets.py made it (unit-norm vectors: L2 gives the same
    # neighbours).
    "MSMARCO8M": dict(kind="beir", beir="msmarco", qrels="dev", learn_pool=502_939, layout="qrels",
                      splits=ROW_SPLITS,
                      n=8_841_823, label="8841823", nq=6_980, d=1024, metrics=["ip"]),
    "DEEP10M": dict(**DEEP1B, n=10_000_000, label="10M"),
    "DEEP50M": dict(**DEEP1B, n=50_000_000, label="50M"),
    "DEEP100M": dict(**DEEP1B, n=100_000_000, label="100M"),
    "MIRACLRU9M": dict(kind="miracl", lang="ru", learn_pool=6_312, layout="qrels",
                       splits={"learn": 5_312, "validation": 1_000},
                       n=9_543_918, label="9543918", nq=1_252, d=1024, metrics=["ip", "l2"]),
    "UCASELAW7M": dict(kind="caselaw", raw="caselaw", url="https://comp21storage.z5.web.core.windows.net/caselaw",
                       base="caselaw_base_embeddings.bin", query="caselaw_query_embeddings.bin",
                       query_pool=89_812, layout="row_splits", splits=ROW_SPLITS,
                       n=7_414_023, label="7M", nq=1_000, d=1536, metrics=["l2"]),
    "T2I10M": dict(**T2I1B, n=10_000_000, label="10M"),
    "T2I50M": dict(**T2I1B, n=50_000_000, label="50M"),
    "T2I100M": dict(**T2I1B, n=100_000_000, label="100M"),
    # Data series (raw float32 records, z-normalized, no header), from the data-series
    # benchmarks; none is public. The base is the series file's first n finite records (Seismic
    # has all-NaN records, probably constant series that z-normalization divided by zero); the
    # queries are the queries file that comes with it, a separate file. No ground truth.
    # Seismic's file has 109,999,999 records, Astro's 277,443,827, SALD's 899,065,213 and the
    # random-walk file 104,857,600.
    "SEISMIC100M": dict(kind="series", file="data_size100M_seismic_len256_znorm.bin",
                        queries="queries_size10K_seismic_len256_znorm.bin",
                        layout="none", splits={}, n=100_000_000, label="100M", nq=10_000, d=256, metrics=[]),
    "ASTRO100M": dict(kind="series", file="data_size270M_astronomy_len256_znorm.bin",
                      queries="queries_ctrl100_astronomy_len256_znorm.bin",
                      layout="none", splits={}, n=100_000_000, label="100M", nq=100, d=256, metrics=[]),
    "SALD100M": dict(kind="series", file="data_size899M_sald_len128_znorm.bin",
                     queries="queries_ctrl100_sald_len128_znorm.bin",
                     layout="none", splits={}, n=100_000_000, label="100M", nq=100, d=128, metrics=[]),
    "RANDOM100M": dict(kind="series", file="data_size104857600random100GB_len256_znorm.bin",
                       queries="queries_ctrl1000_random_len256_znorm.bin",
                       layout="none", splits={}, n=100_000_000, label="100M", nq=1_000, d=256, metrics=[]),
}


# ---------------------------------------------------------------------------- files

def size_label(n):
    return f"{n // 1000}K" if n % 1000 == 0 else str(n)


def split_counts(name):
    """{split: queries} of a dataset, testing queries included."""
    info = DATASETS[name]
    return {**info["splits"], "query": info["nq"]}


def base_path(root, name):
    return f"{root}/{name}/base.{DATASETS[name]['label']}.fvecs"


def split_path(root, name, split):
    return f"{root}/{name}/{split}.{size_label(split_counts(name)[split])}.fvecs"


def query_path(root, name):
    return split_path(root, name, "query")


def gt_prefix(root, name, split, metric, k):
    return f"{root}/{name}/{GT_PREFIXES[metric]}{split}.groundtruth.{size_label(split_counts(name)[split])}.k{k}"


def vecs_shape(path):
    """(rows, dim) of an .fvecs/.ivecs file, or None if missing or malformed."""
    if not os.path.exists(path) or os.path.getsize(path) < 4:
        return None
    dim = int(np.fromfile(path, dtype='<i4', count=1)[0])
    size = os.path.getsize(path)
    return (size // (4 * (dim + 1)), dim) if dim > 0 and size % (4 * (dim + 1)) == 0 else None


def needs_writing(path, shape):
    """True if `path` is missing, False if it is there with `shape`. A file there with another
    shape stops the run: existing files are never replaced (other projects may use them)."""
    have = vecs_shape(path)
    if have == shape:
        return False
    if os.path.exists(path):
        sys.exit(f"{path} exists with shape {have}, not {shape}. Existing files are never replaced: "
                 f"move it away to rebuild it")
    return True


def refuse_existing(path):
    if os.path.exists(path):
        sys.exit(f"{path} already exists; existing files are never replaced")


def bin_type(path):
    """Element type of a big-ann-benchmarks binary file: .fbin float32, .u8bin uint8."""
    return np.uint8 if path.endswith(".u8bin") else np.dtype('<f4')


def open_vectors(path):
    """A read-only (rows, dim) view of an .fvecs, .fbin or .u8bin file."""
    if path.endswith((".fbin", ".u8bin")):
        rows, dim = np.fromfile(path, dtype='<u4', count=2)
        return np.memmap(path, dtype=bin_type(path), mode='r', offset=8, shape=(int(rows), int(dim)))
    dim = int(np.fromfile(path, dtype='<i4', count=1)[0])
    return np.memmap(path, dtype='<f4', mode='r').reshape(-1, dim + 1)[:, 1:]


def append_vecs(f, vectors, dtype='<f4'):
    vectors = np.asarray(vectors)
    records = np.empty((len(vectors), vectors.shape[1] + 1), dtype=dtype)
    records.view('<i4')[:, 0] = vectors.shape[1]
    records[:, 1:] = vectors
    records.tofile(f)


def write_vecs(path, vectors, dtype='<f4'):
    """Write a whole array to a new file, through path.tmp."""
    refuse_existing(path)
    with open(path + ".tmp", "wb") as f:
        append_vecs(f, vectors, dtype)
    os.replace(path + ".tmp", path)


def copy_rows(batches, rows, dim, desc, f=None, exact=False):
    """Take `rows` rows from an iterable of (k, dim) arrays or of (ids, (k, dim) array) pairs,
    appending them to the open file f (if given), and return the ids of the pairs, in order.
    With exact, the source must have exactly `rows` rows and is read to its end (MIRACL's
    archive lists its ids after its vectors); otherwise it may have more, which are not read."""
    taken, ids = 0, []
    with tqdm(total=rows, desc=desc, unit="vec") as bar:
        for batch in batches:
            if isinstance(batch, tuple):
                batch_ids, batch = batch
                ids.extend(batch_ids or [])
            batch = np.asarray(batch, dtype=np.float32)
            if exact and taken + len(batch) > rows:
                raise ValueError(f"{desc}: the source has more than {rows} rows")
            batch = batch[:rows - taken]
            if batch.ndim != 2 or batch.shape[1] != dim or not np.all(np.isfinite(batch)):
                raise ValueError(f"{desc}: a batch of shape {batch.shape} or with non-finite values")
            if f is not None:
                append_vecs(f, batch)
            taken += len(batch)
            bar.update(len(batch))
            if taken == rows and not exact:
                break
    if taken != rows:
        raise ValueError(f"{desc}: the source has only {taken} of {rows} rows")
    if ids and len(ids) != rows:
        raise ValueError(f"{desc}: {len(ids)} ids for {rows} rows")
    return ids


def write_fvecs(path, batches, rows, dim, desc, exact=False):
    """Write exactly `rows` rows of copy_rows' source to a new file, through path.tmp; return their ids."""
    refuse_existing(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        ids = copy_rows(batches, rows, dim, desc, f, exact)
    os.replace(path + ".tmp", path)
    print(f">> wrote {path}: {rows} x {dim}", flush=True)
    return ids


def ids_path(path):
    """The ids file of an .fvecs file: one id per line, of the vector on the same row."""
    return path.removesuffix(".fvecs") + ".ids.txt"


def write_lines(path, lines):
    """Write lines to a new file, through path.tmp."""
    refuse_existing(path)
    with open(path + ".tmp", "w") as f:
        f.writelines(f"{line}\n" for line in lines)
    os.replace(path + ".tmp", path)
    print(f">> wrote {path}", flush=True)


def chunks(vectors, rows=200_000):
    for start in range(0, len(vectors), rows):
        yield vectors[start:start + rows]


# ---------------------------------------------------------------------------- downloads

def download(url, path):
    """Download a whole file to `path`, through <path>.part, which a rerun continues (HTTP)."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    have = os.path.getsize(part) if os.path.exists(part) and url.startswith("http") else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as error:
        if error.code != 416:
            raise
        os.replace(part, path)  # nothing left to fetch: .part is the whole file
        return path
    with response:
        if have and response.status != 206:
            have = 0  # the server sends the whole file again
        length = response.headers.get("Content-Length")
        total = have + int(length) if length else None
        with open(part, "ab" if have else "wb") as f, \
                tqdm(total=total, initial=have, unit="B", unit_scale=True, desc=os.path.basename(path)) as bar:
            while chunk := response.read(1 << 24):
                f.write(chunk)
                bar.update(len(chunk))
    if total is not None and os.path.getsize(part) != total:
        sys.exit(f"The download of {url} stopped early; run again to continue it")
    os.replace(part, path)
    print(f">> downloaded {path}", flush=True)
    return path


def download_rows(url, path, rows):
    """The first `rows` vectors of a remote .fbin/.u8bin file (the 1B-vector files are 128 to
    800 GB), as <name>.first.<ext>, which a later call for more rows extends. Its header counts
    only completely written rows, so a stopped download continues after them."""
    if os.path.exists(path):
        return path  # the whole file is already here
    stem, ext = os.path.splitext(path)
    first = f"{stem}.first{ext}"
    item = np.dtype(bin_type(path)).itemsize
    have = int(np.fromfile(first, dtype='<u4', count=1)[0]) if os.path.exists(first) else 0
    if have >= rows:
        return first
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-7"})) as response:
        count, dim = np.frombuffer(response.read(), dtype='<u4')
    rows = min(rows, int(count))
    start, end = 8 + item * int(dim) * have, 8 + item * int(dim) * rows
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
    with open(first, "r+b" if have else "w+b") as f, urllib.request.urlopen(request) as response:
        if response.status != 206:
            sys.exit(f"{url} does not send parts of files; download it whole to {path}")
        f.write(np.array([have, dim], dtype='<u4').tobytes())
        f.truncate(start)
        f.seek(start)
        with tqdm(total=end - start, unit="B", unit_scale=True, desc=f"{os.path.basename(first)} rows {have}-{rows}") as bar:
            while chunk := response.read(1 << 24):
                f.write(chunk)
                bar.update(len(chunk))
        if f.tell() != end:
            sys.exit(f"The download of {url} stopped early; run again to continue it")
        f.seek(0)
        f.write(np.array([rows, dim], dtype='<u4').tobytes())  # count the rows only now
    print(f">> downloaded the first {rows} vectors of {url} to {first}", flush=True)
    return first


def hf_files(hf_dir, repo, patterns, complete=False):
    """Local copies of a Hugging Face dataset repo's files matching the patterns, downloaded if
    no file matches a pattern, or, with complete, always checked against the repo (files already
    whole are skipped; a stopped download is finished)."""
    patterns = [patterns] if isinstance(patterns, str) else list(patterns)
    local = f"{hf_dir}/{repo}"
    if complete or not all(glob.glob(f"{local}/{p}") for p in patterns):
        from huggingface_hub import snapshot_download
        print(f">> downloading {repo}: {patterns}", flush=True)
        snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=patterns, local_dir=local, max_workers=8)
    files = [path for p in patterns for path in sorted(glob.glob(f"{local}/{p}"))]
    if not files:
        raise FileNotFoundError(f"no files match {patterns} in {local}")
    return files


def md5sum(path):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def texmex_files(info, raw):
    """Download and unpack a texmex archive (its .fvecs files) into its raw folder, once."""
    folder = f"{raw}/{info['raw']}"
    if all(os.path.exists(f"{folder}/{info[key]}") for key in ("base", "query", "learn")):
        return folder
    archive = download(f"{TEXMEX}/{info['archive']}", f"{folder}/{info['archive']}")
    if info["md5"] and md5sum(archive) != info["md5"]:
        os.remove(archive)
        sys.exit(f"{archive} does not match its MD5 sum; it was deleted, run again to download it")
    unpacking = f"{folder}/unpacking"
    shutil.rmtree(unpacking, ignore_errors=True)
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(unpacking, filter="data")
        except TypeError:  # Python < 3.12 has no filter argument
            tar.extractall(unpacking)
    for entry in os.listdir(unpacking):
        os.replace(f"{unpacking}/{entry}", f"{folder}/{entry}")
    os.rmdir(unpacking)
    os.remove(archive)
    return folder


# ---------------------------------------------------------------------------- query splits

def row_splits(pool, starts, query):
    """{split: vectors} from the rows of a learning file (create_datasets.py's row_splits):
    learn keeps the 10K rows from starts["learn"] and validation the 1K rows from
    starts["validation"]; tuning (1K) and the 70K added learning rows are the first rows from
    their start whose vectors are in no split yet (the queries included)."""
    learn = np.array(pool[starts["learn"]:starts["learn"] + 10_000])
    validation = np.array(pool[starts["validation"]:starts["validation"] + 1_000])
    taken = {vector.tobytes() for vectors in (learn, validation, query) for vector in vectors}

    def new_rows(row, count):
        picked = []
        while len(picked) < count:
            vector = np.array(pool[row])
            if vector.tobytes() not in taken:
                taken.add(vector.tobytes())
                picked.append(vector)
            row += 1
        return np.array(picked)

    tuning = new_rows(starts["tuning"], 1_000)
    learn = np.concatenate([learn, new_rows(starts["more_learn"], 70_000)])
    return {"learn": learn, "tuning": tuning, "validation": validation, "query": query}


def sampled_splits(info, pool, query_ids, vector_of, qrels):
    """Splits of a dataset whose queries are judged in qrels: the testing queries
    are the judged testing ids; the other splits are seeded samples of the learning pool
    (sorted ids), drawn in this order, each sorted: learn (its first 10K), validation, then,
    for an 80K learn, tuning and the other 70K learning queries (create_datasets_2.py, and
    create_datasets.py's msmarco_query_splits). So a larger learn starts with the smaller one,
    and validation stays the same."""
    if len(query_ids) != info["nq"] or len(pool) != info["learn_pool"] or not set(query_ids).isdisjoint(pool):
        raise ValueError(f"unexpected qrels: {len(query_ids)} testing and {len(pool)} learning queries")
    counts = info["splits"]
    first = min(counts["learn"], 10_000)
    draws = [("learn", first), ("validation", counts["validation"])]
    draws += [("tuning", counts["tuning"])] if "tuning" in counts else []
    draws += [("more_learn", counts["learn"] - first)] if counts["learn"] > first else []
    rng = np.random.default_rng(SEED)
    remaining, rows = np.arange(len(pool)), {}
    for split, count in draws:
        rows[split] = np.sort(rng.choice(len(pool) if not rows else remaining, count, replace=False))
        remaining = np.setdiff1d(remaining, rows[split])
    rows["learn"] = np.concatenate([rows["learn"], rows.pop("more_learn", [])]).astype(np.int64)
    ids = {split: [pool[r] for r in split_rows] for split, split_rows in rows.items()}
    ids["query"] = query_ids
    splits = Splits({split: np.array([vector_of[qid] for qid in split_ids]) for split, split_ids in ids.items()})
    splits.ids = ids
    splits.qrels = qrels
    return splits


class Splits(dict):
    """{split: vectors}, with, for the datasets judged in qrels, the query id of every row
    ({split: ids}) and the lines of the testing queries' qrels.tsv."""
    ids = {}
    qrels = None


def qrels_lines(query_ids, corpus_ids, scores):
    """The lines of a query.<n>.qrels.tsv file (create_datasets.py's and create_datasets_2.py's)."""
    return ["query_id\tcorpus_id\tscore"] + [f"{q}\t{c}\t{s}" for q, c, s in zip(query_ids, corpus_ids, scores)]


# ---------------------------------------------------------------------------- sources by kind
# Each kind gives base_batches(info, ctx) (arrays of rows, in order, or (ids, rows) pairs for
# the datasets with ids) and splits(info, ctx) ({split: vectors} for "query" and the dataset's
# other splits, or Splits).

def texmex_base(info, ctx):
    return chunks(open_vectors(f"{texmex_files(info, ctx.raw)}/{info['base']}"))


def texmex_splits(info, ctx):
    folder = texmex_files(info, ctx.raw)
    query = np.array(open_vectors(f"{folder}/{info['query']}")[:info["nq"]])
    return row_splits(open_vectors(f"{folder}/{info['learn']}")[:LEARN_ROWS], LEARN_FILE_STARTS, query)


def fbin_rows(info, ctx, key, rows):
    return download_rows(f"{info['url']}/{info[key]}", f"{ctx.raw}/{info['raw']}/{info[key]}", rows)


def fbin_base(info, ctx):
    return chunks(open_vectors(fbin_rows(info, ctx, "base", info["n"]))[:info["n"]])


def fbin_splits(info, ctx):
    # As float32 (SIFT's files are uint8), so row_splits compares learning rows and queries alike.
    query = np.asarray(open_vectors(fbin_rows(info, ctx, "query", info["nq"]))[:info["nq"]], dtype=np.float32)
    pool = np.asarray(open_vectors(fbin_rows(info, ctx, "learn", LEARN_ROWS))[:LEARN_ROWS], dtype=np.float32)
    return row_splits(pool, LEARN_FILE_STARTS, query)


def glove_zip(info, ctx, key):
    return download(info[f"{key}_url"], f"{ctx.raw}/{info['raw']}/{info[f'{key}_zip']}")


def glove_base(info, ctx):
    # As ann-benchmarks' glove() and train_test_split(test_size=10000, random_state=1): each
    # line's values after the word, as float64, then all but the first 10K permuted rows.
    with zipfile.ZipFile(glove_zip(info, ctx, "base")) as z, z.open("glove.twitter.27B.100d.txt") as f:
        vectors = np.array([np.array([float(x) for x in line.strip().split()[1:]]) for line in f])
    train = np.random.RandomState(1).permutation(len(vectors))[10_000:]
    return chunks(vectors[train].astype(np.float32))


def glove_splits(info, ctx):
    """As create_datasets.py's glove_files: the earlier learn, validation and query rows of
    glove.6B.100d.txt (docs/datasets/glove1m_query_rows.txt), then a learning file of the
    earlier learn and validation words followed by all other words in seeded order."""
    with zipfile.ZipFile(glove_zip(info, ctx, "words")) as z, z.open("glove.6B.100d.txt") as f:
        words = np.array([[float(x) for x in line.decode("utf8").rstrip("\n").split(" ")[1:]] for line in f],
                         dtype=np.float32)
    rows = {"learn": [], "validation": [], "query": []}
    with open(f"{REPO}/docs/datasets/glove1m_query_rows.txt") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                split, row = line.split()
                rows[split].append(int(row))
    earlier = {split: words[split_rows] for split, split_rows in rows.items()}
    taken = {vector.tobytes() for vectors in earlier.values() for vector in vectors}
    others = np.array([vector for vector in words if vector.tobytes() not in taken])
    others = others[np.random.default_rng(SEED).permutation(len(others))]
    pool = np.concatenate([earlier["learn"], earlier["validation"], others])
    return row_splits(pool, LEARN_FILE_STARTS, earlier["query"])


def open_caselaw(path):
    """(view of all vectors, rows of the single-vector documents) of a caselaw file: the
    document count, dimension and vector count (uint32), each document's vector count, then
    all vectors (float32), document after document."""
    with open(path, 'rb') as f:
        documents, dim, rows = map(int, np.fromfile(f, dtype='<u4', count=3))
        counts = np.fromfile(f, dtype='<u4', count=documents)
    offset = 12 + 4 * documents
    if os.path.getsize(path) != offset + 4 * rows * dim or counts.sum() != rows:
        raise ValueError(f"{path} does not match its header")
    first_rows = np.cumsum(counts, dtype=np.int64) - counts
    return np.memmap(path, dtype='<f4', mode='r', offset=offset, shape=(rows, dim)), first_rows[counts == 1]


def caselaw_file(info, ctx, key):
    return download(f"{info['url']}/{info[key]}", f"{ctx.raw}/{info['raw']}/{info[key]}")


def caselaw_base(info, ctx):
    vectors, single = open_caselaw(caselaw_file(info, ctx, "base"))
    for start in range(0, len(single), 100_000):
        yield vectors[single[start:start + 100_000]]


def caselaw_splits(info, ctx):
    vectors, single = open_caselaw(caselaw_file(info, ctx, "query"))
    if len(single) != info["query_pool"]:
        raise ValueError(f"expected {info['query_pool']} single-vector query documents, found {len(single)}")
    pool = vectors[single]
    query = np.array(pool[CASELAW_STARTS["query"]:CASELAW_STARTS["query"] + info["nq"]])
    return row_splits(pool, CASELAW_STARTS, query)


def parquet_embeddings(files, d, columns=("_id", "emb")):
    """(ids, vectors) batches of embedding parquet files, in file order."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000, columns=list(columns)):
            embeddings = batch.column(columns[1])
            if not np.all(pc.list_value_length(embeddings).to_numpy() == d):
                raise ValueError(f"embeddings of another dimension than {d} in {path}")
            vectors = embeddings.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(-1, d)
            yield batch.column(columns[0]).to_pylist(), vectors


def beir_base(info, ctx):
    """(document ids, vectors) of the corpus, in file order."""
    yield from parquet_embeddings(hf_files(ctx.hf, COHERE_REPO, f"{info['beir']}/corpus/*.parquet"), info["d"])


def beir_splits(info, ctx):
    import pyarrow.parquet as pq
    b = info["beir"]
    [testing] = hf_files(ctx.hf, COHERE_REPO, f"{b}/qrels/{info['qrels']}.parquet")
    [train] = hf_files(ctx.hf, COHERE_REPO, f"{b}/qrels/train.parquet")
    qrels = pq.read_table(testing, columns=["query_id", "corpus_id", "score"])
    query_ids = sorted(set(qrels.column("query_id").to_pylist()))
    pool = sorted(set(pq.read_table(train, columns=["query_id"]).column("query_id").to_pylist()))
    wanted, vector_of = set(query_ids) | set(pool), {}
    for ids, vectors in parquet_embeddings(hf_files(ctx.hf, COHERE_REPO, f"{b}/queries/*.parquet"), info["d"]):
        for qid, vector in zip(ids, vectors):
            if qid in wanted:
                vector_of[qid] = vector
    lines = qrels_lines(*(qrels.column(c).to_pylist() for c in ("query_id", "corpus_id", "score")))
    return sampled_splits(info, pool, query_ids, vector_of, lines)


def faiss_flat_vectors(fileobj, d, batch_rows=50_000):
    """(None, vectors) batches of a Pyserini Faiss flat index inside a .tar.gz stream
    (<name>/index), and (document ids, no vectors) once its <name>/docid list is read."""
    found = False
    with tarfile.open(fileobj=fileobj, mode="r|gz") as archive:
        for member in archive:
            name = os.path.basename(member.name)
            if name == "docid":
                yield archive.extractfile(member).read().decode("utf-8").split(), np.empty((0, d), np.float32)
            if name != "index":
                continue
            found = True
            index = archive.extractfile(member)
            # IndexFlat header: "IxFI", d (int32), ntotal (int64), two unused int64, is_trained
            # (uint8), metric (int32, 0 = inner product), then the float count (int64).
            magic, dim, ntotal, _, _, _, metric, count = struct.unpack("<4siqqqBiq", index.read(45))
            if magic != b"IxFI" or dim != d or count != ntotal * d:
                raise ValueError(f"not a {d}-dimensional Faiss flat index")
            for start in range(0, ntotal, batch_rows):
                rows = min(batch_rows, ntotal - start)
                buffer = index.read(rows * d * 4)
                if len(buffer) != rows * d * 4:
                    raise ValueError("truncated Faiss index")
                yield None, np.frombuffer(buffer, dtype='<f4').reshape(rows, d)
    if not found:
        raise ValueError("no Faiss index in the archive")


def miracl_base(info, ctx):
    [archive] = hf_files(ctx.hf, BGE_M3_MIRACL_REPO, f"dense/{info['lang']}.tar.gz")
    with open(archive, "rb") as f:
        yield from faiss_flat_vectors(f, info["d"])


def read_topics(path):
    with open(path, encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("\t", 1) for line in f if line.strip()]
    return [row[0] for row in rows], [row[1] for row in rows]


def miracl_splits(info, ctx):
    """Testing: the dev queries (ids of the dev qrels). Learning pool: the train, test-a and
    test-b topics. All topics are encoded, in file order and batches of 32, as the BGE-M3
    authors' MIRACL runs do (Pyserini's AutoQueryEncoder, --pooling cls --l2-norm)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    lang = info["lang"]
    base = f"miracl-v1.0-{lang}"
    [dev] = hf_files(ctx.hf, MIRACL_REPO, f"{base}/qrels/qrels.{base}-dev.tsv")
    with open(dev, encoding="utf-8") as f:
        # TREC lines: query_id Q0 corpus_id score (judged non-relevant rows at score 0)
        judged = [line.split() for line in f if line.strip()]
    if any(len(row) != 4 for row in judged):
        raise ValueError(f"malformed TREC qrels line in {dev}")
    query_ids = sorted({row[0] for row in judged})
    lines = qrels_lines([row[0] for row in judged], [row[2] for row in judged], [int(row[3]) for row in judged])
    pool = sorted({qid for split in ("train", "test-a", "test-b")
                   for qid in read_topics(hf_files(ctx.hf, MIRACL_REPO, f"{base}/topics/topics.{base}-{split}.tsv")[0])[0]})
    topic_ids, texts = [], []
    for path in hf_files(ctx.hf, MIRACL_REPO, f"{base}/topics/*.tsv"):
        ids, path_texts = read_topics(path)
        topic_ids += ids
        texts += path_texts
    torch.set_num_threads(ctx.threads)
    cache = f"{ctx.hf}/models"
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", cache_dir=cache)
    model = AutoModel.from_pretrained("BAAI/bge-m3", cache_dir=cache).eval()
    encoded = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), 32), desc=f"encoding MIRACL {lang} topics"):
            inputs = tokenizer(texts[start:start + 32], padding=True, truncation=True, return_tensors="pt")
            first_token = model(**inputs).last_hidden_state[:, 0]
            encoded.append(torch.nn.functional.normalize(first_token, dim=-1).numpy())
    vectors = np.concatenate(encoded).astype(np.float32)
    wanted, vector_of = set(query_ids) | set(pool), {}
    for qid, vector in zip(topic_ids, vectors):
        if qid in wanted:
            if qid in vector_of:
                raise ValueError(f"query {qid} appears twice in the MIRACL {lang} topics")
            vector_of[qid] = vector
    return sampled_splits(info, pool, query_ids, vector_of, lines)


def series_records(info, ctx, key, at_least):
    """A series dataset's base file (key "file") or queries file (key "queries") as a
    read-only (records, d) float32 view: raw records, no header (the data-series .bin
    format). Found through --series-file / --series-queries NAME=PATH, else under its usual
    name in --series-dir or one folder below it, else downloaded from --series-url."""
    name = next(n for n, i in DATASETS.items() if i is info)
    override = ctx.series_files.get((name, key))
    candidates = [override] if override else \
        [f"{ctx.series_dir}/{info[key]}"] + sorted(glob.glob(f"{ctx.series_dir}/*/{info[key]}"))
    path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    if not os.path.exists(path):
        if not ctx.series_url or override:
            option = "--series-file" if key == "file" else "--series-queries"
            raise FileNotFoundError(f"{path} is missing. It is not public: copy it from a server that has it (or pass "
                                    f"{option} {name}=PATH, --series-dir or --series-url), or copy the prepared "
                                    f"{ctx.data}/{name}/ directory instead.")
        download(f"{ctx.series_url}/{info[key]}", path)
    size, record = os.path.getsize(path), 4 * info["d"]
    if size % record or size // record < at_least:
        raise FileNotFoundError(f"{path} cannot be {name}'s {key} file: it has {size:,} bytes, and {name} needs "
                                f"at least {at_least:,} float32 records of {info['d']} values")
    print(f">> {name} {'base' if key == 'file' else 'queries'}: {path} ({size // record:,} records)", flush=True)
    return np.memmap(path, dtype='<f4', mode='r', shape=(size // record, info["d"]))


def series_base(info, ctx):
    """The series file's first n finite records (write_fvecs takes n and fails if there are
    fewer); records with a non-finite value are skipped."""
    skipped = 0
    try:
        for block in chunks(series_records(info, ctx, "file", info["n"])):
            block = np.asarray(block)
            good = block[np.isfinite(block).all(axis=1)]
            skipped += len(block) - len(good)
            yield good
    finally:
        if skipped:
            print(f">> {info['file']}: skipped {skipped} records with non-finite values", flush=True)


def series_splits(info, ctx):
    """The queries: the first nq records of the queries file (write_fvecs rejects non-finite ones)."""
    return {"query": np.array(series_records(info, ctx, "queries", info["nq"])[:info["nq"]])}


SOURCES = {
    "texmex": (texmex_base, texmex_splits), "fbin": (fbin_base, fbin_splits),
    "glove": (glove_base, glove_splits), "caselaw": (caselaw_base, caselaw_splits),
    "beir": (beir_base, beir_splits), "miracl": (miracl_base, miracl_splits),
    "series": (series_base, series_splits),
}


# ---------------------------------------------------------------------------- ground truth

def log_time(ctx, name, step, seconds, split="", metric="", queries=""):
    """Record one timed ground-truth step in DATA_ROOT/<name>/groundtruth_times.csv, or, if an
    earlier run left that file, in a new groundtruth_times.<date>.csv: existing files are never
    changed. A run appends its own rows to the file it started."""
    path = ctx.time_logs.get(name)
    if path is None:
        path = f"{ctx.data}/{name}/groundtruth_times.csv"
        if os.path.exists(path):
            path = f"{ctx.data}/{name}/groundtruth_times.{datetime.now().strftime('%Y%m%dT%H%M%S')}.csv"
            refuse_existing(path)
        ctx.time_logs[name] = path
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write("date,host,library,threads,k,step,split,metric,queries,seconds\n")
        f.write(f"{datetime.now().isoformat(timespec='seconds')},{os.uname().nodename},{ctx.gt_engine},{ctx.threads},"
                f"{ctx.gt_k},{step},{split},{metric},{queries},{seconds:.1f}\n")
    print(f">> {name} {step} {split} {metric}: {seconds:.1f} s", flush=True)


def faiss_index(path, metric, threads, d):
    """An exact flat Faiss index of a base file, added in chunks (no second copy in memory)."""
    import faiss
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexFlatL2(d) if metric == "l2" else faiss.IndexFlatIP(d)
    for chunk in tqdm(chunks(open_vectors(path), 1_000_000), desc="building the index"):
        index.add(np.ascontiguousarray(chunk, dtype=np.float32))
    return index


def groundtruth(name, ctx):
    """Exact top-k of every split still missing it, for each metric of the dataset."""
    info = DATASETS[name]
    counts = split_counts(name)
    splits = [s for s in (ctx.gt_splits or counts) if s in counts]
    for metric in info["metrics"]:
        missing = [s for s in splits
                   if any([needs_writing(f"{gt_prefix(ctx.data, name, s, metric, ctx.gt_k)}.{ext}", (counts[s], ctx.gt_k))
                           for ext in ("ivecs", "fvecs")])]
        if not missing:
            print(f">> keeping the {metric} ground truth of {name}")
            continue
        print(f">> {name} {metric} ground truth with {ctx.gt_engine}, k={ctx.gt_k}: {missing}", flush=True)
        if ctx.gt_engine == "messi":
            for split in missing:
                started = time.time()
                command = [ctx.gt_tool, "--base", base_path(ctx.data, name), "--queries", split_path(ctx.data, name, split),
                           "--k", str(ctx.gt_k), "--metric", metric, "--threads", str(ctx.threads),
                           "--out", gt_prefix(ctx.data, name, split, metric, ctx.gt_k)]
                if subprocess.run(command).returncode != 0:
                    sys.exit(f"daisy_groundtruth failed for {name} {split} {metric}")
                log_time(ctx, name, "search", time.time() - started, split, metric, counts[split])
            continue
        started = time.time()
        index = faiss_index(base_path(ctx.data, name), metric, ctx.threads, info["d"])
        log_time(ctx, name, "build index", time.time() - started, metric=metric)
        for split in missing:
            queries = np.ascontiguousarray(open_vectors(split_path(ctx.data, name, split)), dtype=np.float32)
            values = np.empty((len(queries), ctx.gt_k), np.float32)
            ids = np.empty((len(queries), ctx.gt_k), np.int64)
            started = time.time()
            for start in tqdm(range(0, len(queries), 1000), desc=f"{name} {split} {metric} ground truth"):
                values[start:start + 1000], ids[start:start + 1000] = index.search(queries[start:start + 1000], ctx.gt_k)
            seconds = time.time() - started
            if (ids < 0).any():
                raise ValueError(f"fewer than {ctx.gt_k} neighbours for {name} {split}")
            prefix = gt_prefix(ctx.data, name, split, metric, ctx.gt_k)
            if needs_writing(f"{prefix}.ivecs", ids.shape):
                write_vecs(f"{prefix}.ivecs", ids.astype(np.int32), '<i4')
            if needs_writing(f"{prefix}.fvecs", values.shape):
                write_vecs(f"{prefix}.fvecs", values)
            print(f">> wrote {prefix}.ivecs/.fvecs", flush=True)
            log_time(ctx, name, "search", seconds, split, metric, len(queries))
        del index


# ---------------------------------------------------------------------------- steps

def has_ids(name):
    """True for the datasets that also get id and qrels files (BEIR, MIRACL)."""
    return DATASETS[name]["kind"] in ("beir", "miracl")


def qrels_path(root, name):
    return query_path(root, name).removesuffix(".fvecs") + ".qrels.tsv"


def prepare(name, ctx):
    """Write every missing base, split, id and qrels file of a dataset. All existing files are
    checked before anything is written, and none is replaced."""
    info = DATASETS[name]
    base_from, splits_from = SOURCES[info["kind"]]
    counts = split_counts(name)
    ids = has_ids(name)
    path = base_path(ctx.data, name)
    vectors_missing = {s for s in counts if needs_writing(split_path(ctx.data, name, s), (counts[s], info["d"]))}
    ids_missing = {s for s in counts if ids and not os.path.exists(ids_path(split_path(ctx.data, name, s)))}
    qrels_missing = ids and not os.path.exists(qrels_path(ctx.data, name))
    base_missing = needs_writing(path, (info["n"], info["d"]))
    base_ids_missing = ids and not os.path.exists(ids_path(path))

    if vectors_missing or ids_missing or qrels_missing:
        splits = splits_from(info, ctx)
        for split in counts:
            split_file = split_path(ctx.data, name, split)
            if split in vectors_missing:
                write_fvecs(split_file, [splits[split]], counts[split], info["d"], f"{name} {split}")
            if split in ids_missing:
                write_lines(ids_path(split_file), splits.ids[split])
        if qrels_missing:
            write_lines(qrels_path(ctx.data, name), splits.qrels)
    else:
        print(f">> keeping the query splits of {name}: {', '.join(counts)}")

    # The base is the first n rows of the Deep1B, T2I and series files, and all rows of the others.
    exact = info["kind"] not in ("fbin", "series")
    if base_missing:
        base_ids = write_fvecs(path, base_from(info, ctx), info["n"], info["d"], f"{name} base", exact)
    elif base_ids_missing:  # the ids alone: read the source again, without writing its vectors
        base_ids = copy_rows(base_from(info, ctx), info["n"], info["d"], f"{name} base ids", exact=exact)
    else:
        print(f">> keeping {path}")
    if base_ids_missing:
        if len(set(base_ids)) != len(base_ids):
            raise ValueError(f"{name}: the corpus has duplicate document ids")
        write_lines(ids_path(path), base_ids)


def download_sources(name, ctx):
    """Download the raw files a dataset is prepared from, by the functions that read them
    (BGE-M3, which encodes MIRACL's queries, is downloaded by the preparation itself)."""
    info = DATASETS[name]
    kind = info["kind"]
    if kind == "fbin":
        fbin_rows(info, ctx, "base", info["n"])
        fbin_rows(info, ctx, "learn", LEARN_ROWS)
        fbin_rows(info, ctx, "query", info["nq"])
    elif kind == "texmex":
        texmex_files(info, ctx.raw)
    elif kind == "glove":
        glove_zip(info, ctx, "base")
        glove_zip(info, ctx, "words")
    elif kind == "caselaw":
        caselaw_file(info, ctx, "base")
        caselaw_file(info, ctx, "query")
    elif kind == "beir":
        b = info["beir"]
        hf_files(ctx.hf, COHERE_REPO, [f"{b}/corpus/*.parquet", f"{b}/queries/*.parquet",
                                       f"{b}/qrels/{info['qrels']}.parquet", f"{b}/qrels/train.parquet"], complete=True)
    elif kind == "miracl":
        base = f"miracl-v1.0-{info['lang']}"
        hf_files(ctx.hf, BGE_M3_MIRACL_REPO, f"dense/{info['lang']}.tar.gz", complete=True)
        hf_files(ctx.hf, MIRACL_REPO, [f"{base}/qrels/qrels.{base}-dev.tsv", f"{base}/topics/*.tsv"], complete=True)
    elif kind == "series":
        series_records(info, ctx, "file", info["n"])
        series_records(info, ctx, "queries", info["nq"])
    print(f">> the raw files of {name} are in place", flush=True)


def status(name, ctx):
    info = DATASETS[name]
    counts = split_counts(name)
    gb = info["n"] * (info["d"] + 1) * 4 / 1e9
    parts = [f"base {'ok' if vecs_shape(base_path(ctx.data, name)) == (info['n'], info['d']) else '-'}"]
    for split, count in counts.items():
        have = vecs_shape(split_path(ctx.data, name, split)) == (count, info["d"])
        gts = sum(vecs_shape(f"{gt_prefix(ctx.data, name, split, m, ctx.gt_k)}.ivecs") == (count, ctx.gt_k)
                  for m in info["metrics"])
        gt = f" gt {gts}/{len(info['metrics'])}" if info["metrics"] else ""
        parts.append(f"{split} {size_label(count)} {'ok' if have else '-'}{gt}")
    return f"{name:12} {info['n']:>11,} x {info['d']:<5} {gb:6.1f} GB  " + "  ".join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("datasets", nargs="*", help=f"datasets to prepare (default: all): {', '.join(DATASETS)}")
    p.add_argument("--data-root", default=os.environ.get("DAISY_DATA_ROOT"),
                   help="where the prepared datasets go (env DAISY_DATA_ROOT); required")
    p.add_argument("--raw-dir", default=os.environ.get("DAISY_RAW_DIR"),
                   help="where downloads go (env DAISY_RAW_DIR; default: <data-root>/../raw)")
    p.add_argument("--hf-dir", default=os.environ.get("DAISY_HF_DIR"),
                   help="Hugging Face downloads and models (env DAISY_HF_DIR; default: <raw-dir>/hf)")
    p.add_argument("--series-dir", default=os.environ.get("DAISY_SERIES_DIR"),
                   help="folder of the series .bin files and their queries files under their usual names, there "
                        "or one folder below (env DAISY_SERIES_DIR; default: <raw-dir>/similarity-search-datasets)")
    p.add_argument("--series-file", action="append", default=[], metavar="NAME=PATH",
                   help="a series .bin file of another name or place, e.g. ASTRO100M=/data/astro.bin (repeatable)")
    p.add_argument("--series-queries", action="append", default=[], metavar="NAME=PATH",
                   help="a series queries .bin file of another name or place (repeatable)")
    p.add_argument("--series-url", default=os.environ.get("DAISY_SERIES_URL"),
                   help="URL prefix to download the non-public series files from (env DAISY_SERIES_URL)")
    p.add_argument("--gt-k", type=int, default=5000, help="neighbours per query in the ground truth (default 5000)")
    p.add_argument("--gt-splits", default="", help="comma-separated splits to compute ground truth for (default: all)")
    p.add_argument("--gt-engine", choices=["faiss", "messi"], default="faiss",
                   help="faiss (exact flat index, the default) or messi (lib/optimizer/daisy_groundtruth)")
    p.add_argument("--gt-tool", default=f"{REPO}/build/lib/optimizer/daisy_groundtruth",
                   help="the daisy_groundtruth binary, for --gt-engine messi")
    p.add_argument("--skip-gt", action="store_true", help="prepare base and query splits only")
    p.add_argument("--download-only", action="store_true",
                   help="only download the raw files (in a small job ahead of the preparation, which then "
                        "only computes); DATA_ROOT is not written")
    p.add_argument("--threads", type=int, default=os.cpu_count(), help="threads for the ground truth and encoders")
    p.add_argument("--list", action="store_true", help="show every dataset and what is already prepared")
    args = p.parse_args()

    if not args.data_root:
        p.error("--data-root (or DAISY_DATA_ROOT) is required")
    names = args.datasets or list(DATASETS)
    unknown = [name for name in names if name not in DATASETS]
    if unknown:
        p.error(f"unknown datasets {unknown}; choose from {', '.join(DATASETS)}")
    ctx = argparse.Namespace(data=os.path.abspath(args.data_root),
                             raw=os.path.abspath(args.raw_dir or f"{args.data_root}/../raw"),
                             series_url=args.series_url, gt_k=args.gt_k, gt_engine=args.gt_engine,
                             gt_splits=[s for s in args.gt_splits.split(",") if s], gt_tool=args.gt_tool,
                             threads=args.threads, time_logs={})
    ctx.hf = os.path.abspath(args.hf_dir or f"{ctx.raw}/hf")
    ctx.series_dir = os.path.abspath(args.series_dir or f"{ctx.raw}/similarity-search-datasets")
    ctx.series_files = {}
    for option, key, items in (("--series-file", "file", args.series_file),
                               ("--series-queries", "queries", args.series_queries)):
        for item in items:
            name, _, path = item.partition("=")
            if DATASETS.get(name, {}).get("kind") != "series" or not path:
                p.error(f"{option} {item}: expected NAME=PATH with NAME one of "
                        f"{', '.join(n for n, i in DATASETS.items() if i['kind'] == 'series')}")
            ctx.series_files[(name, key)] = os.path.abspath(path)

    if args.list:
        print(f"data root {ctx.data}, raw dir {ctx.raw}, ground truth k={ctx.gt_k} "
              f"(gt a/b: a of the dataset's b metrics present)")
        for name in names:
            print(status(name, ctx))
        return
    failed = []
    for name in names:
        started = time.time()
        print(f">> {name} ({DATASETS[name]['kind']})", flush=True)
        try:
            if args.download_only:
                download_sources(name, ctx)
                continue
            os.makedirs(f"{ctx.data}/{name}", exist_ok=True)
            prepare(name, ctx)
            if not args.skip_gt:
                groundtruth(name, ctx)
        except FileNotFoundError as error:  # a source that is not here (the non-public series)
            print(f">> {name} skipped: {error}", flush=True)
            failed.append(name)
            continue
        print(f">> {name} done in {time.time() - started:.0f} s", flush=True)
    if failed:
        sys.exit(f"not prepared (missing sources): {', '.join(failed)}")


if __name__ == "__main__":
    main()
