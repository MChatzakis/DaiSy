#!/usr/bin/env python3
"""Downloads and prepares the benchmark datasets of DaiSy on any server, as on the lab server.

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
The files are the same as the lab server's (erato), built by the same rules:

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
  MSMARCO8M    BEIR + Cohere embed-english-v3               learn 80K, tuning 1K, validation 1K, inner product, L2
                                                            query 6,980 (dev)
  DEEP100M     Yandex Deep1B: first 100M base rows          learn 80K, tuning 1K, validation 1K, L2
                                                            query 10K (public)
  MIRACLRU9M   MIRACL ru + BGE-M3                           learn 5,312, validation 1K,          inner product, L2
                                                            query 1,252
  UCASELAW7M   big-ann caselaw, single-vector documents     learn 80K, tuning 1K, validation 1K, L2
                                                            query 1K (all from the query file)
  T2I100M      Yandex Text-to-Image: first 100M base rows   learn 80K, tuning 1K, validation 1K, inner product
                                                            query 10K (of the public 100K)
  SEISMIC100M  series file (not public): first 100M finite  query 10K (the file's last records)  L2
               records
  ASTRO100M    series file (not public): first 100M records query 10K (the file's last records)  L2
  SALD100M     SALD brain fMRI (INDI S3, CC BY-NC)          query 10K (a held-out subject)       L2
The data series have no learning splits. learn/tuning/validation come from each dataset's
own learning queries: by row (create_datasets.py's rules, LEARN_FILE_STARTS) or as seeded
samples of the judged train queries (BEIR, MIRACL). SEISMIC100M and ASTRO100M need their
series .bin file (raw float32 records, no header): in --series-dir (default
RAW_DIR/similarity-search-datasets/) under its usual name, or anywhere with --series-file
NAME=PATH, or downloaded with --series-url; or their prepared DATA_ROOT/<NAME>/ directory
copied from another server (then the .bin file is not needed).

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
SALD_URL = "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_tar"
SALD_ARCHIVES = ["sub-031274_sub-031323", "sub-031324_sub-031373", "sub-031374_sub-031423",
                 "sub-031424_sub-031473", "sub-031474-sub-031523", "sub-031524_sub-031573",
                 "sub-031574_sub-031623", "sub-031624_sub-031673", "sub-031674_sub-031723",
                 "sub-031724_sub-031767"]  # as on the SALD page; note the '-' in the fifth

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
    "MSMARCO8M": dict(kind="beir", beir="msmarco", qrels="dev", learn_pool=502_939, layout="qrels",
                      splits=ROW_SPLITS,
                      n=8_841_823, label="8841823", nq=6_980, d=1024, metrics=["ip", "l2"]),
    "DEEP100M": dict(kind="fbin", raw="DEEP1B", url=f"{YANDEX}/DEEP", base="base.1B.fbin",
                     query="query.public.10K.fbin", learn="learn.350M.fbin", layout="row_splits", splits=ROW_SPLITS,
                     n=100_000_000, label="100M", nq=10_000, d=96, metrics=["l2"]),
    "MIRACLRU9M": dict(kind="miracl", lang="ru", learn_pool=6_312, layout="qrels",
                       splits={"learn": 5_312, "validation": 1_000},
                       n=9_543_918, label="9543918", nq=1_252, d=1024, metrics=["ip", "l2"]),
    "UCASELAW7M": dict(kind="caselaw", raw="caselaw", url="https://comp21storage.z5.web.core.windows.net/caselaw",
                       base="caselaw_base_embeddings.bin", query="caselaw_query_embeddings.bin",
                       query_pool=89_812, layout="row_splits", splits=ROW_SPLITS,
                       n=7_414_023, label="7M", nq=1_000, d=1536, metrics=["l2"]),
    "T2I100M": dict(kind="fbin", raw="t2i", url=f"{YANDEX}/T2I", base="base.1B.fbin", query="query.public.100K.fbin",
                    learn="query.learn.50M.fbin", layout="row_splits", splits=ROW_SPLITS,
                    n=100_000_000, label="100M", nq=10_000, d=200, metrics=["ip"]),
    # Data series (raw float32, z-normalized): the base is the file's first n finite records,
    # the queries its last nq; records with non-finite values are skipped (Seismic has all-NaN
    # records, probably constant series that z-normalization divided by zero).
    "SEISMIC100M": dict(kind="series", raw="similarity-search-datasets", file="data_size100M_seismic_len256_znorm.bin",
                        layout="none", splits={}, n=100_000_000, label="100M", nq=10_000, d=256, metrics=["l2"]),
    "ASTRO100M": dict(kind="series", raw="similarity-search-datasets", file="data_size270M_astronomy_len256_znorm.bin",
                      layout="none", splits={}, n=100_000_000, label="100M", nq=10_000, d=256, metrics=["l2"]),
    # SALD brain fMRI: every brain voxel's 242-point time course cut into z-normalized windows
    # of 128 with stride 8. The first subject read is held out and gives the queries.
    "SALD100M": dict(kind="sald", raw="SALD", layout="none", splits={},
                     n=100_000_000, label="100M", nq=10_000, d=128, metrics=["l2"]),
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
    """Write a whole array, through path.tmp."""
    with open(path + ".tmp", "wb") as f:
        append_vecs(f, vectors, dtype)
    os.replace(path + ".tmp", path)


def write_fvecs(path, batches, rows, dim, desc):
    """Write exactly `rows` rows taken from an iterable of (k, dim) arrays, through path.tmp."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    written = 0
    with open(path + ".tmp", "wb") as f, tqdm(total=rows, desc=desc, unit="vec") as bar:
        for batch in batches:
            batch = np.asarray(batch, dtype=np.float32)[:rows - written]
            if batch.ndim != 2 or batch.shape[1] != dim or not np.all(np.isfinite(batch)):
                raise ValueError(f"{desc}: a batch of shape {batch.shape} or with non-finite values")
            append_vecs(f, batch)
            written += len(batch)
            bar.update(len(batch))
            if written == rows:
                break
    if written != rows:
        raise ValueError(f"{desc}: the source has only {written} of {rows} rows")
    os.replace(path + ".tmp", path)
    print(f">> wrote {path}: {rows} x {dim}", flush=True)


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


def hf_files(hf_dir, repo, patterns):
    """Local copies of a Hugging Face dataset repo's files matching the patterns (downloaded if missing)."""
    patterns = [patterns] if isinstance(patterns, str) else list(patterns)
    local = f"{hf_dir}/{repo}"
    if not all(glob.glob(f"{local}/{p}") for p in patterns):
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


def sampled_splits(info, pool, query_ids, vector_of):
    """{split: vectors} of a dataset whose queries are judged in qrels: the testing queries
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
    return {split: np.array([vector_of[qid] for qid in split_ids]) for split, split_ids in ids.items()}


# ---------------------------------------------------------------------------- sources by kind
# Each kind gives base_batches(info, ctx) (arrays of rows, in order) and splits(info, ctx)
# ({split: vectors} for "query" and the dataset's other splits).

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
    files = hf_files(ctx.hf, COHERE_REPO, f"{info['beir']}/corpus/*.parquet")
    for _, vectors in parquet_embeddings(files, info["d"]):
        yield vectors


def qrels_ids(path):
    import pyarrow.parquet as pq
    return sorted(set(pq.read_table(path, columns=["query_id"]).column("query_id").to_pylist()))


def beir_splits(info, ctx):
    b = info["beir"]
    [testing] = hf_files(ctx.hf, COHERE_REPO, f"{b}/qrels/{info['qrels']}.parquet")
    [train] = hf_files(ctx.hf, COHERE_REPO, f"{b}/qrels/train.parquet")
    query_ids, pool = qrels_ids(testing), qrels_ids(train)
    wanted, vector_of = set(query_ids) | set(pool), {}
    for ids, vectors in parquet_embeddings(hf_files(ctx.hf, COHERE_REPO, f"{b}/queries/*.parquet"), info["d"]):
        for qid, vector in zip(ids, vectors):
            if qid in wanted:
                vector_of[qid] = vector
    return sampled_splits(info, pool, query_ids, vector_of)


def faiss_flat_vectors(fileobj, d, batch_rows=50_000):
    """Vectors of a Pyserini Faiss flat index inside a .tar.gz stream (<name>/index)."""
    with tarfile.open(fileobj=fileobj, mode="r|gz") as archive:
        for member in archive:
            if os.path.basename(member.name) != "index":
                continue
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
                yield np.frombuffer(buffer, dtype='<f4').reshape(rows, d)
            return
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
        query_ids = sorted({line.split()[0] for line in f if line.strip()})
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
    return sampled_splits(info, pool, query_ids, vector_of)


def series_file(info, ctx):
    """The series file as (records, d) float32: raw records, no header (the data-series .bin
    format). Read from --series-file NAME=PATH, else from --series-dir under its usual name."""
    name = next(n for n, i in DATASETS.items() if i is info)
    path = ctx.series_files.get(name, f"{ctx.series_dir}/{info['file']}")
    if not os.path.exists(path):
        if not ctx.series_url or name in ctx.series_files:
            raise FileNotFoundError(f"{path} is missing. It is not public: copy it from a server that has it (or pass "
                                    f"--series-file {name}=PATH, --series-dir or --series-url), or copy the prepared "
                                    f"{ctx.data}/{name}/ directory instead.")
        download(f"{ctx.series_url}/{info['file']}", path)
    size, record = os.path.getsize(path), 4 * info["d"]
    if size % record or size // record < info["n"] + info["nq"]:
        raise FileNotFoundError(f"{path} cannot be {name}'s series file: it has {size:,} bytes, and {name} needs "
                                f"at least {info['n'] + info['nq']:,} float32 records of {info['d']} values")
    return np.memmap(path, dtype='<f4', mode='r', shape=(size // record, info["d"]))


def finite_rows(block):
    return block[np.isfinite(block).all(axis=1)]


def series_split(info, ctx):
    """(records, q): the queries are the last nq finite records, which start at record q."""
    records = series_file(info, ctx)
    found, q = 0, len(records)
    while found < info["nq"] and q > 0:
        lo = max(0, q - 100_000)
        good = np.flatnonzero(np.isfinite(np.asarray(records[lo:q])).all(axis=1))
        if found + len(good) >= info["nq"]:
            return records, lo + good[len(good) - (info["nq"] - found)]
        found, q = found + len(good), lo
    raise ValueError(f"{info['file']} has fewer than {info['nq']} finite records")


def series_base(info, ctx):
    """The first n finite records. They stop before the queries; write_fvecs fails if there
    are fewer than n of them."""
    records, q = series_split(info, ctx)
    skipped = 0
    try:
        for block in chunks(records[:q]):
            good = finite_rows(np.asarray(block))
            skipped += len(block) - len(good)
            yield good
    finally:
        if skipped:
            print(f">> {info['file']}: skipped {skipped} records with non-finite values", flush=True)


def series_splits(info, ctx):
    records, q = series_split(info, ctx)
    return {"query": finite_rows(np.asarray(records[q:]))}


# SALD: windows of every brain voxel's time course (voxel mean above 10% of the scan's highest
# voxel mean), 128 points with stride 8, z-normalized; the first subject read gives the queries.
SALD_LENGTH, SALD_STRIDE, SALD_MASK, SALD_SEED = 128, 8, 0.1, 0
NIFTI_TYPES = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64, 512: np.uint16}


def sald_scans(info, ctx):
    """(subject, gzipped NIfTI bytes) of every resting-state scan, archive by archive (no extraction)."""
    for name in SALD_ARCHIVES:
        path = download(f"{SALD_URL}/{name}.tar.gz", f"{ctx.raw}/{info['raw']}/{name}.tar.gz")
        with tarfile.open(path, "r|gz") as tar:
            for member in tar:
                base = os.path.basename(member.name)
                if member.isfile() and base.endswith("_task-rest_bold.nii.gz"):
                    yield base.split("_")[0], tar.extractfile(member).read()


def sald_windows(scan):
    import gzip
    raw = gzip.decompress(scan)
    if struct.unpack("<i", raw[0:4])[0] != 348:
        raise ValueError("not a little-endian NIfTI-1 file")
    nx, ny, nz, nt = struct.unpack("<8h", raw[40:56])[1:5]
    datatype = struct.unpack("<h", raw[70:72])[0]
    offset = int(struct.unpack("<f", raw[108:112])[0])
    slope, inter = struct.unpack("<ff", raw[112:120])
    data = np.frombuffer(raw, dtype=NIFTI_TYPES[datatype], count=nx * ny * nz * nt, offset=offset)
    series = data.reshape(nt, nx * ny * nz).T.astype(np.float32)  # x fastest, time slowest
    if slope not in (0.0, 1.0) or inter != 0.0:
        series = series * (slope if slope != 0.0 else 1.0) + inter
    mean, std = series.mean(axis=1), series.std(axis=1)
    series = series[(mean > SALD_MASK * mean.max()) & (std > 0)]
    if series.shape[1] < SALD_LENGTH:
        return np.empty((0, SALD_LENGTH), np.float32)
    w = np.lib.stride_tricks.sliding_window_view(series, SALD_LENGTH, axis=1)[:, ::SALD_STRIDE, :].reshape(-1, SALD_LENGTH)
    mean, std = w.mean(axis=1, keepdims=True), w.std(axis=1, keepdims=True)
    keep = std[:, 0] > 1e-6  # a constant window cannot be z-normalized
    return ((w[keep] - mean[keep]) / std[keep]).astype(np.float32)


def sald_base(info, ctx):
    scans = sald_scans(info, ctx)
    next(scans)  # the held-out subject
    for _, scan in scans:
        yield sald_windows(scan)


def sald_splits(info, ctx):
    subject, scan = next(sald_scans(info, ctx))
    w = sald_windows(scan)
    print(f">> SALD queries: {info['nq']} random windows (seed {SALD_SEED}) of held-out subject {subject}")
    return {"query": w[np.random.default_rng(SALD_SEED).permutation(len(w))[:info["nq"]]]}


SOURCES = {
    "texmex": (texmex_base, texmex_splits), "fbin": (fbin_base, fbin_splits),
    "glove": (glove_base, glove_splits), "caselaw": (caselaw_base, caselaw_splits),
    "beir": (beir_base, beir_splits), "miracl": (miracl_base, miracl_splits),
    "series": (series_base, series_splits), "sald": (sald_base, sald_splits),
}


# ---------------------------------------------------------------------------- ground truth

def log_time(ctx, name, step, seconds, split="", metric="", queries=""):
    """Append one timed ground-truth step to DATA_ROOT/<name>/groundtruth_times.csv."""
    path = f"{ctx.data}/{name}/groundtruth_times.csv"
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
                   if any(vecs_shape(f"{gt_prefix(ctx.data, name, s, metric, ctx.gt_k)}.{ext}") != (counts[s], ctx.gt_k)
                          for ext in ("ivecs", "fvecs"))]
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
            write_vecs(f"{prefix}.ivecs", ids.astype(np.int32), '<i4')
            write_vecs(f"{prefix}.fvecs", values)
            print(f">> wrote {prefix}.ivecs/.fvecs", flush=True)
            log_time(ctx, name, "search", seconds, split, metric, len(queries))
        del index


# ---------------------------------------------------------------------------- steps

def prepare(name, ctx):
    info = DATASETS[name]
    base_from, splits_from = SOURCES[info["kind"]]
    counts = split_counts(name)
    missing = [s for s in counts if vecs_shape(split_path(ctx.data, name, s)) != (counts[s], info["d"])]
    if missing:
        splits = splits_from(info, ctx)
        for split in missing:
            write_fvecs(split_path(ctx.data, name, split), [splits[split]], counts[split], info["d"], f"{name} {split}")
    else:
        print(f">> keeping the query splits of {name}: {', '.join(counts)}")
    path = base_path(ctx.data, name)
    if vecs_shape(path) == (info["n"], info["d"]):
        print(f">> keeping {path}")
    else:
        write_fvecs(path, base_from(info, ctx), info["n"], info["d"], f"{name} base")


def status(name, ctx):
    info = DATASETS[name]
    counts = split_counts(name)
    gb = info["n"] * (info["d"] + 1) * 4 / 1e9
    parts = [f"base {'ok' if vecs_shape(base_path(ctx.data, name)) == (info['n'], info['d']) else '-'}"]
    for split, count in counts.items():
        have = vecs_shape(split_path(ctx.data, name, split)) == (count, info["d"])
        gts = sum(vecs_shape(f"{gt_prefix(ctx.data, name, split, m, ctx.gt_k)}.ivecs") == (count, ctx.gt_k)
                  for m in info["metrics"])
        parts.append(f"{split} {size_label(count)} {'ok' if have else '-'} gt {gts}/{len(info['metrics'])}")
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
                   help="folder of the series .bin files under their usual names (env DAISY_SERIES_DIR; "
                        "default: <raw-dir>/similarity-search-datasets)")
    p.add_argument("--series-file", action="append", default=[], metavar="NAME=PATH",
                   help="a series .bin file of another name or place, e.g. ASTRO100M=/data/astro.bin (repeatable)")
    p.add_argument("--series-url", default=os.environ.get("DAISY_SERIES_URL"),
                   help="URL prefix to download the non-public series files from (env DAISY_SERIES_URL)")
    p.add_argument("--gt-k", type=int, default=5000, help="neighbours per query in the ground truth (default 5000)")
    p.add_argument("--gt-splits", default="", help="comma-separated splits to compute ground truth for (default: all)")
    p.add_argument("--gt-engine", choices=["faiss", "messi"], default="faiss",
                   help="faiss (exact flat index, the default) or messi (lib/optimizer/daisy_groundtruth)")
    p.add_argument("--gt-tool", default=f"{REPO}/build/lib/optimizer/daisy_groundtruth",
                   help="the daisy_groundtruth binary, for --gt-engine messi")
    p.add_argument("--skip-gt", action="store_true", help="prepare base and query splits only")
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
                             threads=args.threads)
    ctx.hf = os.path.abspath(args.hf_dir or f"{ctx.raw}/hf")
    ctx.series_dir = os.path.abspath(args.series_dir or f"{ctx.raw}/similarity-search-datasets")
    ctx.series_files = {}
    for item in args.series_file:
        name, _, path = item.partition("=")
        if DATASETS.get(name, {}).get("kind") != "series" or not path:
            p.error(f"--series-file {item}: expected NAME=PATH with NAME one of "
                    f"{', '.join(n for n, i in DATASETS.items() if i['kind'] == 'series')}")
        ctx.series_files[name] = os.path.abspath(path)

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
