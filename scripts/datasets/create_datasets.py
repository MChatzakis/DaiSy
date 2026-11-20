"""Organize the benchmark datasets: base vectors, four query splits and their exact
ground truth, in DATA_DIR/<dataset>/, as ENDORDataLoader (src/endor/endor_dataloader.hpp) reads them.

Query splits:
    learn       learn.80K.fvecs       learning queries; the first 10K are the earlier learn.10K.fvecs
    tuning      tuning.1K.fvecs       tuning queries
    validation  validation.1K.fvecs   validation queries (the first 1K of the earlier validation file)
    query       query.<size>.fvecs    testing queries: all the official ones, or T2I's first
                                      10K (ENDORDataLoader in src/endor/endor_dataloader.hpp searches a
                                      sample of them)
Learning, tuning and validation come from the dataset's own learning queries, by row
(LEARN_FILE_STARTS, CASELAW_STARTS), or are seeded samples of MS MARCO's training queries, so
every run gives the same splits. The raw learning files repeat some vectors: tuning and the
added learning queries skip every vector already in a split. The earlier files keep theirs:
eight GIST1M validation queries, and one DEEP10M official query, are also in learn.10K.fvecs.
The script lists every query found in two splits. Benchmark runners filter these overlaps
at read time (learning, then tuning, then validation take priority; testing excludes all
preparation vectors), so existing datasets and ground truth do not need regeneration.

Ground truth: the GT_K nearest neighbours of every query, for each metric of the dataset, by
exact search with Faiss (the default) or DaiSy (GT_LIBRARY):
    <l2.|innprod.><split>.groundtruth.<size>.k5000.ivecs   ids
    <l2.|innprod.><split>.groundtruth.<size>.k5000.fvecs   distances (squared L2) or scores
    groundtruth_times.csv                                  wall-clock seconds of every step
The times are appended, one row per step: loading the base (which depends on the disk and its
cache), building the search index, and searching each split.

Generated workloads: for the datasets of WORKLOAD_DATASETS, the testing workloads of controlled
hardness (easy, medium, hard, veryhard) with their ground truth and sanity check, in
DATA_DIR/<dataset>/generated/, by query_generator.py's generate_workloads (see its docstring),
once the dataset's own ground truth is there.

Files already there are kept. Before writing a split, every existing <split>.*.fvecs file
(such as learn.10K.fvecs) must agree with the new split on its first rows, or the script
stops. Missing files are written, so a rerun resumes.

Raw files missing from RAW_DIR are downloaded from the URLs in DATASETS, so a dataset can be
organized from nothing:
    DEEP and T2I    only the rows used, from Yandex's .fbin files (download_rows): the first
                    10M, 50M or 100M base vectors, the first LEARN_ROWS learning queries and the
                    testing queries, into <file>.first.fbin, which a larger dataset extends
    GIST1M          the texmex archive, checked against its MD5 sum and unpacked
    UCASELAW7M      both embedding files, whole (a stopped download resumes)
    MSMARCO8M       its Hugging Face files (hf_files)
    GLOVE1M         Stanford's two GloVe zip files, whole (see GLOVE1M's entry)

A benchmark run prepares its data and runs every method on one server. The base vectors and the
splits taken by row are the same on any server; the splits drawn at random (seed 42) and the
ground truth can differ slightly with other numpy, BLAS or Faiss versions (docs/DESIGN.md).

Run with requirements.txt installed, hephaestus included (and daisy-exact-search for
GT_LIBRARY = "daisy"):
    python scripts/dataset_organizer.py [DATASET ...]   (default: all)
"""
import glob
import hashlib
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime

import faiss
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

import query_generator

RAW_DIR = "/mnt/hddhelp/mchatzakis/datasets"
DATA_DIR = "/mnt/hddhelp/mchatzakis/datasets/processed"
# MS MARCO's Hugging Face parquet files (about 48 GB) live on the root disk: /mnt/hddhelp has
# no room for them next to the processed vectors.
HF_DIR = "/home/mchatzakis/hf_datasets"
SEED = 42
GT_K = 5000
GT_PREFIXES = {"L2": "l2.", "inner-product": "innprod."}
# The exact k-NN search that computes the ground truth: "faiss" (brute force with a flat index,
# as the earlier files were made) or "daisy" (the MESSI index of DaiSy, see daisy_search).
GT_LIBRARY = "faiss"
THREADS = os.cpu_count()  # both libraries use every core
faiss.omp_set_num_threads(THREADS)

# Datasets that also get the generated testing workloads (query_generator.py), one per metric.
WORKLOAD_DATASETS = ["GLOVE1M", "DEEP10M"]

# Queries per split; testing has each dataset's own count ("testing_queries").
SPLIT_COUNTS = {"learn": 80_000, "tuning": 1_000, "validation": 1_000}
# Rows of a raw learning file downloaded when it is missing: every split's rows (LEARN_FILE_STARTS)
# and plenty of room for the repeated vectors that row_splits skips.
LEARN_ROWS = 1_000_000

# The row where each split starts in a raw learning file. The earlier splits were the 10K
# learning rows from 0, the validation rows from 10K (10K or 1K of them) and the calibration
# rows (now tuning) from 20K; "more_learn" is where the 70K added learning rows start, after
# all of them.
LEARN_FILE_STARTS = {"learn": 0, "validation": 10_000, "tuning": 20_000, "more_learn": 30_000}
# The same in caselaw's 89,812 single-vector query documents, which also give its 1K testing
# queries. The earlier splits used rows 0-13K, in the same places. The documents are in no
# particular order (row and filing date are uncorrelated).
CASELAW_STARTS = {"learn": 0, "validation": 10_000, "tuning": 11_000, "query": 12_000, "more_learn": 13_000}

COHERE_REPO = "CohereLabs/beir-embed-english-v3"

# Deep1B image descriptors and its 10K public queries. Each DEEP dataset takes the first 10M,
# 50M or 100M base vectors; their query splits are the same. The files are those of
# https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search.
YANDEX_URL = "https://storage.yandexcloud.net/yandex-research/ann-datasets"
DEEP1B = {
    "kind": "files",
    "base_file": f"{RAW_DIR}/DEEP1B/base.1B.fbin",
    "base_url": f"{YANDEX_URL}/DEEP/base.1B.fbin",
    "learn_file": f"{RAW_DIR}/DEEP1B/learn.350M.fbin",
    "learn_url": f"{YANDEX_URL}/DEEP/learn.350M.fbin",
    "query_file": f"{RAW_DIR}/DEEP1B/query.public.10K.fbin",
    "query_url": f"{YANDEX_URL}/DEEP/query.public.10K.fbin",
    "testing_queries": 10_000,
    "d": 96, "metrics": ["L2"],
}

# Yandex Text-to-Image (T2I-1B of big-ann-benchmarks), by inner product: image embeddings as
# base vectors, text embeddings as queries (unit norm). Each T2I dataset takes the first 10M, 50M
# or 100M base vectors; their query splits are the same. Testing: the first 10K of its 100K
# public queries, as the earlier query.10K.fvecs.
T2I1B = {
    "kind": "files",
    "base_file": f"{RAW_DIR}/t2i/base.1B.fbin",
    "base_url": f"{YANDEX_URL}/T2I/base.1B.fbin",
    "learn_file": f"{RAW_DIR}/t2i/query.learn.50M.fbin",
    "learn_url": f"{YANDEX_URL}/T2I/query.learn.50M.fbin",
    "query_file": f"{RAW_DIR}/t2i/query.public.100K.fbin",
    "query_url": f"{YANDEX_URL}/T2I/query.public.100K.fbin",
    "testing_queries": 10_000,
    "d": 200, "metrics": ["inner-product"],
}

DATASETS = {
    "DEEP10M": {**DEEP1B, "base_vectors": 10_000_000, "base_size": "10M"},
    # GIST1M: its 1M base vectors and 1K queries; its learning file has 500K vectors. All three
    # come in one archive of http://corpus-texmex.irisa.fr/, with the MD5 sum it publishes.
    "GIST1M": {
        "kind": "files",
        "base_file": f"{RAW_DIR}/GIST1M/gist/gist_base.fvecs",
        "learn_file": f"{RAW_DIR}/GIST1M/gist/gist_learn.fvecs",
        "query_file": f"{RAW_DIR}/GIST1M/gist/gist_query.fvecs",
        "archive_url": "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz",
        "archive_md5": "31185e0f00854f74d27e8ad8d52628a9",
        "archive_dir": f"{RAW_DIR}/GIST1M",  # it unpacks into gist/
        "base_vectors": 1_000_000, "base_size": "1M",
        "testing_queries": 1_000,
        "d": 960, "metrics": ["L2"],
    },
    # GloVe, not normalized, so maximum inner product search (https://nlp.stanford.edu/projects/glove/,
    # downloaded from Stanford's mirror, downloads.cs.stanford.edu).
    # Base: the 1,183,514 "train" vectors of ann-benchmarks' glove-100-angular.hdf5, whose
    # download (http://ann-benchmarks.com) turns scripts away. So they are rebuilt as
    # ann-benchmarks' datasets.py builds them, from glove.twitter.27B.zip: the 100-d vectors, then
    # all but the first 10K rows of numpy's RandomState(1) permutation of them (scikit-learn's
    # train_test_split(test_size=10000, random_state=1)). The result is the HDF5's "train" set,
    # bit for bit (glove_base_vectors).
    # Queries: word vectors of glove.6B.100d.txt (400K words, in glove.6B.zip), another GloVe model.
    # The earlier splits came from an unseeded shuffle of those words: learn.10K.fvecs,
    # validation.10K.fvecs and query.10K.fvecs (the official GLOVE1M queries). Their rows in
    # glove.6B.100d.txt are in docs/glove1m_query_rows.txt, from which this script rebuilds them
    # (glove_files): the learning file is their 20K learning and validation rows, as before, then
    # the words in none of the three files in seeded random order.
    "GLOVE1M": {
        "kind": "glove",
        "base_zip": f"{RAW_DIR}/GloVe/misc/glove.twitter.27B.zip",
        "base_url": "https://downloads.cs.stanford.edu/nlp/data/glove.twitter.27B.zip",
        "words_zip": f"{RAW_DIR}/GloVe/misc/glove.6B.zip",
        "words_url": "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip",
        "query_rows": f"{os.path.dirname(os.path.abspath(__file__))}/../docs/glove1m_query_rows.txt",
        "base_vectors": 1_183_514, "base_size": "1183514",
        "testing_queries": 10_000,
        "d": 100, "metrics": ["inner-product"],
    },
    "T2I10M": {**T2I1B, "base_vectors": 10_000_000, "base_size": "10M"},
    # Caselaw (COLD Cases embedded with OpenAI text-embedding-3-small). The files hold
    # multi-vector documents; only the documents embedded as a single vector are used: 7,414,023
    # base documents and 89,812 query documents. Unit norm. The files are those of big-ann-
    # benchmarks' dataset_preparation/caselaw_dataset.md.
    "UCASELAW7M": {
        "kind": "caselaw",
        "base_file": f"{RAW_DIR}/caselaw/caselaw_base_embeddings.bin",
        "base_url": "https://comp21storage.z5.web.core.windows.net/caselaw/caselaw_base_embeddings.bin",
        "query_file": f"{RAW_DIR}/caselaw/caselaw_query_embeddings.bin",
        "query_url": "https://comp21storage.z5.web.core.windows.net/caselaw/caselaw_query_embeddings.bin",
        "base_vectors": 7_414_023, "base_size": "7M",
        "query_pool": 89_812, "testing_queries": 1_000,
        "d": 1536, "metrics": ["L2"],
    },
    # BEIR MS MARCO passages and queries embedded with Cohere embed-english-v3.0, by inner
    # product (unit norm, so L2 gives the same neighbours). Testing: the 6,980 dev queries.
    # Learning, validation and tuning: samples of the 502,939 train queries. The row of each
    # vector in its file is the BEIR id on the same line of <file>.ids.txt.
    "MSMARCO8M": {
        "kind": "msmarco",
        "corpus": "msmarco/corpus/*.parquet",
        "queries": "msmarco/queries/*.parquet",
        "query_qrels": "msmarco/qrels/dev.parquet",
        "learn_qrels": "msmarco/qrels/train.parquet",
        "base_vectors": 8_841_823, "base_size": "8841823",
        "testing_queries": 6_980, "learn_pool": 502_939,
        "d": 1024, "metrics": ["inner-product"],
    },
    # Larger DEEP and T2I bases for the scalability study, with DEEP10M's and T2I10M's queries.
    "DEEP50M": {**DEEP1B, "base_vectors": 50_000_000, "base_size": "50M"},
    "DEEP100M": {**DEEP1B, "base_vectors": 100_000_000, "base_size": "100M"},
    "T2I50M": {**T2I1B, "base_vectors": 50_000_000, "base_size": "50M"},
    "T2I100M": {**T2I1B, "base_vectors": 100_000_000, "base_size": "100M"},
}


def split_count(name, split):
    return DATASETS[name]["testing_queries"] if split == "query" else SPLIT_COUNTS[split]

def size(name, split):
    """The size in a file name: base.10M, learn.80K, query.6980, ..."""
    if split == "base":
        return DATASETS[name]["base_size"]
    count = split_count(name, split)
    return f"{count // 1000}K" if count % 1000 == 0 else str(count)

def vectors_path(name, split):
    return f"{DATA_DIR}/{name}/{split}.{size(name, split)}.fvecs"

def groundtruth_path(name, split, metric, extension):
    return f"{DATA_DIR}/{name}/{GT_PREFIXES[metric]}{split}.groundtruth.{size(name, split)}.k{GT_K}.{extension}"


def open_vectors(path):
    """A read-only (rows, dim) view of an .fvecs or .fbin file."""
    if path.endswith(".fbin"):
        rows, dim = np.fromfile(path, dtype='<u4', count=2)
        return np.memmap(path, dtype='<f4', mode='r', offset=8, shape=(int(rows), int(dim)))
    dim = int(np.fromfile(path, dtype='<i4', count=1)[0])
    return np.memmap(path, dtype='<f4', mode='r').reshape(-1, dim + 1)[:, 1:]

def open_caselaw(path):
    """(view of all vectors, row of each single-vector document) of a caselaw file.

    The file holds the document count, dimension and vector count (uint32), the vector count
    of each document (uint32), then all vectors (float32), document after document.
    """
    with open(path, 'rb') as f:
        documents, dim, rows = map(int, np.fromfile(f, dtype='<u4', count=3))
        counts = np.fromfile(f, dtype='<u4', count=documents)
    offset = 12 + 4 * documents
    if os.path.getsize(path) != offset + 4 * rows * dim or counts.sum() != rows:
        raise ValueError(f"{path} does not match its header")
    first_rows = np.cumsum(counts, dtype=np.int64) - counts
    return np.memmap(path, dtype='<f4', mode='r', offset=offset, shape=(rows, dim)), first_rows[counts == 1]

def vecs_shape(path):
    """(rows, dim) of an .fvecs/.ivecs file, or None if it is missing."""
    if not os.path.exists(path):
        return None
    dim = int(np.fromfile(path, dtype='<i4', count=1)[0])
    return os.path.getsize(path) // (4 * (dim + 1)), dim

def append_vecs(f, vectors, dtype='<f4'):
    """Append rows to an open file as .fvecs ('<f4') or .ivecs ('<i4') records."""
    vectors = np.asarray(vectors)
    records = np.empty((len(vectors), vectors.shape[1] + 1), dtype=dtype)
    records.view('<i4')[:, 0] = vectors.shape[1]
    records[:, 1:] = vectors
    records.tofile(f)

def write_vecs(path, vectors, dtype='<f4'):
    # Through a temporary file, so a stopped run never leaves a partial file.
    with open(path + ".tmp", 'wb') as f:
        append_vecs(f, vectors, dtype)
    os.replace(path + ".tmp", path)
    print(f">> Wrote {path}: {vecs_shape(path)}")

def write_lines(path, values):
    with open(path, 'w') as f:
        f.writelines(f"{value}\n" for value in values)


def download(url, path):
    """Download a whole file from `url` to `path`, through <path>.part, which a rerun continues
    (over HTTP; an FTP download starts over)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    have = os.path.getsize(part) if os.path.exists(part) and url.startswith("http") else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as error:
        if error.code != 416:
            raise
        # Nothing left after <path>.part: it holds the whole file already.
        os.replace(part, path)
        return
    with response:
        if have and response.status != 206:
            have = 0  # the server sends the whole file again
        total = have + int(response.headers["Content-Length"])
        with open(part, "ab" if have else "wb") as f, tqdm(total=total, initial=have, unit="B", unit_scale=True, desc=os.path.basename(path)) as bar:
            while chunk := response.read(1 << 24):
                f.write(chunk)
                bar.update(len(chunk))
    if os.path.getsize(part) != total:
        sys.exit(f"The download of {url} stopped early; run again to continue it")
    os.replace(part, path)
    print(f">> Downloaded {path}")

def download_rows(url, path, rows):
    """Download the first `rows` vectors of the .fbin file at `url` (Deep1B's and T2I's 1B-vector
    files are 384 and 800 GB) into <path without .fbin>.first.fbin, an .fbin file of only those
    rows, and return its path. A call that needs more rows than it holds downloads the rest. Its
    header counts only the rows completely written, so a stopped download continues after them."""
    first = path.removesuffix(".fbin") + ".first.fbin"
    have = int(np.fromfile(first, dtype='<u4', count=1)[0]) if os.path.exists(first) else 0
    if have >= rows:
        return first
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # The remote file's header: its vector count and dimension.
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-7"})) as response:
        count, dim = np.frombuffer(response.read(), dtype='<u4')
    rows = min(rows, int(count))
    if have >= rows:
        return first
    start, end = 8 + 4 * int(dim) * have, 8 + 4 * int(dim) * rows
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
    with open(first, "r+b" if have else "w+b") as f, urllib.request.urlopen(request) as response:
        if response.status != 206:
            sys.exit(f"{url} does not send parts of the file; download it whole to {path} instead")
        # The header, then the rows after the counted ones (dropping what a stopped download left).
        f.write(np.array([have, dim], dtype='<u4').tobytes())
        f.truncate(start)
        f.seek(start)
        with tqdm(total=end - start, unit="B", unit_scale=True, desc=f"{os.path.basename(first)}: rows {have}-{rows}") as bar:
            while chunk := response.read(1 << 24):
                f.write(chunk)
                bar.update(len(chunk))
        if f.tell() != end:
            sys.exit(f"The download of {url} stopped early; run again to continue it")
        # Count the new rows only now that they are all written.
        f.seek(0)
        f.write(np.array([rows, dim], dtype='<u4').tobytes())
    print(f">> Downloaded the first {rows} vectors of {url} to {first}")
    return first

def raw_vectors(name, key, rows):
    """A view of the first `rows` vectors of a dataset's raw base, learning or query file (key
    "base", "learn" or "query"). If the file is not in RAW_DIR, its rows are downloaded
    (download_rows), or the archive that holds it (GIST1M's, download_archive)."""
    info = DATASETS[name]
    path = info[f"{key}_file"]
    if not os.path.exists(path) and path.endswith(".fbin"):
        path = download_rows(info[f"{key}_url"], path, rows)
    elif not os.path.exists(path):
        download_archive(info)
    return open_vectors(path)[:rows]

def download_archive(info):
    """Download a dataset's .tar.gz archive into its archive_dir, check its MD5 sum and unpack it
    there (through a temporary folder, so a stopped run leaves no partial files), then delete it."""
    directory = info["archive_dir"]
    archive = f"{directory}/{os.path.basename(info['archive_url'])}"
    if not os.path.exists(archive):
        download(info["archive_url"], archive)
    md5 = hashlib.md5()
    with open(archive, "rb") as f:
        while chunk := f.read(1 << 24):
            md5.update(chunk)
    if md5.hexdigest() != info["archive_md5"]:
        os.remove(archive)
        sys.exit(f"{archive} does not match its MD5 sum; it was deleted, run again to download it")
    unpacked = f"{directory}/unpacking"
    shutil.rmtree(unpacked, ignore_errors=True)
    with tarfile.open(archive) as tar:
        tar.extractall(unpacked, filter="data")
    for entry in os.listdir(unpacked):
        if os.path.exists(f"{directory}/{entry}"):
            sys.exit(f"{directory}/{entry} is incomplete; move it away, then run again to unpack {archive}")
        os.replace(f"{unpacked}/{entry}", f"{directory}/{entry}")
    os.rmdir(unpacked)
    os.remove(archive)
    print(f">> Unpacked {archive} into {directory}")


def hf_files(pattern):
    """The local copies of COHERE_REPO's files matching a pattern, downloaded if missing."""
    local_dir = f"{HF_DIR}/{COHERE_REPO}"
    if not glob.glob(f"{local_dir}/{pattern}"):
        snapshot_download(repo_id=COHERE_REPO, repo_type="dataset", allow_patterns=[pattern],
                          local_dir=local_dir, max_workers=8)
    return sorted(glob.glob(f"{local_dir}/{pattern}"))

def msmarco_embeddings(pattern, d):
    """(ids, vectors) batches of an embedding parquet file set, in file order."""
    for path in hf_files(pattern):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000, columns=["_id", "emb"]):
            embeddings = batch.column("emb")
            if not np.all(pc.list_value_length(embeddings).to_numpy() == d):
                raise ValueError(f"Embeddings of another dimension than {d} in {path}")
            vectors = embeddings.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(-1, d)
            yield batch.column("_id").to_pylist(), vectors

def msmarco_query_splits(info):
    """{split: (vectors, ids)}. Testing is the dev queries; the other splits are seeded
    samples of the train queries, drawn in the earlier order (learning 10K, then validation
    1K) so those stay the same, then tuning 1K and 70K more learning queries."""
    dev_ids = sorted(set(pq.read_table(hf_files(info["query_qrels"])[0]).column("query_id").to_pylist()))
    pool = sorted(set(pq.read_table(hf_files(info["learn_qrels"])[0]).column("query_id").to_pylist()))
    if len(dev_ids) != info["testing_queries"] or len(pool) != info["learn_pool"] or not set(dev_ids).isdisjoint(pool):
        raise ValueError(f"Unexpected MS MARCO qrels: {len(dev_ids)} dev and {len(pool)} train queries")
    rng = np.random.default_rng(SEED)
    first_learn = np.sort(rng.choice(len(pool), 10_000, replace=False))
    remaining = np.setdiff1d(np.arange(len(pool)), first_learn)
    validation = np.sort(rng.choice(remaining, 1_000, replace=False))
    remaining = np.setdiff1d(remaining, validation)
    tuning = np.sort(rng.choice(remaining, 1_000, replace=False))
    remaining = np.setdiff1d(remaining, tuning)
    more_learn = np.sort(rng.choice(remaining, 70_000, replace=False))
    split_ids = {"learn": [pool[row] for row in np.concatenate([first_learn, more_learn])],
                 "validation": [pool[row] for row in validation],
                 "tuning": [pool[row] for row in tuning],
                 "query": dev_ids}

    wanted = {qid for ids in split_ids.values() for qid in ids}
    found = {}
    for ids, vectors in tqdm(msmarco_embeddings(info["queries"], info["d"]), desc="MSMARCO8M queries"):
        for qid, vector in zip(ids, vectors):
            if qid in wanted:
                found[qid] = vector
    return {split: (np.array([found[qid] for qid in ids]), ids) for split, ids in split_ids.items()}


def row_splits(pool, starts, query):
    """{split: vectors} from the rows of a raw learning file (pool). Learning keeps its earlier
    first 10K rows and validation its earlier first 1K rows; tuning and the 70K added learning
    rows are the first rows from their start whose vectors are in no split yet."""
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

def glove_files(info):
    """(learning file, official queries) of GloVe, rebuilt as GLOVE1M's comment describes."""
    if not os.path.exists(info["words_zip"]):
        download(info["words_url"], info["words_zip"])
    with zipfile.ZipFile(info["words_zip"]) as z, z.open("glove.6B.100d.txt") as f:
        # One word per line, then its 100 values, separated by single spaces; parsed as the
        # earlier files were (to float64, then float32), so equal vectors compare equal.
        words = np.array([[float(x) for x in line.decode("utf8").rstrip("\n").split(" ")[1:]] for line in f], dtype=np.float32)
    # The earlier splits, from their rows (docs/glove1m_query_rows.txt: "<split> <row>" lines).
    rows = {"learn": [], "validation": [], "query": []}
    with open(info["query_rows"]) as f:
        for line in f:
            if not line.startswith("#"):
                split, row = line.split()
                rows[split].append(int(row))
    earlier = {split: words[split_rows] for split, split_rows in rows.items()}
    taken = {vector.tobytes() for vectors in earlier.values() for vector in vectors}
    others = np.array([vector for vector in words if vector.tobytes() not in taken])
    others = others[np.random.default_rng(SEED).permutation(len(others))]
    return np.concatenate([earlier["learn"], earlier["validation"], others]), earlier["query"]

def query_splits(name):
    """{split: (vectors, ids or None)} of the four query splits, from the raw files."""
    info = DATASETS[name]
    if info["kind"] == "msmarco":
        return msmarco_query_splits(info)
    if info["kind"] == "caselaw":
        if not os.path.exists(info["query_file"]):
            download(info["query_url"], info["query_file"])
        vectors, single_rows = open_caselaw(info["query_file"])
        if len(single_rows) != info["query_pool"]:
            raise ValueError(f"Expected {info['query_pool']} single-vector queries in {info['query_file']}")
        pool = vectors[single_rows]
        query = pool[CASELAW_STARTS["query"]:CASELAW_STARTS["query"] + info["testing_queries"]]
        splits = row_splits(pool, CASELAW_STARTS, query)
    else:
        if info["kind"] == "glove":
            learn, official = glove_files(info)
        else:
            learn = raw_vectors(name, "learn", LEARN_ROWS)
            official = raw_vectors(name, "query", info["testing_queries"])
        query = np.array(official[:info["testing_queries"]])
        splits = row_splits(learn, LEARN_FILE_STARTS, query)
    return {split: (vectors, None) for split, vectors in splits.items()}

def base_batches(name, batch_rows=100_000):
    """(ids or None, vectors) batches of the base vectors, from the raw files."""
    info = DATASETS[name]
    if info["kind"] == "msmarco":
        yield from msmarco_embeddings(info["corpus"], info["d"])
    elif info["kind"] == "caselaw":
        if not os.path.exists(info["base_file"]):
            download(info["base_url"], info["base_file"])
        vectors, single_rows = open_caselaw(info["base_file"])
        for start in range(0, len(single_rows), batch_rows):
            yield None, vectors[single_rows[start:start + batch_rows]]
    elif info["kind"] == "glove":
        if not os.path.exists(info["base_zip"]):
            download(info["base_url"], info["base_zip"])
        # As ann-benchmarks' glove() and train_test_split(): each line's values after the word,
        # as float64, then the rows after the first 10K of RandomState(1)'s permutation.
        with zipfile.ZipFile(info["base_zip"]) as z, z.open("glove.twitter.27B.100d.txt") as f:
            vectors = np.array([np.array([float(x) for x in line.strip().split()[1:]]) for line in f])
        train = np.random.RandomState(1).permutation(len(vectors))[10_000:]
        for start in range(0, len(train), batch_rows):
            yield None, vectors[train[start:start + batch_rows]].astype(np.float32)
    else:
        vectors = raw_vectors(name, "base", info["base_vectors"])
        for start in range(0, info["base_vectors"], batch_rows):
            yield None, vectors[start:min(start + batch_rows, info["base_vectors"])]


def organize_base(name):
    info = DATASETS[name]
    path = vectors_path(name, "base")
    if vecs_shape(path) == (info["base_vectors"], info["d"]):
        print(f">> Keeping {path}")
        return
    ids = []
    with open(path + ".tmp", 'wb') as f:
        for batch_ids, vectors in tqdm(base_batches(name), desc=f"{name} base"):
            if not np.all(np.isfinite(vectors)):
                raise ValueError(f"Non-finite {name} base vectors")
            append_vecs(f, vectors)
            ids.extend(batch_ids or [])
    if ids:
        write_lines(path.removesuffix(".fvecs") + ".ids.txt", ids)
    os.replace(path + ".tmp", path)
    if vecs_shape(path) != (info["base_vectors"], info["d"]):
        raise ValueError(f"{path} has shape {vecs_shape(path)}")
    print(f">> Wrote {path}")

def organize_queries(name):
    info = DATASETS[name]
    splits = query_splits(name)

    # List the queries found in two splits (compared by value): only the earlier files have them.
    owner = {}
    for split, (vectors, _) in splits.items():
        if vectors.shape != (split_count(name, split), info["d"]) or not np.all(np.isfinite(vectors)):
            raise ValueError(f"{name} {split}: shape {vectors.shape} or non-finite values")
        for row, vector in enumerate(vectors):
            other = owner.setdefault(vector.tobytes(), (split, row))
            if other[0] != split:
                print(f">> Note: {name} {split} row {row} is also {other[0]} row {other[1]}")

    # The earlier files of each split (learn.10K.fvecs, ...) and its own file, if there, must
    # agree with the new split on their first rows. All are checked before anything is written.
    for split, (vectors, _) in splits.items():
        for existing in sorted(glob.glob(f"{DATA_DIR}/{name}/{split}.*.fvecs")):
            old = open_vectors(existing)
            common = min(len(old), len(vectors))
            if (existing == vectors_path(name, split) and len(old) != len(vectors)) or not np.array_equal(old[:common], vectors[:common]):
                sys.exit(f"{existing} differs from the new {name} {split} split; move it away to continue")
            print(f">> {existing} agrees with the new {split} split on its first {common} rows")

    for split, (vectors, ids) in splits.items():
        path = vectors_path(name, split)
        if not os.path.exists(path):
            write_vecs(path, vectors)
        if ids is not None and not os.path.exists(path.removesuffix(".fvecs") + ".ids.txt"):
            write_lines(path.removesuffix(".fvecs") + ".ids.txt", ids)

    if info["kind"] == "msmarco":
        # The relevance judgements (qrels) of the testing queries.
        qrels_path = vectors_path(name, "query").removesuffix(".fvecs") + ".qrels.tsv"
        if not os.path.exists(qrels_path):
            qrels = pq.read_table(hf_files(info["query_qrels"])[0])
            write_lines(qrels_path, ["query_id\tcorpus_id\tscore"] + [
                f"{q}\t{c}\t{s}" for q, c, s in zip(*(qrels.column(column).to_pylist()
                                                     for column in ("query_id", "corpus_id", "score")))])


def faiss_search(base, metric):
    """A function that gives the exact top-GT_K (distances, ids) of a batch of queries: squared
    L2 distances (ascending) or inner-product scores (descending), by brute force with a flat
    Faiss index."""
    index = faiss.IndexFlatL2(base.shape[1]) if metric == "L2" else faiss.IndexFlatIP(base.shape[1])
    index.add(base)
    return lambda queries: index.search(queries, GT_K)

def daisy_search(base, metric):
    """The same as faiss_search, with the MESSI index of DaiSy (pip install daisy-exact-search).

    MESSI computes squared L2 only. Inner product is reduced to it: each base vector x gets
    sqrt(M^2 - |x|^2) appended (M, the largest base norm) and each query q a 0, so the squared
    distance is |q|^2 + M^2 - 2 q.x, smallest for the largest q.x. MESSI also needs the dimension
    to be a multiple of its 16 PAA segments (another dimension gives wrong answers without an
    error), so zero columns pad it; they change no distance. daisy-exact-search 1.0.9 can pair
    an id with another vector's distance (at tied distances, and on about 1 in 3,000 queries
    even of random data, with any number of workers) and miss neighbours when whole PAA
    segments are zero. So every answer is checked, and the run stops on the first wrong one:
    each returned distance must be its id's, and the first 10 queries of every batch must get
    Faiss's distances.
    """
    import daisy
    lift = metric == "inner-product"
    norms = np.einsum("ij,ij->i", base, base)
    width = -(-(base.shape[1] + lift) // 16) * 16

    def extend(vectors, extra):
        extended = np.zeros((len(vectors), width), np.float32)
        extended[:, :vectors.shape[1]] = vectors
        if lift:
            extended[:, vectors.shape[1]] = extra
        return extended

    messi = daisy.Messi(daisy.DistanceType.L2_SQUARED)
    messi.setNumThreads(THREADS)
    messi.setIndexWorkers(THREADS)
    messi.setSearchWorkers(THREADS)
    # MESSI reads this array during every search without keeping it alive, so search() does.
    extended_base = extend(base, np.sqrt(np.maximum(norms.max() - norms, 0)))
    messi.buildIndex(extended_base)
    reference = faiss_search(base, metric)

    def search(queries):
        assert extended_base is not None  # keeps the array MESSI reads alive
        ids, values = messi.searchIndex(extend(queries, 0), GT_K)
        ids, values = np.asarray(ids).astype(np.int64), np.asarray(values)
        if lift:
            values = (np.einsum("ij,ij->i", queries, queries)[:, None] + norms.max() - values) / 2
        right = np.allclose(values[:10], reference(queries[:10])[0], rtol=1e-3, atol=1e-3)
        for query, row_ids, row_values in zip(queries, ids, values):
            exact = base[row_ids] @ query if lift else ((base[row_ids] - query) ** 2).sum(1)
            right = right and np.allclose(exact, row_values, rtol=1e-3, atol=1e-3)
        if not right:
            raise ValueError("DaiSy's answer is not the exact one; use GT_LIBRARY = \"faiss\"")
        return values, ids
    return search

def log_time(name, step, seconds, split="", metric="", queries=""):
    """Append one timed step of the ground truth to groundtruth_times.csv."""
    path = f"{DATA_DIR}/{name}/groundtruth_times.csv"
    new = not os.path.exists(path)
    with open(path, 'a') as f:
        if new:
            f.write("date,host,library,threads,k,step,split,metric,queries,seconds\n")
        f.write(f"{datetime.now().isoformat(timespec='seconds')},{os.uname().nodename},{GT_LIBRARY},{THREADS},"
                f"{GT_K},{step},{split},{metric},{queries},{seconds:.1f}\n")
    print(f">> {name} {step} {split} {metric}: {seconds:.1f} s")

def compute_groundtruths(name, query_batch=1_000):
    """Ground truth of every split that does not have it yet, for every metric."""
    info = DATASETS[name]
    missing = [split for split in ("learn", "tuning", "validation", "query")
               if any(vecs_shape(groundtruth_path(name, split, metric, extension)) !=
                      (vecs_shape(vectors_path(name, split))[0], GT_K)
                      for metric in info["metrics"] for extension in ("ivecs", "fvecs"))]
    if not missing:
        print(f">> Keeping the ground truth of {name}")
        return
    print(f">> Ground truth of {name} with {GT_LIBRARY}: splits {missing}, metrics {info['metrics']}, k={GT_K}")
    started = time.perf_counter()
    base = np.array(open_vectors(vectors_path(name, "base")))
    log_time(name, "load base", time.perf_counter() - started)
    for metric in info["metrics"]:
        started = time.perf_counter()
        search = faiss_search(base, metric) if GT_LIBRARY == "faiss" else daisy_search(base, metric)
        log_time(name, "build index", time.perf_counter() - started, metric=metric)
        for split in missing:
            queries = np.array(open_vectors(vectors_path(name, split)))
            distances = np.empty((len(queries), GT_K), np.float32)
            ids = np.empty((len(queries), GT_K), np.int32)
            started = time.perf_counter()
            for start in tqdm(range(0, len(queries), query_batch), desc=f"{name} {split} ground truth"):
                distances[start:start + query_batch], ids[start:start + query_batch] = search(queries[start:start + query_batch])
            seconds = time.perf_counter() - started  # the search alone, without writing
            if (ids < 0).any():
                raise ValueError(f"Fewer than {GT_K} neighbours for {name} {split}")
            write_vecs(groundtruth_path(name, split, metric, "ivecs"), ids, '<i4')
            write_vecs(groundtruth_path(name, split, metric, "fvecs"), distances)
            # Logged once the files are written, so a stopped run leaves no row for this split.
            log_time(name, "search", seconds, split, metric, len(queries))


def main():
    names = sys.argv[1:] or list(DATASETS)
    for name in names:
        if name not in DATASETS:
            sys.exit(f"Unknown dataset: {name}. Choose from: {', '.join(DATASETS)}")
    for name in names:
        print(f">> {name}")
        os.makedirs(f"{DATA_DIR}/{name}", exist_ok=True)
        organize_base(name)
        organize_queries(name)
        compute_groundtruths(name)
        if name in WORKLOAD_DATASETS:
            (metric,) = DATASETS[name]["metrics"]
            query_generator.generate_workloads(name, vectors_path(name, "base"), metric, vectors_path(name, "query"),
                                               groundtruth_path(name, "query", metric, "ivecs"))

if __name__ == "__main__":
    main()