import glob
import os
import struct
import sys
import tarfile

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

# Raw Hugging Face parquet files (about 48 GB). They live on the root disk because
# /mnt/hddhelp has no room for them next to the processed vectors (about 96 GB).
# Environment overrides for other machines: HF_DOWNLOAD_DIRECTORY, DATASET_DIRECTORY (as in
# experiments/config.sh), NUM_THREADS.
DIRECTORY_TO_DOWNLOAD = os.environ.get("HF_DOWNLOAD_DIRECTORY", "/home/mchatzakis/hf_datasets")
DIRECTORY_TO_SAVE = os.environ.get("DATASET_DIRECTORY", "/mnt/hddhelp/mchatzakis/datasets/processed")
SEED = 42
# Ground-truth depth: exact neighbors (IDs and
# distances) stored per query. Both metrics are computed, each with its file prefix.
GT_K = 5000
GT_PREFIXES = {"inner-product": "innprod.", "L2": "l2."}
THREADS = int(os.environ.get("NUM_THREADS", "48"))
torch.set_num_threads(THREADS)

COHERE_REPO = "CohereLabs/beir-embed-english-v3"
# The BGE-M3 authors' Pyserini Faiss indexes of MIRACL, and MIRACL's topics and qrels.
BGE_M3_MIRACL_REPO = "hanhainebula/bge-m3_miracl_2cr"
MIRACL_REPO = "miracl/miracl"
# Query splits; learn and validation are disjoint samples of the train queries.
SPLITS = ("learn", "validation", "query")
# Each source is (Hugging Face dataset repo, file pattern or list of patterns inside it).
# A dataset has either learn_qrels or learn_topics: the files listing its train queries.
SOURCE_KEYS = ("corpus", "queries", "query_qrels", "learn_qrels", "learn_topics")

dataset_info = {
    # BEIR MSMARCO passages embedded with Cohere embed-english-v3.0 (unit norm).
    # Evaluation: the 6,980 dev queries. Learning (DARTH training) and validation: 10K and 1K
    # disjoint samples of the 502,939 train queries.
    "MSMARCO8M": {
        "corpus": (COHERE_REPO, "msmarco/corpus/*.parquet"),
        "queries": (COHERE_REPO, "msmarco/queries/*.parquet"),
        "query_qrels": (COHERE_REPO, "msmarco/qrels/dev.parquet"),
        "learn_qrels": (COHERE_REPO, "msmarco/qrels/train.parquet"),
        "id_column": "_id",
        "embedding_column": "emb",
        "normalize": False,

        "base_vectors": 8841823,
        "query_vectors": 6980,
        "learn_pool_vectors": 502939,
        "learn_vectors": 10000,
        "validation_vectors": 1000,
        "d": 1024
    },
    # BEIR HotpotQA embedded with Cohere embed-english-v3.0 (unit norm).
    # Evaluation: the 7,405 test queries. Learning and validation: 10K and 1K of the 85,000 train queries.
    "HOTPOTQA5M": {
        "corpus": (COHERE_REPO, "hotpotqa/corpus/*.parquet"),
        "queries": (COHERE_REPO, "hotpotqa/queries/*.parquet"),
        "query_qrels": (COHERE_REPO, "hotpotqa/qrels/test.parquet"),
        "learn_qrels": (COHERE_REPO, "hotpotqa/qrels/train.parquet"),
        "id_column": "_id",
        "embedding_column": "emb",
        "normalize": False,

        "base_vectors": 5233329,
        "query_vectors": 7405,
        "learn_pool_vectors": 85000,
        "learn_vectors": 10000,
        "validation_vectors": 1000,
        "d": 1024
    },
    # BEIR FEVER (Wikipedia abstracts) embedded with Cohere embed-english-v3.0 (unit norm).
    # Evaluation: the 6,666 test claims. Learning and validation: 10K and 1K of the 109,810 train claims.
    "FEVER5M": {
        "corpus": (COHERE_REPO, "fever/corpus/*.parquet"),
        "queries": (COHERE_REPO, "fever/queries/*.parquet"),
        "query_qrels": (COHERE_REPO, "fever/qrels/test.parquet"),
        "learn_qrels": (COHERE_REPO, "fever/qrels/train.parquet"),
        "id_column": "_id",
        "embedding_column": "emb",
        "normalize": False,

        "base_vectors": 5416568,
        "query_vectors": 6666,
        "learn_pool_vectors": 109810,
        "learn_vectors": 10000,
        "validation_vectors": 1000,
        "d": 1024
    },
    # MIRACL Japanese Wikipedia passages embedded with BGE-M3 (dense, unit norm): the BGE-M3
    # authors' Faiss flat index. Queries are text, encoded here with the same model.
    # Evaluation: the 860 dev queries (MIRACL's test qrels are hidden). Learning and validation:
    # the 5,268 train, test-A and test-B queries; DARTH learns from exact neighbors only, so the
    # hidden test qrels don't matter, and all but the 1K validation queries are learned from.
    "MIRACLJA6M": {
        "corpus": (BGE_M3_MIRACL_REPO, "dense/ja.tar.gz"),
        "corpus_format": "faiss",
        "queries": (MIRACL_REPO, "miracl-v1.0-ja/topics/*.tsv"),
        "query_encoder": "BAAI/bge-m3",
        "query_qrels": (MIRACL_REPO, "miracl-v1.0-ja/qrels/qrels.miracl-v1.0-ja-dev.tsv"),
        "learn_topics": (MIRACL_REPO, [f"miracl-v1.0-ja/topics/topics.miracl-v1.0-ja-{split}.tsv"
                                       for split in ("train", "test-a", "test-b")]),
        "normalize": False,

        "base_vectors": 6953614,
        "query_vectors": 860,
        "learn_pool_vectors": 5268,
        "learn_vectors": 4268,
        "validation_vectors": 1000,
        "d": 1024
    },
    # MIRACL Russian, as MIRACLJA6M. Evaluation: the 1,252 dev queries; learning and validation:
    # the 6,312 train, test-A and test-B queries.
    "MIRACLRU9M": {
        "corpus": (BGE_M3_MIRACL_REPO, "dense/ru.tar.gz"),
        "corpus_format": "faiss",
        "queries": (MIRACL_REPO, "miracl-v1.0-ru/topics/*.tsv"),
        "query_encoder": "BAAI/bge-m3",
        "query_qrels": (MIRACL_REPO, "miracl-v1.0-ru/qrels/qrels.miracl-v1.0-ru-dev.tsv"),
        "learn_topics": (MIRACL_REPO, [f"miracl-v1.0-ru/topics/topics.miracl-v1.0-ru-{split}.tsv"
                                       for split in ("train", "test-a", "test-b")]),
        "normalize": False,

        "base_vectors": 9543918,
        "query_vectors": 1252,
        "learn_pool_vectors": 6312,
        "learn_vectors": 5312,
        "validation_vectors": 1000,
        "d": 1024
    },
}

def size_label(n):
    return f"{n // 1000}K" if n % 1000 == 0 else str(n)

def processed_filepaths(ds_name):
    """Vector, id and qrels files of a dataset, named as hnswlib/runners/helpers.hpp expects."""
    info = dataset_info[ds_name]
    directory = f"{DIRECTORY_TO_SAVE}/{ds_name}"
    base = size_label(info["base_vectors"])
    paths = {
        "base": f"{directory}/base.{base}.fvecs",
        # Row i of each .fvecs file is the BEIR document/query with the i-th listed id.
        "base_ids": f"{directory}/base.{base}.ids.txt",
        "query_qrels": f"{directory}/query.{size_label(info['query_vectors'])}.qrels.tsv",
    }
    for split in SPLITS:
        size = size_label(info[f"{split}_vectors"])
        paths[split] = f"{directory}/{split}.{size}.fvecs"
        paths[f"{split}_ids"] = f"{directory}/{split}.{size}.ids.txt"
    return paths

def groundtruth_filepaths(ds_name, split, metric):
    """(IDs .ivecs, distances .fvecs) of a split's ground truth for one metric."""
    info = dataset_info[ds_name]
    stem = (f"{DIRECTORY_TO_SAVE}/{ds_name}/{GT_PREFIXES[metric]}{split}.groundtruth."
            f"{size_label(info[f'{split}_vectors'])}.k{GT_K}")
    return f"{stem}.ivecs", f"{stem}.fvecs"

def write_vecs(f, vectors, dtype='<f4'):
    """Write rows as .fvecs ('<f4') or .ivecs ('<i4') records to a path or an open binary file."""
    vectors = np.asarray(vectors)
    records = np.empty((len(vectors), vectors.shape[1] + 1), dtype=dtype)
    records.view('<i4')[:, 0] = vectors.shape[1]
    records[:, 1:] = vectors
    records.tofile(f)

def vecs_shape(file_path):
    """(rows, dim) of an .fvecs/.ivecs file from its first header and size; None if absent or malformed."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 4:
        return None
    dim = int(np.fromfile(file_path, dtype='<i4', count=1)[0])
    size = os.path.getsize(file_path)
    if dim <= 0 or size % (4 * (dim + 1)):
        return None
    return size // (4 * (dim + 1)), dim

def check_vecs(file_path, rows, dim):
    shape = vecs_shape(file_path)
    if shape != (rows, dim):
        raise ValueError(f"{file_path} has shape {shape}, expected {(rows, dim)}")
    print(f">> Verified {os.path.basename(file_path)}: {rows} x {dim}")

def load_fvecs(file_path, desc, chunk_rows=1_000_000):
    """Read an .fvecs file into one contiguous array, checking every header and value."""
    rows, dim = vecs_shape(file_path)
    records = np.memmap(file_path, dtype='<f4', mode='r', shape=(rows, dim + 1))
    vectors = np.empty((rows, dim), dtype=np.float32)
    for start in tqdm(range(0, rows, chunk_rows), desc=desc):
        chunk = records[start:start + chunk_rows]
        if not (np.all(chunk.view('<i4')[:, 0] == dim) and np.all(np.isfinite(chunk[:, 1:]))):
            raise ValueError(f"Malformed rows or non-finite values in {file_path}")
        vectors[start:start + len(chunk)] = chunk[:, 1:]
    return vectors

def write_lines(file_name, values):
    with open(file_name, 'w') as f:
        f.writelines(f"{value}\n" for value in values)

def source_patterns(source):
    repo, patterns = source
    return repo, [patterns] if isinstance(patterns, str) else list(patterns)

def source_files(source):
    repo, patterns = source_patterns(source)
    files = []
    for pattern in patterns:
        matches = sorted(glob.glob(f"{DIRECTORY_TO_DOWNLOAD}/{repo}/{pattern}"))
        if not matches:
            raise FileNotFoundError(f"No files match {DIRECTORY_TO_DOWNLOAD}/{repo}/{pattern}")
        files += matches
    return files

def download_sources(ds_name):
    """Fetch only the files a dataset needs; already downloaded files are skipped."""
    info = dataset_info[ds_name]
    patterns = {}
    for key in SOURCE_KEYS:
        if key in info:
            repo, key_patterns = source_patterns(info[key])
            patterns.setdefault(repo, []).extend(key_patterns)
    for repo, allow_patterns in patterns.items():
        print(f">> Downloading {repo}: {allow_patterns}")
        snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=allow_patterns,
                          local_dir=f"{DIRECTORY_TO_DOWNLOAD}/{repo}", max_workers=8)

def checked_vectors(vectors, info, path):
    """The vectors after rejecting non-finite values, normalized if the dataset asks for it."""
    if not np.all(np.isfinite(vectors)):
        raise ValueError(f"Non-finite embedding values in {path}")
    if info["normalize"]:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.all(norms > 0):
            raise ValueError(f"Zero-norm embedding in {path}")
        vectors = vectors / norms
    return vectors

def iter_embeddings(files, info, desc, batch_rows=50_000):
    """Yield (ids, float32 vectors) batches in file order, normalized if the dataset asks for it."""
    id_column, embedding_column, d = info["id_column"], info["embedding_column"], info["d"]
    total = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    with tqdm(total=total, desc=desc, unit="vec") as progress:
        for path in files:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=batch_rows, columns=[id_column, embedding_column]):
                ids, embeddings = batch.column(id_column), batch.column(embedding_column)
                if ids.null_count or embeddings.null_count or not np.all(pc.list_value_length(embeddings).to_numpy() == d):
                    raise ValueError(f"Missing ids or non-{d}-dimensional embeddings in {path}")
                vectors = embeddings.flatten().to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(-1, d)
                vectors = checked_vectors(vectors, info, path)
                progress.update(len(vectors))
                yield ids.to_pylist(), vectors

def write_faiss_corpus(path, info, f, desc, batch_rows=50_000):
    """Copy the vectors of a Pyserini Faiss flat index (a .tar.gz of <name>/index and <name>/docid)
    into the open .fvecs file f, streaming the archive, and return the document ids in row order."""
    d = info["d"]
    ids, rows = None, 0
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            name = os.path.basename(member.name)
            if name == "docid":
                ids = archive.extractfile(member).read().decode("utf-8").split()
            elif name == "index":
                index = archive.extractfile(member)
                # Faiss IndexFlat header: "IxFI", d (int32), ntotal (int64), two unused int64s,
                # is_trained (uint8), metric (int32, 0 = inner product), then the float count (int64).
                magic, dim, ntotal, _, _, _, metric, count = struct.unpack("<4siqqqBiq", index.read(45))
                if magic != b"IxFI" or dim != d or metric != 0 or count != ntotal * d:
                    raise ValueError(f"{path} is not a {d}-dimensional Faiss inner-product flat index")
                with tqdm(total=ntotal, desc=desc, unit="vec") as progress:
                    while rows < ntotal:
                        n = min(batch_rows, ntotal - rows)
                        buffer = index.read(n * d * 4)
                        if len(buffer) != n * d * 4:
                            raise ValueError(f"Truncated Faiss index in {path}")
                        write_vecs(f, checked_vectors(np.frombuffer(buffer, dtype='<f4').reshape(n, d), info, path))
                        rows += n
                        progress.update(n)
    if ids is None or rows == 0 or len(ids) != rows:
        raise ValueError(f"{path} holds {rows} vectors and {0 if ids is None else len(ids)} document ids")
    return ids

def read_topics(path):
    """(query ids, texts) of a topics file with one "query_id<TAB>text" line per query, as MIRACL ships them."""
    with open(path, encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("\t", 1) for line in f if line.strip()]
    if any(len(row) != 2 for row in rows):
        raise ValueError(f"Malformed topics line in {path}")
    return [row[0] for row in rows], [row[1] for row in rows]

def read_qrels(path):
    """(query_id, corpus_id, score) table of a BEIR qrels parquet or of a TREC qrels file
    ("query_id Q0 corpus_id score", as MIRACL ships them, with judged non-relevant rows at score 0)."""
    if path.endswith(".parquet"):
        return pq.read_table(path, columns=["query_id", "corpus_id", "score"])
    with open(path, encoding="utf-8") as f:
        rows = [line.split() for line in f if line.strip()]
    if any(len(row) != 4 for row in rows):
        raise ValueError(f"Malformed TREC qrels line in {path}")
    return pa.table({"query_id": [row[0] for row in rows], "corpus_id": [row[2] for row in rows],
                     "score": [int(row[3]) for row in rows]})

def encode_queries(model_name, texts, desc, batch_size=32):
    """Unit-norm dense query embeddings, as Pyserini's AutoQueryEncoder computes them with --pooling cls
    --l2-norm (the BGE-M3 authors' MIRACL runs): the first token's final hidden state, normalized."""
    # Imported here: only datasets whose queries are text need transformers.
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    batches = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size), desc=desc):
            inputs = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, return_tensors="pt")
            first_token = model(**inputs).last_hidden_state[:, 0]
            batches.append(torch.nn.functional.normalize(first_token, dim=-1).numpy())
    return np.concatenate(batches).astype(np.float32, copy=False)

def iter_query_embeddings(info, desc):
    """Yield (ids, vectors) batches of a dataset's queries: precomputed embeddings, or topic texts
    encoded with info["query_encoder"]."""
    if "query_encoder" not in info:
        yield from iter_embeddings(source_files(info["queries"]), info, desc)
        return
    ids, texts = [], []
    for path in source_files(info["queries"]):
        path_ids, path_texts = read_topics(path)
        ids += path_ids
        texts += path_texts
    yield ids, encode_queries(info["query_encoder"], texts, desc)

def split_query_ids(ds_name):
    """({split: query ids} for learn, validation and query, evaluation qrels) of a dataset.

    Splits come from the qrels rather than the query files, so every split holds judged queries.
    Learn is drawn first and validation from the remaining train queries, so adding the
    validation split left the learn sample unchanged. MIRACL lists its train queries in
    topic files (learn_topics), since some of them have no public qrels.
    """
    info = dataset_info[ds_name]
    [query_qrels_path] = source_files(info["query_qrels"])
    query_qrels = read_qrels(query_qrels_path)
    query_ids = sorted(set(query_qrels.column("query_id").to_pylist()))
    if "learn_topics" in info:
        learn_pool = sorted({qid for path in source_files(info["learn_topics"]) for qid in read_topics(path)[0]})
    else:
        [learn_qrels_path] = source_files(info["learn_qrels"])
        learn_pool = sorted(set(read_qrels(learn_qrels_path).column("query_id").to_pylist()))
    if len(query_ids) != info["query_vectors"] or len(learn_pool) != info["learn_pool_vectors"]:
        raise ValueError(f"{ds_name} qrels list {len(query_ids)} evaluation and {len(learn_pool)} train queries; "
                         f"expected {info['query_vectors']} and {info['learn_pool_vectors']}")
    if not set(query_ids).isdisjoint(learn_pool):
        raise ValueError(f"{ds_name} evaluation and train qrels share queries")
    rng = np.random.default_rng(SEED)
    learn_rows = np.sort(rng.choice(len(learn_pool), info["learn_vectors"], replace=False))
    remaining_rows = np.setdiff1d(np.arange(len(learn_pool)), learn_rows)
    validation_rows = np.sort(rng.choice(remaining_rows, info["validation_vectors"], replace=False))
    return ({"learn": [learn_pool[row] for row in learn_rows],
             "validation": [learn_pool[row] for row in validation_rows],
             "query": query_ids}, query_qrels)

def queries_organized(ds_name):
    info, paths = dataset_info[ds_name], processed_filepaths(ds_name)
    return (all(vecs_shape(paths[split]) == (info[f"{split}_vectors"], info["d"]) and os.path.exists(paths[f"{split}_ids"])
                for split in SPLITS) and os.path.exists(paths["query_qrels"]))

def base_organized(ds_name):
    info, paths = dataset_info[ds_name], processed_filepaths(ds_name)
    return vecs_shape(paths["base"]) == (info["base_vectors"], info["d"]) and os.path.exists(paths["base_ids"])

def organize_queries(ds_name):
    """Write the learn, validation and query vectors with their ids, and the evaluation qrels."""
    info = dataset_info[ds_name]
    paths = processed_filepaths(ds_name)
    d = info["d"]
    if queries_organized(ds_name):
        print(f">> Reusing the existing query splits of {ds_name}; delete them to rebuild")
        return
    os.makedirs(os.path.dirname(paths["base"]), exist_ok=True)
    print(f">> Organizing the query splits of {ds_name}")

    split_ids, query_qrels = split_query_ids(ds_name)
    all_ids = [qid for split in SPLITS for qid in split_ids[split]]
    split_rows = {qid: row for row, qid in enumerate(all_ids)}
    split_vectors = np.empty((len(split_rows), d), dtype=np.float32)
    found = np.zeros(len(split_rows), dtype=bool)
    for ids, vectors in iter_query_embeddings(info, f"{ds_name} queries"):
        for source_row, qid in enumerate(ids):
            row = split_rows.get(qid)
            if row is None:
                continue
            if found[row]:
                raise ValueError(f"Query {qid} appears twice in the {ds_name} query files")
            split_vectors[row] = vectors[source_row]
            found[row] = True
    if not found.all():
        raise ValueError(f"{np.count_nonzero(~found)} split queries are missing from the {ds_name} query files")

    for split in SPLITS:
        ids = split_ids[split]
        write_vecs(paths[split], split_vectors[[split_rows[qid] for qid in ids]])
        check_vecs(paths[split], len(ids), d)
        write_lines(paths[f"{split}_ids"], ids)
    write_lines(paths["query_qrels"], ["query_id\tcorpus_id\tscore"] + [
        f"{q}\t{c}\t{s}" for q, c, s in zip(*(query_qrels.column(name).to_pylist()
                                             for name in ("query_id", "corpus_id", "score")))])

def organize_base(ds_name):
    """Write the base vectors with their ids, after checking the evaluation qrels against them."""
    info = dataset_info[ds_name]
    paths = processed_filepaths(ds_name)
    if base_organized(ds_name):
        print(f">> Reusing the existing base vectors of {ds_name}; delete them to rebuild")
        return
    os.makedirs(os.path.dirname(paths["base"]), exist_ok=True)
    print(f">> Organizing the base vectors of {ds_name}")
    base_ids = []
    with open(paths["base"], 'wb') as f:
        if info.get("corpus_format") == "faiss":
            [corpus_path] = source_files(info["corpus"])
            base_ids = write_faiss_corpus(corpus_path, info, f, f"{ds_name} base")
        else:
            for ids, vectors in iter_embeddings(source_files(info["corpus"]), info, f"{ds_name} base"):
                write_vecs(f, vectors)
                base_ids.extend(ids)
    check_vecs(paths["base"], info["base_vectors"], info["d"])
    base_id_set = set(base_ids)
    if len(base_id_set) != len(base_ids):
        raise ValueError(f"{ds_name} corpus has duplicate document ids")
    _, query_qrels = split_query_ids(ds_name)
    missing = set(query_qrels.column("corpus_id").to_pylist()) - base_id_set
    if missing:
        raise ValueError(f"{len(missing)} documents of the {ds_name} evaluation qrels are not in the corpus")
    # Written last, so a run interrupted earlier is not mistaken for a finished one.
    write_lines(paths["base_ids"], base_ids)

def exact_knn(base, base_norms, queries, k, base_block_size):
    """Exact top-k by inner product (scores, descending) and by squared L2 (ascending).

    One product per base block serves both metrics; squared L2 is expanded as
    |q|^2 + |x|^2 - 2 q.x and clamped at zero, as FAISS computes it.
    """
    best = {}
    for start in range(0, len(base), base_block_size):
        scores = queries @ base[start:start + base_block_size].T
        candidates = {"inner-product": (scores, True),
                      "L2": (scores.mul(-2).add_(base_norms[start:start + base_block_size]), False)}
        for metric, (values, largest) in candidates.items():
            top_values, top_ids = torch.topk(values, min(k, values.shape[1]), dim=1, largest=largest)
            top_ids += start
            if metric in best:
                kept_values, kept_ids = best[metric]
                top_values, rows = torch.topk(torch.cat([kept_values, top_values], 1), k, dim=1, largest=largest)
                top_ids = torch.gather(torch.cat([kept_ids, top_ids], 1), 1, rows)
            best[metric] = top_values, top_ids
    l2_values, l2_ids = best["L2"]
    best["L2"] = (l2_values + (queries * queries).sum(1, keepdim=True)).clamp_(min=0), l2_ids
    return best

def groundtruth_complete(ds_name, split):
    count = dataset_info[ds_name][f"{split}_vectors"]
    return all(vecs_shape(path) == (count, GT_K)
               for metric in GT_PREFIXES for path in groundtruth_filepaths(ds_name, split, metric))

def compute_groundtruths(ds_name, query_batch_size=1000, base_block_size=250_000):
    """Exact top-GT_K neighbors of every query split still missing them, for both metrics."""
    info = dataset_info[ds_name]
    paths = processed_filepaths(ds_name)
    d, k = info["d"], GT_K
    splits = [split for split in SPLITS if not groundtruth_complete(ds_name, split)]
    if not splits:
        print(f">> Reusing the existing ground truth of {ds_name}; delete it to recompute")
        return
    print(f">> Ground truth for {ds_name}: splits={splits}, metrics={list(GT_PREFIXES)}, k={k}")
    check_vecs(paths["base"], info["base_vectors"], d)
    if k > info["base_vectors"]:
        raise ValueError(f"Ground-truth depth {k} exceeds the {info['base_vectors']} base vectors of {ds_name}")
    base = torch.from_numpy(load_fvecs(paths["base"], f"{ds_name} base"))
    base_norms = torch.cat([(block * block).sum(1) for block in base.split(base_block_size)])

    for split in splits:
        count = info[f"{split}_vectors"]
        check_vecs(paths[split], count, d)
        queries = torch.from_numpy(load_fvecs(paths[split], f"{ds_name} {split}"))
        results = {metric: (np.empty((count, k), dtype=np.float32), np.empty((count, k), dtype=np.int32))
                   for metric in GT_PREFIXES}
        for start in tqdm(range(0, count, query_batch_size), desc=f"{ds_name} {split} groundtruth"):
            batch = exact_knn(base, base_norms, queries[start:start + query_batch_size], k, base_block_size)
            for metric, (values, ids) in batch.items():
                results[metric][0][start:start + len(values)] = values.numpy()
                results[metric][1][start:start + len(values)] = ids.numpy()
        for metric, (distances, indices) in results.items():
            ids_path, distances_path = groundtruth_filepaths(ds_name, split, metric)
            write_vecs(ids_path, indices, '<i4')
            write_vecs(distances_path, distances)
            check_vecs(ids_path, count, k)
            check_vecs(distances_path, count, k)
    print(f">> Done ground truth for {ds_name}")


def main():
    # Downloads, organizes and computes both ground truths of every dataset (or only of the
    # named ones). Finished steps are verified and skipped, so a rerun resumes:
    #   python scripts/e2e_dataset_organizer.py [DATASET ...]
    all_datasets = sys.argv[1:] or list(dataset_info)
    for ds_name in all_datasets:
        if ds_name not in dataset_info:
            sys.exit(f"Unknown dataset: {ds_name}. Choose from: {', '.join(dataset_info)}")

    for ds_name in all_datasets:
        download_sources(ds_name)
        organize_queries(ds_name)
        organize_base(ds_name)
        compute_groundtruths(ds_name)

if __name__ == "__main__":
    main()