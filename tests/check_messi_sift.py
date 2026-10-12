"""Manual SIFT exactness audit. Requires numpy, faiss-cpu and built daisy bindings.

python tests/check_messi_sift.py --dataset /path/to/sift --output report.json
Dataset must contain sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs.
Uses raw descriptors, exact squared L2, and permits arbitrary IDs at exact ties.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import faiss
import daisy


def vectors(path, dtype):
    raw = np.memmap(path, dtype=np.int32, mode="r")
    width = int(raw[0])
    rows = raw.reshape(-1, width + 1)
    if not np.all(rows[:, 0] == width):
        raise ValueError(f"Invalid dimension headers: {path}")
    return rows.view(dtype)[:, 1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()
    base = np.ascontiguousarray(vectors(args.dataset / "sift_base.fvecs", np.float32))
    query = vectors(args.dataset / "sift_query.fvecs", np.float32)[:args.queries]
    published = vectors(args.dataset / "sift_groundtruth.ivecs", np.int32)[:args.queries]
    if base.shape != (1000000, 128):
        raise ValueError(f"Expected original SIFT1M, got {base.shape}")
    # Use direct L2 rather than cancellation-prone norm/dot-product evaluation.
    faiss.omp_set_num_threads(4)
    faiss.cvar.distance_compute_blas_threshold = 1000000000
    report = {"dataset": str(args.dataset.resolve()), "normalized_input": False,
              "query_shape": list(query.shape), "query_strides": list(query.strides),
              "module": daisy._core.__file__, "cases": []}
    reference = {}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def expected(size):
        if size not in reference:
            index = faiss.IndexFlatL2(base.shape[1])
            index.add(base[:size])
            distances, _ = index.search(np.ascontiguousarray(query), 1000)
            reference[size] = distances
            if size == len(base):
                stored = ((base[published].astype(np.float64) -
                           query.astype(np.float64)[:, None, :]) ** 2).sum(axis=2)
                np.testing.assert_array_equal(distances[:, :published.shape[1]], stored)
                report["published_groundtruth_matches"] = True
            print(f"reference ready: {size}", flush=True)
        return reference[size]

    # Small static reproduction, full static index, and large streaming index.
    for stages in ((20000,), (1000000,), (750000, 875000, 1000000)):
        for normalized in (False, True):
            for workers in (1, 4):
                index = daisy.Messi(daisy.DistanceType.L2_SQUARED)
                index.setNormalized(normalized)
                index.setSearchWorkers(workers)
                index.setIndexWorkers(2)
                initial = np.ascontiguousarray(base[:stages[0]])
                index.buildIndex(initial)
                previous = stages[0]
                for size in stages:
                    if size != previous:
                        index.insertBatch(base[previous:size])
                    previous = size
                    gt = expected(size)
                    for k in (1, 10, 100, 1000):
                        for repeat in range(2):
                            # Deliberately retain fvecs header strides at the Python API.
                            ids, distances = index.searchIndex(query, k)
                            ids = np.asarray(ids, dtype=np.int64)
                            physical = ((base[ids].astype(np.float64) -
                                         query.astype(np.float64)[:, None, :]) ** 2).sum(axis=2)
                            case = {"initial": stages[0], "size": size,
                                    "mode": "gaussian" if normalized else "adaptive",
                                    "workers": workers, "k": k, "repeat": repeat,
                                    "wrong_distances": int(np.count_nonzero(distances != gt[:, :k])),
                                    "invalid_id_distance_pairs": int(np.count_nonzero(distances != physical)),
                                    "duplicate_rows": sum(len(set(row)) != k for row in ids)}
                            report["cases"].append(case)
                            save()
                            print(json.dumps(case), flush=True)
                            if any(case[key] for key in ("wrong_distances", "invalid_id_distance_pairs", "duplicate_rows")):
                                return 1
    report["passed"] = True
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
