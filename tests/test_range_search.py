"""
Correctness test for range (distance-r) search across DaiSy algorithms.
Ground truth is computed on-the-fly using FAISS IndexFlatL2.range_search().

Usage:
    python tests/test_range_search.py
    python tests/test_range_search.py --dim 96 --n_db 50000 --n_query 100 --r 20.0
    python tests/test_range_search.py --seed 7 --r 30.0 --tol 0.02

The FAISS L2 index stores squared Euclidean distances, matching DaiSy's
DistanceType.L2_SQUARED.  r is therefore an L2-squared threshold.
"""

import argparse
import sys
import os
import tempfile
import shutil

import numpy as np

try:
    import faiss
except ImportError:
    print("faiss not found. Install with:  pip install faiss-cpu")
    sys.exit(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from daisy import DistanceType, SearchConfig, QueryType
from daisy import BruteForceSearch, Messi, Fresh, DumpyOS, LbBruteforce, Sing, Hercules, HerculesConfig, ParIS

try:
    from daisy import Sofa
    _sofa_available = True
except ImportError:
    _sofa_available = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def faiss_range_gt(db: np.ndarray, queries: np.ndarray, r: float):
    """Run FAISS range search and return per-query sorted (indices, distances)."""
    index = faiss.IndexFlatL2(db.shape[1])
    index.add(db)
    lims, D_flat, I_flat = index.range_search(queries, r)

    result_I, result_D = [], []
    for qi in range(len(queries)):
        sl = slice(int(lims[qi]), int(lims[qi + 1]))
        order = np.argsort(D_flat[sl])
        result_I.append(I_flat[sl][order].tolist())
        result_D.append(D_flat[sl][order].tolist())
    return result_I, result_D


def run_daisy_range(algo, db: np.ndarray, queries: np.ndarray, r: float, db_path: str = None):
    """Build index and run range search; return per-query sorted (indices, distances)."""
    if db_path is not None:
        algo.buildIndex(db_path, db.shape[1], db.shape[0])
    else:
        algo.buildIndex(db)
    cfg = SearchConfig()
    cfg.type = QueryType.RANGE
    cfg.r = float(r)
    I_raw, D_raw = algo.searchIndex(queries, cfg)

    result_I, result_D = [], []
    for qi in range(len(queries)):
        idx = np.array(I_raw[qi], dtype=np.int64)
        dist = np.array(D_raw[qi], dtype=np.float32)
        order = np.argsort(idx)
        result_I.append(idx[order].tolist())
        result_D.append(dist[order].tolist())
    return result_I, result_D


def compare(name: str, gt_I, algo_I, algo_D, r: float, tol: float) -> bool:
    """
    Check two things:
      1. No false positives: every returned distance is <= r*(1+tol).
      2. No excessive false negatives: algo misses at most tol fraction of GT results.
    """
    n_query = len(gt_I)
    failures = []

    for qi in range(n_query):
        # false positives
        for d in algo_D[qi]:
            if d > r * (1.0 + tol):
                failures.append(f"  query {qi}: result dist {d:.6f} > r={r:.4f} (tol={tol})")
                break

        # false negatives
        gt_set = set(gt_I[qi])
        algo_set = set(algo_I[qi])
        missing = gt_set - algo_set
        allowed_miss = max(1, int(tol * len(gt_set)))
        if len(missing) > allowed_miss:
            failures.append(
                f"  query {qi}: {len(missing)}/{len(gt_set)} GT results missing "
                f"(allowed {allowed_miss})"
            )

    if failures:
        print(f"[FAIL] {name}")
        for msg in failures[:5]:      # cap output
            print(msg)
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more")
        return False

    avg = np.mean([len(x) for x in algo_I])
    print(f"[PASS] {name}  (avg {avg:.1f} results/query)")
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dim',     type=int,   default=96)
    parser.add_argument('--n_db',    type=int,   default=50000)
    parser.add_argument('--n_query', type=int,   default=100)
    parser.add_argument('--r',       type=float, default=None,
                        help='L2-squared radius (default: 0.20 * 2 * dim)')
    parser.add_argument('--tol',     type=float, default=0.05,
                        help='Allowed fraction of missing/extra results (default 0.05)')
    parser.add_argument('--seed',    type=int,   default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    db      = np.random.randn(args.n_db,    args.dim).astype(np.float32)
    queries = np.random.randn(args.n_query, args.dim).astype(np.float32)

    r = args.r if args.r is not None else 0.20 * 2.0 * args.dim

    print(f"n_db={args.n_db}, dim={args.dim}, n_query={args.n_query}, r={r:.4f}, tol={args.tol}")

    print("Computing FAISS ground truth ...")
    gt_I, gt_D = faiss_range_gt(db, queries, r)
    avg_gt = np.mean([len(x) for x in gt_I])
    print(f"  Ground truth: avg {avg_gt:.1f} results/query\n")

    hercules_dir = tempfile.mkdtemp(prefix="daisy_hercules_range_")
    hercules_cfg = HerculesConfig()
    hercules_cfg.index_dir = hercules_dir

    paris_dir = tempfile.mkdtemp(prefix="daisy_paris_range_")
    paris_db_file = os.path.join(paris_dir, "db.bin")
    db.tofile(paris_db_file)

    algorithms = [
        ("BruteForce",   BruteForceSearch(DistanceType.L2_SQUARED), None),
        ("Messi",        Messi(DistanceType.L2_SQUARED),             None),
        ("Fresh",        Fresh(DistanceType.L2_SQUARED),             None),
        ("DumpyOS",      DumpyOS(DistanceType.L2_SQUARED),           None),
        ("LbBruteforce", LbBruteforce(DistanceType.L2_SQUARED),      None),
        ("Sing",         Sing(DistanceType.L2_SQUARED),              None),
        ("Hercules",     Hercules(DistanceType.L2_SQUARED, hercules_cfg), None),
        ("ParIS",        ParIS(DistanceType.L2_SQUARED),             paris_db_file),
    ]
    if _sofa_available:
        algorithms.append(("Sofa", Sofa(DistanceType.L2_SQUARED), None))

    passed, failed = [], []
    try:
        for name, algo, db_path in algorithms:
            try:
                algo_I, algo_D = run_daisy_range(algo, db, queries, r, db_path)
                ok = compare(name, gt_I, algo_I, algo_D, r, args.tol)
                (passed if ok else failed).append(name)
            except Exception as exc:
                print(f"[ERROR] {name}: {exc}")
                failed.append(name)
    finally:
        shutil.rmtree(hercules_dir, ignore_errors=True)
        shutil.rmtree(paris_dir, ignore_errors=True)

    print(f"\nPassed: {passed}")
    print(f"Failed: {failed}")
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
