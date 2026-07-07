import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from daisy import DistanceType, SearchConfig, QueryType, Messi, BruteForceSearch


def main():
    n_database = 100000
    dim = 256
    n_query = 5

    np.random.seed(42)
    db    = np.random.randn(n_database, dim).astype(np.float32)
    query = np.random.randn(n_query, dim).astype(np.float32)

    index = Messi(DistanceType.L2_SQUARED)
    index.setNumThreads(4)
    index.buildIndex(db)

    # --- top-k search (existing API) ---
    k = 10
    I_topk, D_topk = index.searchIndex(query, k)

    print(f"=== Top-k Search (k={k}) ===")
    for qi in range(n_query):
        print(f"  Query {qi}: {k} nearest distances (L2sq): "
              f"{[round(float(d), 4) for d in D_topk[qi]]}")

    # --- range search (SearchConfig API) ---
    # r is the L2-squared distance threshold: return every vector with L2sq <= r
    r = 60.0

    cfg = SearchConfig()
    cfg.type = QueryType.RANGE
    cfg.r = r

    I_range, D_range = index.searchIndex(query, cfg)

    print(f"\n=== Range Search (r={r}) ===")
    for qi in range(n_query):
        n_res = len(I_range[qi])
        print(f"  Query {qi}: {n_res} results within L2sq <= {r}", end="")
        if n_res > 0:
            print(f"  [closest={D_range[qi][0]:.4f}, farthest={D_range[qi][-1]:.4f}]", end="")
        print()

    # --- verify against brute force ---
    bf = BruteForceSearch(DistanceType.L2_SQUARED)
    bf.buildIndex(db)
    I_bf, D_bf = bf.searchIndex(query, cfg)

    print(f"\n=== Verification against BruteForce ===")
    all_match = True
    for qi in range(n_query):
        messi_set = set(I_range[qi])
        bf_set    = set(I_bf[qi])
        match     = messi_set == bf_set
        all_match = all_match and match
        print(f"  Query {qi}: Messi={len(I_range[qi])}, BruteForce={len(I_bf[qi])}, "
              f"sets equal={match}")

    print(f"\nAll queries match: {all_match}")


if __name__ == "__main__":
    main()
