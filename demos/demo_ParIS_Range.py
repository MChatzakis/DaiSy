import sys
import os
import tempfile
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from daisy import DistanceType, SearchConfig, QueryType, ParIS, BruteForceSearch


def main():
    n_database = 50000
    dim = 96
    n_query = 5
    r = 0.20 * 2.0 * dim

    np.random.seed(42)
    db    = np.random.randn(n_database, dim).astype(np.float32)
    query = np.random.randn(n_query, dim).astype(np.float32)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
    temp_db_file = temp_file.name
    temp_file.close()

    try:
        db.tofile(temp_db_file)

        index = ParIS(DistanceType.L2_SQUARED)
        index.setNumThreads(4)
        index.buildIndex(temp_db_file, dim, n_database)

        cfg = SearchConfig()
        cfg.type = QueryType.RANGE
        cfg.r = r

        I, D = index.searchIndex(query, cfg)

        print(f"=== ParIS Range Search (r={r:.1f}) ===")
        for qi in range(n_query):
            n_res = len(I[qi])
            print(f"  Query {qi}: {n_res} results within L2sq <= {r}", end="")
            if n_res > 0:
                print(f"  [closest={D[qi][0]:.4f}, farthest={D[qi][-1]:.4f}]", end="")
            print()

        bf = BruteForceSearch(DistanceType.L2_SQUARED)
        bf.buildIndex(db)
        I_bf, _ = bf.searchIndex(query, cfg)

        print(f"\n=== Verification against BruteForce ===")
        all_match = True
        for qi in range(n_query):
            match = set(I[qi]) == set(I_bf[qi])
            all_match = all_match and match
            print(f"  Query {qi}: ParIS={len(I[qi])}, BruteForce={len(I_bf[qi])}, sets equal={match}")

        print(f"\nAll queries match: {all_match}")

    finally:
        if os.path.exists(temp_db_file):
            os.remove(temp_db_file)


if __name__ == "__main__":
    main()
