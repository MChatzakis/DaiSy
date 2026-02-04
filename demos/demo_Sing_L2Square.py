"""
Demo Sing L2Square — build index from NumPy array and search (same data/seeds as demo_Messi_L2Square).
Requires diNoSimilaritySearch built with Sing (SING_CUDA_ENABLED).
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from diNoSimilaritySearch import DistanceType, Sing
except ImportError as e:
    print("Error: Sing is not available in the Python bindings.")
    print("Build the project with CUDA (SING_CUDA=ON) so that Sing is compiled and exposed.")
    print("See README or INSTALL.md for build instructions.")
    sys.exit(1)


def main():
    # 0. Configuration (same as C++ demo_Sing_L2Square and demo_Messi_L2Square)
    n_database = 200_000
    dim = 96
    n_query = 10
    k = 5

    print("=== Sing L2Square demo (same dataset as demo_Messi_L2Square) ===")
    print(f"n_database={n_database} dim={dim} n_query={n_query} k={k}")

    # 1. Generate random data and queries (seeds 100 database, 50 query)
    np.random.seed(100)
    db = np.random.randn(n_database, dim).astype(np.float32)
    np.random.seed(50)
    query = np.random.randn(n_query, dim).astype(np.float32)
    print(f"Loaded {n_database} database points and {n_query} query points with dimension {dim}")

    # 2. Create Sing search object (L2 squared)
    sing_search = Sing(DistanceType.L2_SQUARED)
    sing_search.setNumThreads(4)

    # 3. Build the index
    t0 = time.perf_counter()
    sing_search.buildIndex(db)
    t1 = time.perf_counter()
    print(f"buildIndex done in {(t1 - t0) * 1000:.2f} ms")

    # 4. Search the index
    print(f"Starting search (n_query={n_query}, k={k})...")
    t_search0 = time.perf_counter()
    I, D = sing_search.searchIndex(query, k)
    t_search1 = time.perf_counter()
    search_ms = (t_search1 - t_search0) * 1000
    print(f"Search done in {search_ms:.2f} ms ({search_ms / n_query:.2f} ms/query)")

    # 5. Print results (same format as demo_Odyssey_L2Square / demo_ParIS_L2Square)
    for i in range(n_query):
        indices_str = " ".join(str(I[i, j]) for j in range(k))
        print(f"Query {i}: {indices_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
