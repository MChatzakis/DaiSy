"""
Demo Sing DTW — build index from NumPy array and run DTW search.
Requires Sing to be built with CUDA (SING_CUDA_ENABLED).
Note: current Sing DTW implementation only supports k = 1.
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from daisy import DistanceType, Sing


def main():
    # Small-ish sizes so the demo runs quickly
    n_database = 50000
    dim = 128
    n_query = 5
    k = 1  # DTW implementation in Sing currently supports only k = 1

    np.random.seed(100)
    db = np.random.randn(n_database, dim).astype(np.float32)
    np.random.seed(50)
    query = np.random.randn(n_query, dim).astype(np.float32)
    print(f"Loaded {n_database} database points and {n_query} query points with dimension {dim}")

    sing_search = Sing(DistanceType.DTW)
    # Optional: use multiple threads
    sing_search.setNumThreads(4)

    sing_search.buildIndex(db)
    print(f"buildIndex (DTW) done")

    print(f"Starting DTW search (n_query={n_query}, k={k})...")
    I, D = sing_search.searchIndex(query, k)
    print(f"Search done)")

    print("\nPython Sing DTW Search Results:")
    for i in range(n_query):
        print(f"Query {i}:")
        print("Distances:", D[i])
        print("Indices:", I[i])
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

