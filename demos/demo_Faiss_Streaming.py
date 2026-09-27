"""FAISS streaming inserts, side by side with the exact bruteforce baseline.

FaissFlat is only present when DaiSy was built with BUILD_FAISS=ON.
"""

import numpy as np

from daisy import BruteForceSearch, DistanceType

try:
    from daisy import FaissFlat
except ImportError:
    FaissFlat = None


def main():
    if FaissFlat is None:
        print("FaissFlat is not available: rebuild DaiSy with -DBUILD_FAISS=ON")
        return

    rng = np.random.default_rng(100)
    stream = rng.normal(size=(1000, 32)).astype(np.float32)
    query = rng.normal(size=(5, 32)).astype(np.float32)

    index = FaissFlat(DistanceType.L2_SQUARED)
    index.buildIndex(stream[:500])
    index.insert(stream[500])
    index.insertBatch(stream[501:])
    print("FaissFlat now contains", index.getNDatabase(), "series")

    indices, distances = index.searchIndex(query, 3)
    print("FaissFlat query 0 indices:", indices[0])
    print("FaissFlat query 0 distances:", distances[0])

    # A flat FAISS index stays exact across inserts, so it must agree with a
    # bruteforce scan over the whole stream.
    truth = BruteForceSearch(DistanceType.L2_SQUARED)
    truth.buildIndex(stream)
    truth_indices, _ = truth.searchIndex(query, 3)
    print("matches bruteforce:", np.array_equal(indices, truth_indices))


if __name__ == "__main__":
    main()
