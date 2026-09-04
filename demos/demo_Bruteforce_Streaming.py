import numpy as np

from daisy import BruteForceSearch, DistanceType, LbBruteforce


def run_streaming(index, name, initial, single, batch, query):
    index.buildIndex(initial)
    index.insert(single)
    index.insertBatch(batch)
    indices, distances = index.searchIndex(query, 3)
    print(f"{name} query 0 indices:", indices[0])
    print(f"{name} query 0 distances:", distances[0])


def main():
    rng = np.random.default_rng(100)
    stream = rng.normal(size=(1000, 32)).astype(np.float32)
    query = rng.normal(size=(5, 32)).astype(np.float32)

    run_streaming(
        BruteForceSearch(DistanceType.L2_SQUARED),
        "BruteForceSearch",
        stream[:500],
        stream[500],
        stream[501:],
        query,
    )
    run_streaming(
        LbBruteforce(DistanceType.L2_SQUARED),
        "LbBruteforce",
        stream[:500],
        stream[500],
        stream[501:],
        query,
    )


if __name__ == "__main__":
    main()

