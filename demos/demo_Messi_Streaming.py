"""Build MESSI once and append streaming batches without rebuilding."""

import numpy as np

from daisy import DistanceType, Messi


rng = np.random.default_rng(7)
stream = rng.normal(size=(6000, 96)).astype(np.float32)
queries = rng.normal(size=(5, 96)).astype(np.float32)

index = Messi(DistanceType.L2_SQUARED)
index.setIndexWorkers(2)
index.setSearchWorkers(4)
index.buildIndex(stream[:5000])


def query_and_report(stage, size):
    """The index stays queryable between updates: search after every addition."""
    indices, distances = index.searchIndex(queries, 5)
    print(f"{stage}: MESSI size {size}")
    print("  Query 0 IDs:", indices[0])
    print("  Query 0 distances:", distances[0])


query_and_report("after build", 5000)

index.insert(stream[5000])
query_and_report("after insert", 5001)

index.insertBatch(stream[5001:])
query_and_report("after batch", 6000)
