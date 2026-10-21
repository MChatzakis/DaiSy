import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from daisy import DistanceType, Coconut

# COCONUT streaming: build on an initial batch, then insert new series into the live index
# as they arrive and query after each step (no rebuild).
def main():
    dim = 96
    initial = 50000
    batch = 25000
    steps = 3
    n_query = 5
    k = 5

    total = initial + steps * batch
    np.random.seed(100)
    stream = np.random.randn(total, dim).astype(np.float32)
    np.random.seed(50)
    query = np.random.randn(n_query, dim).astype(np.float32)

    index = Coconut(DistanceType.L2_SQUARED)
    index.setNumThreads(2)
    index.setQueryBatchSize(32)  # shared bounded candidate reads; independent exact heaps
    index.setBufferCapacity(10_000)  # bounded RAM table; full buffers persist and schedule merges
    index.buildIndex(stream[:initial])         # build on the initial batch

    def show(when):
        I, D = index.searchIndex(query, k)
        print(f"[{when}] query 0 -> indices {I[0]}, distances {D[0]}")

    show("after build")

    seen = initial
    for s in range(steps):
        index.insertBatch(stream[seen:seen + batch])   # stream the next batch
        index.waitForMerges()  # include this barrier when timing completed insertions
        seen += batch
        print(f"streamed {seen} series; disk={index.getDiskRecords()}, "
              f"RAM buffer={index.getBufferedRecords()}, merges={index.getMergeCount()}")
        show("after insert")
        recent_ids, _ = index.searchIndexRecent(query, 1, batch)
        print("top-1 in the most recent batch:", recent_ids[:, 0])
        print("SIMS phases and reads:", index.getLastSearchStats())

if __name__ == "__main__":
    main()
