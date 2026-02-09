import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from daisy import DistanceType, Messi

def main():
    
    n_database = 200000
    dim = 96
    n_query = 10
    k = 5

    np.random.seed(100)   
    db = np.random.randn(n_database, dim).astype(np.float32)

    np.random.seed(50)
    query = np.random.randn(n_query, dim).astype(np.float32)

    index = Messi(DistanceType.DTW)
    index.setWarpingWindow(int(dim * 0.1))

    index.setNumThreads(1)
    index.buildIndex(db)

    I, D = index.searchIndex(query, k)

    for query_num in range(n_query):
        print(f"Query {query_num}:")
        print("Distances:", D[query_num])
        print("Indices:", I[query_num])
        print()

if __name__ == "__main__":
    main()