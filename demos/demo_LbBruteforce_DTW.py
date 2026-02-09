import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from daisy import DistanceType, LbBruteforce

def main():
    n_database = 200000
    dim = 96
    n_query = 10
    k = 5

    # 1. Generate random data and queries
    np.random.seed(100)   
    db = np.random.randn(n_database, dim).astype(np.float32)

    np.random.seed(50)
    query = np.random.randn(n_query, dim).astype(np.float32)

    # 2. Create a LbBruteforce search object with DTW distance
    index = LbBruteforce(DistanceType.DTW)
    print("Created LbBruteforce with DTW distance type")

    # 3. Build the index
    index.setNumThreads(1)
    index.buildIndex(db)
    print("Index built successfully")

    # 4. Search the index
    print("Starting DTW search...")
    I, D = index.searchIndex(query, k)
    print("DTW search completed successfully!")

    for query_num in range(n_query):
        print(f"Query {query_num}:")
        print("Distances:", D[query_num])
        print("Indices:", I[query_num])
        print()

    print("Python DTW demo completed successfully!")

if __name__ == "__main__":
    main()
