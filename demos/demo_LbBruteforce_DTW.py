import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diNoSimilaritySearch import DistanceType, LbBruteforce

def main():
    # 0. Configuration of the variables
    n_database = 1000  # Smaller dataset for DTW testing
    dim = 96
    n_query = 5
    k = 3

    # 1. Generate random data and queries
    np.random.seed(100)  # to match the seed from C++ if needed
    db = np.random.randn(n_database, dim).astype(np.float32)

    np.random.seed(50)
    query = np.random.randn(n_query, dim).astype(np.float32)

    print(f"Generated {n_database} database points and {n_query} query points with dimension {dim}")

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

    # 5. Print the results
    for i in range(n_query):
        print(f"Query {i} DTW nearest neighbors:", end=" ")
        for j in range(k):
            print(f"({I[i, j]}, {D[i, j]:.4f})", end=" ")
        print()

    print("Python DTW demo completed successfully!")

if __name__ == "__main__":
    main()
