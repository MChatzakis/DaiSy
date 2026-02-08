import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# GUI import not needed for this demo - uncomment if using GUI
# from scripts.gui_config_param import get_config, search_classes

from daisy import DistanceType, BruteForceSearch

def main():
    # 0. Configuration of the variables
    n_database = 200000
    dim = 96
    n_query = 10
    k = 5

    # 1. Generate random data and queries
    np.random.seed(100)   
    db = np.random.randn(n_database, dim).astype(np.float32)

    np.random.seed(50)
    query = np.random.randn(n_query, dim).astype(np.float32)

    # 2. Create a brute-force search object
    index = BruteForceSearch(DistanceType.DTW)

    # 3. Build the index
    index.setNumThreads(1)
    index.buildIndex(db)

    # 4. Search the index
    I, D = index.searchIndex(query, k)
    
    # 5. Print the results
    for query_num in range(n_query):
        print(f"Query {query_num}:")
        print("Distances:", D[query_num])
        print("Indices:", I[query_num])
        print()

if __name__ == "__main__":
    main()