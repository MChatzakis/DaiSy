import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Optional, Tuple
from scripts.config_param import get_config, search_options, param_gui, find_data_files

from diNoSimilaritySearch import BruteForceSearch, DistanceType

def main():
    # Without the GUI
    # 0. Configuration of the variables
    n_database = 200000
    dim = 96
    n_query = 10
    k = 5

    # 1. Generate random data and queries
    np.random.seed(100)  # to match the seed from C++ if needed
    db = np.random.randn(n_database, dim).astype(np.float32)

    np.random.seed(50)
    query = np.random.randn(n_query, dim).astype(np.float32)

    # 2. Create a brute-force search object
    index = BruteForceSearch(DistanceType.L2_SQUARED)

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

# With the GUI
    # 0. Configuration of the variables
    params = get_config()

    if not params:
        return 

    db_name       = params["Dataset"]
    nq            = params["Query Number"]
    knn           = params["k-Nearest Neighbors"]
    dist_metric   = params["Distance Metric"]
    search_method = params["Search Method"]
    num_thread    = params["Threads"]

    filename_db, filename_query, d, nb = find_data_files(db_name)

    # 1. Loading data and queries
    filename_db, filename_query, d, nb = find_data_files(db_name)

    db = np.fromfile(filename_db, dtype='float32').reshape((nb, d))

    query_all = np.fromfile(filename_query, dtype='float32').reshape((-1, d))
    query = query_all[:nq]

    # 2. Create a brute-force search object
    index_class = search_classes[search_method]
    index = index_class(getattr(DistanceType, dist_metric))

    # 3. Build the index
    index.setNumThreads(num_thread)
    index.buildIndex(db)

    # 4. Search the index
    I, D = index.searchIndex(query, knn)
    
    # 5. Print the results
    for query_num in range(n_query):
        print(f"Query {query_num}:")
        print("Distances:", D[query_num])
        print("Indices:", I[query_num])
        print() 

if __name__ == "__main__":
    main()