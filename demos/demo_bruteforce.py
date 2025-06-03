import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.config_param import get_config, search_classes

from diNoSimilaritySearch import DistanceType

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
    # 0. Get parameters from the GUI
    params = get_config()
    if not params:
        print("No parameters provided. Exiting.")
        return

    filename_db   = params["Dataset Path"]
    filename_query= params["Query Path"]
    n_database    = params["Number of Database Vectors"]
    dim           = params["Vector Dimensionality"]    
    nq            = params["Query Number"]
    knn           = params["k-Nearest Neighbors"]
    dist_metric   = params["Distance Metric"]
    search_method = params["Search Method"]
    num_thread    = params["Threads"]

    # 1. Loading
        # database vectors from file and reshape based on input dims
    db = np.fromfile(filename_db, dtype='float32')
    try:
        db = db.reshape((n_database, dim))
    except ValueError:
        raise ValueError(f"Could not reshape database vectors to ({n_database}, {dim})")

        # queries
    query_all = np.fromfile(filename_query, dtype='float32')
    try:
        query_all = query_all.reshape((-1, dim))
    except ValueError:
        raise ValueError(f"Could not reshape query vectors with dimension {dim}")

    query = query_all[:nq]

    # 2. Initialize the search object
    index_class = search_classes[search_method]
    index = index_class(getattr(DistanceType, dist_metric))

    # 3. Build index and set threads
    index.setNumThreads(num_thread)
    index.buildIndex(db)

    # 4. Perform the search
    I, D = index.searchIndex(query, knn)

    # 5. Output results
    for query_num in range(nq):
        print(f"Query {query_num}:")
        print("Distances:", D[query_num])
        print("Indices:", I[query_num])
        print()

if __name__ == "__main__":
    main()