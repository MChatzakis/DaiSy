import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.gui_config_param import get_config, search_classes

from diNoSimilaritySearch import DistanceType #, BruteForceSearch, LbBruteforce, Odyssey, Messi, ParIS, Sing

def main():
# With the GUI
    # 0. Get parameters from the GUI
    params = get_config()
    if not params:
        print("No parameters provided. Exiting.")
        return

    filename_db   = params["Dataset Path"]
    filename_query= params["Query Path"]
    use_subset    = params["Use Dataset Subset"]
    subset_size   = params["Dataset Subset Size"]
    use_query_subset = params["Use Query Subset"]
    query_subset_size = params["Query Subset Size"]
    n_database    = params["Number of Database Vectors"]
    dim           = params["Vector Dimensionality"]    
    nq            = params["Query Number"]
    knn           = params["k-Nearest Neighbors"]
    dist_metric   = params["Distance Metric"]
    search_method = params["Search Method"]
    num_thread    = params["Threads"]

    # 1. Loading
        # database vectors from file and reshape based on input dims
    if use_subset and subset_size:
        print(f"Loading subset of {subset_size} vectors from dataset...")
        bytes_to_read = subset_size * dim * 4  # 4 bytes per float32
        with open(filename_db, 'rb') as f:
            db_bytes = f.read(bytes_to_read)
        db = np.frombuffer(db_bytes, dtype='float32')
        try:
            db = db.reshape((subset_size, dim))
        except ValueError:
            raise ValueError(f"Could not reshape database subset to ({subset_size}, {dim})")
        print(f"Successfully loaded {db.shape[0]} vectors from dataset subset")
    else:
        print(f"Loading full dataset with {n_database} vectors...")
        db = np.fromfile(filename_db, dtype='float32')
        try:
            db = db.reshape((n_database, dim))
        except ValueError:
            raise ValueError(f"Could not reshape database vectors to ({n_database}, {dim})")
        print(f"Successfully loaded {db.shape[0]} vectors from full dataset")

        # queries
    if use_query_subset and query_subset_size:
        print(f"Loading subset of {query_subset_size} queries from query file...")
        query_bytes_to_read = query_subset_size * dim * 4  # 4 bytes per float32
        with open(filename_query, 'rb') as f:
            query_bytes = f.read(query_bytes_to_read)
        query_all = np.frombuffer(query_bytes, dtype='float32')
        try:
            query_all = query_all.reshape((query_subset_size, dim))
        except ValueError:
            raise ValueError(f"Could not reshape query subset to ({query_subset_size}, {dim})")
        print(f"Successfully loaded {query_all.shape[0]} queries from query subset")
    else:
        print("Loading full query file...")
        query_all = np.fromfile(filename_query, dtype='float32')
        try:
            query_all = query_all.reshape((-1, dim))
        except ValueError:
            raise ValueError(f"Could not reshape query vectors with dimension {dim}")
        print(f"Successfully loaded {query_all.shape[0]} queries from full query file")

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