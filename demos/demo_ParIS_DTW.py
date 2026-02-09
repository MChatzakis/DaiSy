import sys
import os
import numpy as np
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from daisy import DistanceType, ParIS
except ImportError:
    print("Error: ParIS is not yet available in Python bindings.")
    print("ParIS requires file-based data and needs to be added to pybinds/setup.cpp")
    print("See demo_ParIS_DTW.cpp for the C++ implementation.")
    sys.exit(1)

def main():
    
    n_database = 200000
    dim = 96
    n_query = 10
    k = 5

    np.random.seed(100)   
    db = np.random.randn(n_database, dim).astype(np.float32)

    np.random.seed(50)
    query = np.random.randn(n_query, dim).astype(np.float32)

    print(f"Loaded {n_database} database points and {n_query} query points with dimension {dim}")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
    temp_db_file = temp_file.name
    temp_file.close()
    
    try:
        
        db.tofile(temp_db_file)
        print(f"Database written to temporary file: {temp_db_file}")

        index = ParIS(DistanceType.DTW)
        index.setNumThreads(4)

        warp_window = max(1, int(dim * 0.1))
        index.setWarpingWindow(warp_window)
        print(f"Warping window set to: {warp_window}")

        index.buildIndex(temp_db_file, dim, n_database)
        print(">>> Finished indexing")

        I, D = index.searchIndex(query, k)
        print(">>> Finished search")

        for query_num in range(n_query):
            print(f"Query {query_num}:")
            print("Distances:", D[query_num])
            print("Indices:", I[query_num])
            print()

    finally:
        
        if os.path.exists(temp_db_file):
            os.remove(temp_db_file)
            print(f"Cleaned up temporary file: {temp_db_file}")

if __name__ == "__main__":
    main()
