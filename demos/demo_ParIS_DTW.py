import sys
import os
import numpy as np
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# GUI import not needed for this demo - uncomment if using GUI
# from scripts.gui_config_param import get_config, search_classes

# Note: ParIS Python bindings are not yet implemented in pybinds/setup.cpp
# This demo shows the expected interface when ParIS bindings are added
# For now, this will fail with ImportError until ParIS is added to the Python bindings
try:
    from daisy import DistanceType, ParIS
except ImportError:
    print("Error: ParIS is not yet available in Python bindings.")
    print("ParIS requires file-based data and needs to be added to pybinds/setup.cpp")
    print("See demo_ParIS_DTW.cpp for the C++ implementation.")
    sys.exit(1)

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

    print(f"Loaded {n_database} database points and {n_query} query points with dimension {dim}")

    # 2. Write database to temporary file (ParIS requires file-based data)
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
    temp_db_file = temp_file.name
    temp_file.close()
    
    try:
        # Write database to binary file
        db.tofile(temp_db_file)
        print(f"Database written to temporary file: {temp_db_file}")

        # 3. Create ParIS search object
        index = ParIS(DistanceType.DTW)
        index.setNumThreads(4)

        # REMEMBER TO SET THE WARPING WINDOW TO 10% OF THE TIME SERIES LENGTH, NOT 10 AS A NUMBER
        warp_window = max(1, int(dim * 0.1))
        index.setWarpingWindow(warp_window)
        print(f"Warping window set to: {warp_window}")

        # 4. Build the index (ParIS requires file-based API: filename, dim, n_database)
        # Note: This is different from other algorithms that accept NumPy arrays directly
        index.buildIndex(temp_db_file, dim, n_database)
        print(">>> Finished indexing")

        # 5. Search the index
        I, D = index.searchIndex(query, k)
        print(">>> Finished search")

        # 6. Print the results
        for query_num in range(n_query):
            print(f"Query {query_num}:")
            print("Distances:", D[query_num])
            print("Indices:", I[query_num])
            print()

    finally:
        # 7. Clean up - remove temporary file
        if os.path.exists(temp_db_file):
            os.remove(temp_db_file)
            print(f"Cleaned up temporary file: {temp_db_file}")

if __name__ == "__main__":
    main()
