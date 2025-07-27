import numpy as np
from joblib import Parallel, delayed
from gt_utils import formatFile_db, formatFile_query, saveOutput, run_all_datasets

try:
    from dtaidistance import dtw
    DTW_LIBRARY = "dtaidistance"
    print(f"Using {DTW_LIBRARY} library for DTW computation")
except ImportError:
    try:
        from fastdtw import fastdtw
        DTW_LIBRARY = "fastdtw"
        print(f"Using {DTW_LIBRARY} library for DTW computation")
    except ImportError:
        print("Error: Neither dtaidistance nor fastdtw library found.")
        print("Please install one of them:")
        print("  pip install dtaidistance")
        print("  pip install fastdtw")
        exit(1)

def compute_dtw_row_dtaidistance(i: int, query: np.ndarray, db: np.ndarray, warping_window: int) -> np.ndarray:
    """
    @brief Compute DTW distances using dtaidistance library (C-optimized).
    
    @param i: Query index (for logging purposes)
    @param query: Single query time series
    @param db: Database of time series
    @param warping_window: Sakoe-Chiba warping window constraint
    @return: Array of DTW distances from query to all database time series
    """
    distances = np.full(len(db), np.inf, dtype=np.float32)
    
    if len(query) == 0:
        print(f"  Warning: Query {i} is empty")
        return distances
    
    # Progress indicator for long computations
    if i % 10 == 0:
        print(f"  Processing query {i}/{len(db)} with dtaidistance...")
    
    try:
        # Use dtaidistance batch computation if available
        if hasattr(dtw, 'distance_matrix_fast'):
            # Try vectorized computation (much faster)
            try:
                # Reshape for batch computation
                query_batch = np.array([query])
                dist_matrix = dtw.distance_matrix_fast(query_batch, db, window=warping_window)
                distances = dist_matrix[0].astype(np.float32)
            except:
                # Fall back to individual computation
                for j, db_ts in enumerate(db):
                    if len(db_ts) > 0 and not np.isnan(db_ts).all():
                        distances[j] = dtw.distance(query, db_ts, window=warping_window)
                    if j % 1000 == 0 and j > 0:
                        print(f"    Computed {j}/{len(db)} distances for query {i}")
        else:
            # Individual distance computation
            for j, db_ts in enumerate(db):
                if len(db_ts) > 0 and not np.isnan(db_ts).all():
                    distances[j] = dtw.distance(query, db_ts, window=warping_window)
                if j % 1000 == 0 and j > 0:
                    print(f"    Computed {j}/{len(db)} distances for query {i}")
                    
    except Exception as e:
        print(f"  Error: DTW computation failed for query {i}: {e}")
        return np.full(len(db), np.inf, dtype=np.float32)
    
    # Check if we got valid results
    finite_count = np.sum(np.isfinite(distances))
    print(f"  Query {i}: {finite_count}/{len(db)} finite distances, min={np.min(distances[np.isfinite(distances)]) if finite_count > 0 else 'inf'}")
    
    return distances

def compute_dtw_row_fastdtw(i: int, query: np.ndarray, db: np.ndarray, warping_window: int) -> np.ndarray:
    """
    @brief Compute DTW distances using fastdtw library (approximation algorithm).
    
    @param i: Query index (for logging purposes)
    @param query: Single query time series
    @param db: Database of time series
    @param warping_window: Radius parameter for FastDTW
    @return: Array of DTW distances from query to all database time series
    """
    distances = np.full(len(db), np.inf, dtype=np.float32)
    
    if len(query) == 0:
        print(f"  Warning: Query {i} is empty")
        return distances
    
    # Progress indicator for long computations
    if i % 10 == 0:
        print(f"  Processing query {i} with fastdtw...")
    
    try:
        for j, db_ts in enumerate(db):
            if len(db_ts) > 0 and not np.isnan(db_ts).all():
                # fastdtw uses radius parameter (similar concept to warping window)
                distance, _ = fastdtw(query, db_ts, radius=warping_window)
                distances[j] = distance
            if j % 1000 == 0 and j > 0:
                print(f"    Computed {j}/{len(db)} distances for query {i}")
                
    except Exception as e:
        print(f"  Error: FastDTW computation failed for query {i}: {e}")
        return np.full(len(db), np.inf, dtype=np.float32)
    
    # Check if we got valid results
    finite_count = np.sum(np.isfinite(distances))
    print(f"  Query {i}: {finite_count}/{len(db)} finite distances, min={np.min(distances[np.isfinite(distances)]) if finite_count > 0 else 'inf'}")
    
    return distances

def compute_dtw_row(i: int, query: np.ndarray, db: np.ndarray) -> np.ndarray:
    """
    @brief Compute DTW distances between a single query and all database time series using selected library.
    
    @param i: Query index (for logging purposes)
    @param query: Single query time series
    @param db: Database of time series
    @return: Array of DTW distances from query to all database time series
    """
    if len(query) == 0:
        print(f"  Warning: Query {i} is empty. Skipping.")
        return np.full(len(db), np.inf, dtype=np.float32)

    # Calculate warping window (10% of sequence length, consistent with other implementations)
    warping_window = max(1, int(0.1 * len(query)))

    if DTW_LIBRARY == "dtaidistance":
        return compute_dtw_row_dtaidistance(i, query, db, warping_window)
    elif DTW_LIBRARY == "fastdtw":
        return compute_dtw_row_fastdtw(i, query, db, warping_window)
    else:
        raise ValueError(f"Unknown DTW library: {DTW_LIBRARY}")

def bruteForceSS_gt(dim: int, 
                    db_file: str, 
                    query_file: str, 
                    num_db: int, 
                    num_queries: int, 
                    db_name: str, 
                    k_tab: list[int] | None = None) -> None:
    """
    @brief Run brute-force nearest neighbor search using alternative DTW libraries and save distances and indices.

    @param dim: Dimensionality of time series (sequence length)
    @param db_file: Path to the database file
    @param query_file: Path to the query file
    @param num_db: Number of time series in the database
    @param num_queries: Number of queries to process
    @param db_name: Name label for the dataset (used in filenames)
    @param k_tab: list[int] | None: list of k values for top-k search
    @return: None
    """    
    print(f"Loading data from {db_file} and {query_file}...")
    db = formatFile_db(db_file, num_db, dim)
    queries = formatFile_query(query_file, dim, num_queries)
    
    if k_tab is None:
        k_tab = [1, 10, 100]    

    print(f"Loaded {num_db} database time series and {num_queries} queries for {DTW_LIBRARY} DTW ground truth generation.")
    print(f"Database shape: {db.shape}, Query shape: {queries.shape}")
    print(f"Running {DTW_LIBRARY} DTW brute-force search for {num_queries} queries...")
    
    # For very large datasets, reduce parallelism to avoid memory issues
    if num_db > 50000 or num_queries > 100:
        n_jobs = min(4, num_queries)  # Limit parallel jobs for large datasets
        print(f"Large dataset detected, using {n_jobs} parallel jobs to avoid memory issues")
    else:
        n_jobs = -1
    
    # Parallel DTW computation for all query-database pairs
    print("Starting parallel DTW computation...")
    all_distances = Parallel(n_jobs=n_jobs, verbose=10, backend='threading')(
        delayed(compute_dtw_row)(i, queries[i], db) for i in range(num_queries)
    )
    all_distances = np.array(all_distances, dtype=np.float32)
    
    print(f"Distance computation completed. Shape: {all_distances.shape}")
    print(f"Distance statistics: min={np.min(all_distances)}, max={np.max(all_distances)}, finite_count={np.sum(np.isfinite(all_distances))}")

    # Generate ground truth for each k value
    for k in k_tab:
        print(f"Generating top-{k} results...")
        
        # Check if we have any finite distances
        finite_mask = np.isfinite(all_distances)
        finite_per_query = np.sum(finite_mask, axis=1)
        queries_with_results = np.sum(finite_per_query > 0)
        
        print(f"Queries with finite distances: {queries_with_results}/{num_queries}")
        
        if queries_with_results == 0:
            print(f"WARNING: No finite distances found! Skipping k={k}")
            continue
            
        # Get indices of k nearest neighbors (smallest distances)
        indices = np.argsort(all_distances, axis=1)[:, :k]
        
        # Get corresponding distances
        distances = np.take_along_axis(all_distances, indices, axis=1)

        # Save ground truth files with library name to distinguish from other implementations
        saveOutput(f"bruteForce_gt{DTW_LIBRARY.upper()}_I_{db_name}_len{dim}_size{num_db}_q{num_queries}_k{k}", indices)
        saveOutput(f"bruteForce_gt{DTW_LIBRARY.upper()}_D_{db_name}_len{dim}_size{num_db}_q{num_queries}_k{k}", distances, is_distance=True)
        
        print(f"Saved ground truth files for k={k}")

    print(f"{DTW_LIBRARY} DTW ground truth generation completed for dataset: {db_name}")

if __name__ == '__main__':
    print(f"--- Generating {DTW_LIBRARY} DTW Ground Truth ---")
    run_all_datasets(bruteForceSS_gt)
