import numpy as np
from joblib import Parallel, delayed
import time
from gt_utils import formatFile_db, formatFile_query, saveOutput, run_all_datasets

def simple_dtw_distance(x: np.ndarray, y: np.ndarray, warping_window: int = None) -> float:
    """
    @brief Simple, fast DTW implementation with optional warping window constraint.
    
    @param x: First time series
    @param y: Second time series  
    @param warping_window: Optional Sakoe-Chiba warping window constraint
    @return: DTW distance
    """
    n, m = len(x), len(y)
    
    if n == 0 or m == 0:
        return np.inf
    
    # Set warping window if not provided (10% of average length)
    if warping_window is None:
        warping_window = max(1, int(0.1 * (n + m) / 2))
    
    # Create cost matrix with infinity padding
    dtw_matrix = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dtw_matrix[0, 0] = 0
    
    # Fill DTW matrix with warping window constraint
    for i in range(1, n + 1):
        start_j = max(1, i - warping_window)
        end_j = min(m + 1, i + warping_window + 1)
        
        for j in range(start_j, end_j):
            cost = (x[i-1] - y[j-1]) ** 2  # Squared Euclidean distance
            
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # Insertion
                dtw_matrix[i, j-1],      # Deletion  
                dtw_matrix[i-1, j-1]     # Match
            )
    
    return dtw_matrix[n, m]

def compute_dtw_row_simple(i: int, query: np.ndarray, db: np.ndarray) -> np.ndarray:
    """
    @brief Compute DTW distances using simple native implementation.
    
    @param i: Query index (for logging purposes)
    @param query: Single query time series
    @param db: Database of time series
    @return: Array of DTW distances from query to all database time series
    """
    distances = np.full(len(db), np.inf, dtype=np.float32)
    
    if len(query) == 0:
        return distances
    
    # Progress indicator
    if i % 10 == 0:
        print(f"  Processing query {i} with simple DTW...")
    
    # Calculate warping window (10% of query length)
    warping_window = max(1, int(0.1 * len(query)))
    
    start_time = time.time()
    valid_distances = 0
    
    for j, db_ts in enumerate(db):
        if len(db_ts) > 0 and not np.isnan(db_ts).all() and not np.isnan(query).all():
            try:
                distances[j] = simple_dtw_distance(query, db_ts, warping_window)
                if np.isfinite(distances[j]):
                    valid_distances += 1
            except Exception as e:
                distances[j] = np.inf
        
        # Progress for large databases
        if j % 5000 == 0 and j > 0:
            elapsed = time.time() - start_time
            print(f"    Query {i}: {j}/{len(db)} done in {elapsed:.1f}s, {valid_distances} valid distances")
    
    elapsed = time.time() - start_time
    print(f"  Query {i}: {valid_distances}/{len(db)} valid distances in {elapsed:.1f}s, min={np.min(distances[np.isfinite(distances)]) if valid_distances > 0 else 'inf'}")
    
    return distances

def bruteForceSS_gt(dim: int, 
                    db_file: str, 
                    query_file: str, 
                    num_db: int, 
                    num_queries: int, 
                    db_name: str, 
                    k_tab: list[int] | None = None) -> None:
    """
    @brief Run brute-force nearest neighbor search using simple DTW and save distances and indices.

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

    print(f"Loaded {num_db} database time series and {num_queries} queries for Simple DTW ground truth generation.")
    print(f"Database shape: {db.shape}, Query shape: {queries.shape}")
    print(f"Data statistics:")
    print(f"  DB: min={np.min(db):.3f}, max={np.max(db):.3f}, mean={np.mean(db):.3f}")
    print(f"  Query: min={np.min(queries):.3f}, max={np.max(queries):.3f}, mean={np.mean(queries):.3f}")
    
    # Use moderate parallelism to avoid memory issues
    n_jobs = min(4, num_queries) if num_db > 10000 else -1
    print(f"Using {n_jobs} parallel jobs")
    
    # Parallel DTW computation
    print("Starting Simple DTW computation...")
    start_time = time.time()
    
    all_distances = Parallel(n_jobs=n_jobs, verbose=5, backend='threading')(
        delayed(compute_dtw_row_simple)(i, queries[i], db) for i in range(num_queries)
    )
    all_distances = np.array(all_distances, dtype=np.float32)
    
    elapsed = time.time() - start_time
    print(f"Distance computation completed in {elapsed:.1f}s. Shape: {all_distances.shape}")
    print(f"Distance statistics: min={np.min(all_distances):.3f}, max={np.max(all_distances):.3f}, finite_count={np.sum(np.isfinite(all_distances))}")

    # Generate ground truth for each k value
    for k in k_tab:
        print(f"Generating top-{k} results...")
        
        # Check if we have any finite distances
        finite_mask = np.isfinite(all_distances)
        finite_per_query = np.sum(finite_mask, axis=1)
        queries_with_results = np.sum(finite_per_query >= k)
        
        print(f"Queries with at least {k} finite distances: {queries_with_results}/{num_queries}")
        
        if queries_with_results == 0:
            print(f"WARNING: No queries have {k} finite distances! Creating files anyway...")
        
        # Get indices of k nearest neighbors (smallest distances)
        indices = np.argsort(all_distances, axis=1)[:, :k]
        
        # Get corresponding distances  
        distances = np.take_along_axis(all_distances, indices, axis=1)

        # Save ground truth files
        saveOutput(f"bruteForce_gtSimpleDTW_I_{db_name}_len{dim}_size{num_db}_q{num_queries}_k{k}", indices)
        saveOutput(f"bruteForce_gtSimpleDTW_D_{db_name}_len{dim}_size{num_db}_q{num_queries}_k{k}", distances, is_distance=True)
        
        print(f"Saved ground truth files for k={k}")

    print(f"Simple DTW ground truth generation completed for dataset: {db_name}")

if __name__ == '__main__':
    print("--- Generating Simple DTW Ground Truth ---")
    run_all_datasets(bruteForceSS_gt)
