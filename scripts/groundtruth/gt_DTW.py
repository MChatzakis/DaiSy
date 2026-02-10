import numpy as np
# this will require pip install tslearn and joblib
from tslearn.metrics import dtw
from joblib import Parallel, delayed
from gt_utils import formatFile_db, formatFile_query, saveOutput, run_all_datasets

def is_valid_ts(ts: np.ndarray) -> bool:
    return ts.size > 0 and not np.isnan(ts).all()

def compute_dtw_row(i, query, db):
    row = np.full(len(db), np.inf, dtype=np.float32)
    if not is_valid_ts(query):
        print(f"  Warning: Query {i} is invalid (all NaNs or empty). Skipping.")
        return row

    # Calculate warping window (10% of sequence length, same as C++)
    warping_window = max(1, int(0.1 * len(query)))

    for j, db_ts in enumerate(db):
        if is_valid_ts(db_ts):
            try:
                # Use constrained DTW with same warping window as C++
                row[j] = dtw(query, db_ts, global_constraint="sakoe_chiba", 
                           sakoe_chiba_radius=warping_window)
            except Exception as e:
                print(f"  Error: DTW failed between query {i} and db {j}: {e}")
    return row

def bruteForceDTW_gt(dim: int,
                     db_file: str,
                     query_file: str,
                     num_db: int,
                     num_queries: int,
                     db_name: str,
                     k_tab: list[int] | None = None) -> None:
    db = formatFile_db(db_file, num_db, dim)
    queries = formatFile_query(query_file, dim, num_queries)
    print(f"Loaded {num_db} database time series and {num_queries} queries for DTW.")

    if k_tab is None:
        k_tab = [1, 10, 100]

    print(f"Running DTW brute-force for {num_queries} queries (Parallelized)...")

    # Parallel DTW computation (each query row)
    all_distances = Parallel(n_jobs=-1, verbose=5)(
        delayed(compute_dtw_row)(i, queries[i], db) for i in range(num_queries)
    )
    all_distances = np.array(all_distances, dtype=np.float32)

    for k in k_tab:
        idx = np.argsort(all_distances, axis=1)[:, :k]
        dists = np.take_along_axis(all_distances, idx, axis=1)

        saveOutput(f"bruteForce_gtDTW_I_{db_name}_len{dim}_size{num_db}_q{num_queries}_k{k}", idx)
        saveOutput(f"bruteForce_gtDTW_D_{db_name}_len{dim}_size{num_db}_q{num_queries}_k{k}", dists, is_distance=True)

if __name__ == '__main__':
    print("--- Generating DTW Ground Truth (Parallel) ---")
    run_all_datasets(bruteForceDTW_gt, dtw_query_percentage=1.0)
