import numpy as np
import faiss
from gt_utils import formatFile_db, formatFile_query, saveOutput, run_all_datasets

def bruteForceSS_gt(dim: int, 
                    db_file: str, 
                    query_file: str, 
                    num_db: int, 
                    num_queries: int, 
                    db_name: str, 
                    k_tab: list[int] | None = None) -> None:
    """
    @brief Run brute-force nearest neighbor search using FAISS (L2) and save distances and indices.

    @param dim: Dimensionality of vectors
    @param db_file: Path to the database file
    @param query_file: Path to the query file
    @param num_db: Number of vectors in the database
    @param num_queries: Number of queries to process
    @param db_name: Name label for the dataset (used in filenames)
    @param k_tab: list[int] | None: list of k values for top-k search
    @return: None
    """    
    db = formatFile_db(db_file, num_db, dim)
    queries = formatFile_query(query_file, dim, num_queries)
    if k_tab is None:
        k_tab = [1, 10, 100]    

    index = faiss.IndexFlatL2(dim)
    index.add(db)

    print(f"Running FAISS L2 search for {num_queries} queries...")

    for k in k_tab:
        D, I = index.search(queries, k)
        saveOutput(f"bruteForce_gt_I_{db_name}_len{dim}_size{num_db}_q{num_queries}_k{k}", I, metric_name="L2")
        saveOutput(f"bruteForce_gt_D_{db_name}_len{dim}_size{num_db}_q{num_queries}_k{k}", D, is_distance=True, metric_name="L2")

if __name__ == '__main__':
    print("--- Generating FAISS L2 Ground Truth ---")
    run_all_datasets(bruteForceSS_gt)