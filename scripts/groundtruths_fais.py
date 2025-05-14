import numpy as np
import faiss
import os
import re

def formatFile_db(filename: str, 
                  num_points: int, 
                  dim: int) -> np.ndarray:
    """
    Load the database and reshape it into a 2D NumPy array.

    @param filename: Path to the binary file
    @param num_points: Number of vectors
    @param dim: Dimensionality of each vector
    @return: np.ndarray of shape (num_points, dim), dtype float32
    """    
    db = np.fromfile(filename, dtype='float32')
    return db.reshape((num_points, dim))

def formatFile_query(filename: str, 
                     dim: int, 
                     nq: int) -> np.ndarray:
    """
    Load the query dataset and reshape it into a 2D NumPy array.

    @param filename: Path to the data
    @param dim: Dimensionality of each query vector
    @param nq: Number of queries to return
    @return: np.ndarray of shape (nq, dim), dtype float32
    @throws ValueError: If requested more queries than available
    """    
    queries = np.fromfile(filename, dtype='float32').reshape((-1, dim))
    if nq > queries.shape[0]:
        raise ValueError(f"Requested {nq} queries, but only {queries.shape[0]} available.")
    return queries[:nq]

def saveOutput(filename_prefix: str, 
               indices: np.ndarray,
               is_distance: bool = False) -> None:
    """
    Save the output array to a .txt file.

    @param filename_prefix: Filename stem
    @param indices: NumPy Array of indices or distances
    @param is_distance: Boolean flag to check if it is distance data (for folder routing)
    @return: None
    """    
    folder = "./tests/gt/" 
    
    if is_distance:
        folder = os.path.join(folder, "Distances")
    else:
        folder = os.path.join(folder, "Indices")

    os.makedirs(folder, exist_ok=True)
    output_filename = os.path.join(folder, f"{filename_prefix}.txt")    
    np.savetxt(output_filename, indices, fmt='%d')
    print(f"Saved output to {output_filename}")

def bruteForceSS_gt(dim: int, 
                    db_file: str, 
                    query_file: str, 
                    num_db: int, 
                    num_queries: int, 
                    db_name: str, 
                    k_tab: list[int] | None = None) -> None:
    """
    Run brute-force nearest neighbor search using FAISS and save distances and indices.

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

    for k in k_tab:
        D, I = index.search(queries, k)
        saveOutput(f"bruteFSS_gt_I_{db_name}_len{dim}_size{num_db}_q{num_queries}_k{k}", I)
        saveOutput(f"bruteFSS_gt_D_{db_name}_len{dim}_size{num_db}_q{num_queries}_k{k}", D, is_distance=True)

def path_to_filename(path: str) -> str:
    """
    Extract the filename from a full file path.

    @param path: Full path to a file
    @return: Just the filename
    """    
    return os.path.basename(path)

def parse_filename_for_config(path: str) -> tuple[str | None, int | None, int | None]:
    """
    Extract dimensionality and database size from filename using regex.

    @param path: File path containing 'len<dim>' and 'size<points>'
    @return: 
            * db_name: str | None
            * dim: int | None
            * n_database: int | None
    """
    filename = path_to_filename(path)

    len_rx = r"len(\d+)" # Regex to extract *length*
    size_rx = r"size(\d+)" # Regex to extract *size*
    db_name_rx = r"([^/]+)\.data" # Regex to extract *name*

    dim = None
    n_database = None
    db_name = None

    len_match = re.search(len_rx, filename)
    dim = int(len_match.group(1)) if len_match else None

    size_match = re.search(size_rx, filename)
    n_database = int(size_match.group(1)) if size_match else None

    db_name_match = re.search(db_name_rx, filename)
    db_name = db_name_match.group(1) if db_name_match else None
    
    return db_name, dim, n_database

def find_dataset_pairs(data_folder: str) -> list[tuple[str, str]]:
    """
    Scan a folder and match database and query file pairs based on filename prefix.

    @param data_folder: Path to the folder containing dataset files
    @return: List of (database_file_path, query_file_path) tuples
    """    
    files = os.listdir(data_folder)
    data_files = [f for f in files if '.data.' in f]
    query_files = [f for f in files if '.query.' in f]

    pairs = []
    for data_file in data_files:
        prefix = data_file.split('.data')[0]
        query_match = next((q for q in query_files if q.startswith(prefix)), None)
        if query_match:
            pairs.append((
                os.path.join(data_folder, data_file),
                os.path.join(data_folder, query_match)
            ))
    return pairs

def run_all_datasets(override_num_queries: int | None = None, 
                     override_k_tab: list[int] | None = None) -> None:
    """
    Run brute-force search on all dataset pairs found in './data'.

    @param override_num_queries: Optional override for number of queries per dataset
    @param override_k_tab: Optional override for list of 'k' values in top-k search
    @return: None
    """    
    data_folder = './data'
    dataset_pairs = find_dataset_pairs(data_folder)

    for db_path, query_path in dataset_pairs:
        DB_NAME, DIM, NUM_DB_POINTS = parse_filename_for_config(db_path)
        _, _, parsed_num_queries = parse_filename_for_config(query_path)

        NUM_QUERIES = override_num_queries if override_num_queries is not None else parsed_num_queries
        K_TAB = override_k_tab if override_k_tab is not None else [1, 10, 100]

        print(f"\nRunning bruteForceSS_gt on dataset: {DB_NAME}")
        print(f"  - DIM = {DIM}")
        print(f"  - NUM_DB_POINTS = {NUM_DB_POINTS}")
        print(f"  - NUM_QUERIES = {NUM_QUERIES}")

        bruteForceSS_gt(DIM, db_path, query_path, NUM_DB_POINTS, NUM_QUERIES, DB_NAME)


if __name__ == '__main__':
    run_all_datasets(override_num_queries=4)
