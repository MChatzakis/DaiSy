import numpy as np
import os
import re

def formatFile_db(filename: str,
                  num_points: int,
                  dim: int) -> np.ndarray:
    """
    @brief Load the database and reshape it into a 2D NumPy array.

    @param filename: Path to the binary file
    @param num_points: Number of vectors (or time series)
    @param dim: Dimensionality of each vector (or length of time series)
    @return: np.ndarray of shape (num_points, dim), dtype float32
    """
    db = np.fromfile(filename, dtype='float32')
    return db.reshape((num_points, dim))

def formatFile_query(filename: str,
                     dim: int,
                     nq: int) -> np.ndarray:
    """
    @brief Load the query dataset and reshape it into a 2D NumPy array.

    @param filename: Path to the data
    @param dim: Dimensionality of each query vector (or length of time series)
    @param nq: Number of queries to return
    @return: np.ndarray of shape (nq, dim), dtype float32
    @throws ValueError: If requested more queries than available
    """
    queries = np.fromfile(filename, dtype='float32').reshape((-1, dim))
    if nq > queries.shape[0]:
        raise ValueError(f"Requested {nq} queries, but only {queries.shape[0]} available.")
    return queries[:nq]

def saveOutput(filename_prefix: str,
               data_array: np.ndarray,
               is_distance: bool = False) -> None:
    """
    @brief Save the output array to a .txt file.

    @param filename_prefix: Filename stem
    @param data_array: NumPy Array of indices or distances
    @param is_distance: Boolean flag to check if it is distance data (for folder routing)
=    @return: None
    @side_effects: Creates output directories and writes text files.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))
    base_folder = os.path.join(project_root, "tests", "groundtruth")

    if is_distance:
        folder = os.path.join(base_folder, "Distances")
        os.makedirs(folder, exist_ok=True)
        output_filename = os.path.join(folder, f"{filename_prefix}.txt")
        np.savetxt(output_filename, data_array, fmt='%.15f')
        print(f"Saved output to {output_filename}")
    else:
        folder = os.path.join(base_folder, "Indices")
        os.makedirs(folder, exist_ok=True)
        output_filename = os.path.join(folder, f"{filename_prefix}.txt")
        np.savetxt(output_filename, data_array, fmt='%.f')
        print(f"Saved output to {output_filename}")

def path_to_filename(path: str) -> str:
    """
    @brief Extract the filename from a full file path.

    @param path: Full path to a file
    @return: Just the filename
    """
    return os.path.basename(path)

def parse_filename_for_config(path: str) -> tuple[str | None, int | None, int | None]:
    """
    @brief Extract dimensionality and database size from filename using regex.

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
    @brief Scan a folder and match database and query file pairs based on filename prefix.

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

def run_all_datasets(groundtruth_function,
                     override_num_queries: int | None = None,
                     override_k_tab: list[int] | None = None,
                     default_num_queries_for_dtw: int = 10,
                     dtw_query_percentage: float = 0.1) -> None:
    """
    @brief Run brute-force search on all dataset pairs found in the project's 'data' directory using a provided groundtruth function.

    @param groundtruth_function: The function to call for generating ground truth (e.g., bruteForceSS_gt, bruteForceDTW_gt)
    @param override_num_queries: Optional override for number of queries per dataset
    @param override_k_tab: Optional override for list of 'k' values in top-k search
    @param default_num_queries_for_dtw: Default number of queries to use if groundtruth_function is DTW specific
                                        and no override_num_queries is provided.
    @param dtw_query_percentage: Percentage of available queries to use for DTW (default 0.1 = 10%)
    @return: None
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))
    data_folder = os.path.join(project_root, 'data')

    dataset_pairs = find_dataset_pairs(data_folder)

    for db_path, query_path in dataset_pairs:
        DB_NAME, DIM, NUM_DB_POINTS = parse_filename_for_config(db_path)
        _, _, parsed_num_queries = parse_filename_for_config(query_path)

        NUM_QUERIES = override_num_queries
        if NUM_QUERIES is None:
            # Check if the function name indicates DTW to apply specific default
            if "DTW" in groundtruth_function.__name__:
                # Use percentage-based approach for DTW, with a minimum from default_num_queries_for_dtw
                percentage_queries = int(parsed_num_queries * dtw_query_percentage)
                NUM_QUERIES = max(percentage_queries, min(parsed_num_queries, default_num_queries_for_dtw))
            else:
                NUM_QUERIES = parsed_num_queries # Use all queries for L2 by default

        K_TAB = override_k_tab if override_k_tab is not None else [1, 10, 100]

        print(f"\nRunning {groundtruth_function.__name__} on dataset: {DB_NAME}")
        print(f"  - Length/Dimension = {DIM}")
        print(f"  - Number of DB Points = {NUM_DB_POINTS}")
        print(f"  - Number of Queries = {NUM_QUERIES}")
        print(f"  - K values = {K_TAB}")

        groundtruth_function(DIM, db_path, query_path, NUM_DB_POINTS, NUM_QUERIES, DB_NAME, K_TAB)