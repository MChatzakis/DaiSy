import numpy as np
import faiss
import os
import re

def formatFile_db(filename: str, 
                  num_points: int, 
                  dim: int) -> np.ndarray:
    """
    Load the database and reshape it into a 2D NumPy array.

    @param filename: Path to the data
    @param num_points: Number of vectors
    @param dim: Dimensionality of each vector
    @return: 2D array of shape (num_points, dim)
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
    @return: 2D NumPy array of shape (nq, dim)
    @throws ValueError: If requested more queries than available
    """    
    queries = np.fromfile(filename, dtype='float32').reshape((-1, dim))
    if nq > queries.shape[0]:
        raise ValueError(f"Requested {nq} queries, but only {queries.shape[0]} available.")
    return queries[:nq]

def saveOutput(filename_prefix: str, 
               indices: np.ndarray):
    """
    Save the output array to a .txt file.

    @param filename_prefix: Output filename
    @param indices: NumPy array 
    @return: None
    """    
    folder = "./tests/gt/" 
    os.makedirs(folder, exist_ok=True)
    output_filename = os.path.join(folder, f"{filename_prefix}.txt")    
    np.savetxt(output_filename, indices, fmt='%d')
    print(f"Saved output to {output_filename}")

def bruteForceSS_gt(dim: int, 
                    db_file: str, 
                    query_file: str, 
                    num_db: int, 
                    num_queries: int, 
                    db_name: str):
    """
    Run brute-force nearest neighbor search using FAISS and save distances and indices.

    @param dim: Dimensionality of vectors
    @param db_file: Path to the database file
    @param query_file: Path to the query file
    @param num_db: Number of vectors in the database
    @param num_queries: Number of queries to process
    @param db_name: Name label for the dataset (used in filenames)
    @return: None
    """    
    db = formatFile_db(db_file, num_db, dim)
    queries = formatFile_query(query_file, dim, num_queries)

    index = faiss.IndexFlatL2(dim)
    index.add(db)

    for k in [1, 10, 100]:
        D, I = index.search(queries, k)
        saveOutput(f"bruteFSS_gt_I_{db_name}_len{dim}_size{num_db}_q{num_queries}_k{k}", I)
        saveOutput(f"bruteFSS_gt_D_{db_name}_len{dim}_size{num_db}_q{num_queries}_k{k}", D)

def path_to_filename(path):
    """
    Extract the filename from a full file path.

    @param path: Full path to a file
    @return: Filename string
    """    
    return os.path.basename(path)

def parse_filename_for_config(path):
    """
    Extract dimensionality and database size from filename using regex.

    @param path: File path containing 'len<dim>' and 'size<points>'
    @return: Tuple (dim, n_database) as integers, 
                or (None, None) if not found
    """
    filename = path_to_filename(path)

    len_rx = r"len(\d+)" # Regex to extract *length*
    size_rx = r"size(\d+)" # Regex to extract *size*

    dim = None
    n_database = None

    len_match = re.search(len_rx, filename)
    if len_match:
        dim = int(len_match.group(1))


    size_match = re.search(size_rx, filename)
    if size_match:
        n_database = int(size_match.group(1))

    return dim, n_database

if __name__ == '__main__':
    NUM_QUERIES = 4

    ##### Random dataset example #####
    DB_FILE = './data/data.randwalk.len96.size200000.znorm.bin'
    QUERY_FILE = './data/query.randwalk.len96.size1000.bin'
    DIM, NUM_DB_POINTS = parse_filename_for_config(DB_FILE)
    DB_NAME = "Random"
    bruteForceSS_gt(DIM, DB_FILE, QUERY_FILE, NUM_DB_POINTS, NUM_QUERIES, DB_NAME)    

    ##### Astronomy dataset example #####
    DB_FILE = "data/astronomy.data.len256.size50000.znorm.bin"
    QUERY_FILE = "data/query.randwalk.len96.size1000.bin"    
    DIM, NUM_DB_POINTS = parse_filename_for_config(DB_FILE)
    DB_NAME = "Astro"
    bruteForceSS_gt(DIM, DB_FILE, QUERY_FILE, NUM_DB_POINTS, NUM_QUERIES, DB_NAME)
