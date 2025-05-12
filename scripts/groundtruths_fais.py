import numpy as np
import faiss
import os

def formatFile_db(filename: str, num_points: int, dim: int) -> np.ndarray:
    db = np.fromfile(filename, dtype='float32')
    return db.reshape((num_points, dim))

def formatFile_query(filename: str, dim: int, nq: int) -> np.ndarray:
    queries = np.fromfile(filename, dtype='float32').reshape((-1, dim))
    if nq > queries.shape[0]:
        raise ValueError(f"Requested {nq} queries, but only {queries.shape[0]} available.")
    return queries[:nq]

def saveOutput(filename_prefix: str, indices: np.ndarray):
    folder = "./tests/" 
    os.makedirs(folder, exist_ok=True)
    output_filename = os.path.join(folder, f"{filename_prefix}_gt.txt")    
    np.savetxt(output_filename, indices, fmt='%d')
    print(f"Saved output to {output_filename}")

def bruteForceSS_gt(dim: int, db_file: str, query_file: str, num_db: int, num_queries: int, db_name: str):
    db = formatFile_db(db_file, num_db, dim)
    queries = formatFile_query(query_file, dim, num_queries)

    index = faiss.IndexFlatL2(dim)
    index.add(db)

    for k in [1, 10, 100]:
        D, I = index.search(queries, k)
        saveOutput(f"bruteForceSS_{db_name}_{k}", I)


if __name__ == '__main__':
    DIM = 96
    NUM_DB_POINTS = 200000
    NUM_QUERIES = 100
    DB_FILE = './data/data.randwalk.len96.size200000.znorm.bin'
    QUERY_FILE = './data/query.randwalk.len96.size1000.bin'

    bruteForceSS_gt(DIM, DB_FILE, QUERY_FILE, NUM_DB_POINTS, NUM_QUERIES, "Random")