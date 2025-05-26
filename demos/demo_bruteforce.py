import sys
import os
import re
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Optional, Tuple
from scripts.config_param import get_config
from diNoSimilaritySearch import BruteForceSearch, DistanceType

search_classes = {
    "BruteForceSearch": BruteForceSearch,
    # "LbBruteForceSearch": LbBruteForceSearch,
    # "Messi": Messi,
    # "Odyssey": Odyssey,
    # "Paris": Paris,
    # "Sing": Sing
}

def printResults(I, D, nq):
    for query_num in range(nq):
        print(f"Query {query_num}:")
        print("Distances:", D[query_num])
        print("Indices:", I[query_num])
        print() 

def param_gui() -> Optional[Tuple[str, int, int, str, str, int]]:
    config = get_config()  # This will open the GUI
    
    if config:
        try:
            return (
                config["Dataset"],
                config["Query Number"],
                config["k-Nearest Neighbors"],
                config["Distance Metric"],
                config["Search Method"],
                config["Threads"]
            )
        except KeyError as e:
            print(f"Configuration key missing: {e}")
    else:
        print("No configuration was provided.")
    
    return None

def find_data_files(db_name: str, data_folder: str = '../data'):
    pattern_db = re.compile(
        rf"{re.escape(db_name)}\.data(?:\.[^.]+)?\.len(\d+)\.size(\d+)\.znorm\.bin"
    )

    pattern_query = re.compile(
        rf"{re.escape(db_name)}\.query(?:\.[^.]+)?\.len(\d+)\.size(\d+)(?:\.znorm)?\.bin"
    )

    dataset_path = None
    query_path = None
    dim = None
    nb = None

    for fname in os.listdir(data_folder):
        # Search for dataset file
        m_db = pattern_db.match(fname)
        if m_db:
            dataset_path = os.path.join(data_folder, fname)
            dim = int(m_db.group(1))
            nb = int(m_db.group(2))

        # Search for query file
        m_query = pattern_query.match(fname)
        if m_query:
            query_path = os.path.join(data_folder, fname)

    if dataset_path is None or query_path is None:
        raise FileNotFoundError(f"Could not find dataset or query files for '{db_name}' in '{data_folder}'")

    return dataset_path, query_path, dim, nb

def loadDataCHECK():
    d = 96
    nb = 200000
    nq = 10    

    filename_db = '/home/gaya/Documents/diNo-lib/data/random.data.randwalk.len96.size200000.znorm.bin'
    db = np.fromfile(filename_db, dtype='float32')
    db = db.reshape((nb, d))

    filename_query = "/home/gaya/Documents/diNo-lib/data/random.query.randwalk.len96.size1000.bin"
    query_all = np.fromfile(filename_query, dtype='float32')
    query_all = query_all.reshape((-1, d)) 
    query = query_all[:nq]

    index = BruteForceSearch(DistanceType.L2_SQUARED)
    index.setNumThreads(1)
    index.buildIndex(db)

    I, D = index.searchIndex(query, 10)

    printResults(I, D, nq)

def loadDataGUI_CHECK():
    params = get_config()

    if not params:
        print("User cancelled or no configuration provided. Exiting...")
        return  

    required_keys = ["Dataset", "Query Number", "k-Nearest Neighbors", "Distance Metric", "Search Method", "Threads"]
    for key in required_keys:
        if key not in params:
            print(f"Configuration key '{key}' missing. Exiting...")
            return

    db_name       = params["Dataset"]
    nq            = params["Query Number"]
    knn           = params["k-Nearest Neighbors"]
    dist_metric   = params["Distance Metric"]
    search_method = params["Search Method"]
    num_thread    = params["Threads"]

    filename_db, filename_query, d, nb = find_data_files(db_name)

    db = np.fromfile(filename_db, dtype='float32').reshape((nb, d))

    query_all = np.fromfile(filename_query, dtype='float32').reshape((-1, d))
    query = query_all[:nq]

    index_class = search_classes[search_method]
    index = index_class(getattr(DistanceType, dist_metric))
    index.setNumThreads(num_thread)
    index.buildIndex(db)

    I, D = index.searchIndex(query, knn)

    printResults(I, D, nq)


if __name__ == "__main__":
    # Without the GUI
    loadDataCHECK()

    # With the GUI
    loadDataGUI_CHECK()

