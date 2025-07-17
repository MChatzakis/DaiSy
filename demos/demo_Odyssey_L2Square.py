import mpi4py.rc

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False

from mpi4py import MPI
import diNoSimilaritySearch as dss
import numpy as np

# 0. Initialize MPI
if not MPI.Is_initialized():
    MPI.Init()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print(f"Python: MPI initialized with {size} processes. I am rank {rank}.")

# 0. Configuration
n_database = 200000
dim = 96
n_query = 10
k = 5

# 1. Generate random data and queries
database = np.random.rand(n_database, dim).astype(np.float32)
query = np.random.rand(n_query, dim).astype(np.float32)

# 2. Create an Odyssey object
odyssey_search = dss.Odyssey(dss.DistanceType.L2_SQUARED)
odyssey_search.buildIndex(database)

# 4. Search the index
I, D = odyssey_search.searchIndex(query, k)

# 5. Print the results (only from rank 0 for clarity)
if rank == 0:
    print("\nPython Search Results (first 5 queries):")
    for i in range(min(5, n_query)):
        print(f"Query {i}: Indices {I[i, :k]} Distances {D[i, :k]}")

# 6. Finalize MPI
if MPI.Is_initialized():
    MPI.Finalize()