import mpi4py.rc

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False

from mpi4py import MPI
import daisy
import numpy as np

if not MPI.Is_initialized():
    MPI.Init()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print(f"Python: MPI initialized with {size} processes. I am rank {rank}.")

n_database = 200000
dim = 96
n_query = 10
k = 5
warp_window = max(1, int(dim * 0.1))  

np.random.seed(100)
db = np.random.randn(n_database, dim).astype(np.float32)

np.random.seed(50)
query = np.random.randn(n_query, dim).astype(np.float32)

odyssey_search = daisy.Odyssey(daisy.DistanceType.DTW)
odyssey_search.setWarpingWindow(warp_window)
odyssey_search.buildIndex(db)

I, D = odyssey_search.searchIndex(query, k)

if rank == 0:
    print("\nPython Search Results (DTW):")
    for i in range(n_query):
        print(f"Query {i}:")
        print("Distances:", D[i])
        print("Indices:", I[i])
        print()

if MPI.Is_initialized():
    MPI.Finalize()
