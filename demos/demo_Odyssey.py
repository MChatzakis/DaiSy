# my_mpi_odyssey_script.py
import mpi4py.rc
# Crucially, disable mpi4py's automatic MPI_Init/Finalize
# because your C++ code (or the calling context) might handle it,
# or you want explicit control.
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False

from mpi4py import MPI
import diNoSimilaritySearch as dss
import numpy as np

# 1. Initialize MPI
# You *must* call MPI.Init() before using any MPI-dependent C++ code.
# This assumes the C++ library doesn't call MPI_Init internally.
# In your case, it seems the OdysseySearch class itself doesn't call it,
# but your demo_Odyssey.cpp does.
# If you run this script with `mpiexec`, mpi4py will handle the `Init()` call
# if `mpi4py.rc.initialize = True` (the default).
# Since you have `MPI_Init` in your C++ `main`, and you're making a Python binding,
# this is where things get a bit tricky.
#
# Option A: Your C++ library (`OdysseySearch`) does NOT call MPI_Init/Finalize.
#           The Python script becomes the "main" and handles MPI lifecycle.
#           This is the recommended approach for a well-behaved library.
#           Your OdysseySearch.cpp doesn't call MPI_Init/Finalize, which is good.
#           So, use this option.

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
# In a real MPI application, you might distribute data generation or loading
# among ranks. For this demo, we'll generate on all ranks for simplicity,
# but in a production setup, rank 0 might load and scatter, or each rank loads a portion.
database = np.random.rand(n_database, dim).astype(np.float32)
query = np.random.rand(n_query, dim).astype(np.float32)

# 2. Create an OdysseySearch object
# This object will use MPI internally if its C++ implementation does.
# The C++ `OdysseySearch` class itself uses `MPI` in its header, but its `searchIndex`
# method doesn't explicitly use MPI communication (it uses OpenMP).
# If you *intend* for OdysseySearch to perform distributed search using MPI,
# then its `searchIndex` method would need MPI calls (e.g., MPI_Gather, MPI_Allreduce etc.).
# As it stands, your `OdysseySearch` class is parallelized with OpenMP,
# but not explicitly distributed with MPI.
# If you want distributed search, your C++ `OdysseySearch::searchIndex` needs to be modified
# to use MPI communication among processes.
# For now, we'll assume a "local" search on potentially replicated data.

# If OdysseySearch needs to be aware of MPI rank/size, you might add methods to it
# or pass the communicator. For now, it seems self-contained.
odyssey_search = dss.OdysseySearch(dss.DistanceType.L2_SQUARED)

# Build the index (assuming each rank builds on its local data or full data)
odyssey_search.buildIndex(database)

# 4. Search the index
# If OdysseySearch performs distributed search, the results (I, D) would reflect
# the combined results from all ranks.
I, D = odyssey_search.searchIndex(query, k)

# 5. Print the results (only from rank 0 for clarity)
if rank == 0:
    print("\nPython Search Results (first 5 queries):")
    for i in range(min(5, n_query)):
        print(f"Query {i}: Indices {I[i, :k]} Distances {D[i, :k]}")

# 6. Finalize MPI
if MPI.Is_initialized():
    MPI.Finalize()