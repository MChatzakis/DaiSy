#include "../commons/dataloaders.hpp"
#include "../lib/algos/OdysseySearch.hpp"
#include <mpi.h> 

int main(int argc, char *argv[]){ // main function needs argc and argv for MPI_Init
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // 0. Configuration of the variables
    diNoLib::idx_t n_database = 200000;
    unsigned long long dim = 96;
    unsigned long long n_query = 10;
    diNoLib::idx_t k = 5;

    // 1. Generate random data and queries
    float *database = loadRandomData(n_database, dim, true, 100);
    float *query = loadRandomData(n_query, dim, true, 50);

    // Get MPI rank and size for informative output
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) { // Only print from rank 0 to avoid redundant output
        printf("MPI initialized with %d processes. I am rank %d.\n", size, rank);
        printf("Loaded %llu database points and %llu query points with dimension %llu\n", n_database, n_query, dim);
    }

    // 2. Create a brute-force search object
    diNoLib::OdysseySearch bf_search(diNoLib::DistanceType::L2_SQUARED);

    // 3. Build the index
    bf_search.buildIndex(database, n_database, dim);

    // 4. Search the index
    diNoLib::idx_t *I = new diNoLib::idx_t[n_query * k];
    float *D = new float[n_query * k];
    bf_search.searchIndex(query, n_query, k, I, D);

    // 5. Print the results (only from rank 0 for clarity in output)
    if (rank == 0) {
        for (diNoLib::idx_t i = 0; i < n_query; i++)
        {
            printf("Query %llu: ", i);
            for (diNoLib::idx_t j = 0; j < k; j++)
            {
                printf("%llu ", I[i * k + j]);
            }
            printf("\n");
        }
    }

    // 6. Clean up
    delete[] database;
    delete[] query;
    delete[] I;
    delete[] D;

    // Finalize MPI
    MPI_Finalize();

    return 0;
}