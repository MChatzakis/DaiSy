/**
 * Demo Odyssey L2Square — IDENTICAL to demo_ParIS_L2Square.cpp
 * Same data, same queries, same flow (file -> build -> search) for result comparison.
 *
 * Run from project root: mpirun -np 4 ./demos/demo_Odyssey_L2Square
 * (use path in CWD so all ranks see the same file; /tmp is local per node in multi-node)
 */
#include "../commons/dataloaders.hpp"
#include "../lib/algos/hodyssey/Odyssey.hpp"
#include "../lib/algos/DataSource.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <cstdlib>

#if ODYSSEY_MPI
#include <mpi.h>
#endif

#if defined(__unix__) || defined(__APPLE__)
#include <limits.h>
#include <stdlib.h>
#endif

int main(int argc, char *argv[])
{
    // 0. Configuration of the variables — IDENTICAL to ParIS
    daisy::idx_t n_database = 200000;
    unsigned long long dim = 96;
    unsigned long long n_query = 10;
    daisy::idx_t k = 5;

    std::string temp_db_file = "odyssey_l2square_db.bin";  // CWD: all ranks see the same file

    // Odyssey initializes MPI in constructor (if ODYSSEY_MPI): create BEFORE any MPI_* call
    daisy::OdysseyConfig config;
    config.search_workers = 2;
    config.index_threads = 4;
    config.query_threads = 2;
    config.leaf_size = 1000;
    config.paa_segments = 16;
    config.replication_groups = 0;

    daisy::Odyssey odyssey(config, daisy::DistanceType::L2_SQUARED, argc, argv);
    int rank = odyssey.getMyRank();
    int size = odyssey.getCommSz();
    (void)size;

    // 1. Generate random data and queries — SAME SEED as ParIS (100, 50)
    float *database = nullptr;
    if (rank == 0)
    {
        remove(temp_db_file.c_str());  // avoid leftover files from previous runs
        database = loadRandomData(n_database, dim, 100, true);
        printf("Loaded %llu database points with dimension %llu\n", n_database, dim);
    }

#if ODYSSEY_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // 2. Write database to temporary file (rank 0 only)
    if (rank == 0)
    {
        FILE *fp = fopen(temp_db_file.c_str(), "wb");
        if (fp == nullptr)
        {
            fprintf(stderr, "Error: Could not create temporary database file\n");
            delete[] database;
            return 1;
        }
        size_t to_write = static_cast<size_t>(n_database) * static_cast<size_t>(dim);
        size_t written = fwrite(database, sizeof(float), to_write, fp);
        fclose(fp);
        delete[] database;
        database = nullptr;
        if (written != to_write)
        {
            fprintf(stderr, "Error: wrote only %zu floats (expected %zu)\n", written, to_write);
            return 1;
        }
    }

#if ODYSSEY_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // Absolute file path: rank 0 computes it and broadcasts to all (so all ranks open the same file)
    static const int PATH_MAX_MPI = 1024;
    char path_buf[PATH_MAX_MPI];
    std::memset(path_buf, 0, PATH_MAX_MPI);
    if (rank == 0)
    {
#if (defined(__unix__) || defined(__APPLE__)) && defined(PATH_MAX)
        char resolved[PATH_MAX];
        if (realpath(temp_db_file.c_str(), resolved) != nullptr)
            std::strncpy(path_buf, resolved, PATH_MAX_MPI - 1);
        else
#endif
            std::strncpy(path_buf, temp_db_file.c_str(), PATH_MAX_MPI - 1);
        path_buf[PATH_MAX_MPI - 1] = '\0';
    }
#if ODYSSEY_MPI
    MPI_Bcast(path_buf, PATH_MAX_MPI, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif
    std::string path_to_use(path_buf);

    // Query: same seed as ParIS (50). Each rank generates the same sequence.
    float *query = loadRandomData(n_query, dim, 50, true);
    if (rank == 0)
        printf("Loaded %llu query points with dimension %llu\n", n_query, dim);

    // 3. Build the index — FileDataSource as ParIS (file-based); all ranks use the same path
    daisy::FileDataSource data_source(path_to_use.c_str(), dim, n_database);
    odyssey.buildIndex(&data_source);
    if (rank == 0)
        printf(">>> Finished indexing\n");

    // 4. Search the index
    daisy::idx_t *I = static_cast<daisy::idx_t *>(std::malloc(sizeof(daisy::idx_t) * static_cast<size_t>(n_query * k)));
    float *D = static_cast<float *>(std::malloc(sizeof(float) * static_cast<size_t>(n_query * k)));
    if (I == nullptr || D == nullptr)
    {
        fprintf(stderr, "Error: Could not allocate I or D\n");
        delete[] query;
        return 1;
    }
    std::memset(I, 0, sizeof(daisy::idx_t) * static_cast<size_t>(n_query * k));
    std::memset(D, 0, sizeof(float) * static_cast<size_t>(n_query * k));

    odyssey.searchIndex(query, n_query, k, I, D);

    if (rank == 0)
        printf(">>> Finished search\n");

    // 5. Print the results — SAME FORMAT as ParIS (indices per query)
    if (rank == 0)
    {
        for (daisy::idx_t i = 0; i < n_query; i++)
        {
            printf("Query %llu: ", static_cast<unsigned long long>(i));
            for (daisy::idx_t j = 0; j < k; j++)
            {
                printf("%llu ", static_cast<unsigned long long>(I[i * k + j]));
            }
            printf("\n");
        }
    }

    // 6. Clean up
    delete[] query;
    std::free(I);
    std::free(D);
    if (rank == 0)
        remove(path_to_use.c_str());

#if ODYSSEY_MPI
    MPI_Finalize();
#endif

    return 0;
}
