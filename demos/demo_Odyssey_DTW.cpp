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
#if (defined(__unix__) || defined(__APPLE__)) && defined(PATH_MAX)
#include <limits.h>
#include <stdlib.h>
#endif

/**
 * Demo Odyssey: build index + search (DTW).
 * Same data and queries as demo_ParIS_DTW for result comparison.
 *
 * Parameters identical to demo_ParIS_DTW: 200000 series, dim 96, 10 queries, k=5, seed 100/50.
 * Warping window: max(1, dim*0.1) = 10 (as ParIS).
 * File: /tmp/paris_test_db.bin (same path as ParIS).
 *
 * To compare: 1) ./demos/demo_ParIS_DTW  2) mpirun -np 4 ./demos/demo_Odyssey_DTW
 */
int main(int argc, char *argv[])
{
    // ========================================================================
    // 1. CONFIGURATION — IDENTICAL to demo_ParIS_DTW
    // ========================================================================
    diNoLib::idx_t n_database = 200000;
    unsigned long long dim = 96;
    unsigned long long n_query = 10;
    diNoLib::idx_t k = 5;
    std::string temp_db_file = "/tmp/paris_test_db.bin";
    int warp_window = std::max(1, static_cast<int>(dim * 0.1));

    // ========================================================================
    // 2. CREATE ODYSSEY OBJECT FIRST (initializes MPI)
    // ========================================================================
    diNoLib::OdysseyConfig config;
    config.search_workers = 2;
    config.index_threads = 2;
    config.query_threads = 2;
    config.leaf_size = 1000;
    config.paa_segments = 16;
    config.replication_groups = 0;
    config.warping_window = warp_window;

    int rank = 0;
    int size = 1;
    std::string path_to_use;

    {
        diNoLib::Odyssey odyssey(config, diNoLib::DistanceType::DTW, argc, argv);

        rank = odyssey.getMyRank();
        size = odyssey.getCommSz();

    // ========================================================================
    // 3. GENERATE AND WRITE DATA TO FILE
    // ========================================================================
    if (rank == 0)
    {
        float *database = loadRandomData(n_database, dim, 100, true);

        FILE *fp = fopen(temp_db_file.c_str(), "wb");
        if (fp == nullptr)
        {
            fprintf(stderr, "[Node 0] Error: Could not create temporary database file\n");
            delete[] database;
            return 1;
        }
        size_t to_write = static_cast<size_t>(n_database) * static_cast<size_t>(dim);
        size_t written = fwrite(database, sizeof(float), to_write, fp);
        fclose(fp);
        delete[] database;
        if (written != to_write)
        {
            fprintf(stderr, "[Node 0] Error: wrote only %zu floats (expected %zu)\n", written, to_write);
            return 1;
        }
        printf("Loaded %llu database points and %llu query points with dimension %llu\n",
               static_cast<unsigned long long>(n_database), static_cast<unsigned long long>(n_query), dim);
    }

#if ODYSSEY_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // Absolute path: rank 0 computes it and broadcasts to all
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
    path_to_use = std::string(path_buf);

#if ODYSSEY_MPI
    long long local_size = -1;
    FILE *check_fp = fopen(path_to_use.c_str(), "rb");
    if (check_fp != nullptr)
    {
        if (fseek(check_fp, 0, SEEK_END) == 0)
            local_size = static_cast<long long>(ftell(check_fp));
        fclose(check_fp);
    }
    long long expected_size = static_cast<long long>(n_database) * static_cast<long long>(dim) * static_cast<long long>(sizeof(float));
    long long size_from_rank0 = (rank == 0) ? expected_size : 0;
    MPI_Bcast(&size_from_rank0, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    if (local_size != size_from_rank0)
    {
        fprintf(stderr, "[Node %d] ERROR: this rank sees a different file than rank 0.\n", rank);
        fprintf(stderr, "  Path: %s\n", path_to_use.c_str());
        fprintf(stderr, "  Likely cause: multi-node with /tmp local per node.\n");
        fprintf(stderr, "  Solution: mpirun -np 4 --bind-to core ./demos/demo_Odyssey_DTW\n");
        fflush(stderr);
        return 1;
    }
#endif

        // ========================================================================
        // 4. BUILD THE INDEX
        // ========================================================================
        try
        {
            diNoLib::FileDataSource data_source(path_to_use.c_str(), dim, n_database);

            odyssey.buildIndex(&data_source);

            if (rank == 0)
                printf(">>> Finished indexing\n>>> Finished indexing \n");

            if (odyssey.getIndex() == nullptr)
            {
                fprintf(stderr, "[Node %d] ERROR: index is NULL!\n", rank);
                return 1;
            }
            if (odyssey.getIndex()->settings == nullptr)
            {
                fprintf(stderr, "[Node %d] ERROR: index->settings is NULL!\n", rank);
                return 1;
            }

            // ========================================================================
            // 5. DTW SEARCH (as in demo ParIS DTW)
            // ========================================================================
            if (rank == 0)
            {
                printf("@ Starting search\n");
                printf("@ Starting search Variables are been set\n");
                printf("@ going searchIndex constructor\n");
            }

            float *query = loadRandomData(n_query, dim, 50, true);
            if (query == nullptr)
            {
                fprintf(stderr, "[Node %d] Error: Could not allocate/generate query data\n", rank);
                return 1;
            }

            diNoLib::idx_t *I = static_cast<diNoLib::idx_t *>(std::malloc(sizeof(diNoLib::idx_t) * static_cast<size_t>(n_query * k)));
            float *D = static_cast<float *>(std::malloc(sizeof(float) * static_cast<size_t>(n_query * k)));
            if (I == nullptr || D == nullptr)
            {
                fprintf(stderr, "[Node %d] Error: Could not allocate I or D\n", rank);
                delete[] query;
                return 1;
            }
            std::memset(I, 0, sizeof(diNoLib::idx_t) * static_cast<size_t>(n_query * k));
            std::memset(D, 0, sizeof(float) * static_cast<size_t>(n_query * k));

            odyssey.searchIndex(query, n_query, k, I, D);

            if (rank == 0)
            {
                printf(">>> Finished querying.\n>>> Finished search \n");
                for (unsigned long long i = 0; i < n_query; i++)
                {
                    printf("Query %llu: ", i);
                    for (diNoLib::idx_t j = 0; j < k; j++)
                    {
                        printf("%llu ", static_cast<unsigned long long>(I[i * k + j]));
                    }
                    printf("\n");
                }
            }

            std::free(I);
            std::free(D);
            delete[] query;
        }
        catch (const std::exception &e)
        {
            fprintf(stderr, "[Node %d] ERROR: Exception: %s\n", rank, e.what());
            return 1;
        }
        catch (...)
        {
            fprintf(stderr, "[Node %d] ERROR: Unknown exception\n", rank);
            return 1;
        }
    }

    // ========================================================================
    // 7. CLEANUP
    // ========================================================================
#if ODYSSEY_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (rank == 0)
    {
        if (remove(path_to_use.c_str()) != 0)
            fprintf(stderr, "[Node 0] Warning: Could not remove temporary file\n");
    }

#if ODYSSEY_MPI
    MPI_Finalize();
#endif

    return 0;
}
