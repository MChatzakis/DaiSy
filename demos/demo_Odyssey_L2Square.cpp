/**
 * Demo Odyssey L2Square — IDENTICA a demo_ParIS_L2Square.cpp
 * Stessi dati, stesse query, stesso flusso (file -> build -> search) per confrontare i risultati.
 *
 * Esegui dalla root del progetto: mpirun -np 4 ./demos/demo_Odyssey_L2Square
 * (usa path in CWD così tutti i rank vedono lo stesso file; /tmp è locale per nodo in multi-node)
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

int main(int argc, char *argv[])
{
    // 0. Configuration of the variables — IDENTICA a ParIS
    diNoLib::idx_t n_database = 200000;
    unsigned long long dim = 96;
    unsigned long long n_query = 10;
    diNoLib::idx_t k = 5;

    std::string temp_db_file = "odyssey_l2square_db.bin";  // CWD: tutti i rank vedono lo stesso file

    // Odyssey inizializza MPI nel costruttore (se ODYSSEY_MPI): creare PRIMA di qualsiasi MPI_*
    diNoLib::OdysseyConfig config;
    config.search_workers = 2;
    config.index_threads = 4;
    config.query_threads = 2;
    config.leaf_size = 1000;
    config.paa_segments = 16;
    config.replication_groups = 0;

    diNoLib::Odyssey odyssey(config, diNoLib::DistanceType::L2_SQUARED, argc, argv);
    int rank = odyssey.getMyRank();
    int size = odyssey.getCommSz();
    (void)size;

    // 1. Generate random data and queries — STESSI SEED di ParIS (100, 50)
    float *database = nullptr;
    if (rank == 0)
    {
        remove(temp_db_file.c_str());  // evita file residui da run precedenti
        database = loadRandomData(n_database, dim, 100);
        printf("Loaded %llu database points with dimension %llu\n", n_database, dim);
    }

#if ODYSSEY_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // 2. Write database to temporary file (solo rank 0)
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

    // Query: stessi seed di ParIS (50). Ogni rank genera la stessa sequenza.
    float *query = loadRandomData(n_query, dim, 50);
    if (rank == 0)
        printf("Loaded %llu query points with dimension %llu\n", n_query, dim);

    // 3. Build the index — FileDataSource come ParIS (file-based)
    diNoLib::FileDataSource data_source(temp_db_file.c_str(), dim, n_database);
    odyssey.buildIndex(&data_source);
    if (rank == 0)
        printf(">>> Finished indexing\n");

    // 4. Search the index
    diNoLib::idx_t *I = static_cast<diNoLib::idx_t *>(std::malloc(sizeof(diNoLib::idx_t) * static_cast<size_t>(n_query * k)));
    float *D = static_cast<float *>(std::malloc(sizeof(float) * static_cast<size_t>(n_query * k)));
    if (I == nullptr || D == nullptr)
    {
        fprintf(stderr, "Error: Could not allocate I or D\n");
        delete[] query;
        return 1;
    }
    std::memset(I, 0, sizeof(diNoLib::idx_t) * static_cast<size_t>(n_query * k));
    std::memset(D, 0, sizeof(float) * static_cast<size_t>(n_query * k));

    odyssey.searchIndex(query, n_query, k, I, D);

    if (rank == 0)
        printf(">>> Finished search\n");

    // 5. Print the results — STESSO FORMATO di ParIS (indici per query)
    if (rank == 0)
    {
        for (diNoLib::idx_t i = 0; i < n_query; i++)
        {
            printf("Query %llu: ", static_cast<unsigned long long>(i));
            for (diNoLib::idx_t j = 0; j < k; j++)
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
        remove(temp_db_file.c_str());

#if ODYSSEY_MPI
    MPI_Finalize();
#endif

    return 0;
}
