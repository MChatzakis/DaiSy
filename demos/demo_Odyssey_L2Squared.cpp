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
 * Demo Odyssey: build index + search (L2 squared).
 * Stessi dati e stesse query della demo ParIS L2Square per confrontare i risultati.
 *
 * Parametri identici a demo_ParIS_L2Square: 200000 serie, dim 96, 10 query, k=5, seed 100/50.
 * File: /tmp/paris_test_db.bin (stesso path di ParIS).
 *
 * Per confrontare: 1) ./demos/demo_ParIS_L2Square  2) mpirun -np 4 ./demos/demo_Odyssey_L2Squared
 */
int main(int argc, char *argv[])
{
    // ========================================================================
    // 1. CONFIGURAZIONE — IDENTICA a demo_ParIS_L2Square
    // ========================================================================
    diNoLib::idx_t n_database = 200000;
    unsigned long long dim = 96;
    unsigned long long n_query = 10;
    diNoLib::idx_t k = 5;
    std::string temp_db_file = "/tmp/paris_test_db.bin";

    // ========================================================================
    // 2. CREA OGGETTO ODYSSEY PRIMA DI TUTTO (inizializza MPI)
    // ========================================================================
    // IMPORTANTE: Odyssey inizializza MPI nel costruttore, quindi deve essere
    // creato PRIMA di qualsiasi chiamata MPI (come MPI_Comm_rank, MPI_Barrier, etc.)
    diNoLib::OdysseyConfig config;
    config.search_workers = 2;
    config.index_threads = 2;
    config.query_threads = 2;
    config.leaf_size = 1000;
    config.paa_segments = 16;
    config.replication_groups = 0; // Auto-calcola in base a comm_sz

    // Variabili per rank e size (saranno settate dopo la creazione di Odyssey)
    int rank = 0;
    int size = 1;
    std::string path_to_use;  // path assoluto del file (broadcast da rank 0), usato anche in cleanup

    // Blocco scope per Odyssey: viene distrutto prima di MPI_Finalize
    {
        diNoLib::Odyssey odyssey(config, diNoLib::DistanceType::L2_SQUARED, argc, argv);

        // Ora possiamo ottenere rank e size da Odyssey (MPI è già inizializzato)
        rank = odyssey.getMyRank();
        size = odyssey.getCommSz();

    // ========================================================================
    // 3. GENERA E SCRIVI DATI SU FILE
    // ========================================================================
    if (rank == 0)
    {
        float *database = loadRandomData(n_database, dim, 100);

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

    // Path assoluto: rank 0 lo calcola e lo invia a tutti (tutti aprono lo stesso file)
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
    // Verifica che tutti i rank vedano lo stesso file (stessa dimensione)
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
        fprintf(stderr, "[Node %d] ERRORE: questo rank vede un file diverso da rank 0.\n", rank);
        fprintf(stderr, "  Path: %s\n", path_to_use.c_str());
        fprintf(stderr, "  Dimensione vista da questo rank: %lld byte\n", static_cast<long long>(local_size));
        fprintf(stderr, "  Dimensione attesa (rank 0):      %lld byte\n", static_cast<long long>(size_from_rank0));
        fprintf(stderr, "  Probabile causa: multi-nodo con /tmp locale per nodo.\n");
        fprintf(stderr, "  Soluzione: lancia su UN SOLO NODO, es: mpirun -np 4 --bind-to core ./demos/demo_Odyssey_L2Squared\n");
        fflush(stderr);
        return 1;
    }
#endif

        // ========================================================================
        // 4. COSTRUISCI L'INDICE
        // ========================================================================
        try
        {
            // Crea FileDataSource — tutti usano lo stesso path (broadcast da rank 0)
            diNoLib::FileDataSource data_source(path_to_use.c_str(), dim, n_database);

            // Chiama buildIndex (questo è il test principale!)
            odyssey.buildIndex(&data_source);

            if (rank == 0)
                printf(">>> Finished indexing\n>>> Finished indexing \n");

            // Verifica che l'indice sia stato creato
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
            if (odyssey.getIndex()->fbl == nullptr)
            {
                fprintf(stderr, "[Node %d] ERROR: index->fbl is NULL!\n", rank);
                return 1;
            }

            // ========================================================================
            // 5. RICERCA L2 SQUARED (come in demo Messi L2Square)
            // ========================================================================
            if (rank == 0)
            {
                printf("@ Starting search\n");
                printf("@ Starting search Variables are been set\n");
                printf("@ going searchIndex constructor\n");
            }

            float *query = loadRandomData(n_query, dim, 50);  // stesso seed 50 di ParIS
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
            fprintf(stderr, "[Node %d] ERROR: Exception during buildIndex: %s\n", rank, e.what());
            return 1;
        }
        catch (...)
        {
            fprintf(stderr, "[Node %d] ERROR: Unknown exception during buildIndex\n", rank);
            return 1;
        }
    } // Fine blocco scope per Odyssey - il distruttore viene chiamato qui

    // ========================================================================
    // 6. CLEANUP
    // ========================================================================
#if ODYSSEY_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (rank == 0)
    {
        if (remove(path_to_use.c_str()) != 0)
            fprintf(stderr, "[Node 0] Warning: Could not remove temporary file\n");
    }

    // Odyssey esce dallo scope qui, quindi il distruttore viene chiamato
    // MPI_Finalize deve essere chiamato dopo che tutti gli oggetti MPI sono stati distrutti
    // Ma in questo caso, chiamiamolo manualmente alla fine per sicurezza

#if ODYSSEY_MPI
    MPI_Finalize();
#endif

    return 0;
}
