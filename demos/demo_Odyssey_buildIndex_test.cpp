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
 * Per confrontare: 1) ./demos/demo_ParIS_L2Square  2) mpirun -np 4 ./demos/demo_Odyssey_buildIndex_test
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

    printf("========================================\n");
    printf("Odyssey BuildIndex Test\n");
    printf("========================================\n");
    printf("Dataset size: %llu\n", n_database);
    printf("Time series dimension: %llu\n", dim);
    printf("Temporary file: %s\n", temp_db_file.c_str());
    printf("[Node %d/%d] Starting test...\n", rank, size);

    // ========================================================================
    // 3. GENERA E SCRIVI DATI SU FILE
    // ========================================================================
    if (rank == 0)
    {
        printf("\n[Node 0] Generating random data (same seed 100 as ParIS)...\n");
        float *database = loadRandomData(n_database, dim, 100);

        printf("[Node 0] Writing database to file %s ...\n", temp_db_file.c_str());
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
        printf("[Node 0] Database written successfully (%llu time series)\n", n_database);
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

        printf("\n[Node %d] Odyssey object created and configured\n", rank);

        // ========================================================================
        // 4. COSTRUISCI L'INDICE
        // ========================================================================
        printf("\n[Node %d] Building index...\n", rank);

        try
        {
            // Crea FileDataSource — tutti usano lo stesso path (broadcast da rank 0)
            diNoLib::FileDataSource data_source(path_to_use.c_str(), dim, n_database);

            printf("[Node %d] FileDataSource created:\n", rank);
            printf("  - Filename: %s\n", data_source.getFilename());
            printf("  - Dimension: %llu\n", data_source.getDim());
            printf("  - Total records: %llu\n", data_source.getTotalRecords());

            // Chiama buildIndex (questo è il test principale!)
            odyssey.buildIndex(&data_source);

            printf("[Node %d] >>> buildIndex() completed successfully!\n", rank);

            // ========================================================================
            // 5. VERIFICA RISULTATI
            // ========================================================================
            printf("\n[Node %d] Verifying index structure...\n", rank);

            // Verifica che l'indice sia stato creato
            if (odyssey.getIndex() == nullptr)
            {
                fprintf(stderr, "[Node %d] ERROR: index is NULL!\n", rank);
                return 1;
            }
            printf("[Node %d] ✓ Index pointer is valid\n", rank);

            // Verifica che index_settings sia stato inizializzato
            if (odyssey.getIndex()->settings == nullptr)
            {
                fprintf(stderr, "[Node %d] ERROR: index->settings is NULL!\n", rank);
                return 1;
            }
            printf("[Node %d] ✓ Index settings initialized\n", rank);
            printf("  - Time series size: %d\n", odyssey.getIndex()->settings->timeseries_size);
            printf("  - PAA segments: %d\n", odyssey.getIndex()->settings->paa_segments);
            printf("  - Leaf size: %d\n", odyssey.getIndex()->settings->max_leaf_size);

            // Verifica che fbl (parallel_first_buffer_layer_ekosmas) sia stato inizializzato
            if (odyssey.getIndex()->fbl == nullptr)
            {
                fprintf(stderr, "[Node %d] ERROR: index->fbl is NULL!\n", rank);
                return 1;
            }
            printf("[Node %d] ✓ FBL (parallel_first_buffer_layer_ekosmas) initialized\n", rank);
            printf("  - Number of buffers: %d\n", odyssey.getIndex()->fbl->number_of_buffers);
            printf("  - Max total size: %d\n", odyssey.getIndex()->fbl->max_total_size);

            // Verifica che first_node sia stato inizializzato (se ci sono dati)
            if (odyssey.getIndex()->first_node != nullptr)
            {
                printf("[Node %d] ✓ First node created (tree structure exists)\n", rank);
            }
            else
            {
                printf("[Node %d] ⚠ First node is NULL (might be empty index or not yet populated)\n", rank);
            }

            // Verifica che root_nodes sia stato inizializzato
            printf("[Node %d] ✓ Root nodes count: %llu\n", rank, 
                   static_cast<unsigned long long>(odyssey.getIndex()->root_nodes));

            // Note: rawfile, replication_data, workstealing_data, bsf_sharing_data
            // sono membri privati e non accessibili direttamente.
            // Se buildIndex() completa senza errori, significa che sono stati inizializzati correttamente.

            printf("\n[Node %d] >>> All checks passed! Index built successfully.\n", rank);

            // ========================================================================
            // 6. RICERCA L2 SQUARED (come in demo Messi L2Square)
            // ========================================================================
            printf("\n[Node %d] Preparing queries and running searchIndex (L2 squared, k=%llu)...\n", rank, static_cast<unsigned long long>(k));

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

            printf("[Node %d] >>> searchIndex completed.\n", rank);

            if (rank == 0)
            {
                printf("\n[Node 0] Search results (stesso formato di ParIS per confronto):\n");
                for (unsigned long long i = 0; i < n_query; i++)
                {
                    printf("Query %llu: ", i);
                    for (diNoLib::idx_t j = 0; j < k; j++)
                    {
                        printf("%llu ", static_cast<unsigned long long>(I[i * k + j]));
                    }
                    printf("\n");
                }
                printf("\n[Node 0] (Distanze: ");
                for (unsigned long long i = 0; i < n_query; i++)
                    for (diNoLib::idx_t j = 0; j < k; j++)
                        printf("%.4f ", D[i * k + j]);
                printf(")\n");
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
        printf("\n[Node 0] Cleaning up temporary file %s ...\n", path_to_use.c_str());
        if (remove(path_to_use.c_str()) == 0)
        {
            printf("[Node 0] Temporary file removed\n");
        }
        else
        {
            fprintf(stderr, "[Node 0] Warning: Could not remove temporary file\n");
        }
    }

    printf("\n[Node %d] ========================================\n", rank);
    printf("[Node %d] Test completed successfully!\n", rank);
    printf("[Node %d] ========================================\n", rank);

    // Odyssey esce dallo scope qui, quindi il distruttore viene chiamato
    // MPI_Finalize deve essere chiamato dopo che tutti gli oggetti MPI sono stati distrutti
    // Ma in questo caso, chiamiamolo manualmente alla fine per sicurezza

#if ODYSSEY_MPI
    MPI_Finalize();
#endif

    return 0;
}
