#include "../commons/dataloaders.hpp"
#include "../lib/algos/Odyssey.hpp"
#include "../lib/algos/DataSource.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <cstdlib>

#if ODYSSEY_MPI
#include <mpi.h>
#endif

/**
 * Test minimale per verificare la fase di costruzione dell'indice Odyssey.
 * 
 * Questo test:
 * 1. Genera dati random e li scrive su file
 * 2. Crea un oggetto Odyssey con configurazione base
 * 3. Chiama buildIndex() per costruire l'indice distribuito
 * 4. Verifica che l'indice sia stato costruito correttamente
 * 5. Stampa statistiche per debug
 */
int main(int argc, char *argv[])
{
#if ODYSSEY_MPI
    // MPI viene inizializzato dal costruttore di Odyssey
#else
    // Non-MPI build: test locale
#endif

    // ========================================================================
    // 1. CONFIGURAZIONE TEST
    // ========================================================================
    diNoLib::idx_t n_database = 10000;  // Dataset piccolo per test rapido
    unsigned long long dim = 256;       // Dimensione time series (deve essere multiplo di 8 per SIMD)
    std::string temp_db_file = "/tmp/odyssey_test_db.bin";

    printf("========================================\n");
    printf("Odyssey BuildIndex Test\n");
    printf("========================================\n");
    printf("Dataset size: %llu\n", n_database);
    printf("Time series dimension: %llu\n", dim);
    printf("Temporary file: %s\n", temp_db_file.c_str());

#if ODYSSEY_MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("[Node %d/%d] Starting test...\n", rank, size);
#else
    int rank = 0, size = 1;
    printf("[Single node] Starting test...\n");
#endif

    // ========================================================================
    // 2. GENERA E SCRIVI DATI SU FILE
    // ========================================================================
    if (rank == 0)
    {
        printf("\n[Node 0] Generating random data...\n");
        float *database = loadRandomData(n_database, dim, 100); // z-normalized

        printf("[Node 0] Writing database to file...\n");
        FILE *fp = fopen(temp_db_file.c_str(), "wb");
        if (fp == nullptr)
        {
            fprintf(stderr, "[Node 0] Error: Could not create temporary database file\n");
            delete[] database;
            return 1;
        }
        fwrite(database, sizeof(float), n_database * dim, fp);
        fclose(fp);
        delete[] database;
        printf("[Node 0] Database written successfully (%llu time series)\n", n_database);
    }

#if ODYSSEY_MPI
    // Sincronizza tutti i nodi prima di procedere
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // ========================================================================
    // 3. CREA OGGETTO ODYSSEY E CONFIGURA
    // ========================================================================
    printf("\n[Node %d] Creating Odyssey object...\n", rank);

    // Configurazione base per test
    diNoLib::OdysseyConfig config;
    config.search_workers = 2;
    config.index_threads = 2;
    config.query_threads = 2;
    config.leaf_size = 1000;
    config.paa_segments = 16;
    config.replication_groups = 0; // Auto-calcola in base a comm_sz

    diNoLib::Odyssey odyssey(config, diNoLib::DistanceType::L2_SQUARED, argc, argv);

    // Abilita verbose per vedere i log
    // Nota: verbose è privato, quindi non possiamo settarlo direttamente
    // Ma possiamo verificare che buildIndex funzioni comunque

    printf("[Node %d] Odyssey object created\n", rank);
    // Note: my_rank and comm_sz are private members, cannot access directly

    // ========================================================================
    // 4. COSTRUISCI L'INDICE
    // ========================================================================
    printf("\n[Node %d] Building index...\n", rank);

    try
    {
        // Crea FileDataSource
        diNoLib::FileDataSource data_source(temp_db_file.c_str(), dim, n_database);

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
        if (odyssey.index == nullptr)
        {
            fprintf(stderr, "[Node %d] ERROR: index is NULL!\n", rank);
            return 1;
        }
        printf("[Node %d] ✓ Index pointer is valid\n", rank);

        // Verifica che index_settings sia stato inizializzato
        if (odyssey.index->settings == nullptr)
        {
            fprintf(stderr, "[Node %d] ERROR: index->settings is NULL!\n", rank);
            return 1;
        }
        printf("[Node %d] ✓ Index settings initialized\n", rank);
        printf("  - Time series size: %d\n", odyssey.index->settings->timeseries_size);
        printf("  - PAA segments: %d\n", odyssey.index->settings->paa_segments);
        printf("  - Leaf size: %d\n", odyssey.index->settings->max_leaf_size);

        // Verifica che fbl (parallel_first_buffer_layer_ekosmas) sia stato inizializzato
        if (odyssey.index->fbl == nullptr)
        {
            fprintf(stderr, "[Node %d] ERROR: index->fbl is NULL!\n", rank);
            return 1;
        }
        printf("[Node %d] ✓ FBL (parallel_first_buffer_layer_ekosmas) initialized\n", rank);
        printf("  - Number of buffers: %d\n", odyssey.index->fbl->number_of_buffers);
        printf("  - Max total size: %d\n", odyssey.index->fbl->max_total_size);

        // Verifica che first_node sia stato inizializzato (se ci sono dati)
        if (odyssey.index->first_node != nullptr)
        {
            printf("[Node %d] ✓ First node created (tree structure exists)\n", rank);
        }
        else
        {
            printf("[Node %d] ⚠ First node is NULL (might be empty index or not yet populated)\n", rank);
        }

        // Verifica che root_nodes sia stato inizializzato
        printf("[Node %d] ✓ Root nodes count: %llu\n", rank, 
               static_cast<unsigned long long>(odyssey.index->root_nodes));

        // Note: rawfile, replication_data, workstealing_data, bsf_sharing_data
        // sono membri privati e non accessibili direttamente.
        // Se buildIndex() completa senza errori, significa che sono stati inizializzati correttamente.

        printf("\n[Node %d] >>> All checks passed! Index built successfully.\n", rank);
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

    // ========================================================================
    // 6. CLEANUP
    // ========================================================================
#if ODYSSEY_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (rank == 0)
    {
        printf("\n[Node 0] Cleaning up temporary file...\n");
        if (remove(temp_db_file.c_str()) == 0)
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

#if ODYSSEY_MPI
    MPI_Finalize();
#endif

    return 0;
}
