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

/**
 * Demo Odyssey: build index + search (L2 squared).
 *
 * Ispirata alla demo Messi L2Square: dopo la costruzione dell'indice
 * esegue anche searchIndex per verificare l'intero flusso.
 *
 * 1. Genera dati random e li scrive su file
 * 2. Crea un oggetto Odyssey con configurazione base (L2_SQUARED)
 * 3. Chiama buildIndex() per costruire l'indice distribuito
 * 4. Verifica che l'indice sia stato costruito correttamente
 * 5. Genera query random e chiama searchIndex() (L2 squared, top-k)
 * 6. Stampa risultati (indici e distanze) e cleanup
 */
int main(int argc, char *argv[])
{
    // ========================================================================
    // 1. CONFIGURAZIONE TEST
    // ========================================================================
    diNoLib::idx_t n_database = 10000;  // Dataset piccolo per test rapido
    unsigned long long dim = 256;       // Dimensione time series (deve essere multiplo di 8 per SIMD)
    unsigned long long n_query = 10;   // Numero di query per la ricerca
    diNoLib::idx_t k = 5;               // Top-k per ogni query (L2 squared)
    std::string temp_db_file = "/tmp/odyssey_test_db.bin";

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

        printf("\n[Node %d] Odyssey object created and configured\n", rank);

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

            float *query = loadRandomData(n_query, dim, 50);
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

            odyssey.searchIndex(query, n_query, k, I, D);

            printf("[Node %d] >>> searchIndex completed.\n", rank);

            if (rank == 0)
            {
                printf("\n[Node 0] Search results (indices and distances):\n");
                for (unsigned long long i = 0; i < n_query; i++)
                {
                    printf("  Query %llu: ", i);
                    for (diNoLib::idx_t j = 0; j < k; j++)
                    {
                        printf("(idx=%llu dist=%.4f) ", static_cast<unsigned long long>(I[i * k + j]), D[i * k + j]);
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

    // Odyssey esce dallo scope qui, quindi il distruttore viene chiamato
    // MPI_Finalize deve essere chiamato dopo che tutti gli oggetti MPI sono stati distrutti
    // Ma in questo caso, chiamiamolo manualmente alla fine per sicurezza

#if ODYSSEY_MPI
    MPI_Finalize();
#endif

    return 0;
}
