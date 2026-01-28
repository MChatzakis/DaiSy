#include "../hodyssey/indexing.hpp"
#include "../isax/SAX.hpp"
#include "../../utils/TimerManager.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace diNoLib
{

    // Helper macro for allocation checking (if not defined elsewhere)
    #ifndef CHECK_ALLOC
    #define CHECK_ALLOC(ptr, rank) \
        if ((ptr) == nullptr) { \
            fprintf(stderr, "[Node %d] Error: Memory allocation failed in %s:%d\n", (rank), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE); \
        }
    #endif

    // ========================================================================
    // index_creation_sequence_worker
    //
    // C++ port of:
    //   void *index_creation_sequence_worker(void *transferdata)
    //
    // NOTE:
    // - La parte relativa a znorm (index->settings->znorm, index->means, index->stds)
    //   è stata rimossa perché nel porting C++ non esistono più questi campi
    //   e la normalizzazione Z viene gestita esternamente (dati già z-normalizzati).
    // ========================================================================
    void *index_creation_sequence_worker(void *transferdata)
    {
        buffer_data_inmemory_ekosmas *input_data =
            static_cast<buffer_data_inmemory_ekosmas *>(transferdata);

        dinoLib::TimerManager *timer_manager = input_data->timer_manager;
        float *rawfile = input_data->rawfile;

        unsigned long ts_num = static_cast<unsigned long>(input_data->ts_num);
        unsigned long total_blocks = ts_num / static_cast<unsigned long>(input_data->readblock);

        isax_index *index = input_data->index;
        unsigned long localcounterblock =
            static_cast<unsigned long>(input_data->workernumber);
        unsigned long *next_block_to_process = input_data->shared_start_number;

        if (input_data->deterministic_index)
        {
            // In modalità deterministica ogni worker incrementa un contatore locale
            next_block_to_process = &localcounterblock;
        }

        int paa_segments = index->settings->paa_segments;
        int sax_byte_size = index->settings->sax_byte_size;

        file_position_type pos;
        sax_type *sax = static_cast<sax_type *>(std::malloc(static_cast<size_t>(sax_byte_size)));
        CHECK_ALLOC(sax, input_data->my_rank);

        // Nel porting C++ non gestiamo più la znorm interna:
        // current_series punta direttamente dentro rawfile.
        ts_type *current_series = nullptr;

        if (input_data->workernumber == 0)
        {
            ::dinoLib::timer_start(reinterpret_cast<::dinoLib::timer_manager_t*>(timer_manager), "BUFFER");
        }

        unsigned long i, block_num, my_ts_start, my_ts_end;
        while (true)
        {
            if (input_data->deterministic_index)
            {
                block_num = __sync_fetch_and_add(next_block_to_process,
                                                 static_cast<unsigned long>(input_data->index_threads));
            }
            else
            {
                block_num = __sync_fetch_and_add(next_block_to_process, 1UL);
            }

            if (block_num > total_blocks)
            {
                break;
            }

            my_ts_start = block_num * static_cast<unsigned long>(input_data->readblock);
            if (block_num == total_blocks)
            {
                // Ultimo blocco: può contenere meno serie del readblock
                my_ts_end = ts_num;
            }
            else
            {
                my_ts_end =
                    (block_num + 1UL) * static_cast<unsigned long>(input_data->readblock);
            }

            for (i = my_ts_start; i < my_ts_end; i++)
            {
                // NIENTE znorm interna: lavoriamo direttamente sui dati di rawfile
                current_series =
                    &rawfile[i * static_cast<unsigned long>(index->settings->timeseries_size)];

                if (sax_from_ts(current_series,
                                sax,
                                index->settings->ts_values_per_paa_segment,
                                paa_segments,
                                index->settings->sax_alphabet_cardinality,
                                index->settings->sax_bit_cardinality) == SUCCESS)
                {
                    pos = static_cast<file_position_type>(
                        i * static_cast<unsigned long>(index->settings->timeseries_size));

                    // Inserisce il record SAX nell'indice usando la versione EKOSMAS
                    isax_pRecBuf_index_insert_inmemory_ekosmas(
                        index,
                        sax,
                        &pos,
                        input_data->lock_firstnode,
                        input_data->workernumber,
                        input_data->index_threads);
                }
                else
                {
                    fprintf(stderr,
                            "error: cannot insert record in index, since sax representation "
                            "failed to be created\n");
                    std::exit(EXIT_FAILURE);
                }
            }
        }

        std::free(sax);

        // Wait for all workers to finish populating buffers before processing
        pthread_barrier_wait(input_data->wait_summaries_to_compute);

        if (input_data->workernumber == 0)
        {
            ::dinoLib::timer_stop(reinterpret_cast<::dinoLib::timer_manager_t*>(timer_manager), "BUFFER");
            ::dinoLib::timer_start(reinterpret_cast<::dinoLib::timer_manager_t*>(timer_manager), "INDEX");
        }

        // Process the populated buffers to build the iSAX tree
        tree_index_creation_from_pRecBuf_fai_blocking(transferdata);

        if (input_data->workernumber == 0)
        {
            ::dinoLib::timer_stop(reinterpret_cast<::dinoLib::timer_manager_t*>(timer_manager), "INDEX");
        }

        return nullptr;
    }

    // ========================================================================
    // tree_index_creation_from_pRecBuf_fai_blocking
    //
    // C++ port of:
    //   void tree_index_creation_from_pRecBuf_fai_blocking(void *transferdata)
    //
    // PURPOSE:
    //   Processes all buffers in the parallel_first_buffer_layer_ekosmas
    //   and inserts records into the iSAX tree structure.
    //   Uses FAI (First Available Index) blocking approach where each worker
    //   atomically increments a counter to get the next buffer to process.
    // ========================================================================
    void tree_index_creation_from_pRecBuf_fai_blocking(void *transferdata)
    {
        buffer_data_inmemory_ekosmas *input_data =
            static_cast<buffer_data_inmemory_ekosmas *>(transferdata);
        isax_index *index = input_data->index;
        int j;
        isax_node_record *r = static_cast<isax_node_record *>(std::malloc(sizeof(isax_node_record)));
        if (r == nullptr)
        {
            fprintf(stderr, "[Node %d] Error: Memory allocation failed for isax_node_record\n",
                    input_data->my_rank);
            std::exit(EXIT_FAILURE);
        }

        idx_t worker_inserts = 0;
        while (true)
        {
            // Atomically get next buffer index to process (FAI - First Available Index)
            j = __sync_fetch_and_add(input_data->node_counter, 1);
            if (j >= index->fbl->number_of_buffers)
            {
                break;
            }

            parallel_first_buffer_layer_ekosmas *p_fbl =
                reinterpret_cast<parallel_first_buffer_layer_ekosmas *>(index->fbl);
            parallel_fbl_soft_buffer_ekosmas *current_fbl_node = &(p_fbl->soft_buffers[j]);

            if (!current_fbl_node->initialized)
            {
                continue;
            }

            // Process all records from all workers for this buffer
            for (int k = 0; k < input_data->index_threads; k++)
            {
                for (int i = 0; i < current_fbl_node->buffer_size[k]; i++)
                {
                    // Set up record structure pointing to SAX and position in buffer
                    r->sax = static_cast<sax_type *>(
                        &(current_fbl_node->sax_records[k][i * index->settings->paa_segments]));
                    r->position = static_cast<file_position_type *>(
                        &(static_cast<file_position_type *>(current_fbl_node->pos_records[k])[i]));
                    r->insertion_mode = static_cast<insertion_mode>(NO_TMP | PARTIAL);

                    // Add record to index tree using in-memory version (no disk files)
                    worker_inserts++;
                    add_record_to_node_inmemory(index, static_cast<isax_node *>(current_fbl_node->node), r, 1);
                }
            }
        }

        printf("[Node %d] Worker %d inserted %llu records\n",
               input_data->my_rank,
               input_data->workernumber,
               static_cast<unsigned long long>(worker_inserts));

        std::free(r);
    }

} // namespace diNoLib

