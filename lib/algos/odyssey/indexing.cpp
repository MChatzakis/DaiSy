#include "../hodyssey/indexing.hpp"
#include "../isax/SAX.hpp"
#include "../../utils/TimerManager.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <climits>

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

        // ::dinoLib::TimerManager *timer_manager = input_data->timer_manager;  // Commented out - only for profiling
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
            // ::dinoLib::timer_start(reinterpret_cast<::dinoLib::timer_manager_t*>(timer_manager), "BUFFER");  // Commented out - only for profiling
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

        // Timer calls commented out - only for profiling
        // if (input_data->workernumber == 0)
        // {
        //     ::dinoLib::timer_stop(reinterpret_cast<::dinoLib::timer_manager_t*>(timer_manager), "BUFFER");
        //     ::dinoLib::timer_start(reinterpret_cast<::dinoLib::timer_manager_t*>(timer_manager), "INDEX");
        // }

        // Process the populated buffers to build the iSAX tree
        tree_index_creation_from_pRecBuf_fai_blocking(transferdata);

        // Timer calls commented out - only for profiling
        // if (input_data->workernumber == 0)
        // {
        //     ::dinoLib::timer_stop(reinterpret_cast<::dinoLib::timer_manager_t*>(timer_manager), "INDEX");
        // }

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

    // Helper functions for tree analysis (faithful C++ port from original C code)
    long int find_total_nodes(isax_node *root_node)
    {
        long int c = 1;
        if (root_node == nullptr)
        {
            return 0;
        }
        else
        {
            c += find_total_nodes(root_node->left_child);
            c += find_total_nodes(root_node->right_child);
            return c;
        }
    }

    long int find_total_leafs_nodes(isax_node *root_node)
    {
        if (root_node == nullptr)
        {
            return 0;
        }

        if (root_node->left_child == nullptr && root_node->right_child == nullptr)
        {
            return 1;
        }
        else
        {
            return find_total_leafs_nodes(root_node->left_child) + 
                   find_total_leafs_nodes(root_node->right_child);
        }
    }

    long int find_tree_height(isax_node *root_node)
    {
        if (root_node == nullptr)
        {
            return -1;
        }
        else
        {
            long int left_depth = find_tree_height(root_node->left_child);
            long int right_depth = find_tree_height(root_node->right_child);

            if (left_depth > right_depth)
            {
                return left_depth + 1;
            }
            else
            {
                return right_depth + 1;
            }
        }
    }

    long int find_total_tree_leafs_depths(isax_node *root_node, long int depth)
    {
        static long int total_leafs_depths = 0;
        if (depth == 0)
        {
            // If we change tree, initialize again the variable
            total_leafs_depths = 0;
        }

        if (root_node == nullptr)
        {
            return 0;
        }
        else if (root_node->left_child == nullptr && root_node->right_child == nullptr)
        {
            // is_leaf
            total_leafs_depths += depth;
        }

        // NOT TRUE: intermediate node
        find_total_tree_leafs_depths(root_node->left_child, depth + 1);
        find_total_tree_leafs_depths(root_node->right_child, depth + 1);

        return total_leafs_depths;
    }

    long int count_ts_in_nodes(isax_node *root_node, const char parallelism_in_subtree, 
                               const char recBuf_helpers_exist)
    {
        long int my_subtree_nodes = 0;

        if (!root_node->is_leaf)
        {
            my_subtree_nodes = count_ts_in_nodes(root_node->left_child, parallelism_in_subtree, recBuf_helpers_exist);
            my_subtree_nodes += count_ts_in_nodes(root_node->right_child, parallelism_in_subtree, recBuf_helpers_exist);
            return my_subtree_nodes;
        }
        else
        {
            // Leaf node: return leaf_size (simplified version - original code had more complex logic)
            // NOTE: Original code checked for LOCKFREE_PARALLELISM_IN_SUBTREE flags and fai_leaf_size,
            // but these are not present in our simplified EKOSMAS version
            return root_node->leaf_size;
        }
    }

    long int find_total_nodes_tmp(isax_node *root_node)
    {
        long int c = 1;
        if (root_node == nullptr)
        {
            return 0;
        }
        else
        {
            c += find_total_nodes(root_node->left_child);
            c += find_total_nodes(root_node->right_child);
            return c;
        }
    }

    long int find_tree_height_tmp(isax_node *root_node)
    {
        if (root_node == nullptr)
        {
            return -1;
        }
        else
        {
            long int left_depth = find_tree_height(root_node->left_child);
            long int right_depth = find_tree_height(root_node->right_child);

            if (left_depth > right_depth)
            {
                return left_depth + 1;
            }
            else
            {
                return right_depth + 1;
            }
        }
    }

    // Debug function to print index statistics (faithful C++ port from original C code)
    void print_index_stats(isax_index *index, int my_rank)
    {
        long int empty_subtrees_buffers = 0;
        int non_empty_subtrees_cnt = 0;

        long int total_nodes = 0;
        long int tree_height = 0;
        // long int total_tree_leafs_depths = 0;

        long int total_leafs_nodes = 0;

        // Cast fbl to parallel_first_buffer_layer_ekosmas (EKOSMAS version)
        parallel_first_buffer_layer_ekosmas *fbl_ekosmas = 
            reinterpret_cast<parallel_first_buffer_layer_ekosmas *>(index->fbl);

        for (int i = 0; i < fbl_ekosmas->number_of_buffers; i++)
        {
            parallel_fbl_soft_buffer_ekosmas *current_fbl_node = &fbl_ekosmas->soft_buffers[i];
            if (!current_fbl_node->initialized)
            {
                empty_subtrees_buffers++;
                continue;
            }

            non_empty_subtrees_cnt++;

            isax_node *subtree_root = current_fbl_node->node;
            long subtree_nodes = find_total_nodes(subtree_root);
            long subtree_height = find_tree_height(subtree_root);
            long subtree_leafs = find_total_leafs_nodes(subtree_root);

            total_nodes += subtree_nodes;
            tree_height += subtree_height;
            total_leafs_nodes += subtree_leafs;

            if (subtree_nodes > 1)
            {
                printf("Subtree[%d]: (height=%lu, nodes=%lu, leafs=%lu)\n", i, subtree_height, subtree_nodes, subtree_leafs);
            }

            // total_tree_leafs_depths += find_total_tree_leafs_depths(subtree_root, 0);
        }

        printf("\n--------------------\n[Node %d]: Tree Stats:\n"
               "Total buffers(2^16): %d\n"
               "Total tree nodes: %ld\n"
               "Total leafs nodes: %ld\n"
               "Average subtree height: %f\n"
               // "Total tree leafs depth: %ld\n"
               // "Average leaf depth: %Lf\n"
               "Empty subtrees: %ld\n"
               "Non Empty subtrees: %d\n-------------------\n",
               my_rank,
               fbl_ekosmas->number_of_buffers,
               total_nodes,
               total_leafs_nodes,
               non_empty_subtrees_cnt > 0 ? (float)tree_height / non_empty_subtrees_cnt : 0.0f,
               // total_tree_leafs_depths,
               // (long double)total_tree_leafs_depths / total_leafs_nodes,
               empty_subtrees_buffers,
               non_empty_subtrees_cnt);
    }

    // Function to create subtree batches for query answering (faithful C++ port from original C code)
    BatchList* create_subtree_batches(NodeList *nodelist, int number_of_batches_to_create, int pq_th)
    {
        // printf("[Node %d]: Node amount: %d\n", MY_RANK, nodelist->node_amount);
        if (nodelist->node_amount < number_of_batches_to_create)
        {
            number_of_batches_to_create = nodelist->node_amount;
        }

        BatchList *batchlist = static_cast<BatchList *>(std::malloc(sizeof(BatchList)));
        if (batchlist == nullptr)
        {
            fprintf(stderr, "Error: Memory allocation failed for BatchList\n");
            std::exit(EXIT_FAILURE);
        }

        batchlist->batch_amount = number_of_batches_to_create;
        // printf("[Node %d]: Batches to create: %d\n",MY_RANK, number_of_batches_to_create);
        batchlist->batches = static_cast<SubtreeBatch *>(std::malloc(sizeof(SubtreeBatch) * number_of_batches_to_create));
        if (batchlist->batches == nullptr)
        {
            fprintf(stderr, "Error: Memory allocation failed for SubtreeBatch array\n");
            std::free(batchlist);
            std::exit(EXIT_FAILURE);
        }

        int batch_size = nodelist->node_amount / batchlist->batch_amount;
        for (int i = 0; i < number_of_batches_to_create; i++)
        {
            std::memset(&batchlist->batches[i], 0, sizeof(SubtreeBatch));

            batchlist->batches[i].id = i;
            batchlist->batches[i].from = i * batch_size;
            batchlist->batches[i].nodelist = nodelist;
            batchlist->batches[i].pq_th = pq_th;

            // patches
            // batchlist->batches[i].pq_amount = 0;
            // batchlist->batches[i].current_subtree_to_process = 0;
            // batchlist->batches[i].current_pq_to_process = 0;
            // batchlist->batches[i].is_getting_help_phase1 = 0;
            // batchlist->batches[i].is_getting_help_phase2 = 0;
            // batchlist->batches[i].is_stolen = 0;

            if (i == number_of_batches_to_create - 1)
            {
                batchlist->batches[i].to = nodelist->node_amount;                    //! padding-like
                batchlist->batches[i].size = nodelist->node_amount - i * batch_size; //! einai swsto auto?
            }
            else
            {
                batchlist->batches[i].to = (i + 1) * batch_size;
                batchlist->batches[i].size = batch_size;
            }

            pthread_mutex_init(&batchlist->batches[i].pq_insert_lock, nullptr);

            batchlist->batches[i].max_pq_index = 0;
            batchlist->batches[i].min_pq_index = INT32_MAX;

            // NULLIFY all the priority queues
            // for (int j = 0; j < MAX_PQs_WORKSTEALING; j++)
            //{
            //    batchlist->batches[i].pq[j] = NULL;
            //}
            // batchlist->batches[i].pq[0] = NULL;
        }

        // printf("Created %d batches\n", number_of_batches_to_create);

        return batchlist;
    }

} // namespace diNoLib

