#include "iSAXIndex.hpp"
#include "SAX.hpp"

#include <float.h>
#include <string.h>
#include <stdlib.h>
#include <cstring>  // for memset
namespace daisy
{

    isax_index_settings *isax_index_settings_init(const char *root_directory, int timeseries_size,
                                                  int paa_segments, int sax_bit_cardinality,
                                                  int max_leaf_size, int min_leaf_size,
                                                  int initial_leaf_buffer_size,
                                                  int max_total_buffer_size, int initial_fbl_buffer_size,
                                                  int total_loaded_leaves, int tight_bound, int aggressive_check, int new_index, char inmemory_flag)
    {
        int i;
        isax_index_settings *settings = (isax_index_settings *)malloc(sizeof(isax_index_settings));
        if (settings == NULL)
        {
            fprintf(stderr, "error: could not allocate memory for index settings.\n");
            return NULL;
        }

        if (new_index)
        {
            if (chdir(root_directory) == 0)
            {
                fprintf(stderr, "WARNING! Target index directory already exists. Please delete or choose a new one.\n");
            }
            if (!inmemory_flag)
            {
            }

            settings->max_total_full_buffer_size = max_total_buffer_size;
            settings->initial_fbl_buffer_size = initial_fbl_buffer_size;
        }
        else
        {
            if (chdir(root_directory) != 0)
            {
                fprintf(stderr, "WARNING! Target index directory does not exist!\n");
            }
            else
            {
                if (chdir("../") != 0) {
                    fprintf(stderr, "Warning: Failed to change directory\n");
                }
            }
            settings->max_total_full_buffer_size = max_total_buffer_size;
            settings->initial_fbl_buffer_size = initial_fbl_buffer_size;
            // settings->max_total_full_buffer_size = 0;
            // settings->initial_fbl_buffer_size = 0;
        }

        if (paa_segments > (int)(8 * (int)sizeof(root_mask_type)))
        {
            fprintf(stderr, "error: Too many paa segments. The maximum value is %zu.\n",
                    8 * sizeof(root_mask_type));
            return NULL;
        }

        if (initial_leaf_buffer_size < max_leaf_size)
        {
            fprintf(stderr, "error: Leaf buffers should be at least as big as leafs.\n");
            return NULL;
        }
        settings->total_loaded_leaves = total_loaded_leaves;
        settings->root_directory = root_directory;
        settings->raw_filename = NULL;

        settings->timeseries_size = timeseries_size;
        settings->paa_segments = paa_segments;
        settings->ts_values_per_paa_segment = timeseries_size / paa_segments;
        settings->max_leaf_size = max_leaf_size;
        settings->min_leaf_size = min_leaf_size;
        settings->initial_leaf_buffer_size = initial_leaf_buffer_size;

        settings->tight_bound = tight_bound;
        settings->aggressive_check = aggressive_check;

        settings->sax_byte_size = (sizeof(sax_type) * paa_segments);
        settings->ts_byte_size = (sizeof(ts_type) * timeseries_size);
        settings->position_byte_size = sizeof(file_position_type);

        settings->full_record_size = settings->sax_byte_size + settings->position_byte_size + settings->ts_byte_size;
        settings->partial_record_size = settings->sax_byte_size + settings->position_byte_size;

        settings->sax_bit_cardinality = sax_bit_cardinality;
        settings->sax_alphabet_cardinality = pow(2, sax_bit_cardinality);

        settings->max_sax_cardinalities = (sax_type *)malloc(sizeof(sax_type) * settings->paa_segments);
        for (i = 0; i < settings->paa_segments; i++)
            settings->max_sax_cardinalities[i] = settings->sax_bit_cardinality;

        // settings->mindist_sqrt = sqrtf((float) settings->timeseries_size /
        //                                (float) settings->paa_segments);
        settings->mindist_sqrt = ((float)settings->timeseries_size /
                                  (float)settings->paa_segments);
        settings->root_nodes_size = pow(2, settings->paa_segments);

        // SEGMENTS * (CARDINALITY)
        float c_size = ceil(log10(settings->sax_alphabet_cardinality + 1));
        settings->max_filename_size = settings->paa_segments *
                                          ((c_size * 2) + 2) +
                                      5 + strlen(root_directory);

        if (paa_segments > sax_bit_cardinality)
        {
            settings->bit_masks = (root_mask_type *)malloc(sizeof(root_mask_type) * (paa_segments + 1));
            if (settings->bit_masks == NULL)
            {
                fprintf(stderr, "error: could not allocate memory for bit masks.\n");
                return NULL;
            }

            for (; paa_segments >= 0; paa_segments--)
            {
                settings->bit_masks[paa_segments] = pow(2, paa_segments);
            }
        }
        else
        {
            settings->bit_masks = (root_mask_type *)malloc(sizeof(root_mask_type) * (sax_bit_cardinality + 1));
            if (settings->bit_masks == NULL)
            {
                fprintf(stderr, "error: could not allocate memory for bit masks.\n");
                return NULL;
            }

            for (; sax_bit_cardinality >= 0; sax_bit_cardinality--)
            {
                settings->bit_masks[sax_bit_cardinality] = pow(2, sax_bit_cardinality);
            }
        }

        if (new_index)
        {
            settings->max_total_buffer_size = (int)((float)(settings->full_record_size /
                                                            (float)settings->partial_record_size) *
                                                    settings->max_total_full_buffer_size);
        }
        else
        {
            settings->max_total_buffer_size = settings->max_total_full_buffer_size;
        }

        return settings;
    }

    first_buffer_layer *initialize_fbl(int initial_buffer_size, int number_of_buffers,
                                       int max_total_buffers_size, isax_index *index)
    {
        first_buffer_layer *fbl = (first_buffer_layer *)malloc(sizeof(first_buffer_layer));

        fbl->max_total_size = max_total_buffers_size;
        fbl->initial_buffer_size = initial_buffer_size;
        fbl->number_of_buffers = number_of_buffers;

        // Allocate a big chunk of memory to store sax data and positions
        long long hard_buffer_size = (long long)(index->settings->sax_byte_size + index->settings->position_byte_size) * (long long)max_total_buffers_size;
        fbl->hard_buffer = (char *)malloc(hard_buffer_size);

        if (fbl->hard_buffer == NULL)
        {
            fprintf(stderr, "Could not initialize hard buffer of size: %lld\n", hard_buffer_size);
            exit(-1);
        }

        // Allocate a set of soft buffers to hold pointers to the hard buffer
        fbl->soft_buffers = (fbl_soft_buffer *)malloc(sizeof(fbl_soft_buffer) * number_of_buffers);
        fbl->current_record_index = 0;
        fbl->current_record = fbl->hard_buffer;
        int i;
        for (i = 0; i < number_of_buffers; i++)
        {
            fbl->soft_buffers[i].initialized = 0;
            fbl->soft_buffers[i].max_buffer_size = 0;
            fbl->soft_buffers[i].buffer_size = 0;
        }
        return fbl;
    }

    void destroy_fbl(first_buffer_layer *fbl)
    {
        free(fbl->hard_buffer);
        free(fbl->soft_buffers);
        free(fbl);
    }

    void destroy_fbl2(first_buffer_layer2 *fbl)
    {
        free(fbl->soft_buffers);
        free(fbl);
    }

    void destroy_parallel_fbl(parallel_first_buffer_layer *fbl)
    {
        if (fbl == NULL) return;
        
        int total_workers = fbl->total_worker_number;
        
        // Free internal allocations of each soft buffer
        if (fbl->soft_buffers != NULL) {
            for (int i = 0; i < fbl->number_of_buffers; i++) {
                parallel_fbl_soft_buffer *sb = &fbl->soft_buffers[i];
                if (sb->initialized) {
                    // Free per-worker arrays
                    if (sb->sax_records != NULL) {
                        for (int w = 0; w < total_workers; w++) {
                            if (sb->sax_records[w] != NULL) {
                                free(sb->sax_records[w]);
                            }
                        }
                        free(sb->sax_records);
                    }
                    if (sb->pos_records != NULL) {
                        for (int w = 0; w < total_workers; w++) {
                            if (sb->pos_records[w] != NULL) {
                                free(sb->pos_records[w]);
                            }
                        }
                        free(sb->pos_records);
                    }
                    if (sb->max_buffer_size != NULL) {
                        free(sb->max_buffer_size);
                    }
                    if (sb->buffer_size != NULL) {
                        free(sb->buffer_size);
                    }
                }
            }
            free(fbl->soft_buffers);
        }
        
        free(fbl->hard_buffer);
        free(fbl);
    }

    parallel_first_buffer_layer *initialize_pRecBuf(int initial_buffer_size, int number_of_buffers,
                                                    int max_total_buffers_size, isax_index *index, int total_workers)
    {
        parallel_first_buffer_layer *fbl = (parallel_first_buffer_layer *)malloc(sizeof(parallel_first_buffer_layer));

        // In the in-memory variant we do not allocate a contiguous hard buffer.
        // Initialize to NULL so destroy_fbl() can free safely.
        fbl->hard_buffer = NULL;

        fbl->max_total_size = max_total_buffers_size;
        fbl->initial_buffer_size = initial_buffer_size;
        fbl->number_of_buffers = number_of_buffers;
        fbl->total_worker_number = total_workers;  // Store for cleanup

        // Allocate a big chunk of memory to store sax data and positions
        // long long hard_buffer_size = (long long)(index->settings->sax_byte_size + index->settings->position_byte_size) * (long long)max_total_buffers_size;
        // fbl->hard_buffer = malloc(hard_buffer_size); // Maybe see here for errors???

        // Allocate a set of soft buffers to hold pointers to the hard buffer
        fbl->soft_buffers = (parallel_fbl_soft_buffer *)malloc(sizeof(parallel_fbl_soft_buffer) * number_of_buffers);
        fbl->current_record_index = 0;
        int i;
        for (i = 0; i < number_of_buffers; i++)
        {
            fbl->soft_buffers[i].initialized = 0;
            fbl->soft_buffers[i].finished = 0;
            fbl->soft_buffers[i].buffer_size = NULL;
        }
        return fbl;
    }

    // EKOSMAS-specific parallel buffer initialization for Odyssey
    // NOTE: total_workers is currently unused, but kept for API symmetry with initialize_pRecBuf.
    parallel_first_buffer_layer_ekosmas *initialize_pRecBuf_ekosmas(int initial_buffer_size,
                                                                     int number_of_buffers,
                                                                     int max_total_buffers_size,
                                                                     isax_index *index,
                                                                     int /*total_workers*/)
    {
        (void)index; // index is not needed for the basic EKOSMAS initialization

        parallel_first_buffer_layer_ekosmas *fbl =
            (parallel_first_buffer_layer_ekosmas *)malloc(sizeof(parallel_first_buffer_layer_ekosmas));
        if (fbl == NULL)
        {
            fprintf(stderr, "Error: could not allocate memory for parallel_first_buffer_layer_ekosmas.\n");
            return NULL;
        }

        fbl->max_total_size = max_total_buffers_size;
        fbl->initial_buffer_size = initial_buffer_size;
        fbl->number_of_buffers = number_of_buffers;
        fbl->current_record_index = 0;
        fbl->current_record = NULL;  // layout compatibility with parallel_first_buffer_layer
        fbl->hard_buffer = NULL;      // layout compatibility with parallel_first_buffer_layer

        // Allocate a set of soft buffers
        fbl->soft_buffers = (parallel_fbl_soft_buffer_ekosmas *)malloc(
            sizeof(parallel_fbl_soft_buffer_ekosmas) * number_of_buffers);
        if (fbl->soft_buffers == NULL)
        {
            fprintf(stderr, "Error: could not allocate memory for parallel_fbl_soft_buffer_ekosmas array.\n");
            free(fbl);
            return NULL;
        }

        for (int i = 0; i < number_of_buffers; i++)
        {
            fbl->soft_buffers[i].initialized = 0;
            fbl->soft_buffers[i].finished = 0;
            fbl->soft_buffers[i].node = NULL;         // EKOSMAS: initially no node is attached
            fbl->soft_buffers[i].sax_records = NULL;  // will be initialized by workers
            fbl->soft_buffers[i].pos_records = NULL;  // will be initialized by workers
            fbl->soft_buffers[i].max_buffer_size = NULL;
            fbl->soft_buffers[i].buffer_size = NULL;
        }

        return fbl;
    }

    isax_index *isax_index_init_inmemory(isax_index_settings *settings)
    {
        isax_index *index = (isax_index *)malloc(sizeof(isax_index));
        if (index == NULL)
        {
            fprintf(stderr, "error: could not allocate memory for index structure.\n");
            return NULL;
        }
        index->memory_info.mem_tree_structure = 0;
        index->memory_info.mem_data = 0;
        index->memory_info.mem_summaries = 0;
        index->memory_info.disk_data_full = 0;
        index->memory_info.disk_data_partial = 0;

        index->settings = settings;
        index->first_node = NULL;
        index->fbl = (first_buffer_layer *)initialize_fbl(settings->initial_fbl_buffer_size,
                                                          pow(2, settings->paa_segments),
                                                          settings->max_total_buffer_size + DISK_BUFFER_SIZE * (PROGRESS_CALCULATE_THREAD_NUMBER - 1), index);

        index->sax_cache = NULL;

        index->total_records = 0;
        index->loaded_records = 0;

        index->root_nodes = 0;
        index->allocated_memory = 0;
        index->has_wedges = 0;
        // index->locations = malloc(sizeof(int) * settings->timeseries_size);

        index->answer = (ts_type *)malloc(sizeof(ts_type) * settings->timeseries_size);
        return index;
    }

    isax_index *isax_index_init_inmemory_ekosmas(isax_index_settings *settings)
    {
        isax_index *index = (isax_index *)malloc(sizeof(isax_index));
        if (index == NULL)
        {
            fprintf(stderr, "error: could not allocate memory for index structure.\n");
            return NULL;
        }

        // Initialize all fields to zero (memset equivalent)
        memset(index, 0, sizeof(isax_index));

        index->settings = settings;
        index->first_node = NULL;

        // NOTE: fbl will be initialized later (e.g., in buildIndexSequence) with parallel_first_buffer_layer_ekosmas
        // This matches the original C implementation where fbl is set up separately

        return index;
    }

    isax_index *isax_index_init(isax_index_settings *settings)
    {
        isax_index *index = (isax_index *)malloc(sizeof(isax_index));
        if (index == NULL)
        {
            fprintf(stderr, "error: could not allocate memory for index structure.\n");
            return NULL;
        }
        index->memory_info.mem_tree_structure = 0;
        index->memory_info.mem_data = 0;
        index->memory_info.mem_summaries = 0;
        index->memory_info.disk_data_full = 0;
        index->memory_info.disk_data_partial = 0;

        index->settings = settings;
        index->first_node = NULL;
        index->fbl = initialize_fbl(settings->initial_fbl_buffer_size,
                                    pow(2, settings->paa_segments),
                                    settings->max_total_buffer_size + DISK_BUFFER_SIZE * (PROGRESS_CALCULATE_THREAD_NUMBER - 1), index);
        char *sax_filename = (char *)malloc((strlen(settings->root_directory) + 15) * sizeof(char));
        sax_filename = strcpy(sax_filename, settings->root_directory);
        sax_filename = strcat(sax_filename, "isax_file.sax");

        if (access(sax_filename, F_OK) != -1)
        {
            index->sax_file = fopen(sax_filename, "rb");
        }
        else
        {
            index->sax_file = fopen(sax_filename, "w+b");
        }

        free(sax_filename);
        index->sax_cache = NULL;

        index->total_records = 0;
        index->loaded_records = 0;

        index->root_nodes = 0;
        index->allocated_memory = 0;
        index->has_wedges = 0;
        // index->locations = malloc(sizeof(int) * settings->timeseries_size);

        index->answer = (float *)malloc(sizeof(ts_type) * settings->timeseries_size);
        return index;
    }

    isax_node_buffer *init_node_buffer(int initial_buffer_size)
    {
        isax_node_buffer *node_buffer = (isax_node_buffer *)malloc(sizeof(isax_node_buffer));
        node_buffer->initial_buffer_size = initial_buffer_size;

        node_buffer->max_full_buffer_size = 0;
        node_buffer->max_partial_buffer_size = 0;
        node_buffer->max_tmp_full_buffer_size = 0;
        node_buffer->max_tmp_partial_buffer_size = 0;
        node_buffer->full_buffer_size = 0;
        node_buffer->partial_buffer_size = 0;
        node_buffer->tmp_full_buffer_size = 0;
        node_buffer->tmp_partial_buffer_size = 0;

        (node_buffer->full_position_buffer) = NULL;
        (node_buffer->full_sax_buffer) = NULL;
        (node_buffer->full_ts_buffer) = NULL;
        (node_buffer->partial_position_buffer) = NULL;
        (node_buffer->partial_sax_buffer) = NULL;
        (node_buffer->tmp_full_position_buffer) = NULL;
        (node_buffer->tmp_full_sax_buffer) = NULL;
        (node_buffer->tmp_full_ts_buffer) = NULL;
        (node_buffer->tmp_partial_position_buffer) = NULL;
        (node_buffer->tmp_partial_sax_buffer = NULL);

        return node_buffer;
    }

    isax_node *isax_leaf_node_init(int initial_buffer_size)
    {
        isax_node *node = (isax_node *)malloc(sizeof(isax_node));
        if (node == NULL)
        {
            fprintf(stderr, "error: could not allocate memory for new node.\n");
            return NULL;
        }
        node->has_partial_data_file = 0;
        node->has_full_data_file = 0;
        node->right_child = NULL;
        node->left_child = NULL;
        node->parent = NULL;
        node->next = NULL;
        node->leaf_size = 0;
        node->filename = NULL;
        node->isax_values = NULL;
        node->isax_cardinalities = NULL;
        node->previous = NULL;
        node->split_data = NULL;
        node->buffer = init_node_buffer(initial_buffer_size);
        node->mask = 0;
        node->wedges = NULL;
        return node;
    }

    isax_node *isax_root_node_init(root_mask_type mask, int initial_buffer_size)
    {
        isax_node *node = isax_leaf_node_init(initial_buffer_size);
        node->mask = mask;
        return node;
    }

    isax_node *insert_to_pRecBuf(parallel_first_buffer_layer *fbl, sax_type *sax,
                                 file_position_type *pos, root_mask_type mask,
                                 isax_index *index, pthread_mutex_t *lock_firstnode, int workernumber, int total_workernumber)
    {
        parallel_fbl_soft_buffer *current_buffer = &fbl->soft_buffers[(int)mask];

        file_position_type *filepointer;
        sax_type *saxpointer;

        int current_buffer_number;
        // char *cd_s, *cd_p;
        //  Check if this buffer is initialized

        if (!current_buffer->initialized)
        {
            pthread_mutex_lock(lock_firstnode);
            if (!current_buffer->initialized)
            {

                current_buffer->max_buffer_size = (int *)malloc(sizeof(int) * total_workernumber);
                current_buffer->buffer_size = (int *)malloc(sizeof(int) * total_workernumber);
                current_buffer->sax_records = (sax_type **)malloc(sizeof(sax_type *) * total_workernumber);
                current_buffer->pos_records = (file_position_type **)malloc(sizeof(file_position_type *) * total_workernumber);
                for (int i = 0; i < total_workernumber; i++)
                {
                    current_buffer->max_buffer_size[i] = 0;
                    current_buffer->buffer_size[i] = 0;
                    current_buffer->pos_records[i] = NULL;
                    current_buffer->sax_records[i] = NULL;
                }
                current_buffer->node = isax_root_node_init(mask, index->settings->initial_leaf_buffer_size);
                current_buffer->node->is_leaf = 1;
                current_buffer->initialized = 1;
                if (index->first_node == NULL)
                {
                    index->first_node = current_buffer->node;
                    pthread_mutex_unlock(lock_firstnode);
                    current_buffer->node->next = NULL;
                    current_buffer->node->previous = NULL;
                }
                else
                {
                    isax_node *prev_first = index->first_node;
                    index->first_node = current_buffer->node;
                    index->first_node->next = prev_first;
                    prev_first->previous = current_buffer->node;
                    pthread_mutex_unlock(lock_firstnode);
                }
                __sync_fetch_and_add(&(index->root_nodes), 1);
            }
            else
            {
                pthread_mutex_unlock(lock_firstnode);
            }
        }

        // Check if this buffer is not full!
        if (current_buffer->buffer_size[workernumber] >= current_buffer->max_buffer_size[workernumber])
        {
            if (current_buffer->max_buffer_size[workernumber] == 0)
            {
                current_buffer->max_buffer_size[workernumber] = fbl->initial_buffer_size;
                current_buffer->sax_records[workernumber] = (sax_type *)malloc(index->settings->sax_byte_size *
                                                                               current_buffer->max_buffer_size[workernumber]);
                current_buffer->pos_records[workernumber] = (file_position_type *)malloc(index->settings->position_byte_size *
                                                                                         current_buffer->max_buffer_size[workernumber]);
            }
            else
            {
                current_buffer->max_buffer_size[workernumber] *= BUFFER_REALLOCATION_RATE;

                current_buffer->sax_records[workernumber] = (sax_type *)realloc(current_buffer->sax_records[workernumber],
                                                                                index->settings->sax_byte_size *
                                                                                    current_buffer->max_buffer_size[workernumber]);
                current_buffer->pos_records[workernumber] = (file_position_type *)realloc(current_buffer->pos_records[workernumber],
                                                                                          index->settings->position_byte_size *
                                                                                              current_buffer->max_buffer_size[workernumber]);
            }
        }

        if (current_buffer->sax_records[workernumber] == NULL || current_buffer->pos_records[workernumber] == NULL)
        {
            fprintf(stderr, "error: Could not allocate memory in FBL.");
            return NULL; // OUT_OF_MEMORY_FAILURE;
        }

        current_buffer_number = current_buffer->buffer_size[workernumber];
        filepointer = (file_position_type *)current_buffer->pos_records[workernumber];
        saxpointer = (sax_type *)current_buffer->sax_records[workernumber];
        // printf("the work number is %d sax is  %d \n",workernumber,saxpointer[current_buffer_number*index->settings->paa_segments]);
        memcpy((void *)(&saxpointer[current_buffer_number * index->settings->paa_segments]), (void *)sax, index->settings->sax_byte_size);
        memcpy((void *)(&filepointer[current_buffer_number]), (void *)pos, index->settings->position_byte_size);

        (current_buffer->buffer_size[workernumber])++;

        return current_buffer->node;
    }

    root_mask_type isax_pRecBuf_index_insert_inmemory(isax_index *index,
                                                      sax_type *sax,
                                                      file_position_type *pos, pthread_mutex_t *lock_firstnode, int workernumber, int total_workernumber)
    {
        // int i, t;
        int totalsize = index->settings->max_total_buffer_size;

        // Create mask for the first bit of the sax representation

        // Step 1: Check if there is a root node that represents the
        //         current node's sax representation

        // TODO: Create INSERTION SHORT AND BINARY SEARCH METHODS.

        root_mask_type first_bit_mask = 0x00;

        CREATE_MASK(first_bit_mask, index, sax);

        insert_to_pRecBuf((parallel_first_buffer_layer *)(index->fbl), sax, pos, first_bit_mask, index, lock_firstnode, workernumber, total_workernumber);
        return first_bit_mask;
    }

    // ========================================================================
    // EKOSMAS-specific versions for Odyssey
    // ========================================================================

    isax_node *insert_to_pRecBuf_ekosmas(parallel_first_buffer_layer_ekosmas *fbl, sax_type *sax,
                                          file_position_type *pos, root_mask_type mask,
                                          isax_index *index, pthread_mutex_t *lock_firstnode, int workernumber, int total_workernumber)
    {
        parallel_fbl_soft_buffer_ekosmas *current_buffer = &fbl->soft_buffers[(int)mask];

        file_position_type *filepointer;
        sax_type *saxpointer;

        int current_buffer_number;

        // Check if this buffer is initialized
        if (!current_buffer->initialized)
        {
            pthread_mutex_lock(lock_firstnode); // Double-check locking pattern
            if (!current_buffer->initialized)
            {
                current_buffer->max_buffer_size = (int *)malloc(sizeof(int) * total_workernumber);
                current_buffer->buffer_size = (int *)malloc(sizeof(int) * total_workernumber);
                current_buffer->sax_records = (sax_type **)malloc(sizeof(sax_type *) * total_workernumber);
                current_buffer->pos_records = (file_position_type **)malloc(sizeof(file_position_type *) * total_workernumber);
                
                if (current_buffer->max_buffer_size == nullptr ||
                    current_buffer->buffer_size == nullptr ||
                    current_buffer->sax_records == nullptr ||
                    current_buffer->pos_records == nullptr)
                {
                    fprintf(stderr, "[Node %d] Error: Memory allocation failed in insert_to_pRecBuf_ekosmas\n", workernumber);
                    pthread_mutex_unlock(lock_firstnode);
                    exit(EXIT_FAILURE);
                }

                for (int i = 0; i < total_workernumber; i++)
                {
                    current_buffer->max_buffer_size[i] = 0;
                    current_buffer->buffer_size[i] = 0;
                    current_buffer->pos_records[i] = NULL;
                    current_buffer->sax_records[i] = NULL;
                }
                
                // NOTE: isax_root_node_init takes only 2 parameters (mask, initial_buffer_size)
                // The third parameter (NULL) in the original C code was likely unused/optional
                current_buffer->node = isax_root_node_init(mask, index->settings->initial_leaf_buffer_size);
                current_buffer->node->is_leaf = 1;
                current_buffer->initialized = 1;
                
                // EKOSMAS: Set index->first_node and increment root_nodes (same as insert_to_pRecBuf)
                if (index->first_node == NULL)
                {
                    index->first_node = current_buffer->node;
                    current_buffer->node->next = NULL;
                    current_buffer->node->previous = NULL;
                }
                else
                {
                    isax_node *prev_first = index->first_node;
                    index->first_node = current_buffer->node;
                    index->first_node->next = prev_first;
                    prev_first->previous = current_buffer->node;
                }
                __sync_fetch_and_add(&(index->root_nodes), 1);
            }
            pthread_mutex_unlock(lock_firstnode);
        }

        // Check if this buffer is not full!
        if (current_buffer->buffer_size[workernumber] >= current_buffer->max_buffer_size[workernumber])
        {
            if (current_buffer->max_buffer_size[workernumber] == 0)
            {
                current_buffer->max_buffer_size[workernumber] = fbl->initial_buffer_size;
                current_buffer->sax_records[workernumber] = (sax_type *)malloc(index->settings->sax_byte_size *
                                                                               current_buffer->max_buffer_size[workernumber]);
                current_buffer->pos_records[workernumber] = (file_position_type *)malloc(index->settings->position_byte_size *
                                                                                          current_buffer->max_buffer_size[workernumber]);
            }
            else
            {
                current_buffer->max_buffer_size[workernumber] *= BUFFER_REALLOCATION_RATE;

                current_buffer->sax_records[workernumber] = (sax_type *)realloc(current_buffer->sax_records[workernumber],
                                                                                index->settings->sax_byte_size *
                                                                                    current_buffer->max_buffer_size[workernumber]);
                current_buffer->pos_records[workernumber] = (file_position_type *)realloc(current_buffer->pos_records[workernumber],
                                                                                          index->settings->position_byte_size *
                                                                                              current_buffer->max_buffer_size[workernumber]);
            }
        }

        if (current_buffer->sax_records[workernumber] == NULL || current_buffer->pos_records[workernumber] == NULL)
        {
            fprintf(stderr, "Node error: Could not allocate memory in FBL. Query Worker ID = %d, Total Query Workers = %d\n", workernumber, total_workernumber);
            exit(EXIT_FAILURE);
        }

        current_buffer_number = current_buffer->buffer_size[workernumber];
        filepointer = (file_position_type *)current_buffer->pos_records[workernumber];
        saxpointer = (sax_type *)current_buffer->sax_records[workernumber];
        
        // EKOSMAS: memcpy per copiare SAX e posizione nel buffer
        memcpy((void *)(&saxpointer[current_buffer_number * index->settings->paa_segments]), (void *)sax, index->settings->sax_byte_size);
        memcpy((void *)(&filepointer[current_buffer_number]), (void *)pos, index->settings->position_byte_size);

        (current_buffer->buffer_size[workernumber])++;

        return (isax_node *)current_buffer->node;
    }

    root_mask_type isax_pRecBuf_index_insert_inmemory_ekosmas(isax_index *index,
                                                               sax_type *sax,
                                                               file_position_type *pos,
                                                               pthread_mutex_t *lock_firstnode,
                                                               int workernumber,
                                                               int total_workernumber)
    {
        root_mask_type first_bit_mask = 0x00;

        CREATE_MASK(first_bit_mask, index, sax);

        // Insert into EKOSMAS parallel buffer layer
        insert_to_pRecBuf_ekosmas(
            (parallel_first_buffer_layer_ekosmas *)(index->fbl),
            sax,
            pos,
            first_bit_mask,
            index,
            lock_firstnode,
            workernumber,
            total_workernumber);

        return first_bit_mask;
    }

    void destroy_node_buffer(isax_node_buffer *node_buffer)
    {
        if (node_buffer->full_position_buffer != NULL)
        {
            free(node_buffer->full_position_buffer);
            node_buffer->full_position_buffer = NULL;
        }
        if (node_buffer->full_sax_buffer != NULL)
        {
            free(node_buffer->full_sax_buffer);
            node_buffer->full_sax_buffer = NULL;
        }
        if (node_buffer->full_ts_buffer != NULL)
        {
            free(node_buffer->full_ts_buffer);
            node_buffer->full_ts_buffer = NULL;
        }
        if (node_buffer->partial_position_buffer != NULL)
        {
            // !!! DON'T FREE THAT IT REMOVES THE DATA!!!!
            free(node_buffer->partial_position_buffer);
            node_buffer->partial_position_buffer = NULL;
        }
        if (node_buffer->partial_sax_buffer != NULL)
        {
            free(node_buffer->partial_sax_buffer);
            node_buffer->partial_sax_buffer = NULL;
        }
        if (node_buffer->tmp_full_position_buffer != NULL)
        {
            free(node_buffer->tmp_full_position_buffer);
            node_buffer->tmp_full_position_buffer = NULL;
        }
        if (node_buffer->tmp_full_sax_buffer != NULL)
        {
            free(node_buffer->tmp_full_sax_buffer);
            node_buffer->tmp_full_sax_buffer = NULL;
        }
        if (node_buffer->tmp_full_ts_buffer != NULL)
        {
            free(node_buffer->tmp_full_ts_buffer);
            node_buffer->tmp_full_ts_buffer = NULL;
        }
        if (node_buffer->tmp_partial_position_buffer != NULL)
        {
            free(node_buffer->tmp_partial_position_buffer);
            node_buffer->tmp_partial_position_buffer = NULL;
        }
        if (node_buffer->tmp_partial_sax_buffer != NULL)
        {
            free(node_buffer->tmp_partial_sax_buffer);
            node_buffer->tmp_partial_sax_buffer = NULL;
        }
        free(node_buffer);
    }

    float sax2paa_word(sax_type sax_word, int cardinality)
    {
        // sax_word is unsigned char, explicitly converted as int
        int sax_value = (int)sax_word;
        int alphabeta_size = pow(2, cardinality);
        int offset = ((alphabeta_size - 1) * (alphabeta_size - 2)) / 2;

        if (sax_value == 0 || sax_value == alphabeta_size - 1)
        {
            return sax_breakpoints[offset + sax_value] + (sax_breakpoints[offset + sax_value] - sax_breakpoints[offset + alphabeta_size / 2 - 1]) / 2;
        }

        return (sax_breakpoints[offset + sax_value - 1] + sax_breakpoints[offset + sax_value]) / 2;
    }

    int informed_split_decision(isax_node_split_data *split_data,
                                isax_index_settings *settings,
                                isax_node_record *records_buffer,
                                int records_buffer_size)
    {
        double *segment_mean = (double *)malloc(sizeof(double) * settings->paa_segments);
        double *segment_stdev = (double *)malloc(sizeof(double) * settings->paa_segments);
        float *sax2paa = (float *)malloc(sizeof(float) * records_buffer_size * settings->paa_segments);

        int i, j;
        for (i = 0; i < settings->paa_segments; i++)
        {
            segment_mean[i] = 0;
            segment_stdev[i] = 0;
        }

        for (i = 0; i < records_buffer_size; i++)
        {
            for (j = 0; j < settings->paa_segments; j++)
            {
                sax2paa[i * settings->paa_segments + j] = sax2paa_word(records_buffer[i].sax[j], settings->sax_bit_cardinality);
                segment_mean[j] += sax2paa[i * settings->paa_segments + j];
            }
        }

        for (i = 0; i < settings->paa_segments; i++)
        {
            segment_mean[i] /= (records_buffer_size);
            // printf("mean: %lf\n", segment_mean[i]);
        }

        for (i = 0; i < records_buffer_size; i++)
        {
            for (j = 0; j < settings->paa_segments; j++)
            {
                segment_stdev[j] += pow(segment_mean[j] - sax2paa[i * settings->paa_segments + j], 2);
            }
        }

        for (i = 0; i < settings->paa_segments; i++)
        {
            segment_stdev[i] = sqrt(segment_stdev[i] / (records_buffer_size));
            // printf("stdev: %lf\n", segment_stdev[i]);
        }

        // Decide split point based on the above calculations
        int segment_to_split = -1;
        float segment_to_split_b = -1;
        for (i = 0; i < settings->paa_segments; i++)
        {
            if (split_data->split_mask[i] + 1 > settings->sax_bit_cardinality - 1)
            {
                continue;
            }
            else
            {
                // TODO: Optimize this.
                // Calculate break point for new cardinality, a bit complex.
                int new_bit_cardinality = split_data->split_mask[i] + 1;
                int break_point_id = records_buffer[0].sax[i];
                break_point_id = (break_point_id >> ((settings->sax_bit_cardinality) - (new_bit_cardinality))) << 1;
                int new_cardinality = pow(2, new_bit_cardinality + 1);
                int offset = (new_cardinality - 1) * (new_cardinality - 2) / 2;
                float b = sax_breakpoints[offset + break_point_id];

                if (segment_to_split == -1)
                {
                    segment_to_split = i;
                    segment_to_split_b = b;
                    continue;
                }

                float left_range = segment_mean[i] - (3 * segment_stdev[i]);
                float right_range = segment_mean[i] + (3 * segment_stdev[i]);
                // printf("%d, %lf -- %lf \n", i, left_range, right_range);

                if (left_range <= b && b <= right_range)
                {
                    if (abs(segment_mean[i] - b) <= abs(segment_mean[segment_to_split] - segment_to_split_b))
                    {
                        segment_to_split = i;
                        segment_to_split_b = b;
                    }
                }
            }
        }

        free(segment_mean);
        free(segment_stdev);
        free(sax2paa);
        return segment_to_split;
    }

    enum response add_to_node_buffer(isax_node_buffer *node_buffer,
                                     isax_node_record *record,
                                     int sax_segments, int ts_segments);

    isax_node *add_record_to_node(isax_index *index,
                                  isax_node *tree_node,
                                  isax_node_record *record,
                                  const char leaf_size_check)
    {
        isax_node *node = tree_node;

        // Traverse tree
        while (!node->is_leaf)
        {
            int location = index->settings->sax_bit_cardinality - 1 -
                           node->split_data->split_mask[node->split_data->splitpoint];

            root_mask_type mask = index->settings->bit_masks[location];
            if (record->sax[node->split_data->splitpoint] & mask)
            {
                node = node->right_child;
            }
            else
            {
                node = node->left_child;
            }
        }
        // Check if split needed
        if ((node->leaf_size) >= index->settings->max_leaf_size && leaf_size_check)
        {
            split_node(index, node);
            add_record_to_node(index, node, record, leaf_size_check);
        }
        else
        {
            if (node->filename == NULL)
            {
                create_node_filename(index, node, record);
            }
            add_to_node_buffer(node->buffer, record, index->settings->paa_segments,
                               index->settings->timeseries_size);
            node->leaf_size++;
        }
        return node;
    }

    // ========================================================================
    // EKOSMAS-specific in-memory versions (no disk file handling)
    // ========================================================================

    enum response initialize_isax_values_and_cardinalities(isax_index *index,
                                                            isax_node *node,
                                                            sax_type *sax)
    {
        int i;
        node->isax_values = static_cast<sax_type *>(malloc(sizeof(sax_type) * index->settings->paa_segments));
        if (node->isax_values == nullptr)
        {
            fprintf(stderr, "error: could not allocate memory for isax_values.\n");
            return FAILURE;
        }

        node->isax_cardinalities = static_cast<sax_type *>(malloc(sizeof(sax_type) * index->settings->paa_segments));
        if (node->isax_cardinalities == nullptr)
        {
            fprintf(stderr, "error: could not allocate memory for isax_cardinalities.\n");
            free(node->isax_values);
            node->isax_values = nullptr;
            return FAILURE;
        }

        // If this has a parent then it is not a root node and as such it does have some
        // split data on its parent about the cardinalities.
        if (node->parent)
        {
            for (i = 0; i < index->settings->paa_segments; i++)
            {
                root_mask_type mask = 0x00;
                int k;
                for (k = 0; k <= node->parent->split_data->split_mask[i]; k++)
                {
                    mask |= (index->settings->bit_masks[index->settings->sax_bit_cardinality - 1 - k] & sax[i]);
                }
                mask = mask >> (index->settings->sax_bit_cardinality - node->parent->split_data->split_mask[i] - 1);

                node->isax_values[i] = static_cast<sax_type>(mask);
                node->isax_cardinalities[i] = node->parent->split_data->split_mask[i] + 1;
            }
        }
        // If it has no parent it is root node and as such it's cardinality is 1.
        else
        {
            root_mask_type mask = 0x00;

            for (i = 0; i < index->settings->paa_segments; i++)
            {
                mask = (index->settings->bit_masks[index->settings->sax_bit_cardinality - 1] & sax[i]);
                mask = mask >> (index->settings->sax_bit_cardinality - 1);

                node->isax_values[i] = static_cast<sax_type>(mask);
                node->isax_cardinalities[i] = 1;
            }
        }

        return SUCCESS;
    }

    isax_node *add_record_to_node_inmemory(isax_index *index,
                                            isax_node *tree_node,
                                            isax_node_record *record,
                                            const char leaf_size_check)
    {
        isax_node *node = tree_node;

        // Traverse tree
        while (!node->is_leaf)
        {
            int location = index->settings->sax_bit_cardinality - 1 -
                           node->split_data->split_mask[node->split_data->splitpoint];

            root_mask_type mask = index->settings->bit_masks[location];
            if (record->sax[node->split_data->splitpoint] & mask)
            {
                node = node->right_child;
            }
            else
            {
                node = node->left_child;
            }
        }

        // Check if split needed
        if ((node->leaf_size) >= index->settings->max_leaf_size && leaf_size_check)
        {
            // NOTE: split_node should work for in-memory nodes (filename == NULL)
            // If it doesn't, we may need to implement split_node_inmemory
            split_node(index, node);
            // EKOSMAS: Recursive call to add_record_to_node_inmemory (not add_record_to_node)
            add_record_to_node_inmemory(index, node, record, leaf_size_check);
        }
        else
        {
            if (node->isax_values == NULL)
            {
                initialize_isax_values_and_cardinalities(index, node, record->sax);
            }
            add_to_node_buffer(node->buffer, record, index->settings->paa_segments,
                               index->settings->timeseries_size);
            node->leaf_size++;
        }
        return node;
    }

    void split_node(isax_index *index, isax_node *node)
    {
        // Bail early if buffer is NULL (node was already split or not initialized)
        if (node == NULL || node->buffer == NULL)
        {
            return;
        }
        
        // Pre-check: if parent exists and all split_mask entries are at max, can't split
        if (node->parent != NULL && node->parent->split_data != NULL)
        {
            int can_split = 0;
            for (int i = 0; i < index->settings->paa_segments; i++)
            {
                if (node->parent->split_data->split_mask[i] < index->settings->sax_bit_cardinality - 1)
                {
                    can_split = 1;
                    break;
                }
            }
            if (!can_split)
            {
                // Cannot split further - all dimensions at max depth
                return;
            }
        }
        
        // Pre-check: if no records to split, don't proceed
        int total_records = node->buffer->full_buffer_size + 
                           node->buffer->partial_buffer_size +
                           node->buffer->tmp_full_buffer_size + 
                           node->buffer->tmp_partial_buffer_size;
        if (total_records == 0)
        {
            return;
        }
        
        // *******************************************************
        // CREATE TWO NEW NODES AND SET OLD ONE AS AN INTERMEDIATE
        // *******************************************************
        int i, sktting;

        node->is_leaf = 0;
        node->leaf_size = 0;

        // Create split_data for this node.
        isax_node_split_data *split_data = (isax_node_split_data *)malloc(sizeof(isax_node_split_data));
        if (split_data == NULL)
        {
            fprintf(stderr, "error: could not allocate memory for node split data.\n");
        }
        split_data->split_mask = (sax_type *)malloc(sizeof(sax_type) * index->settings->paa_segments);
        if (split_data->split_mask == NULL)
        {
            fprintf(stderr, "error: could not allocate memory for node split mask.\n");
        }

        if (node->parent == NULL)
        {
            for (i = 0; i < index->settings->paa_segments; i++)
            {
                split_data->split_mask[i] = 0;
            }
            split_data->splitpoint = 0;
        }
        else
        {
            for (i = 0; i < index->settings->paa_segments; i++)
            {
                split_data->split_mask[i] = node->parent->split_data->split_mask[i];
            }
        }

        __sync_fetch_and_add(&(index->memory_info.mem_tree_structure), 2);

        isax_node *left_child = isax_leaf_node_init(index->settings->initial_leaf_buffer_size);
        isax_node *right_child = isax_leaf_node_init(index->settings->initial_leaf_buffer_size);
        left_child->is_leaf = 1;
        right_child->is_leaf = 1;
        left_child->parent = node;
        right_child->parent = node;
        node->split_data = split_data;
        node->left_child = left_child;
        node->right_child = right_child;

        // ############ S P L I T   D A T A #############
        // Count in-memory records
        int inmem_count = node->buffer->full_buffer_size + 
                         node->buffer->partial_buffer_size +
                         node->buffer->tmp_full_buffer_size + 
                         node->buffer->tmp_partial_buffer_size;
        
        // Count records in files by checking file sizes
        int file_record_count = 0;
        if (node->filename != NULL) {
            // Count .full file records
            char *full_fname = (char *)malloc(sizeof(char) * (strlen(node->filename) + 6));
            strcpy(full_fname, node->filename);
            strcat(full_fname, ".full");
            FILE *f = fopen(full_fname, "r");
            if (f != NULL) {
                fseek(f, 0, SEEK_END);
                long fsize = ftell(f);
                // Each full record = position + sax + ts
                int full_record_size = index->settings->position_byte_size + 
                                      index->settings->sax_byte_size + 
                                      index->settings->ts_byte_size;
                file_record_count += (fsize / full_record_size) + 1;
                fclose(f);
            }
            free(full_fname);
            
            // Count .part file records
            char *part_fname = (char *)malloc(sizeof(char) * (strlen(node->filename) + 6));
            strcpy(part_fname, node->filename);
            strcat(part_fname, ".part");
            f = fopen(part_fname, "r");
            if (f != NULL) {
                fseek(f, 0, SEEK_END);
                long fsize = ftell(f);
                // Each partial record = position + sax (no ts)
                int part_record_size = index->settings->position_byte_size + 
                                      index->settings->sax_byte_size;
                file_record_count += (fsize / part_record_size) + 1;
                fclose(f);
            }
            free(part_fname);
        }
        
        // Allocate buffer for all records with safety margin
        int split_buffer_size = inmem_count + file_record_count + 100;
        isax_node_record *split_buffer = (isax_node_record *)malloc(sizeof(isax_node_record) * split_buffer_size);
        if (split_buffer == NULL) {
            fprintf(stderr, "ERROR: Failed to allocate split_buffer of size %d\n", split_buffer_size);
            return;
        }

        int split_buffer_index = 0;

        // ********************************************************
        // SPLIT SAX BUFFERS CONTAINED IN *RAM* AND PUT IN CHILDREN
        // ********************************************************
        // Split both sax and ts data and move to the new leafs

        if (node->buffer->full_buffer_size > 0)
            for (i = node->buffer->full_buffer_size - 1; i >= 0; i--)
            {
                split_buffer[split_buffer_index].sax = node->buffer->full_sax_buffer[i];
                split_buffer[split_buffer_index].ts = node->buffer->full_ts_buffer[i];
                split_buffer[split_buffer_index].position = node->buffer->full_position_buffer[i];
                split_buffer[split_buffer_index].insertion_mode = (insertion_mode)(NO_TMP | FULL);
                split_buffer_index++;
            }
        node->buffer->full_buffer_size = 0;

        if (node->buffer->partial_buffer_size > 0)
            for (i = node->buffer->partial_buffer_size - 1; i >= 0; i--)
            {
                split_buffer[split_buffer_index].sax = node->buffer->partial_sax_buffer[i];
                split_buffer[split_buffer_index].ts = NULL;
                split_buffer[split_buffer_index].position = node->buffer->partial_position_buffer[i];
                split_buffer[split_buffer_index].insertion_mode = (insertion_mode)(NO_TMP | PARTIAL);
                split_buffer_index++;
            }
        node->buffer->partial_buffer_size = 0;

        if (node->buffer->tmp_full_buffer_size > 0)
            for (i = node->buffer->tmp_full_buffer_size - 1; i >= 0; i--)
            {
                split_buffer[split_buffer_index].sax = node->buffer->tmp_full_sax_buffer[i];
                split_buffer[split_buffer_index].ts = node->buffer->tmp_full_ts_buffer[i];
                split_buffer[split_buffer_index].position = node->buffer->tmp_full_position_buffer[i];
                split_buffer[split_buffer_index].insertion_mode = (insertion_mode)(TMP | FULL);
                split_buffer_index++;
            }
        node->buffer->tmp_full_buffer_size = 0;

        if (node->buffer->tmp_partial_buffer_size > 0)
            for (i = node->buffer->tmp_partial_buffer_size - 1; i >= 0; i--)
            {
                split_buffer[split_buffer_index].sax = node->buffer->tmp_partial_sax_buffer[i];
                split_buffer[split_buffer_index].ts = NULL;
                split_buffer[split_buffer_index].position = node->buffer->tmp_partial_position_buffer[i];
                split_buffer[split_buffer_index].insertion_mode = (insertion_mode)(TMP | PARTIAL);
                split_buffer_index++;
            }
        node->buffer->tmp_partial_buffer_size = 0;

        destroy_node_buffer(node->buffer);
        node->buffer = NULL;

        // *****************************************************
        // SPLIT BUFFERS CONTAINED ON *DISK* AND PUT IN CHILDREN
        // *****************************************************

        // File is split in two files but it is not
        // removed from disk. It is going to be used in the end.
        if (node->filename != NULL)
        {
            char *full_fname = (char *)malloc(sizeof(char) * (strlen(node->filename) + 6));
            strcpy(full_fname, node->filename);
            strcat(full_fname, ".full");
            // COUNT_INPUT_TIME_START
            FILE *full_file = fopen(full_fname, "r");
            // COUNT_INPUT_TIME_END

            // If it can't open exit;
            if (full_file != NULL)
            {

                // COUNT_INPUT_TIME_START
                while (!feof(full_file))
                {
                    // Bounds check to prevent buffer overflow
                    if (split_buffer_index >= split_buffer_size - 1) {
                        fprintf(stderr, "WARNING: split_buffer overflow prevented (full file)\n");
                        break;
                    }
                    split_buffer[split_buffer_index].position = (file_position_type *)malloc(index->settings->position_byte_size);
                    split_buffer[split_buffer_index].sax = (sax_type *)malloc(index->settings->sax_byte_size);
                    split_buffer[split_buffer_index].ts = (ts_type *)malloc(index->settings->ts_byte_size);
                    split_buffer[split_buffer_index].insertion_mode = (insertion_mode)(FULL | TMP);

                    // If it can't read continue.
                    if (!fread(split_buffer[split_buffer_index].position, sizeof(file_position_type),
                               1, full_file))
                    {
                        // Free because it is not inserted in the tree
                        free(split_buffer[split_buffer_index].position);
                        continue;
                    }
                    else
                    {
                        if (!fread(split_buffer[split_buffer_index].sax, sizeof(sax_type),
                                   index->settings->paa_segments, full_file))
                        {
                            // Free because it is not inserted in the tree
                            free(split_buffer[split_buffer_index].position);
                            free(split_buffer[split_buffer_index].sax);
                            free(split_buffer[split_buffer_index].ts);
                            continue;
                        }
                        else
                        {
                            if (!fread(split_buffer[split_buffer_index].ts, sizeof(ts_type),
                                       index->settings->timeseries_size, full_file))
                            {
                                // Free because it is not inserted in the tree
                                free(split_buffer[split_buffer_index].position);
                                free(split_buffer[split_buffer_index].sax);
                                free(split_buffer[split_buffer_index].ts);
                                continue;
                            }
                            else
                            {
                                // Increase leaf size (from 0) so that we keep track how many raw time series we
                                // have to move in the finalization step.
                                node->leaf_size++;
                                split_buffer_index++;
                                index->allocated_memory += index->settings->full_record_size;
                            }
                        }
                    }
                }
                // COUNT_INPUT_TIME_END

                // COUNT_OUTPUT_TIME_START
                remove(full_fname);
                // COUNT_OUTPUT_TIME_END
                //  COUNT_INPUT_TIME_START
                fclose(full_file);
                // COUNT_INPUT_TIME_END
            }
            free(full_fname);

            char *partial_fname = (char *)malloc(sizeof(char) * (strlen(node->filename) + 6));
            strcpy(partial_fname, node->filename);
            strcat(partial_fname, ".part");
            // COUNT_INPUT_TIME_START
            FILE *partial_file = fopen(partial_fname, "r");
            // COUNT_INPUT_TIME_END

            // If it can't open exit;
            if (partial_file != NULL)
            {

                // COUNT_INPUT_TIME_START

                while (!feof(partial_file))
                {
                    // Bounds check to prevent buffer overflow
                    if (split_buffer_index >= split_buffer_size - 1) {
                        fprintf(stderr, "WARNING: split_buffer overflow prevented (partial file)\n");
                        break;
                    }
                    split_buffer[split_buffer_index].position = (file_position_type *)malloc(index->settings->position_byte_size);
                    split_buffer[split_buffer_index].sax = (sax_type *)malloc(index->settings->sax_byte_size);
                    split_buffer[split_buffer_index].insertion_mode = (insertion_mode)(PARTIAL | TMP);
                    // If it can't read continue.
                    if (!fread(split_buffer[split_buffer_index].position, sizeof(file_position_type),
                               1, partial_file))
                    {
                        // Free because it is not inserted in the tree
                        free(split_buffer[split_buffer_index].position);
                        free(split_buffer[split_buffer_index].sax);

                        continue;
                    }
                    else
                    {
                        if (!fread(split_buffer[split_buffer_index].sax, sizeof(sax_type),
                                   index->settings->paa_segments, partial_file))
                        {
                            // Free because it is not inserted in the tree
                            free(split_buffer[split_buffer_index].position);
                            free(split_buffer[split_buffer_index].sax);
                            continue;
                        }
                        else
                        {
                            node->leaf_size++;
                            split_buffer_index++;
                            index->allocated_memory += index->settings->partial_record_size;
                        }
                    }
                }
                // COUNT_INPUT_TIME_END
                // COUNT_OUTPUT_TIME_START
                remove(partial_fname);
                // COUNT_OUTPUT_TIME_END
                //  COUNT_INPUT_TIME_START
                //  printf("this is before sktting\n");

                sktting = fclose(partial_file);
                // printf("this is skating%d\n",sktting);
                // COUNT_INPUT_TIME_END
            }

            free(partial_fname);
        }

        // printf("sizeof split buffer: %d\n", split_buffer_index);
        //  Insert buffered data in children

        // Decide split point...
        // printf("informed decision: %d\n",
        //       informed_split_decision(split_data, index->settings, split_buffer, (split_buffer_index)));

        split_data->splitpoint = informed_split_decision(split_data, index->settings, split_buffer, (split_buffer_index));
        // Decide split point...
        // split_data->splitpoint = simple_split_decision(split_data, index->settings);

        // printf("not informed decision: %d\n", split_data->splitpoint);
        if (split_data->splitpoint < 0)
        {
            // This shouldn't happen due to pre-check, but handle gracefully
            fprintf(stderr, "warning: split_node failed to find valid splitpoint\n");
            // Revert changes and return
            node->is_leaf = 1;
            destroy_node_buffer(left_child->buffer);
            free(left_child);
            destroy_node_buffer(right_child->buffer);
            free(right_child);
            node->left_child = NULL;
            node->right_child = NULL;
            free(split_data->split_mask);
            free(split_data);
            node->split_data = NULL;
            free(split_buffer);
            return;
        }

        if (++split_data->split_mask[split_data->splitpoint] > index->settings->sax_bit_cardinality - 1)
        {
            // This shouldn't happen due to pre-check, but handle gracefully  
            fprintf(stderr, "warning: split_node exceeded max cardinality\n");
            --split_data->split_mask[split_data->splitpoint];
            node->is_leaf = 1;
            destroy_node_buffer(left_child->buffer);
            free(left_child);
            destroy_node_buffer(right_child->buffer);
            free(right_child);
            node->left_child = NULL;
            node->right_child = NULL;
            free(split_data->split_mask);
            free(split_data);
            node->split_data = NULL;
            free(split_buffer);
            return;
        }

        root_mask_type mask = index->settings->bit_masks[index->settings->sax_bit_cardinality -
                                                         split_data->split_mask[split_data->splitpoint] - 1];

        while (split_buffer_index > 0)
        {
            split_buffer_index--;
            if (mask & split_buffer[split_buffer_index].sax[split_data->splitpoint])
            {
                add_record_to_node(index, right_child, &split_buffer[split_buffer_index], 1);
            }
            else
            {
                add_record_to_node(index, left_child, &split_buffer[split_buffer_index], 1);
            }
        }

        free(split_buffer);
        // printf("Splitted\n");
    }

    enum response create_node_filename(isax_index *index,
                                       isax_node *node,
                                       isax_node_record *record)
    {
        int i;

        node->filename = (char *)malloc(sizeof(char) * index->settings->max_filename_size);
        sprintf(node->filename, "%s", index->settings->root_directory);
        int l = (int)strlen(index->settings->root_directory);

        // If this has a parent then it is not a root node and as such it does have some
        // split data on its parent about the cardinalities.
        node->isax_values = (sax_type *)malloc(sizeof(sax_type) * index->settings->paa_segments);
        node->isax_cardinalities = (sax_type *)malloc(sizeof(sax_type) * index->settings->paa_segments);

        if (node->parent)
        {
            for (i = 0; i < index->settings->paa_segments; i++)
            {
                root_mask_type mask = 0x00;
                int k;
                for (k = 0; k <= node->parent->split_data->split_mask[i]; k++)
                {
                    mask |= (index->settings->bit_masks[index->settings->sax_bit_cardinality - 1 - k] &
                             record->sax[i]);
                }
                // mask = mask >> index->settings->sax_bit_cardinality - node->parent->split_data->split_mask[i] - 1;
                mask = mask >> (index->settings->sax_bit_cardinality - node->parent->split_data->split_mask[i] - 1);

                node->isax_values[i] = (int)mask;
                node->isax_cardinalities[i] = node->parent->split_data->split_mask[i] + 1;

                if (i == 0)
                {
                    l += sprintf(node->filename + l, "%d.%d", node->isax_values[i], node->isax_cardinalities[i]);
                }
                else
                {
                    l += sprintf(node->filename + l, "_%d.%d", node->isax_values[i], node->isax_cardinalities[i]);
                }
            }
        }
        // If it has no parent it is root node and as such it's cardinality is 1.
        else
        {
            root_mask_type mask = 0x00;

            for (i = 0; i < index->settings->paa_segments; i++)
            {

                mask = (index->settings->bit_masks[index->settings->sax_bit_cardinality - 1] & record->sax[i]);
                // mask = mask >> index->settings->sax_bit_cardinality - 1;
                mask = mask >> (index->settings->sax_bit_cardinality - 1);

                node->isax_values[i] = (int)mask;
                node->isax_cardinalities[i] = 1;

                if (i == 0)
                {
                    l += sprintf(node->filename + l, "%d.1", (int)mask);
                }
                else
                {
                    l += sprintf(node->filename + l, "_%d.1", (int)mask);
                }
            }
        }

        return SUCCESS;
    }

    enum response add_to_node_buffer(isax_node_buffer *node_buffer,
                                     isax_node_record *record,
                                     int sax_segments, int ts_segments)
    {
        if (record->insertion_mode & TMP)
        {
            if (record->insertion_mode & FULL)
            {
                if (node_buffer->max_tmp_full_buffer_size == 0)
                {
                    node_buffer->max_tmp_full_buffer_size = node_buffer->initial_buffer_size;
                    node_buffer->tmp_full_position_buffer = (file_position_type **)malloc(sizeof(file_position_type *) *
                                                                                          node_buffer->max_tmp_full_buffer_size);
                    node_buffer->tmp_full_sax_buffer = (sax_type **)malloc(sizeof(sax_type *) *
                                                                           node_buffer->max_tmp_full_buffer_size);
                    node_buffer->tmp_full_ts_buffer = (ts_type **)malloc(sizeof(ts_type *) *
                                                                         node_buffer->max_tmp_full_buffer_size);
                }
                else if (node_buffer->max_tmp_full_buffer_size <= node_buffer->tmp_full_buffer_size)
                {
                    node_buffer->max_tmp_full_buffer_size *= BUFFER_REALLOCATION_RATE;
                    node_buffer->tmp_full_position_buffer = (file_position_type **)realloc(node_buffer->tmp_full_position_buffer,
                                                                                           sizeof(file_position_type *) *
                                                                                               node_buffer->max_tmp_full_buffer_size);
                    node_buffer->tmp_full_sax_buffer = (sax_type **)realloc(node_buffer->tmp_full_sax_buffer,
                                                                            sizeof(sax_type *) *
                                                                                node_buffer->max_tmp_full_buffer_size);
                    node_buffer->tmp_full_ts_buffer = (ts_type **)realloc(node_buffer->tmp_full_ts_buffer,
                                                                          sizeof(ts_type *) *
                                                                              node_buffer->max_tmp_full_buffer_size);
                }
                node_buffer->tmp_full_position_buffer[node_buffer->tmp_full_buffer_size] = record->position;
                node_buffer->tmp_full_sax_buffer[node_buffer->tmp_full_buffer_size] = record->sax;
                node_buffer->tmp_full_ts_buffer[node_buffer->tmp_full_buffer_size] = record->ts;
                node_buffer->tmp_full_buffer_size++;
            }
            if (record->insertion_mode & PARTIAL)
            {
                if (node_buffer->max_tmp_partial_buffer_size == 0)
                {
                    node_buffer->max_tmp_partial_buffer_size = node_buffer->initial_buffer_size;
                    node_buffer->tmp_partial_position_buffer = (file_position_type **)malloc(sizeof(file_position_type *) *
                                                                                             node_buffer->max_tmp_partial_buffer_size);
                    node_buffer->tmp_partial_sax_buffer = (sax_type **)malloc(sizeof(sax_type *) *
                                                                              node_buffer->max_tmp_partial_buffer_size);
                }
                else if (node_buffer->max_tmp_partial_buffer_size <= node_buffer->tmp_partial_buffer_size)
                {
                    node_buffer->max_tmp_partial_buffer_size *= BUFFER_REALLOCATION_RATE;
                    node_buffer->tmp_partial_position_buffer = (file_position_type **)realloc(node_buffer->tmp_partial_position_buffer,
                                                                                              sizeof(file_position_type *) *
                                                                                                  node_buffer->max_tmp_partial_buffer_size);
                    node_buffer->tmp_partial_sax_buffer = (sax_type **)realloc(node_buffer->tmp_partial_sax_buffer,
                                                                               sizeof(sax_type *) *
                                                                                   node_buffer->max_tmp_partial_buffer_size);
                }
                node_buffer->tmp_partial_position_buffer[node_buffer->tmp_partial_buffer_size] = record->position;
                node_buffer->tmp_partial_sax_buffer[node_buffer->tmp_partial_buffer_size] = record->sax;
                node_buffer->tmp_partial_buffer_size++;
            }
        }
        else if (record->insertion_mode & NO_TMP)
        {
            if (record->insertion_mode & FULL)
            {
                if (node_buffer->max_full_buffer_size == 0)
                {
                    node_buffer->max_full_buffer_size = node_buffer->initial_buffer_size;
                    node_buffer->full_position_buffer = (file_position_type **)malloc(sizeof(file_position_type *) *
                                                                                      node_buffer->max_full_buffer_size);
                    node_buffer->full_sax_buffer = (sax_type **)malloc(sizeof(sax_type *) *
                                                                       node_buffer->max_full_buffer_size);
                    node_buffer->full_ts_buffer = (ts_type **)malloc(sizeof(ts_type *) *
                                                                     node_buffer->max_full_buffer_size);
                }
                else if (node_buffer->max_full_buffer_size <= node_buffer->full_buffer_size)
                {
                    node_buffer->max_full_buffer_size *= BUFFER_REALLOCATION_RATE;
                    node_buffer->full_position_buffer = (file_position_type **)realloc(node_buffer->full_position_buffer,
                                                                                       sizeof(file_position_type *) *
                                                                                           node_buffer->max_full_buffer_size);
                    node_buffer->full_sax_buffer = (sax_type **)realloc(node_buffer->full_sax_buffer,
                                                                        sizeof(sax_type *) *
                                                                            node_buffer->max_full_buffer_size);
                    node_buffer->full_ts_buffer = (ts_type **)realloc(node_buffer->full_ts_buffer,
                                                                      sizeof(ts_type *) *
                                                                          node_buffer->max_full_buffer_size);
                }
                node_buffer->full_position_buffer[node_buffer->full_buffer_size] = record->position;
                node_buffer->full_sax_buffer[node_buffer->full_buffer_size] = record->sax;
                node_buffer->full_ts_buffer[node_buffer->full_buffer_size] = record->ts;
                node_buffer->full_buffer_size++;
            }
            if (record->insertion_mode & PARTIAL)
            {
                if (node_buffer->max_partial_buffer_size == 0)
                {
                    node_buffer->max_partial_buffer_size = node_buffer->initial_buffer_size;
                    node_buffer->partial_position_buffer = (file_position_type **)malloc(sizeof(file_position_type *) *
                                                                                         node_buffer->max_partial_buffer_size);
                    node_buffer->partial_sax_buffer = (sax_type **)malloc(sizeof(sax_type *) *
                                                                          node_buffer->max_partial_buffer_size);
                }
                else if (node_buffer->max_partial_buffer_size <= node_buffer->partial_buffer_size)
                {
                    node_buffer->max_partial_buffer_size *= BUFFER_REALLOCATION_RATE;
                    node_buffer->partial_position_buffer = (file_position_type **)realloc(node_buffer->partial_position_buffer,
                                                                                          sizeof(file_position_type *) *
                                                                                              node_buffer->max_partial_buffer_size);
                    node_buffer->partial_sax_buffer = (sax_type **)realloc(node_buffer->partial_sax_buffer,
                                                                           sizeof(sax_type *) *
                                                                               node_buffer->max_partial_buffer_size);
                }
                node_buffer->partial_position_buffer[node_buffer->partial_buffer_size] = record->position;
                node_buffer->partial_sax_buffer[node_buffer->partial_buffer_size] = record->sax;
                node_buffer->partial_buffer_size++;
            }
        }

        return SUCCESS;
    }

    enum response flush_subtree_leaf_buffers_inmemory(isax_index *index, isax_node *node)
    {

        if (node->is_leaf && node->filename != NULL)
        {
            // Set that unloaded data exist in disk
            if (node->buffer->partial_buffer_size > 0 || node->buffer->tmp_partial_buffer_size > 0)
            {
                node->has_partial_data_file = 1;
            }
            // Set that the node has flushed full data in the disk
            if (node->buffer->full_buffer_size > 0 || node->buffer->tmp_full_buffer_size > 0)
            {
                node->has_full_data_file = 1;
            }

            if (node->has_full_data_file)
            {
                int prev_rec_count = node->leaf_size - (node->buffer->full_buffer_size + node->buffer->tmp_full_buffer_size);

                int previous_page_size = ceil((float)(prev_rec_count * index->settings->full_record_size) / (float)PAGE_SIZE);
                int current_page_size = ceil((float)(node->leaf_size * index->settings->full_record_size) / (float)PAGE_SIZE);
                __sync_fetch_and_add(&(index->memory_info.disk_data_full), (current_page_size - previous_page_size));
                // index->memory_info.disk_data_full += (current_page_size - previous_page_size);
            }
            if (node->has_partial_data_file)
            {
                int prev_rec_count = node->leaf_size - (node->buffer->partial_buffer_size + node->buffer->tmp_partial_buffer_size);

                int previous_page_size = ceil((float)(prev_rec_count * index->settings->partial_record_size) / (float)PAGE_SIZE);
                int current_page_size = ceil((float)(node->leaf_size * index->settings->partial_record_size) / (float)PAGE_SIZE);

                // index->memory_info.disk_data_partial += (current_page_size - previous_page_size);
                __sync_fetch_and_add(&(index->memory_info.disk_data_partial), (current_page_size - previous_page_size));
            }
            if (node->has_full_data_file && node->has_partial_data_file)
            {
                printf("WARNING: (Mem size counting) this leaf has both partial and full data.\n");
            }
            // index->memory_info.disk_data_full += (node->buffer->full_buffer_size +
            // node->buffer->tmp_full_buffer_size);
            __sync_fetch_and_add(&(index->memory_info.disk_data_full), (node->buffer->full_buffer_size + node->buffer->tmp_full_buffer_size));
            // index->memory_info.disk_data_partial += (node->buffer->partial_buffer_size +
            // node->buffer->tmp_partial_buffer_size);
            __sync_fetch_and_add(&(index->memory_info.disk_data_partial), (node->buffer->partial_buffer_size + node->buffer->tmp_partial_buffer_size));
            // flush_node_buffer(node->buffer, index->settings->paa_segments,
            // index->settings->timeseries_size,
            // node->filename);
        }
        else if (!node->is_leaf)
        {
            flush_subtree_leaf_buffers_inmemory(index, node->left_child);
            flush_subtree_leaf_buffers_inmemory(index, node->right_child);
        }

        return SUCCESS;
    }

    int cmp_pri(double next, double curr)
    {
        return (next > curr);
    }

    double get_pri(void *a)
    {
        return (double)((query_result *)a)->distance;
    }

    void set_pri(void *a, double pri)
    {
        ((query_result *)a)->distance = (float)pri;
    }

    size_t get_pos(void *a)
    {
        return ((query_result *)a)->pqueue_position;
    }

    void set_pos(void *a, size_t pos)
    {
        ((query_result *)a)->pqueue_position = pos;
    }

    float calculate_minimum_distance_inmemory(isax_index *index, isax_node *node, ts_type *raw_query, ts_type *query)
    {
        // printf("Calculating minimum distance...\n");
        float bsfLeaf = minidist_paa_to_isax(query, node->isax_values,
                                             node->isax_cardinalities,
                                             index->settings->sax_bit_cardinality,
                                             index->settings->sax_alphabet_cardinality,
                                             index->settings->paa_segments,
                                             MINVAL, MAXVAL,
                                             index->settings->mindist_sqrt);
        float bsfRecord = FLT_MAX;
        // printf("---> Distance: %lf\n", bsfLeaf);
        // sax_print(node->isax_values, 1,  index->settings->sax_bit_cardinality);

        if (!index->has_wedges)
        {
            //      printf("--------------\n");
            int i = 0;

            if (node->buffer != NULL)
            {
                for (i = 0; i < node->buffer->partial_buffer_size; i++)
                {
                    float mindist = minidist_paa_to_isax_raw_SIMD(query, node->buffer->partial_sax_buffer[i], index->settings->max_sax_cardinalities,
                                                                  index->settings->sax_bit_cardinality,
                                                                  index->settings->sax_alphabet_cardinality,
                                                                  index->settings->paa_segments, MINVAL, MAXVAL,
                                                                  index->settings->mindist_sqrt);
                    //              printf("+[PARTIAL] %lf\n", mindist);
                    if (mindist < bsfRecord)
                    {
                        bsfRecord = mindist;
                    }
                }

                for (i = 0; i < node->buffer->tmp_partial_buffer_size; i++)
                {
                    float mindist = minidist_paa_to_isax_raw_SIMD(query, node->buffer->tmp_partial_sax_buffer[i], index->settings->max_sax_cardinalities,
                                                                  index->settings->sax_bit_cardinality,
                                                                  index->settings->sax_alphabet_cardinality,
                                                                  index->settings->paa_segments, MINVAL, MAXVAL,
                                                                  index->settings->mindist_sqrt);
                    //              printf("+[TMP_PARTIAL] %lf\n", mindist);
                    if (mindist < bsfRecord)
                    {
                        bsfRecord = mindist;
                    }
                }
            }
        }
        else
        {
            int i = 0;
            if (node->wedges[0] == FLT_MIN)
            {
                bsfRecord = FLT_MAX;
            }
            else
            {
                bsfRecord = 0;
                ts_type *min_wedge = &node->wedges[0];
                ts_type *max_wedge = &node->wedges[index->settings->timeseries_size];
                if (raw_query[i] > max_wedge[i])
                {
                    bsfRecord += (raw_query[i] - max_wedge[i]) * (raw_query[i] - max_wedge[i]);
                }
                else if (raw_query[i] < max_wedge[i] && raw_query[i] > min_wedge[i])
                {
                    // bound += 0;
                }
                else
                {
                    bsfRecord += (min_wedge[i] - raw_query[i]) * (min_wedge[i] - raw_query[i]);
                }
                // bsfRecord = sqrtf(bsfRecord);
            }
        }
        float bsf = (bsfRecord == FLT_MAX) ? bsfLeaf : bsfRecord;
        //  printf("\t%.2lf - %d [%d] : %s.%s\n",bsfRecord, node->leaf_size, node->is_leaf, node->filename, node->has_full_data_file ? ".full" : ".part");

        // printf("---> Final: %lf\n", bsf);
        return bsf;
    }

    enum response flush_node_buffer(isax_node_buffer *node_buffer, int sax_segments, int ts_segments, const char *filename)
    {
        // WRITE TWO DIFFERENT FILES!
        // 1. .FULL (full records)
        // 2. .PART (partial records)
        int i;
        if (node_buffer->full_buffer_size > 0 || node_buffer->tmp_full_buffer_size > 0)
        {
            FILE *full_file;
            char *full_filename = (char *)malloc(strlen(filename) + 6);
            sprintf(full_filename, "%s.full", filename);
            // COUNT_OUTPUT_TIME_START
            full_file = fopen(full_filename, "a+");
            // Flushing full records
            for (i = 0; i < node_buffer->full_buffer_size; i++)
            {
                ;
                fwrite(node_buffer->full_position_buffer[i],
                       sizeof(file_position_type), 1, full_file);
                fwrite(node_buffer->full_sax_buffer[i],
                       sizeof(sax_type), sax_segments, full_file);
                fwrite(node_buffer->full_ts_buffer[i],
                       sizeof(ts_type), ts_segments, full_file);
            }
            for (i = 0; i < node_buffer->tmp_full_buffer_size; i++)
            {
                fwrite(node_buffer->tmp_full_position_buffer[i],
                       sizeof(file_position_type), 1, full_file);
                fwrite(node_buffer->tmp_full_sax_buffer[i],
                       sizeof(sax_type), sax_segments, full_file);
                fwrite(node_buffer->tmp_full_ts_buffer[i],
                       sizeof(ts_type), ts_segments, full_file);
            }
            fclose(full_file);
            // COUNT_OUTPUT_TIME_END
            free(full_filename);
        }

        if (node_buffer->partial_buffer_size > 0 || node_buffer->tmp_partial_buffer_size > 0)
        {
            FILE *partial_file;
            char *partial_filename = (char *)malloc(strlen(filename) + 6);
            // sax_type *midium_ptr=malloc(sizeof(sax_type)*(sax_segments+8));
            sprintf(partial_filename, "%s.part", filename);
            // COUNT_OUTPUT_TIME_START
            partial_file = fopen(partial_filename, "a+");
            // Flushing partial records
            // printf("the large is %d\n",node_buffer->partial_buffer_size);
            for (i = 0; i < node_buffer->partial_buffer_size; i++)
            {
                // memcpy(midium_ptr, (sax_type*)(node_buffer->partial_position_buffer[i]),8);
                // memcpy(&(midium_ptr[8]), (sax_type*)(node_buffer->partial_sax_buffer[i]),sax_segments);
                // fwrite(midium_ptr, sizeof(sax_type), sax_segments+8, partial_file);
                // memset(midium_ptr,0,sizeof(sax_type)+8);

                fwrite(node_buffer->partial_position_buffer[i],
                       sizeof(file_position_type), 1, partial_file);
                fwrite(node_buffer->partial_sax_buffer[i],
                       sizeof(sax_type), sax_segments, partial_file);
            }
            // printf("hello word\n");
            for (i = 0; i < node_buffer->tmp_partial_buffer_size; i++)
            {
                // memcpy(midium_ptr, (sax_type*)(node_buffer->tmp_partial_position_buffer[i]),8);
                // memcpy(&(midium_ptr[8]), (sax_type*)(node_buffer->tmp_partial_sax_buffer[i]),sax_segments);
                // fwrite(midium_ptr, sizeof(sax_type), sax_segments+8, partial_file);
                // memset(midium_ptr,0,sizeof(sax_type)+8);

                fwrite(node_buffer->tmp_partial_position_buffer[i],
                       sizeof(file_position_type), 1, partial_file);
                fwrite(node_buffer->tmp_partial_sax_buffer[i],
                       sizeof(sax_type), sax_segments, partial_file);
            }

            // free(midium_ptr);
            fclose(partial_file);
            // COUNT_OUTPUT_TIME_END

            free(partial_filename);
        }
        return SUCCESS;
    }

    enum response flush_subtree_leaf_buffers(isax_index *index, isax_node *node)
    {
        if (node->is_leaf && node->filename != NULL)
        {
            // Set that unloaded data exist in disk
            if (node->buffer->partial_buffer_size > 0 || node->buffer->tmp_partial_buffer_size > 0)
            {
                node->has_partial_data_file = 1;
            }
            // Set that the node has flushed full data in the disk
            if (node->buffer->full_buffer_size > 0 || node->buffer->tmp_full_buffer_size > 0)
            {
                node->has_full_data_file = 1;
            }

            if (node->has_full_data_file)
            {
                int prev_rec_count = node->leaf_size - (node->buffer->full_buffer_size + node->buffer->tmp_full_buffer_size);

                int previous_page_size = ceil((float)(prev_rec_count * index->settings->full_record_size) / (float)PAGE_SIZE);
                int current_page_size = ceil((float)(node->leaf_size * index->settings->full_record_size) / (float)PAGE_SIZE);

                index->memory_info.disk_data_full += (current_page_size - previous_page_size);
            }
            if (node->has_partial_data_file)
            {
                int prev_rec_count = node->leaf_size - (node->buffer->partial_buffer_size + node->buffer->tmp_partial_buffer_size);

                int previous_page_size = ceil((float)(prev_rec_count * index->settings->partial_record_size) / (float)PAGE_SIZE);
                int current_page_size = ceil((float)(node->leaf_size * index->settings->partial_record_size) / (float)PAGE_SIZE);

                index->memory_info.disk_data_partial += (current_page_size - previous_page_size);
            }
            if (node->has_full_data_file && node->has_partial_data_file)
            {
                printf("WARNING: (Mem size counting) this leaf has both partial and full data.\n");
            }
            index->memory_info.disk_data_full += (node->buffer->full_buffer_size +
                                                  node->buffer->tmp_full_buffer_size);

            index->memory_info.disk_data_partial += (node->buffer->partial_buffer_size +
                                                     node->buffer->tmp_partial_buffer_size);

            flush_node_buffer(node->buffer, index->settings->paa_segments,
                              index->settings->timeseries_size,
                              node->filename);
        }
        else if (!node->is_leaf)
        {
            flush_subtree_leaf_buffers(index, node->left_child);
            flush_subtree_leaf_buffers(index, node->right_child);
        }

        return SUCCESS;
    }

    enum response flush_subtree_leaf_buffers_m(isax_index *index, isax_node *node, pthread_mutex_t *lock_index, pthread_mutex_t *lock_write)
    {
        if (node->is_leaf && node->filename != NULL)
        {
            // Set that unloaded data exist in disk
            if (node->buffer->partial_buffer_size > 0 || node->buffer->tmp_partial_buffer_size > 0)
            {
                node->has_partial_data_file = 1;
            }
            // Set that the node has flushed full data in the disk
            if (node->buffer->full_buffer_size > 0 || node->buffer->tmp_full_buffer_size > 0)
            {
                node->has_full_data_file = 1;
            }

            if (node->has_full_data_file)
            {
                int prev_rec_count = node->leaf_size - (node->buffer->full_buffer_size + node->buffer->tmp_full_buffer_size);

                int previous_page_size = ceil((float)(prev_rec_count * index->settings->full_record_size) / (float)PAGE_SIZE);
                int current_page_size = ceil((float)(node->leaf_size * index->settings->full_record_size) / (float)PAGE_SIZE);

                index->memory_info.disk_data_full += (current_page_size - previous_page_size);
            }
            if (node->has_partial_data_file)
            {
                int prev_rec_count = node->leaf_size - (node->buffer->partial_buffer_size + node->buffer->tmp_partial_buffer_size);

                int previous_page_size = ceil((float)(prev_rec_count * index->settings->partial_record_size) / (float)PAGE_SIZE);
                int current_page_size = ceil((float)(node->leaf_size * index->settings->partial_record_size) / (float)PAGE_SIZE);

                index->memory_info.disk_data_partial += (current_page_size - previous_page_size);
            }
            if (node->has_full_data_file && node->has_partial_data_file)
            {
                printf("WARNING: (Mem size counting) this leaf has both partial and full data.\n");
            }
            index->memory_info.disk_data_full += (node->buffer->full_buffer_size +
                                                  node->buffer->tmp_full_buffer_size);

            index->memory_info.disk_data_partial += (node->buffer->partial_buffer_size +
                                                     node->buffer->tmp_partial_buffer_size);

            pthread_mutex_lock(lock_write);
            flush_node_buffer(node->buffer, index->settings->paa_segments,
                              index->settings->timeseries_size,
                              node->filename);
            pthread_mutex_unlock(lock_write);
        }
        else if (!node->is_leaf)
        {
            flush_subtree_leaf_buffers_m(index, node->left_child, lock_index, lock_write);
            flush_subtree_leaf_buffers_m(index, node->right_child, lock_index, lock_write);
        }

        return SUCCESS;
    }

    enum response clear_node_buffer(isax_node_buffer *node_buffer, enum buffer_cleaning_mode clean_mode)
    {
        // ONLY SELECTIVELY CLEAR STUFF THAT COME FROM THE INSERT
        // TO INDEX FUNCTION BECAUSE THEY MAY COME FROM REUSABLE
        // MEMORY SEGMENTS SUCH AS THE FBL BUFFERS WHICH ARE USED
        // AGAIN FOR THE NEXT EPOC OF DATA LOADING WITHOUT BEING
        // FREED.
        int i;
        if (clean_mode == FULL_CLEAN || clean_mode == TMP_AND_TS_CLEAN)
            for (i = 0; i < node_buffer->full_buffer_size; i++)
            {
                if (clean_mode == FULL_CLEAN || clean_mode == TMP_AND_TS_CLEAN)
                {
                    // IMPORTANT: THIS MEANS THAT NOBODY SHOULD INSERT A FULL TS FROM SHARED MEMORY
                    // THE TS_BUFFER WILL *ALWAYS* BE CLEARED! AND AN INVALID FREE WILL BE RAISED!
                    free(node_buffer->full_ts_buffer[i]);
                }

                if (clean_mode == FULL_CLEAN)
                {
                    free(node_buffer->full_sax_buffer[i]);
                    free(node_buffer->full_position_buffer[i]);
                }
            }

        if (clean_mode == FULL_CLEAN)
            for (i = 0; i < node_buffer->partial_buffer_size; i++)
            {
                if (clean_mode == FULL_CLEAN)
                {
                    free(node_buffer->partial_sax_buffer[i]);
                    free(node_buffer->partial_position_buffer[i]);
                }
            }

        // FULL CLEAN TEMPORARY STUFF
        for (i = 0; i < node_buffer->tmp_full_buffer_size; i++)
        {
            free(node_buffer->tmp_full_ts_buffer[i]);
            free(node_buffer->tmp_full_sax_buffer[i]);
            free(node_buffer->tmp_full_position_buffer[i]);
        }
        for (i = 0; i < node_buffer->tmp_partial_buffer_size; i++)
        {
            free(node_buffer->tmp_partial_sax_buffer[i]);
            free(node_buffer->tmp_partial_position_buffer[i]);
        }

        // Set to 0 so that the LBLs will be refilled.

        node_buffer->tmp_full_buffer_size = 0;
        node_buffer->full_buffer_size = 0;
        node_buffer->tmp_partial_buffer_size = 0;
        node_buffer->partial_buffer_size = 0;

        return SUCCESS;
    }

    void isax_index_clear_node_buffers(isax_index *index, isax_node *node, enum node_cleaning_mode node_cleaning_mode, enum buffer_cleaning_mode buffer_clean_mode)
    {
        if (node == NULL)
        {
            // TODO: OPTIMIZE TO FLUSH WITHOUT TRAVERSAL!
            isax_node *subtree_root = index->first_node;

            while (subtree_root != NULL)
            {

                isax_index_clear_node_buffers(index, subtree_root, node_cleaning_mode, buffer_clean_mode);
                subtree_root = subtree_root->next;
            }
        }
        else
        {
            // Traverse tree
            // printf("this is the time 2\n");

            if (!node->is_leaf && node_cleaning_mode == INCLUDE_CHILDREN)
            {
                isax_index_clear_node_buffers(index, node->right_child, node_cleaning_mode, buffer_clean_mode);
                isax_index_clear_node_buffers(index, node->left_child, node_cleaning_mode, buffer_clean_mode);
            }
            else if (node->is_leaf && node->buffer != NULL)
            {
                clear_node_buffer(node->buffer, buffer_clean_mode);
            }
        }
    }

    enum response flush_fbl(first_buffer_layer *fbl, isax_index *index)
    {
        int c = 1;
        int j;
        isax_node_record *r = (isax_node_record *)malloc(sizeof(isax_node_record));
        for (j = 0; j < fbl->number_of_buffers; j++)
        {

            fbl_soft_buffer *current_fbl_node = &index->fbl->soft_buffers[j];
            if (!current_fbl_node->initialized)
            {
                continue;
            }

            int i;
            if (current_fbl_node->buffer_size > 0)
            {
                // For all records in this buffer
                // COUNT_CAL_TIME_START
                for (i = 0; i < current_fbl_node->buffer_size; i++)
                {
                    r->sax = (sax_type *)current_fbl_node->sax_records[i];
                    r->position = (file_position_type *)current_fbl_node->pos_records[i];
                    r->insertion_mode = (insertion_mode)(NO_TMP | PARTIAL);
                    // Add record to index

                    add_record_to_node(index, current_fbl_node->node, r, 1);
                }

                // flush index node
                // COUNT_CAL_TIME_START
                flush_subtree_leaf_buffers(index, current_fbl_node->node);
                // COUNT_CAL_TIME_END
                //  clear FBL records moved in LBL buffers
                free(current_fbl_node->sax_records);
                free(current_fbl_node->pos_records);
                // clear records read from files (free only prev sax buffers)

                isax_index_clear_node_buffers(index, current_fbl_node->node, INCLUDE_CHILDREN, TMP_AND_TS_CLEAN);

                index->allocated_memory = 0;
                // Set to 0 in order to re-allocate original space for buffers.
                current_fbl_node->buffer_size = 0;
                current_fbl_node->max_buffer_size = 0;
            }
        }

        free(r);
        fbl->current_record_index = 0;
        fbl->current_record = fbl->hard_buffer;

        return SUCCESS;
    }

    isax_node *insert_to_fbl(first_buffer_layer *fbl, sax_type *sax, file_position_type *pos, root_mask_type mask, isax_index *index)
    {

        fbl_soft_buffer *current_buffer = &fbl->soft_buffers[(int)mask];

        // Check if this buffer is initialized
        if (!current_buffer->initialized)
        {

            current_buffer->initialized = 1;
            current_buffer->max_buffer_size = 0;
            current_buffer->buffer_size = 0;

            current_buffer->node = isax_root_node_init(mask, index->settings->initial_leaf_buffer_size);
            index->root_nodes++;
            current_buffer->node->is_leaf = 1;

            if (index->first_node == NULL)
            {
                index->first_node = current_buffer->node;
                current_buffer->node->next = NULL;
                current_buffer->node->previous = NULL;
            }
            else
            {
                isax_node *prev_first = index->first_node;
                index->first_node = current_buffer->node;
                index->first_node->next = prev_first;
                prev_first->previous = current_buffer->node;
            }
        }
        // Check if this buffer is not full!
        if (current_buffer->buffer_size >= current_buffer->max_buffer_size)
        {

            if (current_buffer->max_buffer_size == 0)
            {
                current_buffer->max_buffer_size = fbl->initial_buffer_size;
                current_buffer->max_buffer_size = fbl->initial_buffer_size;
                current_buffer->sax_records = (sax_type **)malloc(sizeof(sax_type *) * current_buffer->max_buffer_size);
                current_buffer->pos_records = (file_position_type **)malloc(sizeof(file_position_type *) * current_buffer->max_buffer_size);
            }
            else
            {
                current_buffer->max_buffer_size *= BUFFER_REALLOCATION_RATE;
                current_buffer->sax_records = (sax_type **)realloc(current_buffer->sax_records, sizeof(sax_type *) * current_buffer->max_buffer_size);
                current_buffer->pos_records = (file_position_type **)realloc(current_buffer->pos_records, sizeof(file_position_type *) * current_buffer->max_buffer_size);
            }
        }
        if (current_buffer->sax_records == NULL || current_buffer->pos_records == NULL)
        {
            fprintf(stderr, "error: Could not allocate memory in FBL.");
            exit(1);
            // return OUT_OF_MEMORY_FAILURE;
        }

        // Copy data to hard buffer and make current buffer point to the hard one
        current_buffer->sax_records[current_buffer->buffer_size] = (sax_type *)fbl->current_record;
        memcpy((void *)fbl->current_record, (void *)sax, index->settings->sax_byte_size);
        fbl->current_record += index->settings->sax_byte_size;

        current_buffer->pos_records[current_buffer->buffer_size] = (file_position_type *)fbl->current_record;
        memcpy((void *)fbl->current_record, (void *)pos, index->settings->position_byte_size);
        fbl->current_record += index->settings->position_byte_size;

        current_buffer->buffer_size++;
        fbl->current_record_index++;

        return current_buffer->node;
    }

    root_mask_type isax_fbl_index_insert(isax_index *index, sax_type *sax, file_position_type *pos)
    {
        // COUNT_OUTPUT_TIME_START
        fwrite(sax, index->settings->sax_byte_size, 1, index->sax_file);
        // COUNT_OUTPUT_TIME_END
        //  Create mask for the first bit of the sax representation

        // Step 1: Check if there is a root node that represents the
        //         current node's sax representation

        // TODO: Create INSERTION SHORT AND BINARY SEARCH METHODS.
        root_mask_type first_bit_mask = 0x00;
        CREATE_MASK(first_bit_mask, index, sax);
        // printf("sax is %d\n",first_bit_mask );
        insert_to_fbl(index->fbl, sax, pos, first_bit_mask, index);
        index->total_records++;

        if ((index->total_records % index->settings->max_total_buffer_size) == 0)
        {
            // FLUSHES++;
            flush_fbl(index->fbl, index);
        }

        return first_bit_mask;
    }

    void init(deque *d, int capacity)
    {
        d->capacity = capacity;
        d->size = 0;
        d->dq = (int *)malloc(sizeof(int) * d->capacity);
        d->f = 0;
        d->r = d->capacity - 1;
    }

    /// Insert to the queue at the back
    void push_back(struct deque *d, int v)
    {
        d->dq[d->r] = v;
        d->r--;
        if (d->r < 0)
            d->r = d->capacity - 1;
        d->size++;
    }

    /// Delete the current (front) element from queue
    void pop_front(struct deque *d)
    {
        d->f--;
        if (d->f < 0)
            d->f = d->capacity - 1;
        d->size--;
    }

    /// Delete the last element from queue
    void pop_back(struct deque *d)
    {
        d->r = (d->r + 1) % d->capacity;
        d->size--;
    }

    /// Get the value at the current position of the circular queue
    int front(struct deque *d)
    {
        int aux = d->f - 1;

        if (aux < 0)
            aux = d->capacity - 1;
        return d->dq[aux];
    }

    /// Get the value at the last position of the circular queueint back(struct deque *d)
    int back(struct deque *d)
    {
        int aux = (d->r + 1) % d->capacity;
        return d->dq[aux];
    }

    /// Check whether or not the queue is empty
    int empty(struct deque *d)
    {
        return d->size == 0;
    }

    /// Destroy the queue
    void destroy(deque *d)
    {
        free(d->dq);
    }

    void lower_upper_lemire(float *t, int len, int r, float *l, float *u)
    {
        struct deque du, dl;

        init(&du, 2 * r + 2);
        init(&dl, 2 * r + 2);

        push_back(&du, 0);
        push_back(&dl, 0);
        int i;

        for (i = 1; i < len; i++)
        {
            if (i > r)
            {
                u[i - r - 1] = t[front(&du)];
                l[i - r - 1] = t[front(&dl)];
            }
            if (t[i] > t[i - 1])
            {
                pop_back(&du);
                while (!empty(&du) && t[i] > t[back(&du)])
                    pop_back(&du);
            }
            else
            {
                pop_back(&dl);
                while (!empty(&dl) && t[i] < t[back(&dl)])
                    pop_back(&dl);
            }
            push_back(&du, i);
            push_back(&dl, i);
            if (i == 2 * r + 1 + front(&du))
                pop_front(&du);
            else if (i == 2 * r + 1 + front(&dl))
                pop_front(&dl);
        }
        for (i = len; i < len + r + 1; i++)
        {
            u[i - r - 1] = t[front(&du)];
            l[i - r - 1] = t[front(&dl)];
            if (i - front(&du) >= 2 * r + 1)
                pop_front(&du);
            if (i - front(&dl) >= 2 * r + 1)
                pop_front(&dl);
        }
        destroy(&du);
        destroy(&dl);
    }

    root_mask_type isax_fbl_index_insert_m(isax_index *index, sax_type *sax, file_position_type *pos,
                                           pthread_mutex_t *lock_record, pthread_mutex_t *lock_fbl,
                                           pthread_mutex_t *lock_cbl, pthread_mutex_t *lock_firstnode,
                                           pthread_mutex_t *lock_index, pthread_mutex_t *lock_disk)
    {
        // Thread-safe version: same logic as isax_fbl_index_insert() but with locks
        // and without the flush check (handled separately by indexconstruction())
        pthread_mutex_lock(lock_disk);
        fwrite(sax, index->settings->sax_byte_size, 1, index->sax_file);
        pthread_mutex_unlock(lock_disk);
        
        root_mask_type first_bit_mask = 0x00;
        CREATE_MASK(first_bit_mask, index, sax);
        
        // insert_to_fbl() accesses shared structures, so we need locks
        pthread_mutex_lock(lock_firstnode);
        pthread_mutex_lock(lock_fbl);
        insert_to_fbl(index->fbl, sax, pos, first_bit_mask, index);
        pthread_mutex_unlock(lock_fbl);
        pthread_mutex_unlock(lock_firstnode);
        
        // Use atomic increment for thread safety
        __sync_fetch_and_add(&(index->total_records), 1);
        
        // Note: flush check is handled separately by indexconstruction() in multi-threaded version
        return first_bit_mask;
    }

    void *indexconstructionworker(void *input)
    {
        isax_index *index = ((trans_fbl_input*)input)->index;
        pthread_mutex_t *lock_index = ((trans_fbl_input*)input)->lock_index;
        pthread_mutex_t *lock_write = ((trans_fbl_input*)input)->lock_write;
        int j, c = 1;
        isax_node_record *r = (isax_node_record *)malloc(sizeof(isax_node_record));
        
        while(1)
        {
            pthread_mutex_lock(((trans_fbl_input*)input)->lock_fbl_conter);
            j = ((trans_fbl_input*)input)->conternumber;
            ((trans_fbl_input*)input)->conternumber++;
            pthread_mutex_unlock(((trans_fbl_input*)input)->lock_fbl_conter);
            
            if(j >= ((trans_fbl_input*)input)->stop_number)
            {
                break;
            }
            
            fbl_soft_buffer *current_fbl_node = &index->fbl->soft_buffers[j];
            if (!current_fbl_node->initialized) {
                continue;
            }
            
            int i;
            if (current_fbl_node->buffer_size > 0) 
            {
                // For all records in this buffer 
                for (i = 0; i < current_fbl_node->buffer_size; i++) 
                {
                    r->sax = (sax_type *) current_fbl_node->sax_records[i];
                    r->position = (file_position_type *) current_fbl_node->pos_records[i];
                    r->insertion_mode = (insertion_mode)(NO_TMP | PARTIAL);
                    // Add record to index
                    add_record_to_node(index, current_fbl_node->node, r, 1);
                }

                // flush index node
                flush_subtree_leaf_buffers_m(index, current_fbl_node->node, lock_index, lock_write);
                
                // clear FBL records moved in LBL buffers
                free(current_fbl_node->sax_records);
                free(current_fbl_node->pos_records);
                
                // clear records read from files (free only prev sax buffers)
                isax_index_clear_node_buffers(index, current_fbl_node->node, 
                                              INCLUDE_CHILDREN,
                                              TMP_AND_TS_CLEAN);
                
                index->allocated_memory = 0;
                // Set to 0 in order to re-allocate original space for buffers.
                current_fbl_node->buffer_size = 0;
                current_fbl_node->max_buffer_size = 0;
            }
        }
        
        free(r);
        return NULL;
    }

    enum response indexconstruction(first_buffer_layer *fbl, isax_index *index, pthread_mutex_t *lock_index,
                          pthread_mutex_t *lock_disk, int calculate_thread)
    {
        int j;
        trans_fbl_input input_data;
        pthread_t threadid[calculate_thread];
        pthread_mutex_t lock_fbl_conter = PTHREAD_MUTEX_INITIALIZER;
        
        // Initialize all fields
        input_data.start_number = 0;
        input_data.stop_number = fbl->number_of_buffers;
        input_data.conternumber = 0;
        input_data.index = index;
        input_data.lock_index = lock_index;
        input_data.lock_fbl_conter = &lock_fbl_conter;
        input_data.lock_write = lock_disk;
        input_data.fbl = fbl;
        input_data.preworkernumber = 0;
        input_data.fbloffset = 0;
        input_data.buffersize = NULL;
        input_data.lock_barrier1 = NULL;
        input_data.lock_barrier2 = NULL;
        input_data.lock_barrier3 = NULL;
        input_data.finished = false;
        
        // Start the worker threads
        for (int k = 0; k < calculate_thread; k++)
        {
            pthread_create(&(threadid[k]), NULL, indexconstructionworker, (void*)&(input_data));
        }
        
        // Wait for all threads to complete
        for (int k = 0; k < calculate_thread; k++)
        {
            pthread_join(threadid[k], NULL);
        }   
        
        fbl->current_record_index = 0;
        fbl->current_record = fbl->hard_buffer;
        
        return SUCCESS;
    }

    void *indexbulkloadingworker(void *transferdata)
    {
        sax_type *sax = (sax_type *)malloc(sizeof(sax_type) * ((index_buffer_data *)transferdata)->index->settings->paa_segments);

        int fin_number = ((index_buffer_data *)transferdata)->fin_number;
        file_position_type *pos = (file_position_type *)malloc(sizeof(file_position_type));
        sax_type *saxv;
        int offset_saxv = ((index_buffer_data *)transferdata)->blocid * ((index_buffer_data *)transferdata)->index->settings->paa_segments;
        int paa_segments = ((index_buffer_data *)transferdata)->index->settings->paa_segments;
        isax_index *index = ((index_buffer_data *)transferdata)->index;
        int i = 0;
        pthread_barrier_t *lock_barrier1, *lock_barrier2;

        while (!((index_buffer_data *)transferdata)->finished)
        {
            saxv = (((index_buffer_data *)transferdata)->saxv);
            lock_barrier1 = ((index_buffer_data *)transferdata)->lock_barrier1;
            lock_barrier2 = ((index_buffer_data *)transferdata)->lock_barrier2;
            for (i = 0; i < fin_number; i++)
            {
                if (sax_from_ts(((ts_type *)(((index_buffer_data *)transferdata)->ts) + i * index->settings->timeseries_size), sax,
                                index->settings->ts_values_per_paa_segment,
                                index->settings->paa_segments, index->settings->sax_alphabet_cardinality,
                                index->settings->sax_bit_cardinality) == SUCCESS)
                {
                    *pos = ((index_buffer_data *)transferdata)->pos + index->settings->timeseries_size * sizeof(ts_type) * i;
                    memcpy(&(saxv[offset_saxv + i * paa_segments]), sax, sizeof(sax_type) * paa_segments);
                    isax_fbl_index_insert_m(index, sax, pos, ((index_buffer_data *)transferdata)->lock_record,
                                           ((index_buffer_data *)transferdata)->lock_fbl,
                                           ((index_buffer_data *)transferdata)->lock_cbl,
                                           ((index_buffer_data *)transferdata)->lock_firstnode,
                                           ((index_buffer_data *)transferdata)->lock_index,
                                           ((index_buffer_data *)transferdata)->lock_disk);
                }
                else
                {
                    fprintf(stderr, "error: cannot insert record in index, since sax representation\
                    failed to be created");
                }
            }
            pthread_barrier_wait(lock_barrier1);
            pthread_barrier_wait(lock_barrier2);
        }
        free(pos);
        free(sax);
        return NULL;
    }

    void isax_index_binary_file_m(const char *ifilename, int ts_num, isax_index *index, int calculate_thread, int read_block_length)
    {
        fprintf(stderr, ">>> Indexing: %s\n", ifilename);
        FILE *ifile;

        ifile = fopen(ifilename, "rb");
        index_buffer_data *input_data = (index_buffer_data *)malloc(sizeof(index_buffer_data) * (calculate_thread - 1));
        file_position_type *pos = (file_position_type *)malloc(sizeof(file_position_type));
        pthread_t threadid[calculate_thread - 1];
        bool sax_fist_time_check = false;
        long int ts_loaded = 0;
        int j, conter = 0;
        long long int i;
        int prev_flush_time = 0, now_flush_time = 0;
        int sax_save_number;
        // initial the locks
        pthread_mutex_t lock_record = PTHREAD_MUTEX_INITIALIZER, lockfbl = PTHREAD_MUTEX_INITIALIZER,
                        lock_index = PTHREAD_MUTEX_INITIALIZER,
                        lock_firstnode = PTHREAD_MUTEX_INITIALIZER, lock_disk = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_t *lockcbl;
        pthread_barrier_t lock_barrier1, lock_barrier2;
        pthread_barrier_init(&lock_barrier1, NULL, calculate_thread);
        pthread_barrier_init(&lock_barrier2, NULL, calculate_thread);

        lockcbl = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * LOCK_SIZE);

        for (i = 0; i < LOCK_SIZE; i++)
        {
            pthread_mutex_init(&lockcbl[i], NULL);
        }

        for (i = 0; i < (calculate_thread - 1); i++)
        {
            input_data[i].index = index;
            input_data[i].lock_fbl = &lockfbl;
            input_data[i].lock_record = &lock_record;
            input_data[i].lock_cbl = lockcbl;
            input_data[i].lock_firstnode = &lock_firstnode;
            input_data[i].lock_index = &lock_index;
            input_data[i].blocid = i * read_block_length;
            input_data[i].lock_disk = &lock_disk;

            input_data[i].lock_barrier1 = &lock_barrier1;
            input_data[i].lock_barrier2 = &lock_barrier2;
            input_data[i].finished = false;
        }
        if (ifile == NULL)
        {
            fprintf(stderr, "File %s not found!\n", ifilename);
            exit(-1);
        }

        fseek(ifile, 0L, SEEK_END);
        file_position_type sz = (file_position_type)ftell(ifile);
        file_position_type total_records = sz / index->settings->ts_byte_size;
        fseek(ifile, 0L, SEEK_SET);

        if (total_records < ts_num)
        {
            fprintf(stderr, "File %s has only %llu records!\n", ifilename, total_records);
            exit(-1);
        }

        // read_block_length is now a parameter to the function
        ts_type *ts = (ts_type *)malloc(sizeof(ts_type) * index->settings->timeseries_size * read_block_length * (calculate_thread - 1));
        ts_type *ts1 = (ts_type *)malloc(sizeof(ts_type) * index->settings->timeseries_size * read_block_length * (calculate_thread - 1));
        ts_type *ts2;
        sax_type *saxv = (sax_type *)malloc(sizeof(sax_type) * index->settings->paa_segments * read_block_length * (calculate_thread - 1));
        sax_type *saxv1 = (sax_type *)malloc(sizeof(sax_type) * index->settings->paa_segments * read_block_length * (calculate_thread - 1));
        sax_type *saxv2;

        index->settings->raw_filename = (char *)malloc(256);
        strcpy(index->settings->raw_filename, ifilename);

        *pos = ftell(ifile);

        if (ts_num > read_block_length * (calculate_thread - 1))
        {
            size_t items_read = fread(ts1, sizeof(ts_type), index->settings->timeseries_size * read_block_length * (calculate_thread - 1), ifile);
            (void)items_read; // Suppress unused variable warning - we trust the file size check above
            ts2 = ts;
            ts = ts1;
            ts1 = ts2;

            for (j = 0; j < (calculate_thread - 1); j++)
            {
                input_data[j].pos = *pos + index->settings->timeseries_size * sizeof(ts_type) * j * read_block_length;
                input_data[j].ts = &(ts[index->settings->timeseries_size * j * read_block_length]);
                input_data[j].saxv = saxv;
                input_data[j].fin_number = read_block_length;
            }
            for (j = 0; j < (calculate_thread - 1); j++)
            {
                pthread_create(&(threadid[j]), NULL, indexbulkloadingworker, (void *)&(input_data[j]));
            }

            for (i = read_block_length * (calculate_thread - 1) * 2; i <= ts_num; i += read_block_length * (calculate_thread - 1))
            {

                *pos = ftell(ifile);
                // read the data of next round
                size_t items_read = fread(ts1, sizeof(ts_type), index->settings->timeseries_size * read_block_length * (calculate_thread - 1), ifile);
                (void)items_read; // Suppress unused variable warning - we trust the file size check above
                // write the sax in disk (last round)
                if (sax_fist_time_check)
                {
                    pthread_mutex_lock(&lock_disk);
                    fwrite(saxv1, index->settings->sax_byte_size, read_block_length * (calculate_thread - 1), index->sax_file);
                    pthread_mutex_unlock(&lock_disk);
                }
                else
                {
                    sax_fist_time_check = true;
                }

                pthread_barrier_wait(&lock_barrier1);
                // wait for the finish of other threads
                ts2 = ts;
                ts = ts1;
                ts1 = ts2;
                __sync_fetch_and_add(&(index->fbl->current_record_index), read_block_length * (calculate_thread - 1));
                now_flush_time = i / (index->settings->max_total_buffer_size);
                if (now_flush_time != prev_flush_time)
                {
                    indexconstruction(index->fbl, index, &lock_index, &lock_disk, calculate_thread);
                }
                saxv2 = saxv;
                saxv = saxv1;
                saxv1 = saxv2;

                prev_flush_time = now_flush_time;
                for (j = 0; j < (calculate_thread - 1); j++)
                {
                    input_data[j].pos = *pos + index->settings->timeseries_size * sizeof(ts_type) * j * read_block_length;
                    input_data[j].ts = &(ts[index->settings->timeseries_size * j * read_block_length]);
                    input_data[j].saxv = saxv;
                    input_data[j].fin_number = read_block_length;
                }
                pthread_barrier_wait(&lock_barrier2);
            }

            pthread_barrier_wait(&lock_barrier1);
            for (j = 0; j < (calculate_thread - 1); j++)
            {
                input_data[j].finished = true;
            }
            // wait for the finish of other threads
            __sync_fetch_and_add(&(index->fbl->current_record_index), read_block_length * (calculate_thread - 1));
            now_flush_time = i / (index->settings->max_total_buffer_size);
            if (now_flush_time != prev_flush_time)
            {
                indexconstruction(index->fbl, index, &lock_index, &lock_disk, calculate_thread);
            }

            prev_flush_time = now_flush_time;
            for (j = 0; j < (calculate_thread - 1); j++)
            {
                input_data[j].pos = *pos + index->settings->timeseries_size * sizeof(ts_type) * j * read_block_length;
                input_data[j].ts = &(ts[index->settings->timeseries_size * j * read_block_length]);
                input_data[j].saxv = saxv;
                input_data[j].fin_number = read_block_length;
            }
            pthread_barrier_wait(&lock_barrier2);
            for (j = 0; j < (calculate_thread - 1); j++)
            {
                pthread_join(threadid[j], NULL);
            }
        }
        *pos = ftell(ifile);
        size_t remainder_size = index->settings->timeseries_size * (ts_num % (read_block_length * (calculate_thread - 1)));
        if (remainder_size > 0)
        {
            size_t items_read = fread(ts1, sizeof(ts_type), remainder_size, ifile);
            (void)items_read; // Suppress unused variable warning - we trust the file size check above
        }
        ts2 = ts;
        ts = ts1;
        ts1 = ts2;

        // handle the rest data
        int conter_ts_number = ts_num % (read_block_length * (calculate_thread - 1));
        sax_save_number = conter_ts_number;

        for (j = 0; j < (calculate_thread - 1); j++)
        {
            input_data[j].pos = *pos + index->settings->timeseries_size * sizeof(ts_type) * j * read_block_length;
            conter++;
            input_data[j].ts = &(ts[index->settings->timeseries_size * j * read_block_length]);
            input_data[j].fin_number = min(conter_ts_number, read_block_length);
            input_data[j].saxv = saxv;
            conter_ts_number = conter_ts_number - read_block_length;

            if (conter_ts_number < 0)
                break;
        }

        pthread_barrier_init(&lock_barrier1, NULL, conter + 1);
        pthread_barrier_init(&lock_barrier2, NULL, conter + 1);
        for (j = 0; j < conter; j++)
        {
            input_data[j].finished = false;
            pthread_create(&(threadid[j]), NULL, indexbulkloadingworker, (void *)&(input_data[j]));
        }

        if (sax_fist_time_check)
        {
            pthread_mutex_lock(&lock_disk);
            fwrite(saxv1, index->settings->sax_byte_size, read_block_length * (calculate_thread - 1), index->sax_file);
            pthread_mutex_unlock(&lock_disk);
        }
        else
        {
            sax_fist_time_check = true;
        }
        pthread_barrier_wait(&lock_barrier1);
        saxv2 = saxv;
        saxv = saxv1;
        saxv1 = saxv2;
        for (j = 0; j < conter; j++)
        {
            input_data[j].finished = true;
        }
        pthread_barrier_wait(&lock_barrier2);

        for (j = 0; j < conter; j++)
        {
            pthread_join(threadid[j], NULL);
        }
        pthread_mutex_lock(&lock_disk);
        fwrite(saxv1, index->settings->sax_byte_size, sax_save_number, index->sax_file);
        fflush(index->sax_file);
        pthread_mutex_unlock(&lock_disk);
        __sync_fetch_and_add(&(index->fbl->current_record_index), sax_save_number);
        indexconstruction(index->fbl, index, &lock_index, &lock_disk, calculate_thread);
        free(ts);
        free(ts1);
        free(input_data);
        free(lockcbl);
        free(saxv);
        free(saxv1);
        free(pos);
        pthread_barrier_destroy(&lock_barrier1);
        pthread_barrier_destroy(&lock_barrier2);
        index->total_records = (unsigned long long)ts_num;
        fclose(ifile);
    }

}