#include "iSAXIndex.hpp"

#include <float.h>
namespace diNoLib
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
                // mkdir(root_directory, 0777);
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
                chdir("../");
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

    parallel_first_buffer_layer *initialize_pRecBuf(int initial_buffer_size, int number_of_buffers,
                                                    int max_total_buffers_size, isax_index *index)
    {
        parallel_first_buffer_layer *fbl = (parallel_first_buffer_layer *)malloc(sizeof(parallel_first_buffer_layer));

        fbl->max_total_size = max_total_buffers_size;
        fbl->initial_buffer_size = initial_buffer_size;
        fbl->number_of_buffers = number_of_buffers;

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
        // COUNT_NEW_NODE()
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
        // pthread_rwlock_wrlock(lockfbl);
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
                // current_buffer->finished=1;
                current_buffer->initialized = 1;
                //__sync_synchronize();
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

        // insert_to_fbl_m(index->fbl, sax, pos,first_bit_mask, index,lock_firstnode,lock_fbl);
        insert_to_pRecBuf((parallel_first_buffer_layer *)(index->fbl), sax, pos, first_bit_mask, index, lock_firstnode, workernumber, total_workernumber);
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

        // #ifdef DEBUG
        // printf("%d -> %f\n", records_buffer[0].sax[0], sax2paa[0]);
        // #endif

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

                // #ifdef DEBUG
                // printf("%d -> %d + %d -> %f\n", break_point_id >> 1, break_point_id, offset, b);
                // #endif

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

    void split_node(isax_index *index, isax_node *node)
    {
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
        // Allocating 1 more position to cover any off-sized allocations happening due to
        // trying to load one more record from a fetched file page which does not exist.
        // e.g. line 284 ( if(!fread... )
        isax_node_record *split_buffer = (isax_node_record *)malloc(sizeof(isax_node_record) *
                                                                    (index->settings->max_leaf_size + 1));

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
                            if (!fread(split_buffer[split_buffer_index].sax, sizeof(ts_type),
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
            fprintf(stderr, "error: cannot split in depth more than %d.\n",
                    index->settings->sax_bit_cardinality);
            exit(-1);
        }

        if (++split_data->split_mask[split_data->splitpoint] > index->settings->sax_bit_cardinality - 1)
        {
            fprintf(stderr, "error: cannot split in depth more than %d.\n",
                    index->settings->sax_bit_cardinality);
            exit(-1);
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

#ifdef DEBUG
        printf("\tCreated filename:\t\t %s\n\n", node->filename);
#endif

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
                    node_buffer->tmp_partial_position_buffer = (file_position_type **)realloc(node_buffer->tmp_full_position_buffer,
                                                                                              sizeof(file_position_type *) *
                                                                                                  node_buffer->max_tmp_partial_buffer_size);
                    node_buffer->tmp_partial_sax_buffer = (sax_type **)realloc(node_buffer->tmp_full_sax_buffer,
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

}