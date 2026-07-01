#include "Fresh.hpp"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "../isax/iSAXPqueue.hpp"
#include "../isax/iSAXIndex.hpp"
#include "../isax/iSAXSearch.hpp"
#include "../distance_computers/DistanceComputer.hpp"

namespace daisy
{

#define FRESH_TS_GROUP_LENGTH 1000

typedef struct {
    volatile unsigned long num __attribute__((aligned(64)));
} fresh_next_ts_group_t;

typedef struct {
    volatile unsigned long num __attribute__((aligned(64)));
} fresh_next_ts_in_group_t;

typedef struct FRESH_buffer_data {
    isax_index *index;
    int workernumber;
    int total_workernumber;
    unsigned long ts_num;
    unsigned long *shared_start_number;
    pthread_barrier_t *wait_summaries_to_compute;
    int *node_counter;
    float *rawfile;
    volatile unsigned char *block_processed;
    volatile unsigned char **group_processed;
    volatile unsigned char *ts_processed;
    fresh_next_ts_group_t *next_ts_group_read_in_block;
    fresh_next_ts_in_group_t **next_ts_read_in_group;
    fresh_next_ts_in_group_t **next_ts_read_in_group_fai;
    volatile unsigned char *all_blocks_processed;
    volatile unsigned char *all_RecBufs_processed;
    volatile unsigned char *block_helper_exist;
    volatile unsigned char **group_helpers_exist;
    unsigned long total_blocks;
    unsigned long total_groups;
    unsigned long read_block_length;
} FRESH_buffer_data;

static volatile int g_fresh_query_id = 0;

// defined in Messi.cpp, same namespace
void calculate_node2_topk_inmemory(isax_index *index, isax_node *node, ts_type *query, ts_type *paa,
                                    pqueue_bsf *pq_bsf, pthread_rwlock_t *lock_queue, float *rawfile);
void calculate_node_DTW2knn_inmemory(isax_index *index, isax_node *node, ts_type *query,
                                      float *uo, float *lo, ts_type *paa, ts_type *paaU, ts_type *paaL,
                                      float bsf, int warpWind, pqueue_bsf *pq_bsf,
                                      pthread_rwlock_t *lock_queue, float *rawfile);

static parallel_first_buffer_layer_lf *initialize_pRecBuf_lf(int initial_buffer_size, int number_of_buffers,
                                                               int max_total_size, isax_index *index)
{
    parallel_first_buffer_layer_lf *fbl = (parallel_first_buffer_layer_lf *)malloc(sizeof(parallel_first_buffer_layer_lf));
    fbl->number_of_buffers = number_of_buffers;
    fbl->initial_buffer_size = initial_buffer_size;
    fbl->max_total_size = max_total_size;
    fbl->current_record_index = 0;
    fbl->current_record = NULL;
    fbl->hard_buffer = NULL;
    fbl->soft_buffers = (parallel_fbl_soft_buffer_lf *)calloc(number_of_buffers, sizeof(parallel_fbl_soft_buffer_lf));
    return fbl;
}

static isax_node *insert_to_pRecBuf_lock_free(parallel_first_buffer_layer_lf *fbl, sax_type *sax,
                                               file_position_type *pos, root_mask_type mask,
                                               isax_index *index, int workernumber, int total_workernumber)
{
    parallel_fbl_soft_buffer_lf *current_buffer = &fbl->soft_buffers[(int)mask];
    current_buffer->mask = mask;

    if (!current_buffer->initialized)
    {
        int *tmp_max_buffer_size = (int *)calloc(total_workernumber, sizeof(int));
        int *tmp_buffer_size = (int *)calloc(total_workernumber, sizeof(int));
        sax_type **tmp_sax_records = (sax_type **)calloc(total_workernumber, sizeof(sax_type *));
        file_position_type **tmp_pos_records = (file_position_type **)calloc(total_workernumber, sizeof(file_position_type *));

        if (!current_buffer->max_buffer_size && !FRESH_CASPTR(&current_buffer->max_buffer_size, NULL, tmp_max_buffer_size))
            free(tmp_max_buffer_size);
        if (!current_buffer->buffer_size && !FRESH_CASPTR(&current_buffer->buffer_size, NULL, tmp_buffer_size))
            free(tmp_buffer_size);
        if (!current_buffer->sax_records && !FRESH_CASPTR(&current_buffer->sax_records, NULL, tmp_sax_records))
            free(tmp_sax_records);
        if (!current_buffer->pos_records && !FRESH_CASPTR(&current_buffer->pos_records, NULL, tmp_pos_records))
            free(tmp_pos_records);

        current_buffer->mask = mask;
        if (!current_buffer->initialized)
            current_buffer->initialized = 1;
    }

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
                index->settings->sax_byte_size * current_buffer->max_buffer_size[workernumber]);
            current_buffer->pos_records[workernumber] = (file_position_type *)realloc(current_buffer->pos_records[workernumber],
                index->settings->position_byte_size * current_buffer->max_buffer_size[workernumber]);
        }
    }

    if (current_buffer->sax_records[workernumber] == NULL || current_buffer->pos_records[workernumber] == NULL)
    {
        fprintf(stderr, "error: Could not allocate memory in FBL.");
        return NULL;
    }

    int current_buffer_number = current_buffer->buffer_size[workernumber];
    file_position_type *filepointer = (file_position_type *)current_buffer->pos_records[workernumber];
    sax_type *saxpointer = (sax_type *)current_buffer->sax_records[workernumber];

    memcpy(&saxpointer[current_buffer_number * index->settings->paa_segments], sax, index->settings->sax_byte_size);
    memcpy(&filepointer[current_buffer_number], pos, index->settings->position_byte_size);

    (current_buffer->buffer_size[workernumber])++;
    return (isax_node *)current_buffer->node;
}

static void store_isax_in_pRecBuf_fresh(FRESH_buffer_data *input_data, isax_index *index,
                                        unsigned long ts_id, sax_type *sax)
{
    file_position_type pos;

    if (sax_from_ts((ts_type *)&input_data->rawfile[ts_id * index->settings->timeseries_size],
                    sax,
                    index->settings->ts_values_per_paa_segment,
                    index->settings->paa_segments,
                    index->settings->sax_alphabet_cardinality,
                    index->settings->sax_bit_cardinality) == SUCCESS)
    {
        pos = (file_position_type)(ts_id * index->settings->timeseries_size);

        root_mask_type first_bit_mask = 0x00;
        CREATE_MASK(first_bit_mask, index, sax);

        insert_to_pRecBuf_lock_free((parallel_first_buffer_layer_lf *)(index->fbl),
                                    sax, &pos, first_bit_mask, index,
                                    input_data->workernumber, input_data->total_workernumber);
    }
    else
    {
        fprintf(stderr, "error: cannot insert record in index, since sax representation failed to be created");
    }
}

static void scan_for_unprocessed_ts_fresh(FRESH_buffer_data *input_data, isax_index *index,
                                          unsigned long ts_start, unsigned long ts_end,
                                          volatile unsigned char *stop, sax_type *sax)
{
    for (unsigned long ts_id = ts_start; ts_id < ts_end && !(*stop); ts_id++)
    {
        if (!input_data->ts_processed[ts_id])
        {
            store_isax_in_pRecBuf_fresh(input_data, index, ts_id, sax);
            if (!input_data->ts_processed[ts_id])
                input_data->ts_processed[ts_id] = 1;
        }
    }
}

static void process_group_fresh(unsigned long ts_group, unsigned long block_num,
                                unsigned long ts_group_start, unsigned long ts_group_end,
                                FRESH_buffer_data *input_data, isax_index *index,
                                char is_helper, sax_type *sax)
{
    unsigned long ts_id, tmp;

    while (!input_data->group_processed[block_num][ts_group])
    {
        if (!input_data->group_helpers_exist[block_num][ts_group])
        {
            tmp = input_data->next_ts_read_in_group[block_num][ts_group].num;
            input_data->next_ts_read_in_group[block_num][ts_group].num = tmp + 1;
        }
        else
        {
            if (!input_data->next_ts_read_in_group_fai[block_num][ts_group].num &&
                input_data->next_ts_read_in_group[block_num][ts_group].num)
            {
                FRESH_CASULONG(&input_data->next_ts_read_in_group_fai[block_num][ts_group].num,
                               0, input_data->next_ts_read_in_group[block_num][ts_group].num);
            }
            tmp = __sync_fetch_and_add(&input_data->next_ts_read_in_group_fai[block_num][ts_group].num, 1);
        }

        ts_id = ts_group_start + tmp;
        if (ts_id >= ts_group_end)
            break;

        if (!input_data->ts_processed[ts_id])
        {
            store_isax_in_pRecBuf_fresh(input_data, index, ts_id, sax);
            if (!input_data->ts_processed[ts_id])
                input_data->ts_processed[ts_id] = 1;
        }
    }

    if ((is_helper || input_data->group_helpers_exist[block_num][ts_group]) &&
        !input_data->group_processed[block_num][ts_group])
    {
        scan_for_unprocessed_ts_fresh(input_data, index, ts_group_start, ts_group_end,
                                      &input_data->group_processed[block_num][ts_group], sax);
    }

    if (!input_data->group_processed[block_num][ts_group])
        input_data->group_processed[block_num][ts_group] = 1;
}

static void scan_for_unprocessed_groups_fresh(unsigned long block_num, FRESH_buffer_data *input_data,
                                              isax_index *index, unsigned long total_groups_in_block,
                                              unsigned long my_ts_start, unsigned long my_ts_end,
                                              volatile unsigned char *stop, sax_type *sax)
{
    for (unsigned long ts_group = 0; ts_group < total_groups_in_block && !*stop; ts_group++)
    {
        unsigned long ts_group_start = my_ts_start + ts_group * FRESH_TS_GROUP_LENGTH;
        unsigned long ts_group_end = (ts_group == total_groups_in_block - 1)
                                     ? my_ts_end
                                     : ts_group_start + FRESH_TS_GROUP_LENGTH;

        if (!input_data->group_helpers_exist[block_num][ts_group])
            input_data->group_helpers_exist[block_num][ts_group] = 1;

        process_group_fresh(ts_group, block_num, ts_group_start, ts_group_end, input_data, index, 1, sax);
    }

    if (!input_data->block_processed[block_num])
        input_data->block_processed[block_num] = 1;
}

static void process_block_fresh(unsigned long block_num, unsigned long total_blocks,
                                unsigned long total_ts_num, FRESH_buffer_data *input_data,
                                isax_index *index, char is_helper, sax_type *sax)
{
    unsigned long my_ts_start = block_num * input_data->read_block_length;
    unsigned long my_ts_end = (block_num == total_blocks - 1)
                              ? total_ts_num
                              : my_ts_start + input_data->read_block_length;

    unsigned long total_groups_in_block = (my_ts_end - my_ts_start) / FRESH_TS_GROUP_LENGTH;
    if (total_groups_in_block * FRESH_TS_GROUP_LENGTH < my_ts_end - my_ts_start)
        total_groups_in_block++;

    unsigned long prev_group_id = 0;
    char helpers_exist = 0;

    while (!input_data->block_processed[block_num])
    {
        unsigned long ts_group;

        if (!input_data->block_helper_exist[block_num])
        {
            ts_group = input_data->next_ts_group_read_in_block[block_num].num;
            input_data->next_ts_group_read_in_block[block_num].num = ts_group + 1;
        }
        else
        {
            ts_group = __sync_fetch_and_add(&input_data->next_ts_group_read_in_block[block_num].num, 1);
        }

        if (ts_group > prev_group_id + 1)
            helpers_exist = 1;

        if (ts_group >= total_groups_in_block)
            break;

        unsigned long ts_group_start = my_ts_start + ts_group * FRESH_TS_GROUP_LENGTH;
        unsigned long ts_group_end = (ts_group == total_groups_in_block - 1)
                                     ? my_ts_end
                                     : ts_group_start + FRESH_TS_GROUP_LENGTH;

        if (is_helper && !input_data->group_helpers_exist[block_num][ts_group])
            input_data->group_helpers_exist[block_num][ts_group] = 1;

        process_group_fresh(ts_group, block_num, ts_group_start, ts_group_end, input_data, index, is_helper, sax);

        prev_group_id = ts_group;
    }

    if ((is_helper || helpers_exist) && !input_data->block_processed[block_num])
        scan_for_unprocessed_groups_fresh(block_num, input_data, index, total_groups_in_block,
                                          my_ts_start, my_ts_end, &input_data->block_processed[block_num], sax);

    if (!input_data->block_processed[block_num])
        input_data->block_processed[block_num] = 1;
}

static void scan_for_unprocessed_blocks_fresh(FRESH_buffer_data *input_data, isax_index *index,
                                              unsigned long total_blocks, sax_type *sax)
{
    unsigned long total_ts_num = input_data->ts_num;
    unsigned long my_id = input_data->workernumber;
    unsigned long start_block_num = (my_id + 1) % total_blocks;
    unsigned long block_num = start_block_num;

    do
    {
        if (*input_data->all_blocks_processed)
            break;

        if (input_data->block_processed[block_num])
        {
            block_num = (block_num + 1) % total_blocks;
            continue;
        }

        if (!input_data->block_helper_exist[block_num])
            input_data->block_helper_exist[block_num] = 1;

        process_block_fresh(block_num, total_blocks, total_ts_num, input_data, index, 1, sax);

        block_num = (block_num + 1) % total_blocks;
    } while (block_num != start_block_num);

    if (!*input_data->all_blocks_processed)
        *input_data->all_blocks_processed = 1;
}

static void *indexCreationWorkerFresh(void *transferdata)
{
    FRESH_buffer_data *input_data = (FRESH_buffer_data *)transferdata;
    isax_index *index = input_data->index;

    unsigned long total_ts_num = input_data->ts_num;
    unsigned long total_blocks = input_data->total_blocks;

    sax_type *sax = (sax_type *)malloc(sizeof(sax_type) * index->settings->paa_segments);

    unsigned long block_num;
    while (!*input_data->all_blocks_processed)
    {
        block_num = __sync_fetch_and_add(input_data->shared_start_number, 1);
        if (block_num >= total_blocks)
            break;

        process_block_fresh(block_num, total_blocks, total_ts_num, input_data, index, 0, sax);
    }

    scan_for_unprocessed_blocks_fresh(input_data, index, total_blocks, sax);

    pthread_barrier_wait(input_data->wait_summaries_to_compute);

    isax_node_record *r = (isax_node_record *)malloc(sizeof(isax_node_record));

    while (1)
    {
        int j = __sync_fetch_and_add(input_data->node_counter, 1);
        if (j >= index->fbl->number_of_buffers)
            break;

        parallel_fbl_soft_buffer_lf *current_fbl_node =
            &((parallel_first_buffer_layer_lf *)(index->fbl))->soft_buffers[j];

        if (!current_fbl_node->initialized)
            continue;

        bool have_record = false;
        for (int k = 0; k < input_data->total_workernumber; k++)
        {
            if (current_fbl_node->buffer_size[k] > 0)
                have_record = true;
            for (int i = 0; i < current_fbl_node->buffer_size[k]; i++)
            {
                r->sax = (sax_type *)&((current_fbl_node->sax_records[k])[i * index->settings->paa_segments]);
                r->position = (file_position_type *)&((file_position_type *)(current_fbl_node->pos_records[k]))[i];
                r->insertion_mode = (insertion_mode)(NO_TMP | PARTIAL);

                if (!current_fbl_node->node)
                {
                    isax_node *root = isax_root_node_init(current_fbl_node->mask, index->settings->initial_leaf_buffer_size);
                    root->is_leaf = 1;
                    if (!FRESH_CASPTR(&current_fbl_node->node, NULL, root))
                        free(root);
                }

                add_record_to_node_inmemory(index, current_fbl_node->node, r, 1);
            }
        }

        if (have_record)
            flush_subtree_leaf_buffers_inmemory(index, current_fbl_node->node);
    }

    free(r);
    free(sax);
    return NULL;
}


static void init_array_list_fresh(fresh_array_list_t *list, int size)
{
    list->Top = (fresh_array_list_node_t *)malloc(sizeof(fresh_array_list_node_t));
    list->Top->data = (fresh_array_element_t *)calloc(size, sizeof(fresh_array_element_t));
    list->Top->num_node = 0;
    list->Top->next = NULL;
    list->element_size = size;
}

static fresh_array_list_node_t *add_array_fresh(fresh_array_list_node_t *last, fresh_array_list_t *list)
{
    fresh_array_list_node_t *tmp = (fresh_array_list_node_t *)malloc(sizeof(fresh_array_list_node_t));
    tmp->data = (fresh_array_element_t *)calloc(list->element_size, sizeof(fresh_array_element_t));
    tmp->num_node = last->num_node + 1;
    tmp->next = last;

    if (!FRESH_CASPTR(&list->Top, last, tmp))
    {
        free(tmp->data);
        free(tmp);
        tmp = list->Top;
        while (tmp->num_node > last->num_node + 1)
            tmp = tmp->next;
    }
    return tmp;
}

static fresh_array_element_t *get_element_at_fresh(unsigned long position, fresh_array_list_t *list)
{
    fresh_array_list_node_t *array_node;
    int array_num_node = position / list->element_size;
    fresh_array_list_node_t *last = list->Top;

    if (array_num_node == last->num_node)
        array_node = last;
    else if (array_num_node > last->num_node)
        array_node = add_array_fresh(last, list);
    else
    {
        array_node = list->Top;
        while (array_node->num_node > array_num_node)
            array_node = array_node->next;
    }
    return &array_node->data[position % list->element_size];
}

static void add_to_array_data_lf(float *paa, isax_node *node, isax_index *index, float bsf,
                                  fresh_array_list_t *array_lists, int *tnumber,
                                  volatile unsigned long *next_queue_data_pos, int n_queues,
                                  const int query_id)
{
    if (node->isax_values == NULL)
        return;

    float distance = minidist_paa_to_isax(paa, node->isax_values,
                                           node->isax_cardinalities,
                                           index->settings->sax_bit_cardinality,
                                           index->settings->sax_alphabet_cardinality,
                                           index->settings->paa_segments,
                                           MINVAL, MAXVAL,
                                           index->settings->mindist_sqrt);

    if (distance < bsf)
    {
        if (node->is_leaf)
        {
            unsigned long queue_data_pos = __sync_fetch_and_add(&next_queue_data_pos[*tnumber], 1);
            fresh_array_element_t *array_elem = get_element_at_fresh(queue_data_pos, &array_lists[*tnumber]);
            array_elem->distance = distance;
            array_elem->node = node;
            *tnumber = (*tnumber + 1) % n_queues;
        }
        else
        {
            if (node->left_child->isax_cardinalities != NULL)
                add_to_array_data_lf(paa, node->left_child, index, bsf, array_lists, tnumber, next_queue_data_pos, n_queues, query_id);
            if (node->right_child->isax_cardinalities != NULL)
                add_to_array_data_lf(paa, node->right_child, index, bsf, array_lists, tnumber, next_queue_data_pos, n_queues, query_id);
        }
    }
}

static int compare_sorted_array_items(const void *a, const void *b)
{
    float dist_a = ((fresh_array_element_t *)a)->distance;
    float dist_b = ((fresh_array_element_t *)b)->distance;
    if (dist_a < dist_b) return -1;
    if (dist_a > dist_b) return 1;
    return 0;
}

static void create_sorted_array_from_data_queue(int pq_id, fresh_array_list_t *array_lists,
                                                volatile unsigned long *next_queue_data_pos,
                                                fresh_sorted_array_t *volatile *sorted_arrays)
{
    fresh_sorted_array_t *local_sa = (fresh_sorted_array_t *)malloc(sizeof(fresh_sorted_array_t));
    unsigned long array_elements = next_queue_data_pos[pq_id];
    local_sa->data = (fresh_array_element_t *)malloc(array_elements * sizeof(fresh_array_element_t));

    size_t j = 0;
    int array_buckets_num = array_lists[pq_id].Top->num_node + 1;
    unsigned long array_elements_traversed = 0;

    // The list is a stack: Top is the newest bucket (highest num_node), .next goes
    // toward older buckets. Positions increase with num_node (bucket i holds positions
    // i*element_size .. (i+1)*element_size-1). We must traverse oldest-first so the
    // array_elements_traversed limit cuts off at the END of the newest bucket, not the
    // middle of the oldest one.
    fresh_array_list_node_t **buckets = (fresh_array_list_node_t **)malloc(
        (size_t)array_buckets_num * sizeof(fresh_array_list_node_t *));
    fresh_array_list_node_t *b = array_lists[pq_id].Top;
    for (int i = array_buckets_num - 1; i >= 0; i--, b = b->next)
        buckets[i] = b;

    for (int i = 0; i < array_buckets_num; i++)
    {
        for (int k = 0; k < array_lists[pq_id].element_size &&
             array_elements_traversed < array_elements && !sorted_arrays[pq_id];
             k++, array_elements_traversed++)
        {
            if (buckets[i]->data[k].node && buckets[i]->data[k].distance >= 0)
            {
                local_sa->data[j].node = buckets[i]->data[k].node;
                local_sa->data[j].distance = buckets[i]->data[k].distance;
                j++;
            }
        }
    }
    free(buckets);

    local_sa->num_elements = j;
    if (!sorted_arrays[pq_id])
        qsort(local_sa->data, j, sizeof(fresh_array_element_t), compare_sorted_array_items);

    if (sorted_arrays[pq_id] || !FRESH_CASPTR(&sorted_arrays[pq_id], NULL, local_sa))
    {
        free(local_sa->data);
        free(local_sa);
    }
}


static int process_sorted_array_element_L2(fresh_array_element_t *n, FRESH_workerdata *wd)
{
    float bsfdistance = wd->pq_bsf->knn[wd->pq_bsf->k - 1];
    if (n->distance > bsfdistance || n->distance > wd->minimum_distance)
        return 0;
    if (n->node->is_leaf && n->distance >= 0)
    {
        calculate_node2_topk_inmemory(wd->index, n->node, wd->ts, wd->paa, wd->pq_bsf, wd->lock_bsf, wd->rawfile);
        n->distance = -1;
    }
    return 1;
}

static int process_sorted_array_element_DTW(fresh_array_element_t *n, FRESH_workerdata *wd)
{
    float bsfdistance = wd->pq_bsf->knn[wd->pq_bsf->k - 1];
    if (n->distance > bsfdistance || n->distance > wd->minimum_distance)
        return 0;
    if (n->node->is_leaf && n->distance >= 0)
    {
        calculate_node_DTW2knn_inmemory(wd->index, n->node, wd->ts, wd->uo, wd->lo,
                                        wd->paa, wd->paaU, wd->paaL, bsfdistance,
                                        wd->warpWind, wd->pq_bsf, wd->lock_bsf, wd->rawfile);
        n->distance = -1;
    }
    return 1;
}

static void help_sorted_array_L2(FRESH_workerdata *wd, fresh_sorted_array_t *sa, int pq_id)
{
    size_t pq_size = sa->num_elements;
    if (pq_size != 0)
    {
        int element_id;
        while ((element_id = (int)__sync_fetch_and_add(&wd->sorted_array_FAI_counter[pq_id], 1)) < (int)pq_size)
        {
            fresh_array_element_t *n = &sa->data[element_id];
            if (!process_sorted_array_element_L2(n, wd))
                break;
        }
        for (int i = 0; i < (int)pq_size && !wd->queue_finished[pq_id]; i++)
        {
            fresh_array_element_t *n = &sa->data[i];
            if (n->distance >= 0 && !process_sorted_array_element_L2(n, wd))
                break;
        }
    }
    if (!wd->queue_finished[pq_id])
        wd->queue_finished[pq_id] = 1;
}

static void help_sorted_array_DTW(FRESH_workerdata *wd, fresh_sorted_array_t *sa, int pq_id)
{
    size_t pq_size = sa->num_elements;
    if (pq_size != 0)
    {
        int element_id;
        while ((element_id = (int)__sync_fetch_and_add(&wd->sorted_array_FAI_counter[pq_id], 1)) < (int)pq_size)
        {
            fresh_array_element_t *n = &sa->data[element_id];
            if (!process_sorted_array_element_DTW(n, wd))
                break;
        }
        for (int i = 0; i < (int)pq_size && !wd->queue_finished[pq_id]; i++)
        {
            fresh_array_element_t *n = &sa->data[i];
            if (n->distance >= 0 && !process_sorted_array_element_DTW(n, wd))
                break;
        }
    }
    if (!wd->queue_finished[pq_id])
        wd->queue_finished[pq_id] = 1;
}

// ── Search workers ─────────────────────────────────────────────────────────────

void *FRESH_topk_search_worker_L2Squared(void *rfdata)
{
    FRESH_workerdata *wd = (FRESH_workerdata *)rfdata;
    isax_index *index = wd->index;
    ts_type *paa = wd->paa;
    float bsfdistance = wd->pq_bsf->knn[wd->pq_bsf->k - 1];
    const int query_id = wd->query_id;
    int tnumber = rand() % wd->n_queues;

    while (1)
    {
        int current_root_node_number = __sync_fetch_and_add(wd->node_counter, 1);
        if (current_root_node_number >= wd->amountnode)
            break;
        add_to_array_data_lf(paa, wd->nodelist[current_root_node_number], index, bsfdistance,
                             wd->array_lists, &tnumber, wd->next_queue_data_pos, wd->n_queues, query_id);
    }

    pthread_barrier_wait(wd->wait_tree_pruning_phase_to_finish);

    int my_pq = wd->workernumber % wd->n_queues;
    if (!wd->sorted_arrays[my_pq] && wd->next_queue_data_pos[my_pq])
        create_sorted_array_from_data_queue(my_pq, wd->array_lists, wd->next_queue_data_pos, wd->sorted_arrays);

    fresh_sorted_array_t *my_array = wd->sorted_arrays[my_pq];
    if (my_array && !wd->queue_finished[my_pq])
        help_sorted_array_L2(wd, my_array, my_pq);

    for (int i = 0; i < wd->n_queues; i++)
    {
        if (!wd->sorted_arrays[i] && wd->next_queue_data_pos[i])
            create_sorted_array_from_data_queue(i, wd->array_lists, wd->next_queue_data_pos, wd->sorted_arrays);
    }

    for (int i = 0; i < wd->n_queues; i++)
    {
        if (wd->queue_finished[i] || !wd->sorted_arrays[i])
            continue;
        if (!wd->helper_queue_exist[i])
            wd->helper_queue_exist[i] = 1;
        help_sorted_array_L2(wd, wd->sorted_arrays[i], i);
    }

    return nullptr;
}

void *FRESH_topk_search_worker_DTW(void *rfdata)
{
    FRESH_workerdata *wd = (FRESH_workerdata *)rfdata;
    isax_index *index = wd->index;
    ts_type *paa = wd->paa;
    float bsfdistance = wd->pq_bsf->knn[wd->pq_bsf->k - 1];
    const int query_id = wd->query_id;
    int tnumber = rand() % wd->n_queues;

    while (1)
    {
        int current_root_node_number = __sync_fetch_and_add(wd->node_counter, 1);
        if (current_root_node_number >= wd->amountnode)
            break;
        add_to_array_data_lf(paa, wd->nodelist[current_root_node_number], index, bsfdistance,
                             wd->array_lists, &tnumber, wd->next_queue_data_pos, wd->n_queues, query_id);
    }

    pthread_barrier_wait(wd->wait_tree_pruning_phase_to_finish);

    int my_pq = wd->workernumber % wd->n_queues;
    if (!wd->sorted_arrays[my_pq] && wd->next_queue_data_pos[my_pq])
        create_sorted_array_from_data_queue(my_pq, wd->array_lists, wd->next_queue_data_pos, wd->sorted_arrays);

    fresh_sorted_array_t *my_array = wd->sorted_arrays[my_pq];
    if (my_array && !wd->queue_finished[my_pq])
        help_sorted_array_DTW(wd, my_array, my_pq);

    for (int i = 0; i < wd->n_queues; i++)
    {
        if (!wd->sorted_arrays[i] && wd->next_queue_data_pos[i])
            create_sorted_array_from_data_queue(i, wd->array_lists, wd->next_queue_data_pos, wd->sorted_arrays);
    }

    for (int i = 0; i < wd->n_queues; i++)
    {
        if (wd->queue_finished[i] || !wd->sorted_arrays[i])
            continue;
        if (!wd->helper_queue_exist[i])
            wd->helper_queue_exist[i] = 1;
        help_sorted_array_DTW(wd, wd->sorted_arrays[i], i);
    }

    return nullptr;
}

// ── Fresh class ────────────────────────────────────────────────────────────────

Fresh::Fresh(DistanceType distance_type)
    : Fresh(distance_type, FreshConfig{})
{
}

Fresh::Fresh(DistanceType distance_type, const FreshConfig &config)
    : SimilaritySearchAlgorithm(distance_type)
{
    this->search_workers = config.search_workers;
    this->index_workers = config.index_workers;
    this->warping_window = config.warping_window;
    this->leaf_size = config.leaf_size;
    this->paa_segments = config.paa_segments;
}

void Fresh::buildIndex(DataSource *data_source)
{
    this->dim = data_source->getDim();
    this->n_database = data_source->getTotalRecords();

    if (this->n_database == 0)
    {
        data_source->reset();
        idx_t count = 0;
        float *dummy = new float[this->dim];
        while (data_source->nextRecord(dummy))
            count++;
        delete[] dummy;
        this->n_database = count;
        data_source->reset();
    }

    data_source->reset();
    const float *raw = data_source->rawPointer();
    if (raw != nullptr)
    {
        this->database = const_cast<float *>(raw);
        this->owns_database = false;
    }
    else
    {
        this->database = new float[this->n_database * this->dim];
        float *record = new float[this->dim];
        idx_t idx = 0;
        while (data_source->nextRecord(record))
        {
            std::copy(record, record + this->dim, this->database + idx * this->dim);
            idx++;
        }
        delete[] record;
        this->owns_database = true;
    }

    this->index_settings = isax_index_settings_init("",
                                                    this->dim,
                                                    this->paa_segments,
                                                    this->sax_cardinality,
                                                    this->leaf_size,
                                                    this->min_leaf_size,
                                                    this->initial_lbl_size,
                                                    this->flush_limit,
                                                    this->initial_fbl_size,
                                                    this->total_loaded_leaves,
                                                    this->tight_bound,
                                                    0,
                                                    1,
                                                    1);

    this->index = isax_index_init_inmemory(this->index_settings);
    isax_index *index = this->index;
    index->sax_file = NULL;

    unsigned long shared_start_number = 0;
    int node_counter = 0;

    int n_buffers = (int)pow(2, index->settings->paa_segments);
    destroy_fbl(index->fbl);
    index->fbl = (first_buffer_layer *)initialize_pRecBuf_lf(
        index->settings->initial_fbl_buffer_size, n_buffers,
        index->settings->max_total_buffer_size + DISK_BUFFER_SIZE * (PROGRESS_CALCULATE_THREAD_NUMBER - 1),
        index);

    unsigned long total_blocks = this->n_database / this->read_block_length;
    if (this->read_block_length * total_blocks < this->n_database)
        total_blocks++;

    unsigned long total_groups = this->read_block_length / FRESH_TS_GROUP_LENGTH;
    if (FRESH_TS_GROUP_LENGTH * total_groups < (unsigned long)this->read_block_length)
        total_groups++;

    volatile unsigned char *block_processed = (volatile unsigned char *)calloc(total_blocks, sizeof(unsigned char));
    volatile unsigned char **group_processed = (volatile unsigned char **)malloc(total_blocks * sizeof(unsigned char *));
    fresh_next_ts_group_t *next_ts_group_read_in_block = (fresh_next_ts_group_t *)calloc(total_blocks, sizeof(fresh_next_ts_group_t));
    fresh_next_ts_in_group_t **next_ts_read_in_group = (fresh_next_ts_in_group_t **)malloc(total_blocks * sizeof(fresh_next_ts_in_group_t *));
    fresh_next_ts_in_group_t **next_ts_read_in_group_fai = (fresh_next_ts_in_group_t **)malloc(total_blocks * sizeof(fresh_next_ts_in_group_t *));
    volatile unsigned char *block_helper_exist = (volatile unsigned char *)calloc(total_blocks, sizeof(unsigned char));
    volatile unsigned char **group_helpers_exist = (volatile unsigned char **)malloc(total_blocks * sizeof(unsigned char *));

    for (unsigned long i = 0; i < total_blocks; i++)
    {
        group_processed[i] = (volatile unsigned char *)calloc(total_groups, sizeof(unsigned char));
        next_ts_read_in_group[i] = (fresh_next_ts_in_group_t *)calloc(total_groups, sizeof(fresh_next_ts_in_group_t));
        next_ts_read_in_group_fai[i] = (fresh_next_ts_in_group_t *)calloc(total_groups, sizeof(fresh_next_ts_in_group_t));
        group_helpers_exist[i] = (volatile unsigned char *)calloc(total_groups, sizeof(unsigned char));
    }

    volatile unsigned char *ts_processed = (volatile unsigned char *)calloc(this->n_database, sizeof(unsigned char));
    volatile unsigned char all_blocks_processed = 0;
    volatile unsigned char all_RecBufs_processed = 0;

    index->sax_cache = (sax_type *)malloc(sizeof(sax_type) * index->settings->paa_segments * this->n_database);

    pthread_barrier_t wait_summaries_to_compute;
    pthread_barrier_init(&wait_summaries_to_compute, NULL, this->index_workers);

    pthread_t threadid[this->index_workers];
    FRESH_buffer_data *input_data = (FRESH_buffer_data *)malloc(sizeof(FRESH_buffer_data) * this->index_workers);

    for (int i = 0; i < this->index_workers; i++)
    {
        input_data[i].index = index;
        input_data[i].workernumber = i;
        input_data[i].total_workernumber = this->index_workers;
        input_data[i].ts_num = this->n_database;
        input_data[i].shared_start_number = &shared_start_number;
        input_data[i].wait_summaries_to_compute = &wait_summaries_to_compute;
        input_data[i].node_counter = &node_counter;
        input_data[i].rawfile = this->database;
        input_data[i].block_processed = block_processed;
        input_data[i].group_processed = group_processed;
        input_data[i].ts_processed = ts_processed;
        input_data[i].next_ts_group_read_in_block = next_ts_group_read_in_block;
        input_data[i].next_ts_read_in_group = next_ts_read_in_group;
        input_data[i].next_ts_read_in_group_fai = next_ts_read_in_group_fai;
        input_data[i].all_blocks_processed = &all_blocks_processed;
        input_data[i].all_RecBufs_processed = &all_RecBufs_processed;
        input_data[i].block_helper_exist = block_helper_exist;
        input_data[i].group_helpers_exist = group_helpers_exist;
        input_data[i].total_blocks = total_blocks;
        input_data[i].total_groups = total_groups;
        input_data[i].read_block_length = this->read_block_length;
    }

    for (int i = 0; i < this->index_workers; i++)
        pthread_create(&threadid[i], NULL, indexCreationWorkerFresh, (void *)&input_data[i]);
    for (int i = 0; i < this->index_workers; i++)
        pthread_join(threadid[i], NULL);

    // Build index->first_node linked list from FBL so that search functions can traverse all roots.
    {
        parallel_first_buffer_layer_lf *fbl = (parallel_first_buffer_layer_lf *)index->fbl;
        for (int i = 0; i < fbl->number_of_buffers; i++)
        {
            if (!fbl->soft_buffers[i].initialized || !fbl->soft_buffers[i].node)
                continue;
            isax_node *node = fbl->soft_buffers[i].node;
            node->next = index->first_node;
            node->previous = NULL;
            if (index->first_node)
                index->first_node->previous = node;
            index->first_node = node;
            __sync_fetch_and_add(&(index->root_nodes), 1);
        }
    }

    __sync_fetch_and_add(&(index->total_records), this->n_database);
    index->sax_cache_size = index->total_records;
    fprintf(stderr, ">>> Finished indexing\n");

    pthread_barrier_destroy(&wait_summaries_to_compute);

    for (unsigned long i = 0; i < total_blocks; i++)
    {
        free((void *)group_processed[i]);
        free(next_ts_read_in_group[i]);
        free(next_ts_read_in_group_fai[i]);
        free((void *)group_helpers_exist[i]);
    }
    free((void *)block_processed);
    free(group_processed);
    free(next_ts_group_read_in_block);
    free(next_ts_read_in_group);
    free(next_ts_read_in_group_fai);
    free((void *)block_helper_exist);
    free(group_helpers_exist);
    free((void *)ts_processed);
    free(input_data);
}

pqueue_bsf Fresh::FRESH_search_topk_L2Squared(ts_type *ts, ts_type *paa, node_list *nodelist, idx_t k)
{
    pqueue_bsf *pq_bsf = pqueue_bsf_init(k);

    approximate_topk_inmemory(ts, paa, index, pq_bsf, this->database);
    this->minimum_distance = pq_bsf->knn[k - 1];

    if (this->minimum_distance == FLT_MAX || min_checked_leaves > 1)
    {
        refine_topk_answer_inmemory(ts, paa, index, pq_bsf, this->minimum_distance, this->min_checked_leaves, this->database);
        this->minimum_distance = pq_bsf->knn[k - 1];
    }

    int n_queues = (this->search_workers == 1) ? 1 : this->search_workers / 2;
    int query_id = __sync_add_and_fetch(&g_fresh_query_id, 1);

    fresh_array_list_t *array_lists = (fresh_array_list_t *)malloc(sizeof(fresh_array_list_t) * n_queues);
    for (int i = 0; i < n_queues; i++)
        init_array_list_fresh(&array_lists[i], index->settings->root_nodes_size / n_queues);

    fresh_sorted_array_t *volatile *sorted_arrays = (fresh_sorted_array_t *volatile *)calloc(n_queues, sizeof(fresh_sorted_array_t *));
    volatile unsigned long *next_queue_data_pos = (volatile unsigned long *)calloc(n_queues, sizeof(unsigned long));
    volatile unsigned char *queue_finished = (volatile unsigned char *)calloc(n_queues, sizeof(unsigned char));
    volatile unsigned char *helper_queue_exist = (volatile unsigned char *)calloc(n_queues, sizeof(unsigned char));
    volatile unsigned long *sorted_array_FAI_counter = (volatile unsigned long *)calloc(n_queues, sizeof(unsigned long));

    volatile int node_counter = 0;

    pthread_barrier_t wait_tree_pruning_phase_to_finish;
    pthread_barrier_init(&wait_tree_pruning_phase_to_finish, NULL, this->search_workers);

    pthread_rwlock_t lock_bsf = PTHREAD_RWLOCK_INITIALIZER;
    pthread_t threadid[this->search_workers];
    FRESH_workerdata workerdata[this->search_workers];

    for (int i = 0; i < this->search_workers; i++)
    {
        workerdata[i].paa = paa;
        workerdata[i].ts = ts;
        workerdata[i].index = index;
        workerdata[i].minimum_distance = this->minimum_distance;
        workerdata[i].pq_bsf = pq_bsf;
        workerdata[i].lock_bsf = &lock_bsf;
        workerdata[i].nodelist = nodelist->nlist;
        workerdata[i].amountnode = nodelist->node_amount;
        workerdata[i].node_counter = &node_counter;
        workerdata[i].array_lists = array_lists;
        workerdata[i].sorted_arrays = sorted_arrays;
        workerdata[i].next_queue_data_pos = next_queue_data_pos;
        workerdata[i].queue_finished = queue_finished;
        workerdata[i].helper_queue_exist = helper_queue_exist;
        workerdata[i].sorted_array_FAI_counter = sorted_array_FAI_counter;
        workerdata[i].sorted_array_counter = nullptr;
        workerdata[i].fai_queue_counters = nullptr;
        workerdata[i].queue_bsf_distance = nullptr;
        workerdata[i].wait_tree_pruning_phase_to_finish = &wait_tree_pruning_phase_to_finish;
        workerdata[i].workernumber = i;
        workerdata[i].n_queues = n_queues;
        workerdata[i].query_id = query_id;
        workerdata[i].rawfile = this->database;
    }

    for (int i = 0; i < this->search_workers; i++)
        pthread_create(&threadid[i], NULL, FRESH_topk_search_worker_L2Squared, (void *)&workerdata[i]);
    for (int i = 0; i < this->search_workers; i++)
        pthread_join(threadid[i], NULL);

    pthread_barrier_destroy(&wait_tree_pruning_phase_to_finish);

    this->minimum_distance = pq_bsf->knn[k - 1];

    for (int i = 0; i < n_queues; i++)
    {
        if (sorted_arrays[i])
        {
            free(sorted_arrays[i]->data);
            free(sorted_arrays[i]);
        }
        fresh_array_list_node_t *node = array_lists[i].Top;
        while (node)
        {
            fresh_array_list_node_t *next = node->next;
            free(node->data);
            free(node);
            node = next;
        }
    }
    free(array_lists);
    free((void *)sorted_arrays);
    free((void *)next_queue_data_pos);
    free((void *)queue_finished);
    free((void *)helper_queue_exist);
    free((void *)sorted_array_FAI_counter);

    pqueue_bsf result = *pq_bsf;
    free(pq_bsf);
    return result;
}

pqueue_bsf Fresh::FRESH_search_topk_DTW(ts_type *ts, node_list *nodelist, idx_t k)
{
    isax_index *index = this->index;
    int warpWind = this->warping_window;
    float *rawfile = this->database;

    ts_type *paa = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);
    paa_from_ts(ts, paa, index->settings->paa_segments, index->settings->ts_values_per_paa_segment);

    pqueue_bsf *pq_bsf = pqueue_bsf_init(k);
    approximate_DTWtopk_inmemory(ts, paa, index, warpWind, pq_bsf, rawfile);
    this->minimum_distance = pq_bsf->knn[k - 1];

    ts_type *upperLemire = (ts_type *)malloc(sizeof(ts_type) * index->settings->timeseries_size);
    ts_type *lowerLemire = (ts_type *)malloc(sizeof(ts_type) * index->settings->timeseries_size);
    ts_type *paaU = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);
    ts_type *paaL = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);

    lower_upper_lemire(ts, index->settings->timeseries_size, warpWind, lowerLemire, upperLemire);
    paa_from_ts(upperLemire, paaU, index->settings->paa_segments, index->settings->ts_values_per_paa_segment);
    paa_from_ts(lowerLemire, paaL, index->settings->paa_segments, index->settings->ts_values_per_paa_segment);

    int n_queues = (this->search_workers == 1) ? 1 : this->search_workers / 2;
    int query_id = __sync_add_and_fetch(&g_fresh_query_id, 1);

    fresh_array_list_t *array_lists = (fresh_array_list_t *)malloc(sizeof(fresh_array_list_t) * n_queues);
    for (int i = 0; i < n_queues; i++)
        init_array_list_fresh(&array_lists[i], index->settings->root_nodes_size / n_queues);

    fresh_sorted_array_t *volatile *sorted_arrays = (fresh_sorted_array_t *volatile *)calloc(n_queues, sizeof(fresh_sorted_array_t *));
    volatile unsigned long *next_queue_data_pos = (volatile unsigned long *)calloc(n_queues, sizeof(unsigned long));
    volatile unsigned char *queue_finished = (volatile unsigned char *)calloc(n_queues, sizeof(unsigned char));
    volatile unsigned char *helper_queue_exist = (volatile unsigned char *)calloc(n_queues, sizeof(unsigned char));
    volatile unsigned long *sorted_array_FAI_counter = (volatile unsigned long *)calloc(n_queues, sizeof(unsigned long));

    volatile int node_counter = 0;

    pthread_barrier_t wait_tree_pruning_phase_to_finish;
    pthread_barrier_init(&wait_tree_pruning_phase_to_finish, NULL, this->search_workers);

    pthread_rwlock_t lock_bsf = PTHREAD_RWLOCK_INITIALIZER;
    pthread_t threadid[this->search_workers];
    FRESH_workerdata workerdata[this->search_workers];

    for (int i = 0; i < this->search_workers; i++)
    {
        workerdata[i].paa = paa;
        workerdata[i].paaU = paaU;
        workerdata[i].paaL = paaL;
        workerdata[i].ts = ts;
        workerdata[i].uo = upperLemire;
        workerdata[i].lo = lowerLemire;
        workerdata[i].index = index;
        workerdata[i].minimum_distance = this->minimum_distance;
        workerdata[i].pq_bsf = pq_bsf;
        workerdata[i].lock_bsf = &lock_bsf;
        workerdata[i].nodelist = nodelist->nlist;
        workerdata[i].amountnode = nodelist->node_amount;
        workerdata[i].node_counter = &node_counter;
        workerdata[i].array_lists = array_lists;
        workerdata[i].sorted_arrays = sorted_arrays;
        workerdata[i].next_queue_data_pos = next_queue_data_pos;
        workerdata[i].queue_finished = queue_finished;
        workerdata[i].helper_queue_exist = helper_queue_exist;
        workerdata[i].sorted_array_FAI_counter = sorted_array_FAI_counter;
        workerdata[i].sorted_array_counter = nullptr;
        workerdata[i].fai_queue_counters = nullptr;
        workerdata[i].queue_bsf_distance = nullptr;
        workerdata[i].wait_tree_pruning_phase_to_finish = &wait_tree_pruning_phase_to_finish;
        workerdata[i].workernumber = i;
        workerdata[i].n_queues = n_queues;
        workerdata[i].warpWind = warpWind;
        workerdata[i].query_id = query_id;
        workerdata[i].rawfile = rawfile;
    }

    for (int i = 0; i < this->search_workers; i++)
        pthread_create(&threadid[i], NULL, FRESH_topk_search_worker_DTW, (void *)&workerdata[i]);
    for (int i = 0; i < this->search_workers; i++)
        pthread_join(threadid[i], NULL);

    pthread_barrier_destroy(&wait_tree_pruning_phase_to_finish);

    this->minimum_distance = pq_bsf->knn[k - 1];

    for (int i = 0; i < n_queues; i++)
    {
        if (sorted_arrays[i])
        {
            free(sorted_arrays[i]->data);
            free(sorted_arrays[i]);
        }
        fresh_array_list_node_t *node = array_lists[i].Top;
        while (node)
        {
            fresh_array_list_node_t *next = node->next;
            free(node->data);
            free(node);
            node = next;
        }
    }
    free(array_lists);
    free((void *)sorted_arrays);
    free((void *)next_queue_data_pos);
    free((void *)queue_finished);
    free((void *)helper_queue_exist);
    free((void *)sorted_array_FAI_counter);
    free(upperLemire);
    free(lowerLemire);
    free(paaU);
    free(paaL);
    free(paa);

    pqueue_bsf result = *pq_bsf;
    free(pq_bsf);
    return result;
}

void Fresh::searchIndexL2Squared(const float *query, idx_t n_query, idx_t k, idx_t *I, float *D)
{
    ts_type *paa = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);

    node_list nodelist;
    nodelist.nlist = (isax_node **)malloc(sizeof(isax_node *) * pow(2, index->settings->paa_segments));
    nodelist.node_amount = 0;
    isax_node *current_root_node = index->first_node;
    while (current_root_node != NULL)
    {
        nodelist.nlist[nodelist.node_amount] = current_root_node;
        current_root_node = current_root_node->next;
        nodelist.node_amount++;
    }

    for (idx_t q_loaded = 0; q_loaded < n_query; q_loaded++)
    {
        const float *ts = query + q_loaded * this->dim;
        paa_from_ts(ts, paa, index->settings->paa_segments, index->settings->ts_values_per_paa_segment);

        pqueue_bsf result = FRESH_search_topk_L2Squared((float *)ts, paa, &nodelist, k);

        std::vector<std::pair<float, long>> pairs;
        pairs.reserve(static_cast<size_t>(k));
        for (idx_t ik = 0; ik < k; ik++)
        {
            if (result.position[ik] >= 0 && result.knn[ik] < FLT_MAX * 0.99f)
                pairs.emplace_back(result.knn[ik], result.position[ik]);
        }
        std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        });
        std::unordered_set<long> seen_pos;
        std::vector<std::pair<float, long>> uniq;
        uniq.reserve(pairs.size());
        for (const auto &p : pairs)
        {
            if (seen_pos.insert(p.second).second)
                uniq.push_back(p);
        }
        long last_pos = 0;
        float last_dist = 0.0f;
        for (idx_t ik = 0; ik < k; ik++)
        {
            if (ik < static_cast<idx_t>(uniq.size()))
            {
                last_dist = uniq[static_cast<size_t>(ik)].first;
                last_pos = uniq[static_cast<size_t>(ik)].second;
            }
            I[q_loaded * k + ik] = static_cast<idx_t>(last_pos >= 0 ? last_pos : 0);
            D[q_loaded * k + ik] = last_dist;
        }

        free(result.position);
        free(result.knn);
        free(result.node);
    }

    free(nodelist.nlist);
    free(paa);
    fprintf(stderr, ">>> Finished querying.\n");
    fflush(stdout);
}

void Fresh::searchIndexDTW(const float *query, idx_t n_query, idx_t k, idx_t *I, float *D)
{
    isax_index *index = this->index;

    node_list nodelist;
    nodelist.nlist = (isax_node **)malloc(sizeof(isax_node *) * pow(2, index->settings->paa_segments));
    nodelist.node_amount = 0;
    isax_node *current_root_node = index->first_node;
    while (current_root_node != NULL)
    {
        nodelist.nlist[nodelist.node_amount] = current_root_node;
        current_root_node = current_root_node->next;
        nodelist.node_amount++;
    }

    for (idx_t q_loaded = 0; q_loaded < n_query; q_loaded++)
    {
        float *ts = (float *)&query[q_loaded * this->dim];
        pqueue_bsf result = FRESH_search_topk_DTW((float *)ts, &nodelist, k);

        std::vector<std::pair<float, long>> pairs;
        pairs.reserve(static_cast<size_t>(k));
        for (idx_t ik = 0; ik < k; ik++)
        {
            if (result.position[ik] >= 0 && result.knn[ik] < FLT_MAX * 0.99f)
                pairs.emplace_back(result.knn[ik], result.position[ik]);
        }
        std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        });
        std::unordered_set<long> seen_pos;
        std::vector<std::pair<float, long>> uniq;
        uniq.reserve(pairs.size());
        for (const auto &p : pairs)
        {
            if (seen_pos.insert(p.second).second)
                uniq.push_back(p);
        }
        long last_pos = 0;
        float last_dist = 0.0f;
        for (idx_t ik = 0; ik < k; ik++)
        {
            if (ik < static_cast<idx_t>(uniq.size()))
            {
                last_dist = uniq[static_cast<size_t>(ik)].first;
                last_pos = uniq[static_cast<size_t>(ik)].second;
            }
            I[q_loaded * k + ik] = static_cast<idx_t>(last_pos >= 0 ? last_pos : 0);
            D[q_loaded * k + ik] = last_dist;
        }

        free(result.position);
        free(result.knn);
        free(result.node);
    }

    free(nodelist.nlist);
    fprintf(stderr, ">>> Finished querying.\n");
}

void Fresh::searchIndex(const float *query, idx_t n_query, idx_t k, idx_t *I, float *D)
{
    if (this->distance_type == DistanceType::L2_SQUARED)
        searchIndexL2Squared(query, n_query, k, I, D);
    else if (this->distance_type == DistanceType::DTW)
        searchIndexDTW(query, n_query, k, I, D);
    else
    {
        fprintf(stderr, "Error: Unsupported distance type for Fresh index.\n");
        exit(1);
    }
}

Fresh::~Fresh()
{
    if (owns_database && database != nullptr)
        delete[] database;

    if (index != nullptr)
    {
        if (index->sax_cache != nullptr)
            free(index->sax_cache);
        if (index->answer != nullptr)
            free(index->answer);
        if (index->fbl != nullptr)
        {
            parallel_first_buffer_layer_lf *fbl = (parallel_first_buffer_layer_lf *)index->fbl;
            for (int i = 0; i < fbl->number_of_buffers; i++)
            {
                parallel_fbl_soft_buffer_lf *sb = &fbl->soft_buffers[i];
                if (!sb->initialized) continue;
                for (int w = 0; w < this->index_workers; w++)
                {
                    if (sb->sax_records && sb->sax_records[w]) free(sb->sax_records[w]);
                    if (sb->pos_records && sb->pos_records[w]) free(sb->pos_records[w]);
                }
                free(sb->max_buffer_size);
                free(sb->buffer_size);
                free(sb->sax_records);
                free(sb->pos_records);
            }
            free(fbl->soft_buffers);
            free(fbl);
        }
        if (index->sax_file != nullptr)
            fclose(index->sax_file);
        free(index);
    }

    if (index_settings != nullptr)
    {
        if (index_settings->bit_masks != nullptr)
            free(index_settings->bit_masks);
        if (index_settings->max_sax_cardinalities != nullptr)
            free(index_settings->max_sax_cardinalities);
        free(index_settings);
    }
}

} // namespace daisy


