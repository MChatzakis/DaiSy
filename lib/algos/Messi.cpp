#include "Messi.hpp"

#include <cmath>

#include "../isax/iSAXPqueue.hpp"
#include "../isax/iSAXIndex.hpp"
#include "../isax/iSAXSearch.hpp"

namespace diNoLib
{

    Messi::Messi(DistanceType distance_type)
        : SimilaritySearchAlgorithm(distance_type)
    {
    }

    void *indexCreationWorker(void *transferdata)
    {
        sax_type *sax = (sax_type *)malloc(sizeof(sax_type) * ((buffer_data_inmemory *)transferdata)->index->settings->paa_segments);

        unsigned long start_number;
        unsigned long stop_number = ((buffer_data_inmemory *)transferdata)->stop_number;
        unsigned long roundfinishednumber = stop_number;
        file_position_type *pos = (file_position_type *)malloc(sizeof(file_position_type));
        isax_index *index = ((buffer_data_inmemory *)transferdata)->index;
        ts_type *ts = (ts_type *)malloc(sizeof(ts_type) * index->settings->timeseries_size);
        int paa_segments = ((buffer_data_inmemory *)transferdata)->index->settings->paa_segments;

        int read_block_length = ((buffer_data_inmemory *)transferdata)->read_block_length; // this->read_block_length;

        unsigned long i = 0;
        float *raw_file = ((buffer_data_inmemory *)transferdata)->ts;
        while (1)
        {
            start_number = __sync_fetch_and_add(((buffer_data_inmemory *)transferdata)->shared_start_number, read_block_length);
            if (start_number > stop_number)
            {
                break;
            }
            else if (start_number > stop_number - read_block_length)
            {
                roundfinishednumber = stop_number;
            }
            else
            {
                roundfinishednumber = min(stop_number, (start_number + read_block_length));
            }
            for (i = start_number; i < roundfinishednumber; i++)
            {
                memcpy(ts, &(raw_file[i * index->settings->timeseries_size]), sizeof(ts_type) * index->settings->timeseries_size);
                if (sax_from_ts(ts, sax, index->settings->ts_values_per_paa_segment,
                                index->settings->paa_segments, index->settings->sax_alphabet_cardinality,
                                index->settings->sax_bit_cardinality) == SUCCESS)
                {
                    *pos = (file_position_type)(i * index->settings->timeseries_size);
                    memcpy(&(index->sax_cache[i * index->settings->paa_segments]), sax, sizeof(sax_type) * index->settings->paa_segments);

                    isax_pRecBuf_index_insert_inmemory(index, sax, pos, ((buffer_data_inmemory *)transferdata)->lock_firstnode, ((buffer_data_inmemory *)transferdata)->workernumber, ((buffer_data_inmemory *)transferdata)->total_workernumber);
                }
                else
                {
                    fprintf(stderr, "error: cannot insert record in index, since sax representation\
                    failed to be created");
                }
            }
        }

        free(pos);
        free(sax);
        free(ts);

        pthread_barrier_wait(((buffer_data_inmemory *)transferdata)->lock_barrier1);
        pthread_barrier_wait(((buffer_data_inmemory *)transferdata)->lock_barrier2);
        bool have_record = false;
        int j;
        isax_node_record *r = (isax_node_record *)malloc(sizeof(isax_node_record));

        while (1)
        {

            j = __sync_fetch_and_add(((buffer_data_inmemory *)transferdata)->node_counter, 1);

            if (j >= index->fbl->number_of_buffers)
            {
                break;
            }
            parallel_fbl_soft_buffer *current_fbl_node = &((parallel_first_buffer_layer *)(index->fbl))->soft_buffers[j];
            if (!current_fbl_node->initialized)
            {
                continue;
            }

            int i;
            have_record = false;
            for (int k = 0; k < ((buffer_data_inmemory *)transferdata)->total_workernumber; k++)
            {
                if (current_fbl_node->buffer_size[k] > 0)
                    have_record = true;
                for (i = 0; i < current_fbl_node->buffer_size[k]; i++)
                {
                    r->sax = (sax_type *)&(((current_fbl_node->sax_records[k]))[i * index->settings->paa_segments]);
                    r->position = (file_position_type *)&((file_position_type *)(current_fbl_node->pos_records[k]))[i];
                    r->insertion_mode = (insertion_mode)(NO_TMP | PARTIAL);

                    add_record_to_node(index, current_fbl_node->node, r, 1);
                }
            }
            if (have_record)
            {
                flush_subtree_leaf_buffers_inmemory(index, current_fbl_node->node);
            }
        }
        free(r);

        return NULL;
    }

    void Messi::buildIndex(const float *database, const idx_t n_database, const idx_t dim)
    {
        this->database = new float[n_database * dim];
        std::copy(database, database + n_database * dim, this->database);
        this->n_database = n_database;
        this->dim = dim;

        this->index_settings = isax_index_settings_init("",                        // INDEX DIRECTORY
                                                        this->dim,                 // TIME SERIES SIZE
                                                        this->paa_segments,        // PAA SEGMENTS
                                                        this->sax_cardinality,     // SAX CARDINALITY IN BITS
                                                        this->leaf_size,           // LEAF SIZE
                                                        this->min_leaf_size,       // MIN LEAF SIZE
                                                        this->initial_lbl_size,    // INITIAL LEAF BUFFER SIZE
                                                        this->flush_limit,         // FLUSH LIMIT
                                                        this->initial_fbl_size,    // INITIAL FBL BUFFER SIZE
                                                        this->total_loaded_leaves, // Leaves to load at each fetch
                                                        this->tight_bound,         // Tightness of leaf bounds
                                                        0,                         // aggressive check
                                                        1,
                                                        1); // new index

        this->index = isax_index_init_inmemory(this->index_settings);
        ;
        isax_index *index = this->index;

        index->sax_file = NULL;
        long int ts_loaded = 0;
        unsigned long shared_start_number = 0;
        int i, j;
        int node_counter = 0;
        pthread_t threadid[this->index_workers];
        buffer_data_inmemory *input_data = (buffer_data_inmemory *)malloc(sizeof(buffer_data_inmemory) * (this->index_workers));
        index->sax_cache = (sax_type *)malloc(sizeof(sax_type) * index->settings->paa_segments * this->n_database);
        pthread_barrier_t lock_barrier1, lock_barrier2;
        pthread_barrier_init(&lock_barrier1, NULL, this->index_workers + 1);
        pthread_barrier_init(&lock_barrier2, NULL, this->index_workers + 1);

        pthread_mutex_t lock_record = PTHREAD_MUTEX_INITIALIZER, lockfbl = PTHREAD_MUTEX_INITIALIZER, lock_index = PTHREAD_MUTEX_INITIALIZER, lock_firstnode = PTHREAD_MUTEX_INITIALIZER, lock_disk = PTHREAD_MUTEX_INITIALIZER;

        destroy_fbl(index->fbl);
        index->fbl = (first_buffer_layer *)initialize_pRecBuf(index->settings->initial_fbl_buffer_size, pow(2, index->settings->paa_segments), index->settings->max_total_buffer_size + DISK_BUFFER_SIZE * (PROGRESS_CALCULATE_THREAD_NUMBER - 1), index);

        int nodeid[index->fbl->number_of_buffers];
        int nodesize[index->fbl->number_of_buffers];

        for (i = 0; i < this->index_workers; i++)
        {
            input_data[i].index = index;
            input_data[i].lock_fbl = &lockfbl;
            input_data[i].lock_record = &lock_record;
            input_data[i].lock_firstnode = &lock_firstnode;
            input_data[i].lock_index = &lock_index;
            input_data[i].ts = this->database;
            input_data[i].lock_disk = &lock_disk;
            input_data[i].workernumber = i;
            input_data[i].total_workernumber = this->index_workers;
            input_data[i].start_number = i * (this->n_database / this->index_workers);
            input_data[i].shared_start_number = &shared_start_number;
            input_data[i].stop_number = this->n_database;
            input_data[i].node_counter = &node_counter;
            input_data[i].lock_barrier1 = &lock_barrier1;
            input_data[i].lock_barrier2 = &lock_barrier2;
            input_data[i].nodeid = nodeid;

            input_data[i].read_block_length = this->read_block_length;
        }

        for (i = 0; i < this->index_workers; i++)
        {
            pthread_create(&(threadid[i]), NULL, indexCreationWorker, (void *)&(input_data[i]));
        }

        pthread_barrier_wait(&lock_barrier1);
        pthread_barrier_wait(&lock_barrier2);

        for (i = 0; i < this->index_workers; i++)
        {
            pthread_join(threadid[i], NULL);
        }
        __sync_fetch_and_add(&(index->total_records), this->n_database);
        index->sax_cache_size = index->total_records;
        fprintf(stderr, ">>> Finished indexing\n");
        free(input_data);
    }

    void Messi::searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        ts_type *paa = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);

        node_list nodelist;
        nodelist.nlist = (isax_node **)malloc(sizeof(isax_node *) * pow(2, index->settings->paa_segments));
        nodelist.node_amount = 0;
        isax_node *current_root_node = index->first_node;
        while (1)
        {
            if (current_root_node != NULL)
            {
                nodelist.nlist[nodelist.node_amount] = current_root_node;
                current_root_node = current_root_node->next;
                nodelist.node_amount++;
            }
            else
            {
                break;
            }
        }

        for (idx_t q_loaded = 0; q_loaded < n_query; q_loaded++)
        {
            const float *ts = query + q_loaded * this->dim;

            //  Parse ts and make PAA representation
            paa_from_ts((float *)ts, paa, index->settings->paa_segments, index->settings->ts_values_per_paa_segment); // check this catastrophic cast and maybe fix?

            pqueue_bsf result = MESSI_search_topk((float *)ts, paa, &nodelist, k); // check cast again..
            for (idx_t ik = 0; ik < k; ik++)                                       // check again if this is correct!
            {
                I[q_loaded * k + ik] = result.position[ik];
                D[q_loaded * k + ik] = result.knn[ik];
            }
        }

        free(paa);
        fprintf(stderr, ">>> Finished querying.\n");
    }

    pqueue_bsf Messi::MESSI_search_topk(ts_type *ts, ts_type *paa, node_list *nodelist, idx_t k)
    {
        pqueue_bsf *pq_bsf = pqueue_bsf_init(k);

        approximate_topk_inmemory(ts, paa, index, pq_bsf, this->database);

        int tight_bound = index->settings->tight_bound;
        int aggressive_check = index->settings->aggressive_check;
        int node_counter = 0;

        if (pq_bsf->knn[k - 1] == FLT_MAX || min_checked_leaves > 1)
        {
            refine_topk_answer_inmemory(ts, paa, index, pq_bsf, this->minimum_distance, this->min_checked_leaves, this->database);
        }
        pqueue_t **allpq = (pqueue_t **)malloc(sizeof(pqueue_t *) * this->n_pqueue);

        pthread_mutex_t ququelock[this->n_pqueue];
        int queuelabel[this->n_pqueue];
        // Insert all root nodes in heap.
        isax_node *current_root_node = index->first_node;

        pthread_t threadid[this->search_workers];
        MESSI_workerdata workerdata[this->search_workers];
        pthread_mutex_t lock_queue = PTHREAD_MUTEX_INITIALIZER, lock_current_root_node = PTHREAD_MUTEX_INITIALIZER;
        pthread_rwlock_t lock_bsf = PTHREAD_RWLOCK_INITIALIZER;
        pthread_barrier_t lock_barrier;
        pthread_barrier_init(&lock_barrier, NULL, this->search_workers);

        for (int i = 0; i < this->n_pqueue; i++)
        {
            allpq[i] = pqueue_init(index->settings->root_nodes_size / this->n_pqueue, cmp_pri, get_pri, set_pri, get_pos, set_pos);
            pthread_mutex_init(&ququelock[i], NULL);
            queuelabel[i] = 1;
        }

        for (int i = 0; i < this->search_workers; i++)
        {
            workerdata[i].paa = paa;
            workerdata[i].ts = ts;
            workerdata[i].lock_queue = &lock_queue;
            workerdata[i].lock_current_root_node = &lock_current_root_node;
            workerdata[i].lock_bsf = &lock_bsf;
            workerdata[i].nodelist = nodelist->nlist;
            workerdata[i].amountnode = nodelist->node_amount;
            workerdata[i].index = index;
            workerdata[i].minimum_distance = minimum_distance;
            workerdata[i].node_counter = &node_counter;
            workerdata[i].pq = allpq[i];
            workerdata[i].lock_barrier = &lock_barrier;
            workerdata[i].alllock = ququelock;
            workerdata[i].allqueuelabel = queuelabel;
            workerdata[i].allpq = allpq;
            workerdata[i].startqueuenumber = i % this->n_pqueue;
            workerdata[i].pq_bsf = pq_bsf;
        }

        query_result *n;

        for (int i = 0; i < this->search_workers; i++)
        {
            pthread_create(&(threadid[i]), NULL, /*exact_topk_worker_inmemory_hybridpqueue*/NULL, (void *)&(workerdata[i])); // Continue here tomorrow!
        }
        for (int i = 0; i < this->search_workers; i++)
        {
            pthread_join(threadid[i], NULL);
        }

        // Free the nodes that where not popped.
        // Free the priority queue.
        pthread_barrier_destroy(&lock_barrier);

        // pqueue_free(pq);
        for (int i = 0; i < this->n_pqueue; i++)
        {
            pqueue_free(allpq[i]);
        }
        free(allpq);

        // free(rfdata);
        return *pq_bsf;

        // Free the nodes that where not popped.
    }

    Messi::~Messi()
    {
        delete[] database;

        // todo add free funcs here:
    }
}