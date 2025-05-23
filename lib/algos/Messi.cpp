#include "Messi.hpp"

#include <cmath>

namespace diNoLib
{

    Messi::Messi(DistanceType distance_type)
        : SimilaritySearchAlgorithm(distance_type)
    {
    }

    void* indexCreationWorker(void *transferdata)
    {
        sax_type *sax = (sax_type *)malloc(sizeof(sax_type) * ((buffer_data_inmemory *)transferdata)->index->settings->paa_segments);

        unsigned long start_number;
        unsigned long stop_number = ((buffer_data_inmemory *)transferdata)->stop_number;
        unsigned long roundfinishednumber = stop_number;
        file_position_type *pos = (file_position_type *)malloc(sizeof(file_position_type));
        isax_index *index = ((buffer_data_inmemory *)transferdata)->index;
        ts_type *ts = (ts_type *) malloc(sizeof(ts_type) * index->settings->timeseries_size);
        int paa_segments = ((buffer_data_inmemory *)transferdata)->index->settings->paa_segments;

        int read_block_length = ((buffer_data_inmemory *)transferdata)->read_block_length;//this->read_block_length;

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
        isax_node_record *r = (isax_node_record *) malloc(sizeof(isax_node_record));
        
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
                    r->insertion_mode = (insertion_mode) (NO_TMP | PARTIAL);
                    
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
    }

    Messi::~Messi()
    {
        delete[] database;

        //todo add free funcs here:
    }
}