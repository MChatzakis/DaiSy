#include "Sing.hpp"
#include "../isax/iSAXIndex.hpp"
#include "../isax/iSAXTypes.hpp"
#include "../isax/SAX.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <pthread.h>

namespace diNoLib
{

    // Funzioni non presenti nel codebase: dichiarate e chiamate, da implementare
    first_buffer_layer2 *initialize_simrec(int initial_buffer_size, int number_of_buffers,
                                           int max_total_buffers_size, isax_index *index);
    void *index_creation_worker2_inmemory(void *arg);

    first_buffer_layer2 *initialize_simrec(int initial_buffer_size, int number_of_buffers,
                                           int max_total_buffers_size, isax_index *index)
    {
        first_buffer_layer2 *fbl = (first_buffer_layer2 *)malloc(sizeof(first_buffer_layer2));

        fbl->max_total_size = max_total_buffers_size;
        fbl->initial_buffer_size = initial_buffer_size;
        fbl->number_of_buffers = number_of_buffers;

        // Allocate a big chunk of memory to store sax data and positions (optional, commented in original)
        // long long hard_buffer_size = (long long)(index->settings->sax_byte_size + index->settings->position_byte_size) * (long long)max_total_buffers_size;
        // fbl->hard_buffer = malloc(hard_buffer_size);

        fbl->soft_buffers = (fbl_soft_buffer2 *)malloc(sizeof(fbl_soft_buffer2) * (size_t)number_of_buffers);
        fbl->current_record_index = 0;
        fbl->current_record = NULL;
        fbl->hard_buffer = NULL;

        for (int i = 0; i < number_of_buffers; i++)
        {
            fbl->soft_buffers[i].initialized = 0;
            fbl->soft_buffers[i].max_buffer_size = 0;
            fbl->soft_buffers[i].buffer_size = 0;
            fbl->soft_buffers[i].node = NULL;
            fbl->soft_buffers[i].sax_records = NULL;
            fbl->soft_buffers[i].pos_records = NULL;
        }
        return fbl;
    }

    void *index_creation_worker2_inmemory(void *transferdata)
    {
        sax_type *sax = (sax_type *)malloc(sizeof(sax_type) * ((buffer_data_inmemory *)transferdata)->index->settings->paa_segments);
        unsigned long start_number = ((buffer_data_inmemory *)transferdata)->start_number;
        unsigned long stop_number = ((buffer_data_inmemory *)transferdata)->stop_number;
        file_position_type *pos = (file_position_type *)malloc(sizeof(file_position_type));
        isax_index *index = ((buffer_data_inmemory *)transferdata)->index;
        ts_type *ts = (ts_type *)malloc(sizeof(ts_type) * (size_t)index->settings->timeseries_size);
        int paa_segments = ((buffer_data_inmemory *)transferdata)->index->settings->paa_segments;

        (void)start_number;
        (void)stop_number;
        (void)paa_segments;

        float *raw_file = ((buffer_data_inmemory *)transferdata)->ts;
        (void)raw_file;

        free(pos);
        free(sax);
        free(ts);

        int j, c = 1, k;
        (void)c;
        isax_node_record *r = (isax_node_record *)malloc(sizeof(isax_node_record));

        int worker_id = ((buffer_data_inmemory *)transferdata)->workernumber;
        int buffers_processed = 0;
        static const int debug_print_max = 10;  /* stampa al più i primi N buffer per worker */

        while (1)
        {
            j = __sync_fetch_and_add(((buffer_data_inmemory *)transferdata)->node_counter, 1);
            if (j >= index->fbl->number_of_buffers)
                break;

            fbl_soft_buffer2 *current_fbl_node = &((first_buffer_layer2 *)(index->fbl))->soft_buffers[j];
            if (!current_fbl_node->initialized)
                continue;

            if (current_fbl_node->buffer_size > 0)
            {
                if (buffers_processed < debug_print_max)
                    fprintf(stderr, "  [worker %d] buffer j=%d buffer_size=%d\n", worker_id, j, current_fbl_node->buffer_size);
                buffers_processed++;
                for (k = 0; k < current_fbl_node->buffer_size; k++)
                {
                    r->sax = &(index->sax_cache[current_fbl_node->pos_records[k] / index->settings->timeseries_size * index->settings->paa_segments]);
                    r->position = &current_fbl_node->pos_records[k];
                    r->insertion_mode = (insertion_mode)(NO_TMP | PARTIAL);
                    add_record_to_node(index, current_fbl_node->node, r, 1);
                }
                flush_subtree_leaf_buffers_inmemory(index, current_fbl_node->node);
            }
        }
        free(r);
        return nullptr;
    }

    Sing::Sing(DistanceType distance_type)
        : SimilaritySearchAlgorithm(distance_type)
    {
    }

    void Sing::setNumThreads(int num_threads)
    {
        int max_threads = omp_get_max_threads();

        if (num_threads > max_threads) 
        {
            std::cerr << "[Warning] " << num_threads 
                    << " threads exceeds max available " << max_threads << " Using the max threads available.\n";
            this->num_threads = max_threads;
        } 
        else if (num_threads < 1) 
        {
            std::cerr << "[Warning] Thread count must be >= 1. Using 1.\n";
            this->num_threads = 1;
        } 
        else 
        {
            this->num_threads = num_threads;
        }
    } 

    int Sing::getNumThreads() const
    {
        return this->num_threads;
    } 

    void Sing::buildIndex(DataSource *data_source)
    {
        this->dim = data_source->getDim();
        this->n_database = data_source->getTotalRecords();

        if (this->n_database == 0)
        {
            data_source->reset();
            idx_t count = 0;
            float *dummy = new float[this->dim];
            while (data_source->nextRecord(dummy))
            {
                count++;
            }
            delete[] dummy;
            this->n_database = count;
            data_source->reset();
        }

        data_source->reset();
        this->database = new float[this->n_database * this->dim];
        float *record = new float[this->dim];
        idx_t ri = 0;
        while (data_source->nextRecord(record))
        {
            std::copy(record, record + this->dim, this->database + ri * this->dim);
            ri++;
        }
        delete[] record;

        const char *index_path = "";  // in-memory: no directory
        this->index_settings = isax_index_settings_init(index_path,         // INDEX DIRECTORY
                                                        this->dim,          // TIME SERIES SIZE
                                                        this->paa_segments, // PAA SEGMENTS
                                                        this->sax_cardinality, // SAX CARDINALITY IN BITS
                                                        this->leaf_size,    // LEAF SIZE
                                                        this->min_leaf_size, // MIN LEAF SIZE
                                                        this->initial_lbl_size,  // INITIAL LEAF BUFFER SIZE
                                                        this->flush_limit,  // FLUSH LIMIT
                                                        this->initial_fbl_size, // INITIAL FBL BUFFER SIZE
                                                        this->total_loaded_leaves, // Leaves to load at each fetch
                                                        this->tight_bound,  // Tightness of leaf bounds
                                                        this->aggressive_check, // aggressive check
                                                        1, 1);             // new index, inmemory

        this->index = isax_index_init_inmemory(this->index_settings);
        this->index->sax_cache_size = this->n_database;
        index_creation_gpu(this->database, this->n_database, this->index);
    }

    void Sing::index_creation_gpu(float *dataset, idx_t dataset_size, isax_index *idx)
    {
        ts_type *rawfile = (ts_type *)dataset;
        long int ts_num = (long int)dataset_size;
        int maxquerythread = this->index_workers;

        fprintf(stderr, "[Sing buildIndex] index_creation_gpu: start, ts_num=%ld, index_workers=%d\n", (long)ts_num, maxquerythread);
        idx->sax_file = NULL;

        long int ts_loaded = 0;
        (void)ts_loaded;
        int i;
        int node_counter = 0;
        pthread_t *threadid = (pthread_t *)malloc(sizeof(pthread_t) * (size_t)maxquerythread);
        buffer_data_inmemory *input_data = (buffer_data_inmemory *)malloc(sizeof(buffer_data_inmemory) * (size_t)maxquerythread);

        idx->sax_cache = (sax_type *)malloc(sizeof(sax_type) * (size_t)(idx->settings->paa_segments * ts_num));

        if (idx->settings->raw_filename != nullptr)
        {
            free(idx->settings->raw_filename);
            idx->settings->raw_filename = nullptr;
        }
        idx->settings->raw_filename = (char *)malloc(256);
        if (idx->settings->raw_filename != nullptr)
            strcpy(idx->settings->raw_filename, "inmemory");

        pthread_mutex_t lock_record = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_t lockfbl = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_t lock_index = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_t lock_firstnode = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_t lock_disk = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_t *lockcbl = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * LOCK_SIZE);
        pthread_barrier_t lock_barrier1;
        pthread_barrier_t lock_barrier2;
        pthread_barrier_init(&lock_barrier1, NULL, maxquerythread);
        pthread_barrier_init(&lock_barrier2, NULL, maxquerythread);

        destroy_fbl(idx->fbl);
        idx->fbl = (first_buffer_layer *)initialize_simrec(
            idx->settings->initial_fbl_buffer_size,
            (int)pow(2, idx->settings->paa_segments),
            idx->settings->max_total_buffer_size + DISK_BUFFER_SIZE * (PROGRESS_CALCULATE_THREAD_NUMBER - 1),
            idx);

        if (idx->fbl == nullptr)
        {
            fprintf(stderr, "Sing::index_creation_gpu: initialize_simrec failed\n");
            throw std::runtime_error("Sing::index_creation_gpu: initialize_simrec failed");
        }
        fprintf(stderr, "[Sing buildIndex] initialize_simrec ok, number_of_buffers=%d\n", ((first_buffer_layer2 *)idx->fbl)->number_of_buffers);

        for (i = 0; i < LOCK_SIZE; i++)
            pthread_mutex_init(&lockcbl[i], NULL);

#pragma omp parallel for num_threads(maxquerythread)
        for (long int j = 0; j < ts_num; j++)
        {
            sax_type *sax = &(idx->sax_cache[j * idx->settings->paa_segments]);
            if (sax_from_ts(&(rawfile[j * idx->settings->timeseries_size]), sax,
                            idx->settings->ts_values_per_paa_segment,
                            idx->settings->paa_segments, idx->settings->sax_alphabet_cardinality,
                            idx->settings->sax_bit_cardinality) == SUCCESS)
            {
                root_mask_type first_bit_mask = 0x00;
                CREATE_MASK(first_bit_mask, idx, sax);
                fbl_soft_buffer2 *current_buffer = &((first_buffer_layer2 *)(idx->fbl))->soft_buffers[(int)first_bit_mask];
                __sync_fetch_and_add(&(current_buffer->max_buffer_size), 1);
            }
        }
        fprintf(stderr, "[Sing buildIndex] phase 1 done: SAX count per buffer\n");

#pragma omp parallel for num_threads(maxquerythread)
        for (i = 0; i < ((first_buffer_layer2 *)(idx->fbl))->number_of_buffers; i++)
        {
            fbl_soft_buffer2 *current_buffer = &((first_buffer_layer2 *)(idx->fbl))->soft_buffers[i];
            if (current_buffer->max_buffer_size != 0)
            {
                current_buffer->initialized = 1;
                current_buffer->sax_records = (sax_type *)malloc(sizeof(sax_type) * idx->settings->paa_segments * (size_t)current_buffer->max_buffer_size);
                current_buffer->pos_records = (file_position_type *)malloc(sizeof(file_position_type) * (size_t)current_buffer->max_buffer_size);
                current_buffer->node = isax_root_node_init((root_mask_type)i, idx->settings->initial_leaf_buffer_size);
                current_buffer->node->is_leaf = 1;
            }
        }
        fprintf(stderr, "[Sing buildIndex] phase 2 done: buffers and nodes allocated\n");

#pragma omp parallel for num_threads(maxquerythread)
        for (long int j = 0; j < ts_num; j++)
        {
            root_mask_type first_bit_mask = 0x00;
            sax_type *sax = &(idx->sax_cache[j * idx->settings->paa_segments]);
            CREATE_MASK(first_bit_mask, idx, sax);
            fbl_soft_buffer2 *current_buffer = &((first_buffer_layer2 *)(idx->fbl))->soft_buffers[(int)first_bit_mask];
            int buffersize = __sync_fetch_and_add(&(current_buffer->buffer_size), 1);
            memcpy(&current_buffer->sax_records[buffersize * idx->settings->paa_segments], sax, sizeof(sax_type) * (size_t)idx->settings->paa_segments);
            current_buffer->pos_records[buffersize] = (file_position_type)(j * idx->settings->timeseries_size);
        }
        fprintf(stderr, "[Sing buildIndex] phase 3 done: sax_records/pos_records filled\n");

        for (i = 0; i < maxquerythread; i++)
        {
            input_data[i].index = idx;
            input_data[i].lock_fbl = &lockfbl;
            input_data[i].lock_record = &lock_record;
            input_data[i].lock_cbl = lockcbl;
            input_data[i].lock_firstnode = &lock_firstnode;
            input_data[i].lock_index = &lock_index;
            input_data[i].lock_nodeconter = nullptr;
            input_data[i].lock_disk = &lock_disk;
            input_data[i].ts = rawfile;
            input_data[i].workernumber = i;
            input_data[i].total_workernumber = maxquerythread;
            input_data[i].start_number = i * (int)(ts_num / maxquerythread);
            input_data[i].stop_number = (i + 1) * (int)(ts_num / maxquerythread);
            input_data[i].node_counter = &node_counter;
            input_data[i].lock_barrier1 = &lock_barrier1;
            input_data[i].lock_barrier2 = &lock_barrier2;
            input_data[i].shared_start_number = nullptr;
            input_data[i].read_block_length = this->read_block_length;
            input_data[i].finished = false;
            input_data[i].nodeid = nullptr;
        }
        input_data[maxquerythread - 1].start_number = (maxquerythread - 1) * (int)(ts_num / maxquerythread);
        input_data[maxquerythread - 1].stop_number = (int)ts_num;

        fprintf(stderr, "[Sing buildIndex] starting %d workers (add_record_to_node + flush)...\n", maxquerythread);
        for (i = 0; i < maxquerythread; i++)
            pthread_create(&(threadid[i]), NULL, index_creation_worker2_inmemory, (void *)&(input_data[i]));
        for (i = 0; i < maxquerythread; i++)
            pthread_join(threadid[i], NULL);
        fprintf(stderr, "[Sing buildIndex] workers finished\n");

        idx->sax_cache_size = (unsigned long)ts_num;

        free(lockcbl);
        free(threadid);
        free(input_data);
        pthread_barrier_destroy(&lock_barrier1);
        pthread_barrier_destroy(&lock_barrier2);
    }
    
    void Sing::searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        #pragma omp parallel num_threads(num_threads)
        {
            #pragma omp for 
            for (idx_t qi = 0; qi < n_query; qi++)
            {   
                std::priority_queue<std::pair<float, idx_t>> pq;
                const float *q_vec = query + qi * this->dim;

                float bound = FLT_MAX;  // initialize bound to max float

                for (idx_t dbi = 0; dbi < this->n_database; ++dbi)
                {
                    const float *db_vec = this->database + dbi * this->dim;
                    float dist = this->distance_computer->compute_dist(const_cast<float *>(q_vec), 
                                                                        const_cast<float *>(db_vec), 
                                                                        this->dim, 
                                                                        bound);
                    if ((idx_t)pq.size() < k) // maintain max-heap
                    {
                        pq.emplace(dist, dbi); // equivalent to pq.push(make_pair(dist, dbi));
                    }
                    else if (dist < pq.top().first) 
                    {
                        pq.pop();
                        pq.emplace(dist, dbi);                         
                        bound = pq.top().first; // update the bound variable for pruning
                    }

                }
                
                // store top-k results in reverse order
                for (idx_t j = k; j > 0; --j)
                {
                    D[qi * k + (j - 1)] = pq.top().first;
                    I[qi * k + (j - 1)] = pq.top().second;
                    pq.pop();
                }               
            }
        }
    }

    void Sing::printBuildIndexDebug() const
    {
        if (index == nullptr)
        {
            fprintf(stderr, "[Sing printBuildIndexDebug] index is NULL (buildIndex not called?)\n");
            return;
        }
        fprintf(stderr, "--- Sing index debug ---\n");
        fprintf(stderr, "  sax_cache_size   = %lu\n", (unsigned long)index->sax_cache_size);
        fprintf(stderr, "  paa_segments     = %d\n", index->settings->paa_segments);
        fprintf(stderr, "  timeseries_size  = %d\n", index->settings->timeseries_size);
        if (index->fbl == nullptr)
        {
            fprintf(stderr, "  fbl             = NULL\n");
            return;
        }
        first_buffer_layer2 *fbl2 = (first_buffer_layer2 *)index->fbl;
        fprintf(stderr, "  fbl number_of_buffers = %d\n", fbl2->number_of_buffers);
        int init_count = 0;
        unsigned long total_records = 0;
        for (int i = 0; i < fbl2->number_of_buffers; i++)
        {
            fbl_soft_buffer2 *sb = &fbl2->soft_buffers[i];
            if (sb->initialized)
            {
                init_count++;
                total_records += (unsigned long)sb->buffer_size;
            }
        }
        fprintf(stderr, "  buffers initialized = %d / %d\n", init_count, fbl2->number_of_buffers);
        fprintf(stderr, "  total records in FBL = %lu\n", total_records);
        fprintf(stderr, "--- end ---\n");
    }

    Sing::~Sing()
    {
        delete[] database;

        if (index != nullptr)
        {
            if (index->sax_cache != nullptr)
                free(index->sax_cache);
            if (index->answer != nullptr)
                free(index->answer);
            if (index->fbl != nullptr)
                destroy_fbl2((first_buffer_layer2 *)index->fbl);
            if (index->sax_file != nullptr)
                fclose(index->sax_file);
            free(index);
            index = nullptr;
        }
        if (index_settings != nullptr)
        {
            if (index_settings->bit_masks != nullptr)
                free(index_settings->bit_masks);
            if (index_settings->max_sax_cardinalities != nullptr)
                free(index_settings->max_sax_cardinalities);
            free(index_settings);
            index_settings = nullptr;
        }
    }
}