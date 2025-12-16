#include "ParIS.hpp"
#include "../isax/SAX.hpp"
#include "../isax/iSAXPqueue.hpp"
#include <stdexcept>
#include <pthread.h>
#include <cstring>

namespace diNoLib
{

    ParIS::ParIS(DistanceType distance_type)
        : SimilaritySearchAlgorithm(distance_type)
    {
    }

    void ParIS::setNumThreads(int num_threads)
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

    int ParIS::getNumThreads() const
    {
        return this->num_threads;
    }

    void ParIS::buildIndex(DataSource *data_source)
    {
        this->dim = data_source->getDim();
        this->n_database = data_source->getTotalRecords();

        // ParIS requires FileDataSource
        FileDataSource *file_source = dynamic_cast<FileDataSource *>(data_source);
        if (file_source == nullptr)
        {
            fprintf(stderr, "Error: ParIS::buildIndex requires FileDataSource\n");
            throw std::runtime_error("ParIS::buildIndex requires FileDataSource");
        }

        const char *filename = file_source->getFilename();
        if (filename == nullptr)
        {
            fprintf(stderr, "Error: FileDataSource does not have a filename\n");
            throw std::runtime_error("FileDataSource does not have a filename");
        }

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
                                                        1,                         // new index
                                                        0);

        this->index = isax_index_init(this->index_settings);
        isax_index *index = this->index;

        // Use the multi-threaded file-based indexing function
        int ts_num = (this->n_database > 0) ? (int)this->n_database : 0;
        int calculate_thread = this->index_workers;

        // If n_database is 0, we need to determine it from file size
        // The isax_index_binary_file_m function will handle this
        if (ts_num == 0)
        {
            // Get file size to determine number of records
            FILE *temp_file = fopen(filename, "rb");
            if (temp_file != nullptr)
            {
                fseek(temp_file, 0L, SEEK_END);
                long file_size = ftell(temp_file);
                fclose(temp_file);
                ts_num = file_size / (sizeof(float) * this->dim);
                this->n_database = ts_num;
            }
            else
            {
                fprintf(stderr, "Error: Could not open file to determine size\n");
                throw std::runtime_error("Could not open file to determine size");
            }
        }

        // Call the multi-threaded indexing function
        isax_index_binary_file_m(filename, ts_num, index, calculate_thread, this->read_block_length);

        // Load sax_cache from sax_file for search
        if (index->sax_file != nullptr && index->total_records > 0)
        {
            rewind(index->sax_file);
            index->sax_cache = (sax_type *)malloc(sizeof(sax_type) * index->settings->paa_segments * index->total_records);
            if (index->sax_cache != nullptr)
            {
                size_t items_read = fread(index->sax_cache, index->settings->sax_byte_size, index->total_records, index->sax_file);
                if (items_read == index->total_records)
                {
                    index->sax_cache_size = index->total_records;
                }
                else
                {
                    fprintf(stderr, "Warning: Could not read all SAX cache entries\n");
                    free(index->sax_cache);
                    index->sax_cache = nullptr;
                }
            }
        }

        fprintf(stderr, ">>> Finished indexing\n");
    }

    void ParIS::searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        if (this->distance_type == DistanceType::L2_SQUARED)
        {
            searchIndexL2Squared(query, n_query, k, I, D);
        }
        else
        {
            fprintf(stderr, "Warning: ParIS::searchIndex for DTW is not yet implemented.\n");
            // Initialize results to invalid values
            for (idx_t qi = 0; qi < n_query; qi++)
            {
                for (idx_t j = 0; j < k; j++)
                {
                    I[qi * k + j] = 0;
                    D[qi * k + j] = FLT_MAX;
                }
            }
        }
    }

    void ParIS::searchIndexL2Squared(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        if (index == nullptr || index->sax_cache == nullptr)
        {
            fprintf(stderr, "Error: Index not built or sax_cache not loaded\n");
            for (idx_t qi = 0; qi < n_query; qi++)
            {
                for (idx_t j = 0; j < k; j++)
                {
                    I[qi * k + j] = 0;
                    D[qi * k + j] = FLT_MAX;
                }
            }
            return;
        }

        ts_type *paa = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);

        for (idx_t q_loaded = 0; q_loaded < n_query; q_loaded++)
        {
            const float *ts = query + q_loaded * this->dim;

            // Parse ts and make PAA representation
            paa_from_ts(ts, paa, index->settings->paa_segments, index->settings->ts_values_per_paa_segment);

            pqueue_bsf result = exact_topk_serial_ParIS((float *)ts, paa, index, this->minimum_distance, this->min_checked_leaves, k, this->search_workers);

            // Extract results
            for (idx_t ik = 0; ik < k; ik++)
            {
                D[q_loaded * k + ik] = result.knn[ik];
                I[q_loaded * k + ik] = result.position[ik];
            }

            // Free only the internal pointers, not the structure itself (it's on the stack)
            if (result.position != nullptr) {
                free(result.position);
            }
            if (result.knn != nullptr) {
                free(result.knn);
            }
            if (result.node != nullptr) {
                free(result.node);
            }
        }

        free(paa);
    }

    ParIS::~ParIS()
    {
        // Save settings pointer before freeing index (they point to the same object)
        isax_index_settings *settings_to_free = nullptr;
        
        // Cleanup iSAX index structures
        if (index != nullptr) {
            if (index->sax_cache != nullptr) {
                free(index->sax_cache);
                index->sax_cache = nullptr;
            }
            if (index->answer != nullptr) {
                free(index->answer);
                index->answer = nullptr;
            }
            if (index->fbl != nullptr) {
                destroy_fbl(index->fbl);
                index->fbl = nullptr;
            }
            if (index->sax_file != nullptr) {
                fclose(index->sax_file);
                index->sax_file = nullptr;
            }
            // Save settings pointer before freeing index
            settings_to_free = index->settings;
            free(index);
            index = nullptr;
        }
        
        // Clean up settings (index->settings and index_settings point to the same object)
        // Use index_settings if settings_to_free is null (index was never created)
        isax_index_settings *settings = (settings_to_free != nullptr) ? settings_to_free : index_settings;
        if (settings != nullptr) {
            // Free raw_filename if it was allocated
            if (settings->raw_filename != nullptr) {
                free(settings->raw_filename);
                settings->raw_filename = nullptr;
            }
            if (settings->bit_masks != nullptr) {
                free(settings->bit_masks);
                settings->bit_masks = nullptr;
            }
            if (settings->max_sax_cardinalities != nullptr) {
                free(settings->max_sax_cardinalities);
                settings->max_sax_cardinalities = nullptr;
            }
            free(settings);
        }
        
        index_settings = nullptr;
    }

    // File-based calculate_node_topk (reads from disk files and in-memory buffers)
    void calculate_node_topk(isax_index *index, isax_node *node, ts_type *query, pqueue_bsf *pq_bsf)
    {
        FILE *raw_file = fopen(index->settings->raw_filename, "rb");
        if (raw_file == nullptr)
        {
            return;
        }

        // Check in-memory buffers first (like in-memory version)
        if (node->buffer != NULL)
        {
            int i;
            // Check full buffers
            for (i = 0; i < node->buffer->full_buffer_size; i++)
            {
                float dist = ts_euclidean_distance_SIMD(query, node->buffer->full_ts_buffer[i],
                                                        index->settings->timeseries_size, pq_bsf->knn[pq_bsf->k - 1]);
                if (dist <= pq_bsf->knn[pq_bsf->k - 1])
                {
                    pqueue_bsf_insert(pq_bsf, dist, 0, node);
                }
            }
            // Check temporary full buffers
            for (i = 0; i < node->buffer->tmp_full_buffer_size; i++)
            {
                float dist = ts_euclidean_distance_SIMD(query, node->buffer->tmp_full_ts_buffer[i],
                                                        index->settings->timeseries_size, pq_bsf->knn[pq_bsf->k - 1]);
                if (dist <= pq_bsf->knn[pq_bsf->k - 1])
                {
                    pqueue_bsf_insert(pq_bsf, dist, 0, node);
                }
            }
            // Check partial buffers (read from raw file)
            for (i = 0; i < node->buffer->partial_buffer_size; i++)
            {
                file_position_type pos = *node->buffer->partial_position_buffer[i];
                fseek(raw_file, pos, SEEK_SET);
                ts_type *ts_buffer = (ts_type *)malloc(index->settings->ts_byte_size);
                size_t items_read = fread(ts_buffer, index->settings->ts_byte_size, 1, raw_file);
                (void)items_read; // Suppress unused variable warning
                float dist = ts_euclidean_distance_SIMD(query, ts_buffer, index->settings->timeseries_size, pq_bsf->knn[pq_bsf->k - 1]);
                if (dist <= pq_bsf->knn[pq_bsf->k - 1])
                {
                    pqueue_bsf_insert(pq_bsf, dist, pos / index->settings->timeseries_size, node);
                }
                free(ts_buffer);
            }
        }

        // Check partial data file
        if (node->filename != NULL && node->has_partial_data_file)
        {
            FILE *node_file = fopen(node->filename, "rb");
            if (node_file != nullptr)
            {
                ts_type *ts_buffer = (ts_type *)malloc(index->settings->ts_byte_size);
                for (int i = 0; i < node->leaf_size; i++)
                {
                    fseek(node_file, i * index->settings->partial_record_size, SEEK_SET);
                    size_t items_read = fread(ts_buffer, index->settings->ts_byte_size, 1, node_file);
                    (void)items_read; // Suppress unused variable warning
                    float dist = ts_euclidean_distance_SIMD(query, ts_buffer, index->settings->timeseries_size, pq_bsf->knn[pq_bsf->k - 1]);
                    if (dist <= pq_bsf->knn[pq_bsf->k - 1])
                    {
                        file_position_type pos = (file_position_type)(i * index->settings->timeseries_size);
                        pqueue_bsf_insert(pq_bsf, dist, pos / index->settings->timeseries_size, node);
                    }
                }
                free(ts_buffer);
                fclose(node_file);
            }
        }

        // Skip full data files in refine_topk_answer - they will be handled by mindistance_worker
        // which scans the SAX cache and finds all candidates. This avoids position mapping issues.
        // The in-memory buffers and partial data files should provide a good BSF for pruning.

        fclose(raw_file);
    }

    // File-based calculate_minimum_distance
    float calculate_minimum_distance(isax_index *index, isax_node *node, ts_type *raw_query, ts_type *query)
    {
        float bsfLeaf = minidist_paa_to_isax(query, node->isax_values,
                                             node->isax_cardinalities,
                                             index->settings->sax_bit_cardinality,
                                             index->settings->sax_alphabet_cardinality,
                                             index->settings->paa_segments,
                                             MINVAL, MAXVAL,
                                             index->settings->mindist_sqrt);
        float bsfRecord = FLT_MAX;

        if (!index->has_wedges)
        {
            if (node->filename != NULL && node->has_partial_data_file)
            {
                FILE *node_file = fopen(node->filename, "rb");
                if (node_file != nullptr)
                {
                    sax_type *sax_buffer = (sax_type *)malloc(index->settings->sax_byte_size);
                    for (int i = 0; i < node->leaf_size; i++)
                    {
                        fseek(node_file, i * index->settings->partial_record_size, SEEK_SET);
                        size_t items_read = fread(sax_buffer, index->settings->sax_byte_size, 1, node_file);
                        (void)items_read; // Suppress unused variable warning
                        float mindist = minidist_paa_to_isax_raw_SIMD(query, sax_buffer, index->settings->max_sax_cardinalities,
                                                                      index->settings->sax_bit_cardinality,
                                                                      index->settings->sax_alphabet_cardinality,
                                                                      index->settings->paa_segments, MINVAL, MAXVAL,
                                                                      index->settings->mindist_sqrt);
                        if (mindist < bsfRecord)
                        {
                            bsfRecord = mindist;
                        }
                    }
                    free(sax_buffer);
                    fclose(node_file);
                }
            }
        }

        return (bsfLeaf < bsfRecord) ? bsfLeaf : bsfRecord;
    }

    // File-based approximate_topk (uses sax_cache loaded from sax_file)
    void approximate_topk(ts_type *ts, ts_type *paa, isax_index *index, pqueue_bsf *pq_bsf)
    {
        sax_type *sax = (sax_type *)malloc(sizeof(sax_type) * index->settings->paa_segments);
        sax_from_paa(paa, sax, index->settings->paa_segments,
                     index->settings->sax_alphabet_cardinality,
                     index->settings->sax_bit_cardinality);

        root_mask_type root_mask = 0;
        CREATE_MASK(root_mask, index, sax);

        if (index->fbl->soft_buffers[(int)root_mask].initialized) {
            isax_node *node = index->fbl->soft_buffers[(int)root_mask].node;
            // Traverse tree

            // Adaptive splitting
            // For file-based indexing, skip adaptive splitting if node has file data
            if (node->is_leaf && !node->has_full_data_file &&
                !node->has_partial_data_file &&
                (node->leaf_size > index->settings->min_leaf_size) &&
                node->buffer != NULL &&
                (node->buffer->full_buffer_size > 0 || 
                 node->buffer->partial_buffer_size > 0 ||
                 node->buffer->tmp_full_buffer_size > 0 ||
                 node->buffer->tmp_partial_buffer_size > 0))
            {
                split_node(index, node);
            }

            while (!node->is_leaf) {
                int location = index->settings->sax_bit_cardinality - 1 -
                node->split_data->split_mask[node->split_data->splitpoint];
                root_mask_type mask = index->settings->bit_masks[location];

                if(sax[node->split_data->splitpoint] & mask)
                {
                    node = node->right_child;
                }
                else
                {
                    node = node->left_child;
                }

                // Adaptive splitting
                // For file-based indexing, skip adaptive splitting if node has file data
                if (node->is_leaf && !node->has_full_data_file &&
                    !node->has_partial_data_file &&
                    (node->leaf_size > index->settings->min_leaf_size) &&
                    node->buffer != NULL &&
                    (node->buffer->full_buffer_size > 0 || 
                     node->buffer->partial_buffer_size > 0 ||
                     node->buffer->tmp_full_buffer_size > 0 ||
                     node->buffer->tmp_partial_buffer_size > 0))
                {
                    split_node(index, node);
                }
            }

            calculate_node_topk(index, node, ts, pq_bsf);
        }
        else {

        }
        for (int i = 0; i < pq_bsf->k-1; ++i)
        {
            pq_bsf->knn[i]=pq_bsf->knn[pq_bsf->k-1];
        }
        free(sax);
    }

    // File-based refine_topk_answer (reads from files instead of in-memory)
    void refine_topk_answer(ts_type *ts, ts_type *paa, isax_index *index, 
              pqueue_bsf *pq_bsf, 
                            float minimum_distance, int limit)  
    {  
       int tight_bound = index->settings->tight_bound; 
      int aggressive_check = index->settings->aggressive_check; 
 
        pqueue_t *pq = pqueue_init(index->settings->root_nodes_size, 
                               cmp_pri, get_pri, set_pri, get_pos, set_pos); 
 
        // Insert all root nodes in heap. 
        isax_node *current_root_node = index->first_node; 
        while (current_root_node != NULL) { 
        query_result * mindist_result = (query_result *)malloc(sizeof(query_result)); 
        mindist_result->distance =  minidist_paa_to_isax(paa, current_root_node->isax_values, 
                                              current_root_node->isax_cardinalities, 
                                              index->settings->sax_bit_cardinality, 
                                              index->settings->sax_alphabet_cardinality, 
                                              index->settings->paa_segments, 
                                              MINVAL, MAXVAL, 
                                              index->settings->mindist_sqrt); 
        mindist_result->node = current_root_node; 
            if (mindist_result->distance < pq_bsf->knn[pq_bsf->k-1])
            {
                pqueue_insert(pq, mindist_result);
            }
            else
            {
                free(mindist_result);
            }
        current_root_node = current_root_node->next; 
        } 
        query_result * n; 
        int checks = 0; 
        while ((n = (query_result *)pqueue_pop(pq))) 
        { 
        // The best node has a worse mindist, so search is finished! 
            if (n->distance >= pq_bsf->knn[pq_bsf->k-1] || n->distance > minimum_distance) {
                pqueue_insert(pq, n);
                break;
            } 
            else { 
          // If it is a leaf, check its real distance. 
            if (n->node->is_leaf) { 
        // *** ADAPTIVE SPLITTING *** 
        // For file-based indexing, skip adaptive splitting if node has file data
        // to avoid double-free issues with FBL-managed memory
            if (!n->node->has_full_data_file && 
          !n->node->has_partial_data_file &&
          (n->node->leaf_size > index->settings->min_leaf_size) &&
          n->node->buffer != NULL &&
          (n->node->buffer->full_buffer_size > 0 || 
           n->node->buffer->partial_buffer_size > 0 ||
           n->node->buffer->tmp_full_buffer_size > 0 ||
           n->node->buffer->tmp_partial_buffer_size > 0)) 
            { 
          // Split and push again in the queue 
                split_node(index, n->node); 
          pqueue_insert(pq, n); 
                continue; 
            } 
        // *** EXTRA BOUNDING *** 
        if(tight_bound) { 
          float mindistance = calculate_minimum_distance(index, n->node, ts, paa); 
                    if(mindistance >= pq_bsf->knn[pq_bsf->k-1])
                    {
                        free(n);
                        continue;
                    }
        } 
        // *** REAL DISTANCE *** 
        checks++; 
        calculate_node_topk(index, n->node, ts, pq_bsf);

                // Continue searching - the termination condition at line 440 will handle stopping
                // when no node can improve the results (n->distance >= pq_bsf->knn[pq_bsf->k-1])
                // Don't break early here, as there might be better candidates still in the queue
            } 
            else { 
              // If it is an intermediate node calculate mindist for children 
              // and push them in the queue 
                if (n->node->left_child != NULL && n->node->left_child->isax_cardinalities != NULL) { 
          if(n->node->left_child->is_leaf && !n->node->left_child->has_partial_data_file && aggressive_check){ 
            calculate_node_topk(index, n->node->left_child, ts, pq_bsf);

          } 
          else { 
                    query_result * mindist_result = (query_result *)malloc(sizeof(query_result)); 
                    mindist_result->distance =  minidist_paa_to_isax(paa, n->node->left_child->isax_values, 
                                                                     n->node->left_child->isax_cardinalities, 
                                                                     index->settings->sax_bit_cardinality, 
                                                                     index->settings->sax_alphabet_cardinality, 
                                                                     index->settings->paa_segments, 
                                                                     MINVAL, MAXVAL, 
                                                                     index->settings->mindist_sqrt); 
          mindist_result->node = n->node->left_child;  
            if (mindist_result->distance < pq_bsf->knn[pq_bsf->k-1])
            {
                pqueue_insert(pq, mindist_result);
            }
            else
            {
                free(mindist_result);
            }
          } 
                } 
                if (n->node->right_child != NULL && n->node->right_child->isax_cardinalities != NULL) { 
          if(n->node->right_child->is_leaf && !n->node->right_child->has_partial_data_file && aggressive_check){ 
            calculate_node_topk(index, n->node->right_child, ts, pq_bsf);
          } 
          else { 
                    query_result * mindist_result = (query_result *)malloc(sizeof(query_result)); 
          mindist_result->distance =  minidist_paa_to_isax(paa, n->node->right_child->isax_values, 
                                                                     n->node->right_child->isax_cardinalities, 
                                                                     index->settings->sax_bit_cardinality, 
                                                                     index->settings->sax_alphabet_cardinality, 
                                                                     index->settings->paa_segments, 
                                                                     MINVAL, MAXVAL, 
                                                                     index->settings->mindist_sqrt); 
                    mindist_result->node = n->node->right_child; 
            if (mindist_result->distance < pq_bsf->knn[pq_bsf->k-1])
            {
                pqueue_insert(pq, mindist_result);
            }
            else
            {
                free(mindist_result);
            }
          } 
                } 
            } 
            } 

            // Free the node currently popped. 
           free(n);
        } 
        // Free the nodes that where not popped. 
        while ((n = (query_result *)pqueue_pop(pq))) 
        { 
            free(n); 
        } 
        // Free the priority queue. 
        for (int i = 0; i < pq_bsf->k-1; ++i)
        {
            pq_bsf->knn[i]=pq_bsf->knn[pq_bsf->k-1];
        }
        pqueue_free(pq); 
    }

    void *mindistance_worker(void *essdata)
    {
        unsigned long i;
        float mindist;
        isax_index *index = ((ParIS_LDCW_data*)essdata)->index;
        unsigned long start_number = ((ParIS_LDCW_data*)essdata)->start_number;
        unsigned long stop_number = ((ParIS_LDCW_data*)essdata)->stop_number;
        ts_type *paa = ((ParIS_LDCW_data*)essdata)->paa;

        ((ParIS_LDCW_data*)essdata)->label_number = (unsigned long *)malloc(sizeof(unsigned long) * 10000);
        ((ParIS_LDCW_data*)essdata)->minidisvector = (float *)malloc(sizeof(float) * 10000);
        unsigned long max_number = 10000;

        for(i = start_number; i < stop_number; i++)
        {
            sax_type *sax = &index->sax_cache[i * index->settings->paa_segments];

            mindist = minidist_paa_to_isax_rawa_SIMD(paa, sax, index->settings->max_sax_cardinalities,
                                                     index->settings->sax_bit_cardinality,
                                                     index->settings->sax_alphabet_cardinality,
                                                     index->settings->paa_segments, MINVAL, MAXVAL,
                                                     index->settings->mindist_sqrt);
            if(mindist <= ((ParIS_LDCW_data*)essdata)->bsfdistance) {
                if (((ParIS_LDCW_data*)essdata)->sum_of_lab >= max_number)
                {
                    max_number = (max_number + 10000);
                    unsigned long* change_lab = ((ParIS_LDCW_data*)essdata)->label_number;
                    float* change_minivec = ((ParIS_LDCW_data*)essdata)->minidisvector;
                    ((ParIS_LDCW_data*)essdata)->label_number = (unsigned long *)malloc(sizeof(unsigned long) * max_number);
                    ((ParIS_LDCW_data*)essdata)->minidisvector = (float *)malloc(sizeof(float) * max_number);
                    memcpy(((ParIS_LDCW_data*)essdata)->label_number, change_lab, sizeof(unsigned long) * (max_number - 10000));
                    memcpy(((ParIS_LDCW_data*)essdata)->minidisvector, change_minivec, sizeof(float) * (max_number - 10000));
                    free(change_lab);
                    free(change_minivec);
                }
                ((ParIS_LDCW_data*)essdata)->label_number[((ParIS_LDCW_data*)essdata)->sum_of_lab] = i;
                ((ParIS_LDCW_data*)essdata)->minidisvector[((ParIS_LDCW_data*)essdata)->sum_of_lab] = mindist;
                ((ParIS_LDCW_data*)essdata)->sum_of_lab++;
            }
        }
        return NULL;
    }

    void *topk_read_worker(void *read_pointer)
    {
        isax_index *index=((ParIS_read_worker_data*)read_pointer)->index;
        pqueue_bsf *pq_bsf=((ParIS_read_worker_data*)read_pointer)->pq_bsf;
        FILE *raw_file = fopen(index->settings->raw_filename, "rb");
        fseek(raw_file, 0, SEEK_SET);
        ts_type *ts_buffer = (ts_type *)malloc(index->settings->ts_byte_size);
        ts_type *ts =((ParIS_read_worker_data*)read_pointer)->ts;
        unsigned long t=0,p;
        unsigned long sum_of_lab=((ParIS_read_worker_data*)read_pointer)->sum_of_lab;
        float *minidisvector=((ParIS_read_worker_data*)read_pointer)->minidisvector;

        float bsf,dist;
        while(1)
        { 
         
            //pthread_rwlock_rdlock(((ParIS_read_worker_data*)read_pointer)->lock_bsf); 
            bsf= pq_bsf->knn[pq_bsf->k-1]; 
            //printf(" t is %ld\n",*(((ParIS_read_worker_data*)read_pointer)->counter));
     
            //pthread_rwlock_unlock(((ParIS_read_worker_data*)read_pointer)->lock_bsf); 
                    //t=*(((ParIS_read_worker_data*)read_pointer)->counter); 
            //*(((ParIS_read_worker_data*)read_pointer)->counter)=*(((ParIS_read_worker_data*)read_pointer)->counter)+1;
            t=__sync_fetch_and_add(((ParIS_read_worker_data*)read_pointer)->counter,1);
            //printf("%ld\n", ((ParIS_read_worker_data*)read_pointer)->sum_of_lab);
            if (t>=sum_of_lab) 
            {    
                break; 
            } 
            
             p=((ParIS_read_worker_data*)read_pointer)->load_point[t];
            //printf("t is %ld!!!\n",p );
            if (minidisvector[t]<bsf) 
            {
                fseek(raw_file, p * index->settings->ts_byte_size, SEEK_SET); 
                size_t items_read = fread(ts_buffer, index->settings->ts_byte_size, 1, raw_file);
                (void)items_read; // Suppress unused variable warning
                //read_time_conter++;                 
                 dist = ts_euclidean_distance_SIMD(ts, ts_buffer, index->settings->timeseries_size, bsf); 
                 //printf("the distance is %f!!\n", dist);
                if(dist <= bsf)  
                {  
                    pthread_rwlock_wrlock(((ParIS_read_worker_data*)read_pointer)->lock_bsf); 
                    pqueue_bsf_insert(pq_bsf,dist,p,NULL);
                    pthread_rwlock_unlock(((ParIS_read_worker_data*)read_pointer)->lock_bsf); 
                } 
            } 
            //printf("the t is :%ld  !!!!!\n",t); 
        }

        free(ts_buffer);
        fclose(raw_file);
        return NULL;
    }

    pqueue_bsf exact_topk_serial_ParIS(ts_type *ts, ts_type *paa, isax_index *index, float minimum_distance, int min_checked_leaves, int k, int maxquerythread)
    {
        FILE *raw_file = fopen(index->settings->raw_filename, "rb");
        if (raw_file == nullptr)
        {
            fprintf(stderr, "Error: Could not open raw file for search\n");
            pqueue_bsf *pq_bsf = pqueue_bsf_init(k);
            pqueue_bsf result = *pq_bsf;
            // Don't free pq_bsf - caller will free result
            return result;
        }
        fseek(raw_file, 0, SEEK_SET);

        pthread_t *threadid = (pthread_t *)malloc(sizeof(pthread_t) * maxquerythread);
        pqueue_bsf *pq_bsf = pqueue_bsf_init(k);
        approximate_topk(ts, paa, index, pq_bsf);
        ts_type *ts_buffer = (ts_type *)malloc(index->settings->ts_byte_size);

        int sum_of_lab = 0;

        // Early termination - perfect match found (distance = 0)
        if (pq_bsf->knn[k-1] == 0) {
            free(ts_buffer);
            free(threadid);
            fclose(raw_file);
            pqueue_bsf result = *pq_bsf;
            // Don't free pq_bsf - caller will free result
            return result;
        }
        
        // Always refine to get a tight BSF distance for exact results
        // This ensures the mindistance_worker phase can prune correctly
        refine_topk_answer(ts, paa, index, pq_bsf, minimum_distance, min_checked_leaves);
        
        unsigned long i;

        ParIS_LDCW_data *essdata = (ParIS_LDCW_data *)malloc(sizeof(ParIS_LDCW_data) * maxquerythread);
        pthread_rwlock_t lock_bsf = PTHREAD_RWLOCK_INITIALIZER;

        for (i = 0; i < (maxquerythread - 1); i++)
        {
            essdata[i].index = index;
            essdata[i].lock_bsf = &lock_bsf;
            essdata[i].start_number = i * (index->sax_cache_size / maxquerythread);
            essdata[i].stop_number = (i + 1) * (index->sax_cache_size / maxquerythread);
            essdata[i].paa = paa;
            essdata[i].ts = ts;
            essdata[i].bsfdistance = pq_bsf->knn[k - 1];
            essdata[i].sum_of_lab = 0;
        }
        essdata[maxquerythread - 1].index = index;
        essdata[maxquerythread - 1].lock_bsf = &lock_bsf;
        essdata[maxquerythread - 1].start_number = (maxquerythread - 1) * (index->sax_cache_size / maxquerythread);
        essdata[maxquerythread - 1].stop_number = index->sax_cache_size;
        essdata[maxquerythread - 1].paa = paa;
        essdata[maxquerythread - 1].ts = ts;
        essdata[maxquerythread - 1].bsfdistance = pq_bsf->knn[k - 1];
        essdata[maxquerythread - 1].sum_of_lab = 0;

        for(i = 0; i < maxquerythread; i++)
        {
            pthread_create(&(threadid[i]), NULL, mindistance_worker, (void*)&(essdata[i]));
        }
        for (i = 0; i < maxquerythread; i++)
        {
            pthread_join(threadid[i], NULL);
            sum_of_lab += essdata[i].sum_of_lab;
        }

        unsigned long* label_number = (unsigned long *)malloc(sizeof(unsigned long) * sum_of_lab);
        float* minidisvector = (float *)malloc(sizeof(float) * sum_of_lab);
        
        sum_of_lab = 0;
        for (i = 0; i < maxquerythread; i++)
        {
            memcpy(&(label_number[sum_of_lab]), essdata[i].label_number, sizeof(unsigned long) * essdata[i].sum_of_lab);
            memcpy(&(minidisvector[sum_of_lab]), essdata[i].minidisvector, sizeof(float) * essdata[i].sum_of_lab);
            free(essdata[i].label_number);
            free(essdata[i].minidisvector);
            sum_of_lab += essdata[i].sum_of_lab;
        }

        pthread_t *readthread = (pthread_t *)malloc(sizeof(pthread_t) * maxquerythread * MAXREADTHREAD);
        ParIS_read_worker_data readpointer;

        readpointer.ts = ts;
        readpointer.index = index;
        unsigned long readcounter = 0;
        readpointer.counter = &readcounter;
        readpointer.load_point = label_number;
        readpointer.lock_bsf = &lock_bsf;
        readpointer.minidisvector = minidisvector;
        readpointer.sum_of_lab = sum_of_lab;
        readpointer.pq_bsf = pq_bsf;

        for (i = 0; i < maxquerythread * MAXREADTHREAD; i++)
        {
            pthread_create(&(readthread[i]), NULL, topk_read_worker, (void*)&(readpointer));
        }
        
        for (i = 0; i < maxquerythread * MAXREADTHREAD; i++)
        {
            pthread_join(readthread[i], NULL);
        }

        free(readthread);
        free(threadid);

        free(essdata);
        free(minidisvector);
        free(label_number);
        free(ts_buffer);
        fclose(raw_file);

        pqueue_bsf result = *pq_bsf;
        // Note: We return by value (stack copy), but the internal pointers (knn, position, node) point to heap memory
        // The caller should free only the internal pointers, not the structure itself
        // Don't free pq_bsf here - the caller will free the internal pointers from the returned copy
        return result;
    }
}