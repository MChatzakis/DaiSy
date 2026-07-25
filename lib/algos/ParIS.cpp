#include "ParIS.hpp"
#include "../isax/SAX.hpp"
#include "../isax/iSAXPqueue.hpp"
#include "../isax/iSAXIndex.hpp"
#include <stdexcept>
#include <pthread.h>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <string>
#include <unistd.h>  // getpid

namespace daisy
{

    ParIS::ParIS(DistanceType distance_type)
        : ParIS(distance_type, ParISConfig{})
    {
    }

    ParIS::ParIS(DistanceType distance_type, const ParISConfig &config)
        : SimilaritySearchAlgorithm(distance_type)
    {
        this->search_workers = config.search_workers;
        this->index_workers = config.index_workers;
        this->warping_window = config.warping_window;
        this->leaf_size = config.leaf_size;
        this->paa_segments = config.paa_segments;
    }

    void ParIS::buildIndex(DataSource *data_source)
    {
        this->dim = data_source->getDim();
        this->n_database = data_source->getTotalRecords();

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

        // Unique per-instance prefix so concurrent ParIS indices don't share on-disk files
        // (root_directory is used as a filename prefix: "<prefix>isax_file.sax", etc.).
        static std::atomic<unsigned long> paris_uid{0};
        this->index_prefix_ = "/tmp/paris_" + std::to_string((unsigned long)getpid()) + "_" +
                              std::to_string(paris_uid.fetch_add(1)) + "_";

        this->index_settings = isax_index_settings_init(this->index_prefix_.c_str(),
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
                                                        0,
                                                        this->bp_mode);

        // Equi-depth breakpoints. ParIS streams from a raw float32 file, so sample leading
        // rows from it to derive the distribution.
        if (this->bp_mode == BP_EQUIDEPTH && this->n_database > 0 && this->dim > 0)
        {
            size_t cap_rows = (size_t)2000000 / (size_t)std::max(1, this->paa_segments);
            size_t sample_rows = std::min((size_t)this->n_database, std::max((size_t)1, cap_rows));
            std::vector<float> sample((size_t)sample_rows * (size_t)this->dim);
            FILE *sf = fopen(filename, "rb");
            if (sf)
            {
                size_t got = fread(sample.data(), sizeof(float), sample.size(), sf);
                fclose(sf);
                compute_equidepth_breakpoints(this->index_settings, sample.data(), got / (size_t)this->dim);
            }
        }
        activateBreakpoints();

        this->index = isax_index_init(this->index_settings);
        isax_index *index = this->index;
        index->sax_cache_size = 0;

        /*
         * If a previous ParIS run left an isax_file.sax on disk, the
         * constructor opens it in read mode (it cannot know whether we are
         * building a new index).  Subsequent fwrite calls inside
         * isax_index_binary_file_m would then silently fail, leaving an
         * outdated cache with fewer entries (e.g., the 100k vs 200k mismatch
         * observed in the random‑walk tests).  Force a fresh truncateable
         * handle here so indexing always writes the full sax cache.
         */
        if (index->sax_file != nullptr)
        {
            char *sax_path = (char *)malloc((strlen(index->settings->root_directory) + 15) * sizeof(char));
            strcpy(sax_path, index->settings->root_directory);
            strcat(sax_path, "isax_file.sax");

            fclose(index->sax_file);
            index->sax_file = fopen(sax_path, "w+b");
            free(sax_path);

            if (index->sax_file == nullptr)
            {
                throw std::runtime_error("ParIS: cannot open sax cache file for writing");
            }
        }

        int ts_num = (this->n_database > 0) ? (int)this->n_database : 0;
        int calculate_thread = this->index_workers;

        if (ts_num == 0)
        {
            
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

        isax_index_binary_file_m(filename, ts_num, index, calculate_thread, this->read_block_length);

        /* ParIS is on-disk and reads SAX from isax_file.sax on demand during search.
         * The cache streamed by isax_index_binary_file_m can end up duplicated/misaligned
         * (two independent write paths), which makes the range search's Phase-1 SAX filter
         * prune valid candidates. Rewrite the cache cleanly from the raw file so record i's
         * SAX is exactly at offset i, then reopen it read-only. */
        if (index->sax_file != nullptr)
        {
            fclose(index->sax_file);
            index->sax_file = nullptr;
        }
        {
            char *sax_path = (char *)malloc((strlen(index->settings->root_directory) + 15) * sizeof(char));
            strcpy(sax_path, index->settings->root_directory);
            strcat(sax_path, "isax_file.sax");

            FILE *raw = fopen(filename, "rb");
            FILE *sf = fopen(sax_path, "wb");
            if (raw != nullptr && sf != nullptr)
            {
                ts_type *ts_buf = (ts_type *)malloc((size_t)index->settings->ts_byte_size);
                sax_type *sax_buf = (sax_type *)malloc((size_t)index->settings->sax_byte_size);
                for (int r = 0; r < ts_num; r++)
                {
                    if (fread(ts_buf, (size_t)index->settings->ts_byte_size, 1, raw) != 1)
                        break;
                    sax_from_ts(ts_buf, sax_buf, index->settings->ts_values_per_paa_segment,
                                index->settings->paa_segments, index->settings->sax_alphabet_cardinality,
                                index->settings->sax_bit_cardinality);
                    fwrite(sax_buf, (size_t)index->settings->sax_byte_size, 1, sf);
                }
                free(ts_buf);
                free(sax_buf);
            }
            if (raw != nullptr) fclose(raw);
            if (sf != nullptr) fclose(sf);

            index->sax_file = fopen(sax_path, "rb");
            free(sax_path);
            index->total_records = (unsigned long long)ts_num;
            index->sax_cache_size = (unsigned long)ts_num;
        }

        fprintf(stderr, ">>> Finished indexing\n");
    }

    void ParIS::searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        activateBreakpoints();
        if (this->distance_type == DistanceType::L2_SQUARED)
        {
            searchIndexL2Squared(query, n_query, k, I, D);
        }
        else if (this->distance_type == DistanceType::DTW)
        {
            searchIndexDTW(query, n_query, k, I, D);
        }
        else
        {
            fprintf(stderr, "Error: Unsupported distance type for ParIS index.\n");
            exit(1);
        }
    }

    void ParIS::searchIndexDTW(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        if (index == nullptr || index->sax_file == nullptr || index->total_records == 0)
        {
            fprintf(stderr, "Error: Index not built or sax file not ready\n");
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

        int warpWind = (this->warping_window > 0) ? this->warping_window : std::max(1, static_cast<int>(this->dim * 0.1));

        for (idx_t q_loaded = 0; q_loaded < n_query; q_loaded++)
        {
            
            const float *ts = query + q_loaded * this->dim;

            pqueue_bsf result = exact_DTWknn_serial_ParIS(
                (float *)ts,
                index,
                warpWind,
                k);

            for (idx_t ik = 0; ik < k; ik++)
            {
                D[q_loaded * k + ik] = result.knn[ik];
                I[q_loaded * k + ik] = result.position[ik];
            }

            if (result.position != nullptr)
            {
                free(result.position);
            }
            if (result.knn != nullptr)
            {
                free(result.knn);
            }
            if (result.node != nullptr)
            {
                free(result.node);
            }
        }
    }

    pqueue_bsf ParIS::exact_DTWknn_serial_ParIS(ts_type *ts, isax_index *index, int warpWind, int k)
    {
        float minimum_distance = this->minimum_distance;
        int min_checked_leaves = this->min_checked_leaves;

        ts_type *paa = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);
        paa_from_ts(ts, paa, index->settings->paa_segments, index->settings->ts_values_per_paa_segment);

        pqueue_bsf *pq_bsf = pqueue_bsf_init(k);

        approximate_topk_dtw(ts, paa, index, pq_bsf, warpWind);

        ts_type *upperLemire = (ts_type *)malloc(sizeof(ts_type) * index->settings->timeseries_size);
        ts_type *lowerLemire = (ts_type *)malloc(sizeof(ts_type) * index->settings->timeseries_size);
        ts_type *paaU = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);
        ts_type *paaL = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);

        lower_upper_lemire(ts, index->settings->timeseries_size, warpWind, lowerLemire, upperLemire);
        paa_from_ts(upperLemire, paaU, index->settings->paa_segments, index->settings->ts_values_per_paa_segment);
        paa_from_ts(lowerLemire, paaL, index->settings->paa_segments, index->settings->ts_values_per_paa_segment);

        ts_type *ts_buffer = (ts_type *)malloc(index->settings->ts_byte_size);
        int sum_of_lab = 0;
        int tight_bound = index->settings->tight_bound;
        int aggressive_check = index->settings->aggressive_check;
        float bsf_distance;
        unsigned long j;
        unsigned long i;

        int maxquerythread = this->search_workers;
        pthread_t *threadid = (pthread_t *)malloc(sizeof(pthread_t) * maxquerythread);
        ParIS_LDCW_data *essdata = (ParIS_LDCW_data *)malloc(sizeof(ParIS_LDCW_data) * maxquerythread);
        pthread_rwlock_t lock_bsf = PTHREAD_RWLOCK_INITIALIZER;

        for (i = 0; i < (maxquerythread - 1); i++)
        {
            essdata[i].index = index;
            essdata[i].lock_bsf = &lock_bsf;
            essdata[i].start_number = i * (index->sax_cache_size / maxquerythread);
            essdata[i].stop_number = (i + 1) * (index->sax_cache_size / maxquerythread);
            essdata[i].paa = paa;
            essdata[i].paaU = paaU;
            essdata[i].paaL = paaL;
            essdata[i].ts = ts;
            essdata[i].bsfdistance = pq_bsf->knn[pq_bsf->k - 1];
            essdata[i].sum_of_lab = 0;
        }
        essdata[maxquerythread - 1].index = index;
        essdata[maxquerythread - 1].lock_bsf = &lock_bsf;
        essdata[maxquerythread - 1].start_number = (maxquerythread - 1) * (index->sax_cache_size / maxquerythread);
        essdata[maxquerythread - 1].stop_number = index->sax_cache_size;
        essdata[maxquerythread - 1].paa = paa;
        essdata[maxquerythread - 1].paaU = paaU;
        essdata[maxquerythread - 1].paaL = paaL;
        essdata[maxquerythread - 1].ts = ts;
        essdata[maxquerythread - 1].bsfdistance = pq_bsf->knn[pq_bsf->k - 1];
        essdata[maxquerythread - 1].sum_of_lab = 0;

        for (i = 0; i < maxquerythread; i++)
        {
            pthread_create(&(threadid[i]), NULL, mindistance_worker_dtw, (void *)&(essdata[i]));
        }
        for (i = 0; i < maxquerythread; i++)
        {
            pthread_join(threadid[i], NULL);
            sum_of_lab += essdata[i].sum_of_lab;
        }

        unsigned long *label_number = (unsigned long *)malloc(sizeof(unsigned long) * sum_of_lab);
        float *minidisvector = (float *)malloc(sizeof(float) * sum_of_lab);
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
        unsigned long readcounter = 0;

        readpointer.ts = ts;
        readpointer.tsU = upperLemire;
        readpointer.tsL = lowerLemire;
        readpointer.index = index;
        readpointer.counter = &readcounter;
        readpointer.load_point = label_number;
        readpointer.lock_bsf = &lock_bsf;
        readpointer.minidisvector = minidisvector;
        readpointer.sum_of_lab = sum_of_lab;
        readpointer.warpWind = warpWind;
        readpointer.pq_bsf = pq_bsf;

        for (i = 0; i < maxquerythread * MAXREADTHREAD; i++)
        {
            pthread_create(&(readthread[i]), NULL, dtwknnreadworker, (void *)&(readpointer));
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
        free(upperLemire);
        free(lowerLemire);
        free(paaU);
        free(paaL);
        free(paa);
        free(ts_buffer);

        pqueue_bsf result = *pq_bsf;
        
        return result;
    }

    void ParIS::searchIndexL2Squared(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        if (index == nullptr || index->sax_file == nullptr || index->total_records == 0)
        {
            fprintf(stderr, "Error: Index not built or sax file not ready\n");
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

            paa_from_ts(ts, paa, index->settings->paa_segments, index->settings->ts_values_per_paa_segment);

            pqueue_bsf result = exact_topk_serial_ParIS((float *)ts, paa, index, this->minimum_distance, this->min_checked_leaves, k, this->search_workers);

            for (idx_t ik = 0; ik < k; ik++)
            {
                D[q_loaded * k + ik] = result.knn[ik];
                I[q_loaded * k + ik] = result.position[ik];
            }

            if (result.position != nullptr)
            {
                free(result.position);
            }
            if (result.knn != nullptr)
            {
                free(result.knn);
            }
            if (result.node != nullptr)
            {
                free(result.node);
            }
        }

        free(paa);
    }

    ParIS::~ParIS()
    {
        
        isax_index_settings *settings_to_free = nullptr;

        if (index != nullptr)
        {
            if (index->sax_cache != nullptr)
            {
                free(index->sax_cache);
                index->sax_cache = nullptr;
            }
            if (index->answer != nullptr)
            {
                free(index->answer);
                index->answer = nullptr;
            }
            if (index->fbl != nullptr)
            {
                destroy_fbl(index->fbl);
                index->fbl = nullptr;
            }
            if (index->sax_file != nullptr)
            {
                fclose(index->sax_file);
                index->sax_file = nullptr;
            }
            
            settings_to_free = index->settings;
            free(index);
            index = nullptr;
        }

        isax_index_settings *settings = (settings_to_free != nullptr) ? settings_to_free : index_settings;
        if (settings != nullptr)
        {
            
            if (settings->raw_filename != nullptr)
            {
                free(settings->raw_filename);
                settings->raw_filename = nullptr;
            }
            if (settings->bit_masks != nullptr)
            {
                free(settings->bit_masks);
                settings->bit_masks = nullptr;
            }
            if (settings->max_sax_cardinalities != nullptr)
            {
                free(settings->max_sax_cardinalities);
                settings->max_sax_cardinalities = nullptr;
            }
            free(settings);
        }

        index_settings = nullptr;
    }

    void *mindistance_worker_dtw(void *essdata)
    {
        unsigned long i;
        float mindist;
        isax_index *index = ((ParIS_LDCW_data *)essdata)->index;
        unsigned long start_number = ((ParIS_LDCW_data *)essdata)->start_number;
        unsigned long stop_number = ((ParIS_LDCW_data *)essdata)->stop_number;
        ts_type *paaU = ((ParIS_LDCW_data *)essdata)->paaU;
        ts_type *paaL = ((ParIS_LDCW_data *)essdata)->paaL;

        ((ParIS_LDCW_data *)essdata)->label_number = (unsigned long *)malloc(sizeof(unsigned long) * 10000);
        ((ParIS_LDCW_data *)essdata)->minidisvector = (float *)malloc(sizeof(float) * 10000);
        unsigned long max_number = 10000;

        /* On-disk: open SAX file and read one record at a time (no in-memory sax_cache) */
        char *sax_path = (char *)malloc((strlen(index->settings->root_directory) + 15) * sizeof(char));
        strcpy(sax_path, index->settings->root_directory);
        strcat(sax_path, "isax_file.sax");
        FILE *sax_file = fopen(sax_path, "rb");
        free(sax_path);
        if (sax_file == nullptr)
        {
            return NULL;
        }
        sax_type *sax_buf = (sax_type *)malloc((size_t)index->settings->sax_byte_size);
        if (sax_buf == nullptr)
        {
            fclose(sax_file);
            return NULL;
        }

        for (i = start_number; i < stop_number; i++)
        {
            if (fseek(sax_file, (long)(i * (unsigned long)index->settings->sax_byte_size), SEEK_SET) != 0)
                continue;
            if (fread(sax_buf, (size_t)index->settings->sax_byte_size, 1, sax_file) != 1)
                continue;

            mindist = minidist_paa_to_isax_raw_DTW_SIMD(paaU, paaL, sax_buf, index->settings->max_sax_cardinalities,
                                                        index->settings->sax_bit_cardinality,
                                                        index->settings->sax_alphabet_cardinality,
                                                        index->settings->paa_segments, MINVAL, MAXVAL,
                                                        index->settings->mindist_sqrt);
            if (mindist <= ((ParIS_LDCW_data *)essdata)->bsfdistance)
            {
                if (((ParIS_LDCW_data *)essdata)->sum_of_lab >= max_number)
                {
                    max_number = (max_number + 10000);
                    unsigned long *change_lab = ((ParIS_LDCW_data *)essdata)->label_number;
                    float *change_minivec = ((ParIS_LDCW_data *)essdata)->minidisvector;
                    ((ParIS_LDCW_data *)essdata)->label_number = (unsigned long *)malloc(sizeof(unsigned long) * max_number);
                    ((ParIS_LDCW_data *)essdata)->minidisvector = (float *)malloc(sizeof(float) * max_number);
                    memcpy(((ParIS_LDCW_data *)essdata)->label_number, change_lab, sizeof(unsigned long) * (max_number - 10000));
                    memcpy(((ParIS_LDCW_data *)essdata)->minidisvector, change_minivec, sizeof(float) * (max_number - 10000));
                    free(change_lab);
                    free(change_minivec);
                }
                ((ParIS_LDCW_data *)essdata)->label_number[((ParIS_LDCW_data *)essdata)->sum_of_lab] = i;
                ((ParIS_LDCW_data *)essdata)->minidisvector[((ParIS_LDCW_data *)essdata)->sum_of_lab] = mindist;
                ((ParIS_LDCW_data *)essdata)->sum_of_lab++;
            }
        }
        free(sax_buf);
        fclose(sax_file);
        return NULL;
    }

    void *dtwknnreadworker(void *read_pointer)
    {
        isax_index *index = ((ParIS_read_worker_data *)read_pointer)->index;
        pqueue_bsf *pq_bsf = ((ParIS_read_worker_data *)read_pointer)->pq_bsf;
        FILE *raw_file = fopen(index->settings->raw_filename, "rb");
        if (raw_file == NULL)
        {
            return NULL; 
        }
        fseek(raw_file, 0, SEEK_SET);
        ts_type *ts_buffer = (ts_type *)malloc(index->settings->ts_byte_size);
        ts_type *ts = ((ParIS_read_worker_data *)read_pointer)->ts;
        unsigned long t = 0, p;
        unsigned long sum_of_lab = ((ParIS_read_worker_data *)read_pointer)->sum_of_lab;
        int warpWind = ((ParIS_read_worker_data *)read_pointer)->warpWind;
        float *minidisvector = ((ParIS_read_worker_data *)read_pointer)->minidisvector;
        float *lowerLemire = ((ParIS_read_worker_data *)read_pointer)->tsL;
        float *upperLemire = ((ParIS_read_worker_data *)read_pointer)->tsU;

        float bsf, dist, dist2;
        float *cb = (float *)calloc(index->settings->timeseries_size, sizeof(float));
        float *cb1 = (float *)calloc(index->settings->timeseries_size, sizeof(float));
        int length = 2 * warpWind + 1;
        float *tSum = (float *)malloc(sizeof(float) * length);
        
        float *pCost = (float *)malloc(sizeof(float) * length);
        
        float *rDist = (float *)malloc(sizeof(float) * length);

        while (1)
        {
            pthread_rwlock_rdlock(((ParIS_read_worker_data *)read_pointer)->lock_bsf);
            bsf = pq_bsf->knn[pq_bsf->k - 1];
            pthread_rwlock_unlock(((ParIS_read_worker_data *)read_pointer)->lock_bsf);

            t = __sync_fetch_and_add(((ParIS_read_worker_data *)read_pointer)->counter, 1);
            if (t >= sum_of_lab)
            {
                break;
            }

            p = ((ParIS_read_worker_data *)read_pointer)->load_point[t];
            if (minidisvector[t] < bsf)
            {
                
                fseek(raw_file, p * index->settings->ts_byte_size, SEEK_SET);
                size_t items_read = fread(ts_buffer, index->settings->ts_byte_size, 1, raw_file);
                (void)items_read; 

                pthread_rwlock_rdlock(((ParIS_read_worker_data *)read_pointer)->lock_bsf);
                bsf = pq_bsf->knn[pq_bsf->k - 1];
                pthread_rwlock_unlock(((ParIS_read_worker_data *)read_pointer)->lock_bsf);

                dist2 = lb_keogh_data_bound(ts_buffer, upperLemire, lowerLemire, cb1, index->settings->timeseries_size, bsf);

                if (dist2 < pq_bsf->knn[pq_bsf->k - 1])
                {
                    cb[index->settings->timeseries_size - 1] = cb1[index->settings->timeseries_size - 1];
                    for (int k = index->settings->timeseries_size - 2; k >= 0; k--)
                        cb[k] = cb[k + 1] + cb1[k];

                    dist = dtwsimdPruned(ts, ts_buffer, cb, index->settings->timeseries_size, warpWind, pq_bsf->knn[pq_bsf->k - 1], tSum, pCost, rDist);

                    if (dist < pq_bsf->knn[pq_bsf->k - 1])
                    {
                        pthread_rwlock_wrlock(((ParIS_read_worker_data *)read_pointer)->lock_bsf);
                        
                        bsf = pq_bsf->knn[pq_bsf->k - 1];
                        if (dist < bsf)
                        {
                            pqueue_bsf_insert(pq_bsf, dist, p, NULL);
                        }
                        pthread_rwlock_unlock(((ParIS_read_worker_data *)read_pointer)->lock_bsf);
                    }
                }
            }
        }

        free(tSum);
        free(pCost);
        free(cb);
        free(cb1);
        free(rDist);
        free(ts_buffer);
        fclose(raw_file);

        return NULL;
    }

    void calculate_node_topk(isax_index *index, isax_node *node, ts_type *query, pqueue_bsf *pq_bsf)
    {
        FILE *raw_file = fopen(index->settings->raw_filename, "rb");
        if (raw_file == nullptr)
        {
            return;
        }

        if (node->buffer != NULL)
        {
            int i;
            
            for (i = 0; i < node->buffer->full_buffer_size; i++)
            {
                float dist = ts_euclidean_distance_SIMD(query, node->buffer->full_ts_buffer[i],
                                                        index->settings->timeseries_size, pq_bsf->knn[pq_bsf->k - 1]);
                if (dist <= pq_bsf->knn[pq_bsf->k - 1])
                {
                    pqueue_bsf_insert(pq_bsf, dist, 0, node);
                }
            }
            
            for (i = 0; i < node->buffer->tmp_full_buffer_size; i++)
            {
                float dist = ts_euclidean_distance_SIMD(query, node->buffer->tmp_full_ts_buffer[i],
                                                        index->settings->timeseries_size, pq_bsf->knn[pq_bsf->k - 1]);
                if (dist <= pq_bsf->knn[pq_bsf->k - 1])
                {
                    pqueue_bsf_insert(pq_bsf, dist, 0, node);
                }
            }
            
            for (i = 0; i < node->buffer->partial_buffer_size; i++)
            {
                file_position_type pos = *node->buffer->partial_position_buffer[i];
                fseek(raw_file, pos, SEEK_SET);
                ts_type *ts_buffer = (ts_type *)malloc(index->settings->ts_byte_size);
                size_t items_read = fread(ts_buffer, index->settings->ts_byte_size, 1, raw_file);
                (void)items_read; 
                float dist = ts_euclidean_distance_SIMD(query, ts_buffer, index->settings->timeseries_size, pq_bsf->knn[pq_bsf->k - 1]);
                if (dist <= pq_bsf->knn[pq_bsf->k - 1])
                {
                    pqueue_bsf_insert(pq_bsf, dist, pos / index->settings->timeseries_size, node);
                }
                free(ts_buffer);
            }
        }

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
                    (void)items_read; 
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

        fclose(raw_file);
    }

    void calculate_node_topk_dtw(isax_index *index, isax_node *node, ts_type *query, pqueue_bsf *pq_bsf, int warpWind)
    {
        FILE *raw_file = fopen(index->settings->raw_filename, "rb");
        if (raw_file == nullptr)
        {
            return;
        }

        float *cb = (float *)calloc(index->settings->timeseries_size, sizeof(float));

        if (node->buffer != NULL)
        {
            int i;
            
            for (i = 0; i < node->buffer->full_buffer_size; i++)
            {
                float dist = dtw(query, node->buffer->full_ts_buffer[i], cb,
                                 index->settings->timeseries_size, warpWind, pq_bsf->knn[pq_bsf->k - 1]);
                if (dist <= pq_bsf->knn[pq_bsf->k - 1])
                {
                    pqueue_bsf_insert(pq_bsf, dist, 0, node);
                }
            }
            
            for (i = 0; i < node->buffer->tmp_full_buffer_size; i++)
            {
                float dist = dtw(query, node->buffer->tmp_full_ts_buffer[i], cb,
                                 index->settings->timeseries_size, warpWind, pq_bsf->knn[pq_bsf->k - 1]);
                if (dist <= pq_bsf->knn[pq_bsf->k - 1])
                {
                    pqueue_bsf_insert(pq_bsf, dist, 0, node);
                }
            }
            
            for (i = 0; i < node->buffer->partial_buffer_size; i++)
            {
                file_position_type pos = *node->buffer->partial_position_buffer[i];
                fseek(raw_file, pos, SEEK_SET);
                ts_type *ts_buffer = (ts_type *)malloc(index->settings->ts_byte_size);
                size_t items_read = fread(ts_buffer, index->settings->ts_byte_size, 1, raw_file);
                (void)items_read; 
                float dist = dtw(query, ts_buffer, cb, index->settings->timeseries_size, warpWind, pq_bsf->knn[pq_bsf->k - 1]);
                if (dist <= pq_bsf->knn[pq_bsf->k - 1])
                {
                    pqueue_bsf_insert(pq_bsf, dist, pos / index->settings->timeseries_size, node);
                }
                free(ts_buffer);
            }
        }

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
                    (void)items_read; 
                    float dist = dtw(query, ts_buffer, cb, index->settings->timeseries_size, warpWind, pq_bsf->knn[pq_bsf->k - 1]);
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

        free(cb);
        fclose(raw_file);
    }

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
                        (void)items_read; 
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

    void approximate_topk(ts_type *ts, ts_type *paa, isax_index *index, pqueue_bsf *pq_bsf)
    {
        sax_type *sax = (sax_type *)malloc(sizeof(sax_type) * index->settings->paa_segments);
        sax_from_paa(paa, sax, index->settings->paa_segments,
                     index->settings->sax_alphabet_cardinality,
                     index->settings->sax_bit_cardinality);

        root_mask_type root_mask = 0;
        CREATE_MASK(root_mask, index, sax);

        if (index->fbl->soft_buffers[(int)root_mask].initialized)
        {
            isax_node *node = index->fbl->soft_buffers[(int)root_mask].node;

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

            while (!node->is_leaf)
            {
                int location = index->settings->sax_bit_cardinality - 1 -
                               node->split_data->split_mask[node->split_data->splitpoint];
                root_mask_type mask = index->settings->bit_masks[location];

                if (sax[node->split_data->splitpoint] & mask)
                {
                    node = node->right_child;
                }
                else
                {
                    node = node->left_child;
                }

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
        else
        {
        }
        for (int i = 0; i < pq_bsf->k - 1; ++i)
        {
            pq_bsf->knn[i] = pq_bsf->knn[pq_bsf->k - 1];
        }
        free(sax);
    }

    void approximate_topk_dtw(ts_type *ts, ts_type *paa, isax_index *index, pqueue_bsf *pq_bsf, int warpWind)
    {
        sax_type *sax = (sax_type *)malloc(sizeof(sax_type) * index->settings->paa_segments);
        sax_from_paa(paa, sax, index->settings->paa_segments,
                     index->settings->sax_alphabet_cardinality,
                     index->settings->sax_bit_cardinality);

        root_mask_type root_mask = 0;
        CREATE_MASK(root_mask, index, sax);

        if (index->fbl->soft_buffers[(int)root_mask].initialized)
        {
            isax_node *node = index->fbl->soft_buffers[(int)root_mask].node;

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

            while (!node->is_leaf)
            {
                int location = index->settings->sax_bit_cardinality - 1 -
                               node->split_data->split_mask[node->split_data->splitpoint];
                root_mask_type mask = index->settings->bit_masks[location];

                if (sax[node->split_data->splitpoint] & mask)
                {
                    node = node->right_child;
                }
                else
                {
                    node = node->left_child;
                }

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

            calculate_node_topk_dtw(index, node, ts, pq_bsf, warpWind);
        }
        else
        {
        }
        for (int i = 0; i < pq_bsf->k - 1; ++i)
        {
            pq_bsf->knn[i] = pq_bsf->knn[pq_bsf->k - 1];
        }
        free(sax);
    }

    void refine_topk_answer(ts_type *ts, ts_type *paa, isax_index *index,
                            pqueue_bsf *pq_bsf,
                            float minimum_distance, int limit)
    {
        int tight_bound = index->settings->tight_bound;
        int aggressive_check = index->settings->aggressive_check;

        pqueue_t *pq = pqueue_init(index->settings->root_nodes_size,
                                   cmp_pri, get_pri, set_pri, get_pos, set_pos);

        isax_node *current_root_node = index->first_node;
        while (current_root_node != NULL)
        {
            query_result *mindist_result = (query_result *)malloc(sizeof(query_result));
            mindist_result->distance = minidist_paa_to_isax(paa, current_root_node->isax_values,
                                                            current_root_node->isax_cardinalities,
                                                            index->settings->sax_bit_cardinality,
                                                            index->settings->sax_alphabet_cardinality,
                                                            index->settings->paa_segments,
                                                            MINVAL, MAXVAL,
                                                            index->settings->mindist_sqrt);
            mindist_result->node = current_root_node;
            if (mindist_result->distance < pq_bsf->knn[pq_bsf->k - 1])
            {
                pqueue_insert(pq, mindist_result);
            }
            else
            {
                free(mindist_result);
            }
            current_root_node = current_root_node->next;
        }
        query_result *n;
        int checks = 0;
        while ((n = (query_result *)pqueue_pop(pq)))
        {
            
            if (n->distance >= pq_bsf->knn[pq_bsf->k - 1] || n->distance > minimum_distance)
            {
                pqueue_insert(pq, n);
                break;
            }
            else
            {
                
                if (n->node->is_leaf)
                {

                    if (!n->node->has_full_data_file &&
                        !n->node->has_partial_data_file &&
                        (n->node->leaf_size > index->settings->min_leaf_size) &&
                        n->node->buffer != NULL &&
                        (n->node->buffer->full_buffer_size > 0 ||
                         n->node->buffer->partial_buffer_size > 0 ||
                         n->node->buffer->tmp_full_buffer_size > 0 ||
                         n->node->buffer->tmp_partial_buffer_size > 0))
                    {
                        
                        split_node(index, n->node);
                        pqueue_insert(pq, n);
                        continue;
                    }
                    
                    if (tight_bound)
                    {
                        float mindistance = calculate_minimum_distance(index, n->node, ts, paa);
                        if (mindistance >= pq_bsf->knn[pq_bsf->k - 1])
                        {
                            free(n);
                            continue;
                        }
                    }
                    
                    checks++;
                    calculate_node_topk(index, n->node, ts, pq_bsf);

                }
                else
                {

                    if (n->node->left_child != NULL && n->node->left_child->isax_cardinalities != NULL)
                    {
                        if (n->node->left_child->is_leaf && !n->node->left_child->has_partial_data_file && aggressive_check)
                        {
                            calculate_node_topk(index, n->node->left_child, ts, pq_bsf);
                        }
                        else
                        {
                            query_result *mindist_result = (query_result *)malloc(sizeof(query_result));
                            mindist_result->distance = minidist_paa_to_isax(paa, n->node->left_child->isax_values,
                                                                            n->node->left_child->isax_cardinalities,
                                                                            index->settings->sax_bit_cardinality,
                                                                            index->settings->sax_alphabet_cardinality,
                                                                            index->settings->paa_segments,
                                                                            MINVAL, MAXVAL,
                                                                            index->settings->mindist_sqrt);
                            mindist_result->node = n->node->left_child;
                            if (mindist_result->distance < pq_bsf->knn[pq_bsf->k - 1])
                            {
                                pqueue_insert(pq, mindist_result);
                            }
                            else
                            {
                                free(mindist_result);
                            }
                        }
                    }
                    if (n->node->right_child != NULL && n->node->right_child->isax_cardinalities != NULL)
                    {
                        if (n->node->right_child->is_leaf && !n->node->right_child->has_partial_data_file && aggressive_check)
                        {
                            calculate_node_topk(index, n->node->right_child, ts, pq_bsf);
                        }
                        else
                        {
                            query_result *mindist_result = (query_result *)malloc(sizeof(query_result));
                            mindist_result->distance = minidist_paa_to_isax(paa, n->node->right_child->isax_values,
                                                                            n->node->right_child->isax_cardinalities,
                                                                            index->settings->sax_bit_cardinality,
                                                                            index->settings->sax_alphabet_cardinality,
                                                                            index->settings->paa_segments,
                                                                            MINVAL, MAXVAL,
                                                                            index->settings->mindist_sqrt);
                            mindist_result->node = n->node->right_child;
                            if (mindist_result->distance < pq_bsf->knn[pq_bsf->k - 1])
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

            free(n);
        }
        
        while ((n = (query_result *)pqueue_pop(pq)))
        {
            free(n);
        }
        
        for (int i = 0; i < pq_bsf->k - 1; ++i)
        {
            pq_bsf->knn[i] = pq_bsf->knn[pq_bsf->k - 1];
        }
        pqueue_free(pq);
    }

    void *mindistance_worker(void *essdata)
    {
        unsigned long i;
        float mindist;
        isax_index *index = ((ParIS_LDCW_data *)essdata)->index;
        unsigned long start_number = ((ParIS_LDCW_data *)essdata)->start_number;
        unsigned long stop_number = ((ParIS_LDCW_data *)essdata)->stop_number;
        ts_type *paa = ((ParIS_LDCW_data *)essdata)->paa;

        ((ParIS_LDCW_data *)essdata)->label_number = (unsigned long *)malloc(sizeof(unsigned long) * 10000);
        ((ParIS_LDCW_data *)essdata)->minidisvector = (float *)malloc(sizeof(float) * 10000);
        unsigned long max_number = 10000;

        /* On-disk: open SAX file and read one record at a time (no in-memory sax_cache) */
        char *sax_path = (char *)malloc((strlen(index->settings->root_directory) + 15) * sizeof(char));
        strcpy(sax_path, index->settings->root_directory);
        strcat(sax_path, "isax_file.sax");
        FILE *sax_file = fopen(sax_path, "rb");
        free(sax_path);
        if (sax_file == nullptr)
        {
            return NULL;
        }
        sax_type *sax_buf = (sax_type *)malloc((size_t)index->settings->sax_byte_size);
        if (sax_buf == nullptr)
        {
            fclose(sax_file);
            return NULL;
        }

        for (i = start_number; i < stop_number; i++)
        {
            if (fseek(sax_file, (long)(i * (unsigned long)index->settings->sax_byte_size), SEEK_SET) != 0)
                continue;
            if (fread(sax_buf, (size_t)index->settings->sax_byte_size, 1, sax_file) != 1)
                continue;

            mindist = minidist_paa_to_isax_rawa_SIMD(paa, sax_buf, index->settings->max_sax_cardinalities,
                                                     index->settings->sax_bit_cardinality,
                                                     index->settings->sax_alphabet_cardinality,
                                                     index->settings->paa_segments, MINVAL, MAXVAL,
                                                     index->settings->mindist_sqrt);
            if (mindist <= ((ParIS_LDCW_data *)essdata)->bsfdistance)
            {
                if (((ParIS_LDCW_data *)essdata)->sum_of_lab >= max_number)
                {
                    max_number = (max_number + 10000);
                    unsigned long *change_lab = ((ParIS_LDCW_data *)essdata)->label_number;
                    float *change_minivec = ((ParIS_LDCW_data *)essdata)->minidisvector;
                    ((ParIS_LDCW_data *)essdata)->label_number = (unsigned long *)malloc(sizeof(unsigned long) * max_number);
                    ((ParIS_LDCW_data *)essdata)->minidisvector = (float *)malloc(sizeof(float) * max_number);
                    memcpy(((ParIS_LDCW_data *)essdata)->label_number, change_lab, sizeof(unsigned long) * (max_number - 10000));
                    memcpy(((ParIS_LDCW_data *)essdata)->minidisvector, change_minivec, sizeof(float) * (max_number - 10000));
                    free(change_lab);
                    free(change_minivec);
                }
                ((ParIS_LDCW_data *)essdata)->label_number[((ParIS_LDCW_data *)essdata)->sum_of_lab] = i;
                ((ParIS_LDCW_data *)essdata)->minidisvector[((ParIS_LDCW_data *)essdata)->sum_of_lab] = mindist;
                ((ParIS_LDCW_data *)essdata)->sum_of_lab++;
            }
        }
        free(sax_buf);
        fclose(sax_file);
        return NULL;
    }

    void *topk_read_worker(void *read_pointer)
    {
        isax_index *index = ((ParIS_read_worker_data *)read_pointer)->index;
        pqueue_bsf *pq_bsf = ((ParIS_read_worker_data *)read_pointer)->pq_bsf;
        FILE *raw_file = fopen(index->settings->raw_filename, "rb");
        fseek(raw_file, 0, SEEK_SET);
        ts_type *ts_buffer = (ts_type *)malloc(index->settings->ts_byte_size);
        ts_type *ts = ((ParIS_read_worker_data *)read_pointer)->ts;
        unsigned long t = 0, p;
        unsigned long sum_of_lab = ((ParIS_read_worker_data *)read_pointer)->sum_of_lab;
        float *minidisvector = ((ParIS_read_worker_data *)read_pointer)->minidisvector;

        float bsf, dist;
        while (1)
        {

            bsf = pq_bsf->knn[pq_bsf->k - 1];

            t = __sync_fetch_and_add(((ParIS_read_worker_data *)read_pointer)->counter, 1);
            
            if (t >= sum_of_lab)
            {
                break;
            }

            p = ((ParIS_read_worker_data *)read_pointer)->load_point[t];

            fseek(raw_file, p * index->settings->ts_byte_size, SEEK_SET);
            size_t items_read = fread(ts_buffer, index->settings->ts_byte_size, 1, raw_file);
            (void)items_read; 
            
            bsf = pq_bsf->knn[pq_bsf->k - 1];
            dist = ts_euclidean_distance_SIMD(ts, ts_buffer, index->settings->timeseries_size, bsf);
            
            if (dist <= bsf)
            {
                pthread_rwlock_wrlock(((ParIS_read_worker_data *)read_pointer)->lock_bsf);
                
                bsf = pq_bsf->knn[pq_bsf->k - 1];
                if (dist <= bsf)
                {
                    pqueue_bsf_insert(pq_bsf, dist, p, NULL);
                }
                pthread_rwlock_unlock(((ParIS_read_worker_data *)read_pointer)->lock_bsf);
            }
            
        }

        free(ts_buffer);
        fclose(raw_file);
        return NULL;
    }

    void *paris_range_read_worker(void *arg)
    {
        paris_range_read_worker_data *wd = (paris_range_read_worker_data *)arg;
        isax_index *index = wd->index;
        float r = wd->r;
        ts_type *ts = wd->ts;
        unsigned long sum_of_lab = wd->sum_of_lab;

        FILE *raw_file = fopen(index->settings->raw_filename, "rb");
        if (raw_file == nullptr)
            return nullptr;

        ts_type *ts_buffer = (ts_type *)malloc(index->settings->ts_byte_size);
        if (ts_buffer == nullptr) {
            fclose(raw_file);
            return nullptr;
        }

        while (1) {
            unsigned long t = __sync_fetch_and_add(wd->counter, 1);
            if (t >= sum_of_lab)
                break;

            unsigned long p = wd->load_point[t];
            fseek(raw_file, (long)(p * (unsigned long)index->settings->ts_byte_size), SEEK_SET);
            size_t items_read = fread(ts_buffer, index->settings->ts_byte_size, 1, raw_file);
            (void)items_read;

            float dist = ts_euclidean_distance_SIMD(ts, ts_buffer, index->settings->timeseries_size, FLT_MAX);
            if (dist <= r) {
                pthread_mutex_lock(wd->lock_results);
                wd->range_results->emplace_back(dist, (idx_t)p);
                pthread_mutex_unlock(wd->lock_results);
            }
        }

        free(ts_buffer);
        fclose(raw_file);
        return nullptr;
    }

    pqueue_bsf exact_topk_serial_ParIS(ts_type *ts, ts_type *paa, isax_index *index, float minimum_distance, int min_checked_leaves, int k, int maxquerythread)
    {
        FILE *raw_file = fopen(index->settings->raw_filename, "rb");
        if (raw_file == nullptr)
        {
            fprintf(stderr, "Error: Could not open raw file for search\n");
            pqueue_bsf *pq_bsf = pqueue_bsf_init(k);
            pqueue_bsf result = *pq_bsf;
            
            return result;
        }
        fseek(raw_file, 0, SEEK_SET);

        pthread_t *threadid = (pthread_t *)malloc(sizeof(pthread_t) * maxquerythread);
        pqueue_bsf *pq_bsf = pqueue_bsf_init(k);
        approximate_topk(ts, paa, index, pq_bsf);
        ts_type *ts_buffer = (ts_type *)malloc(index->settings->ts_byte_size);

        int sum_of_lab = 0;

        if (pq_bsf->knn[k - 1] == 0)
        {
            free(ts_buffer);
            free(threadid);
            fclose(raw_file);
            pqueue_bsf result = *pq_bsf;
            
            return result;
        }

        float approximate_bsf = pq_bsf->knn[k - 1];

        refine_topk_answer(ts, paa, index, pq_bsf, minimum_distance, min_checked_leaves);

        float bsf_for_mindistance = (approximate_bsf != FLT_MAX) ? approximate_bsf : FLT_MAX;

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
            essdata[i].bsfdistance = bsf_for_mindistance;
            essdata[i].sum_of_lab = 0;
        }
        essdata[maxquerythread - 1].index = index;
        essdata[maxquerythread - 1].lock_bsf = &lock_bsf;
        essdata[maxquerythread - 1].start_number = (maxquerythread - 1) * (index->sax_cache_size / maxquerythread);
        essdata[maxquerythread - 1].stop_number = index->sax_cache_size;
        essdata[maxquerythread - 1].paa = paa;
        essdata[maxquerythread - 1].ts = ts;
        essdata[maxquerythread - 1].bsfdistance = bsf_for_mindistance;
        essdata[maxquerythread - 1].sum_of_lab = 0;

        for (i = 0; i < maxquerythread; i++)
        {
            pthread_create(&(threadid[i]), NULL, mindistance_worker, (void *)&(essdata[i]));
        }
        for (i = 0; i < maxquerythread; i++)
        {
            pthread_join(threadid[i], NULL);
            sum_of_lab += essdata[i].sum_of_lab;
        }

        unsigned long *label_number = (unsigned long *)malloc(sizeof(unsigned long) * sum_of_lab);
        float *minidisvector = (float *)malloc(sizeof(float) * sum_of_lab);

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
            pthread_create(&(readthread[i]), NULL, topk_read_worker, (void *)&(readpointer));
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

        return result;
    }

    void ParIS::searchIndex(const float *query, idx_t n_query, const SearchConfig &config,
                            std::vector<std::vector<idx_t>> &I,
                            std::vector<std::vector<float>> &D)
    {
        activateBreakpoints();
        if (config.type == QueryType::TOP_K) {
            SimilaritySearchAlgorithm::searchIndex(query, n_query, config, I, D);
            return;
        }

        if (index == nullptr || index->sax_file == nullptr || index->total_records == 0) {
            fprintf(stderr, "Error: Index not built or sax file not ready\n");
            I.assign(n_query, {});
            D.assign(n_query, {});
            return;
        }

        float r = config.r;
        int maxquerythread = this->search_workers;
        ts_type *paa = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);

        I.assign(n_query, {});
        D.assign(n_query, {});

        for (idx_t q_loaded = 0; q_loaded < n_query; q_loaded++) {
            const float *ts = query + q_loaded * this->dim;
            paa_from_ts(ts, paa, index->settings->paa_segments, index->settings->ts_values_per_paa_segment);

            /* Phase 1: SAX-level filter — collect candidate record indices with lb <= r */
            pthread_t *threadid = (pthread_t *)malloc(sizeof(pthread_t) * (size_t)maxquerythread);
            ParIS_LDCW_data *essdata = (ParIS_LDCW_data *)malloc(sizeof(ParIS_LDCW_data) * (size_t)maxquerythread);

            for (int i = 0; i < maxquerythread - 1; i++) {
                essdata[i].index = index;
                essdata[i].lock_bsf = nullptr;
                essdata[i].start_number = (unsigned long)i * (index->sax_cache_size / (unsigned long)maxquerythread);
                essdata[i].stop_number = (unsigned long)(i + 1) * (index->sax_cache_size / (unsigned long)maxquerythread);
                essdata[i].paa = paa;
                essdata[i].ts = (ts_type *)ts;
                essdata[i].bsfdistance = r;
                essdata[i].sum_of_lab = 0;
                essdata[i].label_number = nullptr;
                essdata[i].minidisvector = nullptr;
            }
            essdata[maxquerythread - 1].index = index;
            essdata[maxquerythread - 1].lock_bsf = nullptr;
            essdata[maxquerythread - 1].start_number = (unsigned long)(maxquerythread - 1) * (index->sax_cache_size / (unsigned long)maxquerythread);
            essdata[maxquerythread - 1].stop_number = index->sax_cache_size;
            essdata[maxquerythread - 1].paa = paa;
            essdata[maxquerythread - 1].ts = (ts_type *)ts;
            essdata[maxquerythread - 1].bsfdistance = r;
            essdata[maxquerythread - 1].sum_of_lab = 0;
            essdata[maxquerythread - 1].label_number = nullptr;
            essdata[maxquerythread - 1].minidisvector = nullptr;

            for (int i = 0; i < maxquerythread; i++)
                pthread_create(&threadid[i], NULL, mindistance_worker, (void *)&essdata[i]);
            for (int i = 0; i < maxquerythread; i++)
                pthread_join(threadid[i], NULL);

            unsigned long sum_of_lab = 0;
            for (int i = 0; i < maxquerythread; i++)
                sum_of_lab += essdata[i].sum_of_lab;

            unsigned long *label_number = (unsigned long *)malloc(sizeof(unsigned long) * (sum_of_lab + 1));
            unsigned long offset = 0;
            for (int i = 0; i < maxquerythread; i++) {
                if (essdata[i].label_number != nullptr) {
                    memcpy(label_number + offset, essdata[i].label_number, sizeof(unsigned long) * essdata[i].sum_of_lab);
                    offset += essdata[i].sum_of_lab;
                    free(essdata[i].label_number);
                    free(essdata[i].minidisvector);
                }
            }
            free(essdata);
            free(threadid);

            /* Phase 2: exact distance computation — collect all with dist <= r */
            std::vector<std::pair<float, idx_t>> hits;
            pthread_mutex_t lock_results = PTHREAD_MUTEX_INITIALIZER;
            unsigned long readcounter = 0;
            int nread = maxquerythread * MAXREADTHREAD;
            pthread_t *readthread = (pthread_t *)malloc(sizeof(pthread_t) * (size_t)nread);
            paris_range_read_worker_data rdwd;
            rdwd.index = index;
            rdwd.ts = (ts_type *)ts;
            rdwd.counter = &readcounter;
            rdwd.load_point = label_number;
            rdwd.sum_of_lab = sum_of_lab;
            rdwd.r = r;
            rdwd.range_results = &hits;
            rdwd.lock_results = &lock_results;

            for (int i = 0; i < nread; i++)
                pthread_create(&readthread[i], NULL, paris_range_read_worker, (void *)&rdwd);
            for (int i = 0; i < nread; i++)
                pthread_join(readthread[i], NULL);

            free(readthread);
            free(label_number);
            pthread_mutex_destroy(&lock_results);

            std::sort(hits.begin(), hits.end());
            I[q_loaded].resize(hits.size());
            D[q_loaded].resize(hits.size());
            for (size_t j = 0; j < hits.size(); j++) {
                D[q_loaded][j] = hits[j].first;
                I[q_loaded][j] = hits[j].second;
            }
        }

        free(paa);
    }
}
