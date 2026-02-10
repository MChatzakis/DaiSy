#include "LbBruteforce.hpp"
#include "../isax/iSAXIndex.hpp"

namespace daisy
{

    LbBruteforce::LbBruteforce(DistanceType distance_type)
        : SimilaritySearchAlgorithm(distance_type)
    {
    }

    void LbBruteforce::setNumThreads(int num_threads)
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

    int LbBruteforce::getNumThreads() const
    {
        return this->num_threads;
    }

    void LbBruteforce::buildIndex(DataSource *data_source)
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

        this->database = new float[this->n_database * this->dim];
        float *record = new float[this->dim];
        idx_t idx = 0;
        while (data_source->nextRecord(record))
        {
            std::copy(record, record + this->dim, this->database + idx * this->dim);
            idx++;
        }
        delete[] record;

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

        this->db_sax_representations = (sax_type **)malloc(n_database * sizeof(sax_type *));

#pragma omp parallel for num_threads(num_threads)
        for (idx_t dbi = 0; dbi < n_database; dbi++)
        {
            this->db_sax_representations[dbi] = (sax_type *)malloc(sizeof(sax_type) * this->index->settings->paa_segments);
            float *vi_vec = this->database + dbi * dim;

            if (!this->distance_computer->compute_sax_from_ts(vi_vec,
                                                              this->db_sax_representations[dbi],
                                                              this->index->settings->ts_values_per_paa_segment,
                                                              this->index->settings->paa_segments,
                                                              this->index->settings->sax_alphabet_cardinality,
                                                              this->index->settings->sax_bit_cardinality))
            {
                fprintf(stderr, "error: cannot insert record in index, since sax representation failed to be created");
                exit(EXIT_FAILURE);
            }
        }
    }

    void LbBruteforce::searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
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
            std::cerr << "Error: Unsupported distance type for LbBruteforce index." << std::endl;
            exit(1);
        }
    }

    void LbBruteforce::searchIndexL2Squared(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
#pragma omp parallel num_threads(num_threads)
        {
#pragma omp for
            for (idx_t qi = 0; qi < n_query; qi++)
            {
                std::priority_queue<std::pair<float, idx_t>> pq;
                const float *q_vec = query + qi * dim;
                float bound = FLT_MAX;

                ts_type *q_paa = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);
                if (q_paa == nullptr)
                {
                    fprintf(stderr, "Error: Failed to allocate memory for PAA representation\n");
                    continue;
                }

                this->distance_computer->compute_paa_from_ts(
                    q_vec, q_paa,
                    index->settings->paa_segments,
                    index->settings->ts_values_per_paa_segment);

                for (idx_t dbi = 0; dbi < n_database; ++dbi)
                {
                    const float *db_vec = database + dbi * dim;

                    float minimum_distance = this->distance_computer->compute_minidist_SIMD(
                        q_paa,
                        db_sax_representations[dbi],
                        (const int *)(index->settings->max_sax_cardinalities),
                        index->settings->sax_bit_cardinality,
                        index->settings->sax_alphabet_cardinality,
                        index->settings->paa_segments,
                        MINVAL,
                        MAXVAL,
                        index->settings->mindist_sqrt);
                    if (minimum_distance < bound)
                    {
                        float dist = this->distance_computer->compute_dist_SIMD(const_cast<float *>(q_vec),
                                                                                const_cast<float *>(db_vec),
                                                                                dim,
                                                                                bound);

                        if ((idx_t)pq.size() < k)
                        {
                            pq.emplace(dist, dbi);
                        }
                        else if (dist < pq.top().first)
                        {
                            pq.pop();
                            pq.emplace(dist, dbi);
                            bound = pq.top().first;
                        }
                    }
                }

                free(q_paa);

                for (idx_t j = k; j > 0; --j)
                {
                    D[qi * k + (j - 1)] = pq.top().first;
                    I[qi * k + (j - 1)] = pq.top().second;
                    pq.pop();
                }
            }
        }
    }

    void LbBruteforce::searchIndexDTW(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
#pragma omp parallel num_threads(num_threads)
        {
#pragma omp for
            for (idx_t qi = 0; qi < n_query; qi++)
            {
                std::priority_queue<std::pair<float, idx_t>> pq;
                const float *q_vec = query + qi * dim;
                float bound = FLT_MAX;

                float *lower_envelope = (float *)malloc(sizeof(float) * dim);
                float *upper_envelope = (float *)malloc(sizeof(float) * dim);
                ts_type *q_paa_upper = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);
                ts_type *q_paa_lower = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);

                if (!lower_envelope || !upper_envelope || !q_paa_upper || !q_paa_lower)
                {
                    fprintf(stderr, "Error: Failed to allocate memory for DTW computation\n");

                    if (lower_envelope)
                        free(lower_envelope);
                    if (upper_envelope)
                        free(upper_envelope);
                    if (q_paa_upper)
                        free(q_paa_upper);
                    if (q_paa_lower)
                        free(q_paa_lower);
                    continue;
                }

                int warping_window = static_cast<int>(dim * 0.1);
                lower_upper_lemire(const_cast<float *>(q_vec), dim, warping_window, lower_envelope, upper_envelope);

                this->distance_computer->compute_paa_from_ts(
                    upper_envelope, q_paa_upper,
                    index->settings->paa_segments,
                    index->settings->ts_values_per_paa_segment);
                this->distance_computer->compute_paa_from_ts(
                    lower_envelope, q_paa_lower,
                    index->settings->paa_segments,
                    index->settings->ts_values_per_paa_segment);

                for (idx_t dbi = 0; dbi < n_database; ++dbi)
                {
                    const float *db_vec = database + dbi * dim;

                    float minimum_distance = this->distance_computer->wrap_minidist_paa_to_isax_DTW(
                        q_paa_upper, q_paa_lower,
                        db_sax_representations[dbi],
                        (sax_type *)(index->settings->max_sax_cardinalities),
                        index->settings->sax_bit_cardinality,
                        index->settings->sax_alphabet_cardinality,
                        index->settings->paa_segments,
                        MINVAL,
                        MAXVAL,
                        index->settings->mindist_sqrt);

                    if (minimum_distance < bound)
                    {

                        float dist = this->distance_computer->compute_dist(const_cast<float *>(q_vec),
                                                                           const_cast<float *>(db_vec),
                                                                           dim,
                                                                           bound);

                        if ((idx_t)pq.size() < k)
                        {
                            pq.emplace(dist, dbi);
                        }
                        else if (dist < pq.top().first)
                        {
                            pq.pop();
                            pq.emplace(dist, dbi);
                            bound = pq.top().first;
                        }
                    }
                }

                for (idx_t j = k; j > 0; --j)
                {
                    D[qi * k + (j - 1)] = pq.top().first;
                    I[qi * k + (j - 1)] = pq.top().second;
                    pq.pop();
                }

                free(lower_envelope);
                free(upper_envelope);
                free(q_paa_upper);
                free(q_paa_lower);
            }
        }
    }

    LbBruteforce::~LbBruteforce()
    {
        delete[] database;

        for (idx_t dbi = 0; dbi < n_database; ++dbi)
        {
            free(db_sax_representations[dbi]);
        }
        free(db_sax_representations);

        if (this->index)
        {
            free(this->index);
            this->index = nullptr;
        }

        if (this->index_settings)
        {
            free(this->index_settings);
            this->index_settings = nullptr;
        }
    }
}