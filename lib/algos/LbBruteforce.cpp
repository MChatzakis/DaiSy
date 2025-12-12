#include "LbBruteforce.hpp"
#include "../isax/iSAXIndex.hpp"

namespace diNoLib
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

    void LbBruteforce::buildIndex(const float *database, const idx_t n_database, const idx_t dim)
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

        // initialize db_sax_representations
        this->db_sax_representations = (sax_type **)malloc(n_database * sizeof(sax_type *));

#pragma omp parallel for num_threads(num_threads)
        for (idx_t dbi = 0; dbi < n_database; dbi++) // replaced n_database*dim with n_database
        {
            this->db_sax_representations[dbi] = (sax_type *)malloc(sizeof(sax_type) * this->index->settings->paa_segments);
            float *vi_vec = this->database + dbi * dim;

            // todo: move those dist functions in the distance computer!
            // if (sax_from_ts(
            //         vi_vec,
            //         this->db_sax_representations[dbi],
            //         this->index->settings->ts_values_per_paa_segment,
            //         this->index->settings->paa_segments,
            //         this->index->settings->sax_alphabet_cardinality,
            //         this->index->settings->sax_bit_cardinality) != SUCCESS)
            // {
            //     fprintf(stderr, "error: cannot insert record in index, since sax representation failed to be created");
            //     exit(EXIT_FAILURE);
            // }
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
                float bound = FLT_MAX; // initialize bound to max float

                ts_type *q_paa = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);
                if (q_paa == nullptr) {
                    fprintf(stderr, "Error: Failed to allocate memory for PAA representation\n");
                    continue; // Skip this query
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
                }

                // Free the allocated PAA memory
                free(q_paa);

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

    void LbBruteforce::searchIndexDTW(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
#pragma omp parallel num_threads(num_threads)
        {
#pragma omp for
            for (idx_t qi = 0; qi < n_query; qi++)
            {
                std::priority_queue<std::pair<float, idx_t>> pq;
                const float *q_vec = query + qi * dim;
                float bound = FLT_MAX; // initialize bound to max float

                // Allocate memory for DTW envelopes and PAA representations
                float *lower_envelope = (float *)malloc(sizeof(float) * dim);
                float *upper_envelope = (float *)malloc(sizeof(float) * dim);
                ts_type *q_paa_upper = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);
                ts_type *q_paa_lower = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);

                // Check memory allocation
                if (!lower_envelope || !upper_envelope || !q_paa_upper || !q_paa_lower) {
                    fprintf(stderr, "Error: Failed to allocate memory for DTW computation\n");
                    // Clean up any successful allocations
                    if (lower_envelope) free(lower_envelope);
                    if (upper_envelope) free(upper_envelope);
                    if (q_paa_upper) free(q_paa_upper);
                    if (q_paa_lower) free(q_paa_lower);
                    continue; // Skip this query
                }

                // Compute DTW envelopes with warping window
                // For DTW we use a default warping window, could be made configurable
                int warping_window = static_cast<int>(dim * 0.1); // 10% of time series length
                lower_upper_lemire(const_cast<float *>(q_vec), dim, warping_window, lower_envelope, upper_envelope);


                // Compute PAA for upper and lower envelopes
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

                    // Use DTW-specific minimum distance computation
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
                        // Compute actual DTW distance
                        float dist = this->distance_computer->compute_dist(const_cast<float *>(q_vec),
                                                                           const_cast<float *>(db_vec),
                                                                           dim,
                                                                           bound);

                        if ((idx_t)pq.size() < k) // maintain max-heap
                        {
                            pq.emplace(dist, dbi); // equivalent to pq.push(make_pair(dist, dbi));
                            fprintf("Inserted into pq 1\n");
                            fflush(stdout);
                        }
                        else if (dist < pq.top().first)
                        {
                            pq.pop();
                            pq.emplace(dist, dbi);
                            bound = pq.top().first; // update the bound variable for pruning
                            fprintf("Inserted into pq 2\n");
                            fflush(stdout);
                        }
                    }
                }

                // Free the allocated memory
                free(lower_envelope);
                free(upper_envelope);
                free(q_paa_upper);
                free(q_paa_lower);

                // store top-k results in reverse order
                for (idx_t j = k; j > 0; --j)
                {
                    D[qi * k + (j - 1)] = pq.top().first;
                    I[qi * k + (j - 1)] = pq.top().second;
                    pq.pop();
                }

                // Clean up allocated memory
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

        // delete the sax representations as well
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