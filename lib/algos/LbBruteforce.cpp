#include "LbBruteforce.hpp"

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

        // todo: Make this parallel!
        for (idx_t dbi = 0; dbi < n_database * dim; dbi++)
        {
            this->db_sax_representations[dbi] = (sax_type *)malloc(sizeof(sax_type) * this->index->settings->paa_segments);
            float *vi_vec = this->database + dbi * dim;

            // todo: move those dist functions in the distance computer!
            if (sax_from_ts(
                    vi_vec,
                    this->db_sax_representations[dbi],
                    this->index->settings->ts_values_per_paa_segment,
                    this->index->settings->paa_segments,
                    this->index->settings->sax_alphabet_cardinality,
                    this->index->settings->sax_bit_cardinality) != SUCCESS)
            {
                fprintf(stderr, "error: cannot insert record in index, since sax representation failed to be created");
                exit(EXIT_FAILURE);
            }
        }
    }

    void LbBruteforce::searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
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
                paa_from_ts((float *)q_vec, q_paa, index->settings->paa_segments, index->settings->ts_values_per_paa_segment);

                for (idx_t dbi = 0; dbi < n_database; ++dbi)
                {
                    const float *db_vec = database + dbi * dim;
                    
                    float minimum_distance = minidist_paa_to_isax_rawa_SIMD(q_paa,
                                                                            db_sax_representations[dbi],
                                                                            index->settings->max_sax_cardinalities,
                                                                            index->settings->sax_bit_cardinality,
                                                                            index->settings->sax_alphabet_cardinality,
                                                                            index->settings->paa_segments,
                                                                            MINVAL,
                                                                            MAXVAL,
                                                                            index->settings->mindist_sqrt);

                    if (minimum_distance < bound)
                    {   
                        // todo: rename this as "compute_dist_SIMD"
                        float dist = this->distance_computer->compute_dist(const_cast<float *>(q_vec),
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
                            // bound = pq.top().first; // update the `bound` variable
                        }
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

    LbBruteforce::~LbBruteforce()
    {
        delete[] database;

        // delete the sax representations as well
        for (idx_t dbi = 0; dbi < n_database; ++dbi)
        {
            free(db_sax_representations[dbi]);
        }
        free(db_sax_representations);

        // todo free the index here as well
    }
}