#include "Paris.hpp"

namespace diNoLib
{

    Paris::Paris(DistanceType distance_type)
        : SimilaritySearchAlgorithm(distance_type)
    {
    }

    void Paris::setNumThreads(int num_threads)
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

    int Paris::getNumThreads() const
    {
        return this->num_threads;
    }

    void Paris::buildIndex(const float *database, const idx_t n_database, const idx_t dim)
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
                                                        1,                         // new index
                                                        0);

        this->index = isax_index_init(this->index_settings);
        isax_index *index = this->index;

        idx_t ts_loaded = 0;
        sax_type *sax = (sax_type *)malloc(sizeof(sax_type) * index->settings->paa_segments);

        while (ts_loaded < this->n_database)
        {
            ts_type *ts = &this->database[ts_loaded * this->dim];
            file_position_type pos = ts_loaded * this->dim;

            if (sax_from_ts(ts, sax, index->settings->ts_values_per_paa_segment, index->settings->paa_segments, index->settings->sax_alphabet_cardinality, index->settings->sax_bit_cardinality) == SUCCESS)
            {
                isax_fbl_index_insert(index, sax, &pos);
                ts_loaded++;
            }
            else
            {
                fprintf(stderr, "error: cannot insert record in index, since sax representation failed to be created");
                exit(EXIT_FAILURE);
            }
        }

        free(sax);

        flush_fbl(index->fbl, index);

        fprintf(stderr, ">>> Finished indexing\n");
    }

    void Paris::searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        //todo
    }

    Paris::~Paris()
    {
        delete[] database;
    }
}