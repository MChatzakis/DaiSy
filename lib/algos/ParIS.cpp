#include "ParIS.hpp"
#include "../isax/SAX.hpp"
#include <stdexcept>

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

        fprintf(stderr, ">>> Finished indexing\n");
    }

    void ParIS::searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        // TODO: Implement ParIS search algorithm using iSAX index
        // For now, return empty results to allow testing of buildIndex phase
        fprintf(stderr, "Warning: ParIS::searchIndex is not yet implemented. Only buildIndex phase is available.\n");
        
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

    ParIS::~ParIS()
    {
        // Cleanup iSAX index structures
        if (index != nullptr) {
            if (index->sax_cache != nullptr) {
                free(index->sax_cache);
            }
            if (index->answer != nullptr) {
                free(index->answer);
            }
            if (index->fbl != nullptr) {
                destroy_fbl(index->fbl);
            }
            if (index->sax_file != nullptr) {
                fclose(index->sax_file);
            }
            free(index);
        }
        
        if (index_settings != nullptr) {
            if (index_settings->bit_masks != nullptr) {
                free(index_settings->bit_masks);
            }
            if (index_settings->max_sax_cardinalities != nullptr) {
                free(index_settings->max_sax_cardinalities);
            }
            free(index_settings);
        }
    }
}