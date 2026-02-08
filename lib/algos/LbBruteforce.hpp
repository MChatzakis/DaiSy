#ifndef LBBRUTEFORCE_HPP
#define LBBRUTEFORCE_HPP

#include "SimilaritySearchAlgorithm.hpp"

#include "../isax/iSAXSearch.hpp"
#include "../isax/iSAXTypes.hpp"

#include <queue>
#include <cfloat>
#include <omp.h>

namespace daisy
{
    /**
     * @class LbBruteforce
     * @brief Lower-bound pruning with brute-force refinement
     *
     * Accelerates similarity search by first computing lower-bound distances using
     * iSAX representations, then pruning candidates before exact distance computation.
     * Supports both L2 and DTW metrics with configurable iSAX indexing parameters.
     * Requires in-memory data; file-based input is not supported.
     *
     * @note Use setPaaSegments(), setSaxCardinality(), etc. to tune performance.
     */
    class LbBruteforce : public SimilaritySearchAlgorithm
    {
    private:
        // iSAX configuration parameters
        int paa_segments = 16;
        int sax_cardinality = 8;
        int leaf_size = 2000;
        int min_leaf_size = 10;
        int initial_lbl_size = 2000;
        int flush_limit = 200000;
        int initial_fbl_size = 100;
        int total_loaded_leaves = 1;
        int tight_bound = 0;
        int num_threads = 1;

        // iSAX index structures
        isax_index_settings *index_settings = nullptr;
        isax_index *index = nullptr;
        sax_type **db_sax_representations = nullptr;

    public:
        LbBruteforce(DistanceType distance_type);
        void setNumThreads(int num_threads);
        int getNumThreads() const;

        int getPaaSegments() const { return paa_segments; }
        int getSaxCardinality() const { return sax_cardinality; }
        int getLeafSize() const { return leaf_size; }
        int getMinLeafSize() const { return min_leaf_size; }
        int getInitialLblSize() const { return initial_lbl_size; }
        int getFlushLimit() const { return flush_limit; }
        int getInitialFblSize() const { return initial_fbl_size; }
        int getTotalLoadedLeaves() const { return total_loaded_leaves; }
        int getTightBound() const { return tight_bound; }

        void setPaaSegments(int paa_segments) { this->paa_segments = paa_segments; }
        void setSaxCardinality(int sax_cardinality) { this->sax_cardinality = sax_cardinality; }
        void setLeafSize(int leaf_size) { this->leaf_size = leaf_size; }
        void setMinLeafSize(int min_leaf_size) { this->min_leaf_size = min_leaf_size; }
        void setInitialLblSize(int initial_lbl_size) { this->initial_lbl_size = initial_lbl_size; }
        void setFlushLimit(int flush_limit) { this->flush_limit = flush_limit; }
        void setInitialFblSize(int initial_fbl_size) { this->initial_fbl_size = initial_fbl_size; }
        void setTotalLoadedLeaves(int total_loaded_leaves) { this->total_loaded_leaves = total_loaded_leaves; }
        void setTightBound(int tight_bound) { this->tight_bound = tight_bound; }

        // Bring base class buildIndex overloads into scope
        using SimilaritySearchAlgorithm::buildIndex;

        void buildIndex(DataSource *data_source) override;

        // LbBruteforce only supports in-memory data
        void buildIndex(const std::string &filename, idx_t dim, idx_t n_database = 0) override
        {
            throw std::runtime_error("LbBruteforce requires in-memory data. Use buildIndex(database, n_database, dim) instead.");
        }

        void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) override;
        void searchIndexL2Squared(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D);
        void searchIndexDTW(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D);

        ~LbBruteforce();
    };

} // namespace daisy

#endif // LBBRUTEFORCE_HPP