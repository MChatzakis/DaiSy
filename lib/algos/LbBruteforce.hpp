#ifndef LBBRUTEFORCE_HPP
#define LBBRUTEFORCE_HPP

#include "SimilaritySearchAlgorithm.hpp"

#include "../isax/iSAXSearch.hpp"

#include <queue>
#include <cfloat>
#include <omp.h>

namespace diNoLib
{
    class LbBruteforce : public SimilaritySearchAlgorithm
    {
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

        void buildIndex(const float *database, const idx_t n_database, const idx_t dim) override;
        void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) override;

        ~LbBruteforce();
    };

} // namespace diNoLib

#endif // LBBRUTEFORCE_HPP