#ifndef MESSI_HPP
#define MESSI_HPP

#include "SimilaritySearchAlgorithm.hpp"

#include <queue>
#include <cfloat>
#include <omp.h>

#include "../isax/iSAXIndex.hpp"

namespace diNoLib
{
    class Messi : public SimilaritySearchAlgorithm
    {
    private:
        float *database = nullptr;
        idx_t n_database = 0;
        idx_t dim = 0;
        //int num_threads = 1;
        int paa_segments = 16;
        int sax_cardinality = 8;
        int leaf_size = 2000;
        int min_leaf_size = 10;
        int initial_lbl_size = 2000;
        int flush_limit = 200000;
        int initial_fbl_size = 100;
        int total_loaded_leaves = 1;
        int tight_bound = 0;

        int search_workers = 2;
        int index_workers = 2;

        int read_block_length = 100000;

        isax_index_settings * index_settings = nullptr;
        isax_index *index = nullptr;

    public:
        Messi(DistanceType distance_type);
        void setNumThreads(int num_threads) {this->search_workers = num_threads; this->index_workers = num_threads;}
        int getNumThreads() const {return this->search_workers;}

        int getPaaSegments() const { return paa_segments; }
        int getSaxCardinality() const { return sax_cardinality; }
        int getLeafSize() const { return leaf_size; }
        int getMinLeafSize() const { return min_leaf_size; }
        int getInitialLblSize() const { return initial_lbl_size; }
        int getFlushLimit() const { return flush_limit; }
        int getInitialFblSize() const { return initial_fbl_size; }
        int getTotalLoadedLeaves() const { return total_loaded_leaves; }
        int getTightBound() const { return tight_bound; }
        int getSearchWorkers() const { return search_workers; }
        int getIndexWorkers() const { return index_workers; }
        int getReadBlockLength() const { return read_block_length; }

        void setPaaSegments(int paa_segments) { this->paa_segments = paa_segments; }
        void setSaxCardinality(int sax_cardinality) { this->sax_cardinality = sax_cardinality; }
        void setLeafSize(int leaf_size) { this->leaf_size = leaf_size; }
        void setMinLeafSize(int min_leaf_size) { this->min_leaf_size = min_leaf_size; }
        void setInitialLblSize(int initial_lbl_size) { this->initial_lbl_size = initial_lbl_size; }
        void setFlushLimit(int flush_limit) { this->flush_limit = flush_limit; }
        void setInitialFblSize(int initial_fbl_size) { this->initial_fbl_size = initial_fbl_size; }
        void setTotalLoadedLeaves(int total_loaded_leaves) { this->total_loaded_leaves = total_loaded_leaves; }
        void setTightBound(int tight_bound) { this->tight_bound = tight_bound; }
        void setSearchWorkers(int search_workers) { this->search_workers = search_workers; }
        void setIndexWorkers(int index_workers) { this->index_workers = index_workers; }
        void setReadBlockLength(int read_block_length) { this->read_block_length = read_block_length; }

        void buildIndex(const float *database, const idx_t n_database, const idx_t dim) override;
        void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) override;

        ~Messi();
    };

} // namespace diNoLib

#endif // MESSI_HPP