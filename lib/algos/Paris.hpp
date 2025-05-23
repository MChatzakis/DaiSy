#ifndef PARIS_HPP
#define PARIS_HPP

#include "SimilaritySearchAlgorithm.hpp"

#include "isax/iSAXIndex.hpp"
#include "isax/iSAXSearch.hpp"

#include <queue>
#include <cfloat>
#include <omp.h> 

namespace diNoLib
{
    class Paris : public SimilaritySearchAlgorithm
    {
    private:
        int num_threads = 1;
        int paa_segments = 16;
        int sax_cardinality = 8;
        int leaf_size = 2000;
        int min_leaf_size = 10;
        int initial_lbl_size = 2000;
        int flush_limit = 200000;
        int initial_fbl_size = 100;
        int total_loaded_leaves = 1;
        int tight_bound = 0;
        int search_workers = 64;
        int index_workers = 32;
        int read_block_length = 100000;
        float minimum_distance = FLT_MAX;
        int min_checked_leaves = -1;
        int n_pqueue = 42;

        isax_index_settings *index_settings = nullptr;
        isax_index *index = nullptr;
                
    public:
        Paris(DistanceType distance_type);
        void setNumThreads(int num_threads);
        int getNumThreads() const;        
        void buildIndex(const float *database, const idx_t n_database, const idx_t dim) override;
        void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) override;     

        ~Paris();

    };

} // namespace diNoLib

#endif // PARIS_HPP