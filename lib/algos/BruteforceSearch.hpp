#ifndef BRUTEFORCESEARCH_HPP
#define BRUTEFORCESEARCH_HPP

#include "SimilaritySearchAlgorithm.hpp"

#include <queue>
#include <cfloat>

namespace diNoLib
{
    class BruteForceSearch : public SimilaritySearchAlgorithm
    {
    private:
        float* database = nullptr;
        idx_t n_database = 0;
        idx_t dim = 0;
                
    public:
        BruteForceSearch(DistanceType distance_type);
        void buildIndex(const float *database, const idx_t n_database, const idx_t dim) override;
        void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) override;
        /**
         * @param query a pointor to an array of float
         * @param n_query number of query vectors
         * @param k number of nearest neighbors
         * @param I output indices of nearest neighbors
         * @param D output distances of nearest neighbors
         */        

        ~BruteForceSearch();

    };

} // namespace diNoLib

#endif // BRUTEFORCESEARCH_HPP