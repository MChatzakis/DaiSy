#ifndef BRUTEFORCE_HPP
#define BRUTEFORCE_HPP

#include "SimilaritySearchAlgorithm.hpp"

#include <queue>
#include <cfloat>
#include <omp.h> 

namespace diNoLib
{
    class BruteForceSearch : public SimilaritySearchAlgorithm
    {                
    public:
        BruteForceSearch(DistanceType distance_type);
        void setNumThreads(int num_threads);
        int getNumThreads() const { return this->num_threads; }         
        void buildIndex(const float *database, const idx_t n_database, const idx_t dim) override;
        void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) override;  

        ~BruteForceSearch();

    };

} // namespace diNoLib

#endif // BRUTEFORCE_HPP