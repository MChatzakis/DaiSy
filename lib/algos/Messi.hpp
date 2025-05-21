#ifndef MESSI_HPP
#define MESSI_HPP

#include "SimilaritySearchAlgorithm.hpp"

#include <queue>
#include <cfloat>
#include <omp.h> 

namespace diNoLib
{
    class Messi : public SimilaritySearchAlgorithm
    {
    private:
        float* database = nullptr;
        idx_t n_database = 0;
        idx_t dim = 0;
        int num_threads = 1;
                
    public:
        Messi(DistanceType distance_type);
        void setNumThreads(int num_threads);
        int getNumThreads() const;        
        void buildIndex(const float *database, const idx_t n_database, const idx_t dim) override;
        void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) override;     

        ~Messi();

    };

} // namespace diNoLib

#endif // MESSI_HPP