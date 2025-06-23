#ifndef SINGSEARCH_HPP
#define SINGSEARCH_HPP

#include "SimilaritySearchAlgorithm.hpp"

#include <queue>
#include <cfloat>
#include <omp.h> 

#if SING_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace diNoLib
{
    class Sing : public SimilaritySearchAlgorithm
    {
    private:
        float* database = nullptr;
        idx_t n_database = 0;
        idx_t dim = 0;
        int num_threads = 1;

        #if SING_CUDA_ENABLED
        float* d_database = nullptr;
        #endif

        bool use_cuda = false;
                
    public:
        Sing(DistanceType distance_type);
        void setNumThreads(int num_threads);
        int getNumThreads() const;        
        void buildIndex(const float *database, const idx_t n_database, const idx_t dim) override;
        void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) override;     

        ~Sing();

    };

} // namespace diNoLib

#endif // SINGSEARCH_HPP