#ifndef ODYSSEY_HPP
#define ODYSSEY_HPP

#include "SimilaritySearchAlgorithm.hpp"

#include <queue>
#include <cfloat>
#include <omp.h> 

#if ODYSSEY_MPI
#include <mpi.h>
#endif

namespace diNoLib
{
    class Odyssey : public SimilaritySearchAlgorithm
    {
    private:
        int num_threads = 1;
                
    public:
        Odyssey(DistanceType distance_type);
        void setNumThreads(int num_threads);
        int getNumThreads() const;        
        void buildIndex(DataSource *data_source) override;
        void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) override;     

        ~Odyssey();

    };

} // namespace diNoLib

#endif // ODYSSEY_HPP