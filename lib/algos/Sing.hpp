#ifndef SINGSEARCH_HPP
#define SINGSEARCH_HPP

#include "SimilaritySearchAlgorithm.hpp"

#include <queue>
#include <cfloat>
#include <omp.h>

#include "../isax/iSAXIndex.hpp"
#include "../isax/iSAXPqueue.hpp"

#if SING_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace diNoLib
{

    struct SingConfig
    {
        int search_workers = 4;   // threads for querying
        int index_workers = 2;     // threads for indexing
        int warping_window = 10;  // warping window size (typically 10% of time series length)
        int leaf_size = 2000;
        int paa_segments = 16;
    };

    class Sing : public SimilaritySearchAlgorithm
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
        int aggressive_check = 0;    // NO
        float minimum_distance = FLT_MAX;
        int serial_scan = 0;         // NO
        int min_checked_leaves = -1;
        int cpu_control_type = 81;   // NO
        int calculate_thread = 8;    // NO

        int read_block_length = 100000;
        int search_workers = 4;
        int index_workers = 2;
        int n_pqueue = 42;
        int warping_window = 10;  // warping window size (typically 10% of time series length)

        isax_index_settings *index_settings = nullptr;
        isax_index *index = nullptr;

        void index_creation_gpu(float *dataset, idx_t dataset_size, isax_index *idx);

        #if SING_CUDA_ENABLED
        float* d_database = nullptr;
        #endif

        bool use_cuda = false;
                
    public:
        Sing(DistanceType distance_type);
        void setNumThreads(int num_threads);
        int getNumThreads() const;        
        void buildIndex(DataSource *data_source) override;
        void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) override;

        /** Debug: stampa statistiche dell'indice dopo buildIndex (FBL, buffer, sax_cache). */
        void printBuildIndexDebug() const;

        ~Sing();

    };

} // namespace diNoLib

#endif // SINGSEARCH_HPP