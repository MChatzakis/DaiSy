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

namespace daisy
{

    typedef struct SING_workerdata
    {
        isax_node *current_root_node;
        ts_type *paa, *paaU, *paaL, *ts, *uo, *lo;
        pqueue_t *pq;
        isax_index *index;
        float minimum_distance;
        int limit;
        pthread_mutex_t *lock_current_root_node;
        pthread_mutex_t *lock_queue;
        pthread_barrier_t *lock_barrier;
        pthread_rwlock_t *lock_bsf;
        query_result *bsf_result;
        int *node_counter;
        isax_node **nodelist;
        int amountnode;
        localStack *localstk;
        localStack *allstk;
        pthread_mutex_t *locallock, *alllock;
        int *queuelabel, *allqueuelabel;
        pqueue_t **allpq;
        int startqueuenumber;
        int warpWind;
        pqueue_bsf *pq_bsf;
        float *lbdmap;
        bool *labelvalue;
        bool *activenode;
        int *offsetvalue;
        float shortrate;
        unsigned long int *gpuoffset;
        unsigned long int *seriesnumber;
        unsigned long int rawdatanumber;
        int n_pqueue;
        float *rawfile;
    } SING_workerdata;

    typedef struct gap_workerdata
    {
        isax_node **nodelist;
        int amountnode, workerstartnode, workerstopnode;
        int *startnode, *stopnode, *gapstartnode, *gapstopnode;
        int *nodecounter, *nodecounter2;
        bool *activechunk;
        bool *activenode;
        isax_index *index;
        int chunknumber;
        float bsf;
        ts_type *paa, *paaU, *paaL, *ts, *uo, *lo;
        pthread_mutex_t *lockposition;
        unsigned long *offsetarray;
        int *chunkcounter;
        int *chunkfinishcounter;
        int **arrangenodearray;
        int *arrangenodearraynumber;
    } gap_workerdata;

    struct SingConfig
    {
        int search_workers = 4;
        int index_workers = 4;
        int warping_window = 10;
        int leaf_size = 128;
        int paa_segments = 16;
    };

    void insert_tree_node_m_hybridpqueueBreakpolyroot(float *paa, isax_node *node, isax_index *index, float bsf,
                                                      pqueue_t **pq, pthread_mutex_t *lock_queue, int *tnumber, int n_pqueue);

    void calculate_node_topk_SING(isax_index *index, isax_node *node, ts_type *query, ts_type *paa,
                                  pqueue_bsf *pq_bsf, pthread_rwlock_t *lock_queue, float *rawfile);

    void calculate_node_cal_topk_inmemory(isax_index *index, isax_node *node, ts_type *query, ts_type *paa,
                                          pqueue_bsf *pq_bsf, pthread_rwlock_t *lock_queue, float *rawfile);

    float minidist_paa_to_isax_Breakpoly(float *paa, sax_type *sax, sax_type *sax_cardinalities,
                                         sax_type max_bit_cardinality, int max_cardinality, int number_of_segments,
                                         int min_val, int max_val, float ratio_sqrt);

    class Sing : public SimilaritySearchAlgorithm
    {
    private:
        int aggressive_check = 0;
        int serial_scan = 0;
        int cpu_control_type = 81;
        int calculate_thread = 8;

        int read_block_length = 100000;
        int search_workers = 4;
        int index_workers = 4;
        int n_pqueue = 42;

        void index_creation_gpu(float *dataset, idx_t dataset_size, isax_index *idx);

        void searchIndexL2Square(const float *query, idx_t n_query, idx_t k, idx_t *I, float *D);
        void searchIndex_DTW(const float *query, idx_t n_query, idx_t k, idx_t *I, float *D);

#if SING_CUDA_ENABLED
        float *d_database = nullptr;
#endif

        bool use_cuda = false;

    public:
        Sing(DistanceType distance_type);
        void setNumThreads(int num_threads);
        int getNumThreads() const;
        void buildIndex(DataSource *data_source) override;
        void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) override;

        void printBuildIndexDebug() const;

        ~Sing();
    };

}

#endif