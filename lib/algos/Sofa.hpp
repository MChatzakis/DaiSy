#ifndef SOFA_HPP
#define SOFA_HPP

#ifdef SOFA_FFTW_ENABLED

#include "SimilaritySearchAlgorithm.hpp"
#include <queue>
#include <cfloat>
#include <omp.h>
#include "../isax/iSAXIndex.hpp"
#include "../isax/iSAXPqueue.hpp"

namespace daisy
{
    struct SofaConfig
    {
        int  search_workers  = 4;
        int  index_workers   = 2;
        int  leaf_size       = 20000;
        int  min_leaf_size   = 20000;
        int  paa_segments    = 16;
        int  sample_size     = 100000;
        int  histogram_type  = 2;
        int  coeff_number    = 0;
        bool is_norm         = false;
    };

    typedef struct SOFA_workerdata
    {
        isax_node        **nodelist;
        int                amountnode;
        ts_type           *fft;
        ts_type           *ts;
        pqueue_t          *pq;
        isax_index        *index;
        float              minimum_distance;
        pthread_mutex_t   *lock_current_root_node;
        pthread_mutex_t   *lock_queue;
        pthread_barrier_t *lock_barrier;
        pthread_rwlock_t  *lock_bsf;
        int               *node_counter;
        pthread_mutex_t   *alllock;
        int               *allqueuelabel;
        pqueue_t         **allpq;
        int                startqueuenumber;
        pqueue_bsf        *pq_bsf;
        int                n_pqueue;
        float             *rawfile;
        float            **bins;
        bool               is_norm;
    } SOFA_workerdata;

    void *SOFA_topk_search_worker(void *rfdata);

    void insert_tree_node_sofa(float **bins, bool is_norm, float *fft,
                               isax_node *node, isax_index *index, float bsf,
                               pqueue_t **pq, pthread_mutex_t *lock_queue,
                               int *tnumber, int n_pqueue);

    void calculate_node_topk_sofa(isax_index *index, isax_node *node,
                                  ts_type *query, float *fft,
                                  float **bins, bool is_norm,
                                  pqueue_bsf *pq_bsf,
                                  pthread_rwlock_t *lock_bsf, float *rawfile);

    class Sofa : public SimilaritySearchAlgorithm
    {
    private:
        int  read_block_length = 100000;
        int  search_workers = 4;
        int  index_workers  = 2;
        int  n_pqueue       = 42;
        bool owns_database  = false;

        float **bins        = nullptr;
        float  *binsv       = nullptr;
        int    *coefficients = nullptr;

        int  sample_size    = 100000;
        int  histogram_type = 2;
        int  coeff_number   = 0;
        bool is_norm        = false;

        void sfaBinsInit();
        void sfaSetBins();
        void sfaFreeBins();

        pqueue_bsf sofaSearchTopkL2Squared(float *ts, float *fft, node_list *nodelist, idx_t k);
        void searchIndexL2Squared(const float *query, idx_t n_query, idx_t k, idx_t *I, float *D);

    public:
        Sofa(DistanceType distance_type);
        Sofa(DistanceType distance_type, const SofaConfig &config);

        using SimilaritySearchAlgorithm::buildIndex;

        void buildIndex(DataSource *data_source) override;

        void buildIndex(const std::string &filename, idx_t dim, idx_t n_database = 0) override
        {
            throw std::runtime_error(
                "Sofa requires in-memory data. Use buildIndex(database, n_database, dim) instead.");
        }

        void searchIndex(const float *query, idx_t n_query, idx_t k, idx_t *I, float *D) override;

        int  getNumThreads()        const { return SimilaritySearchAlgorithm::num_threads; }
        int  getReadBlockLength()   const { return read_block_length; }
        int  getSearchWorkers()     const { return search_workers; }
        int  getIndexWorkers()      const { return index_workers; }
        int  getPaaSegments()       const { return paa_segments; }
        int  getSaxCardinality()    const { return sax_cardinality; }
        int  getLeafSize()          const { return leaf_size; }
        int  getMinLeafSize()       const { return min_leaf_size; }
        int  getInitialLblSize()    const { return initial_lbl_size; }
        int  getFlushLimit()        const { return flush_limit; }
        int  getInitialFblSize()    const { return initial_fbl_size; }
        int  getTotalLoadedLeaves() const { return total_loaded_leaves; }
        int  getTightBound()        const { return tight_bound; }
        int  getSampleSize()        const { return sample_size; }
        int  getHistogramType()     const { return histogram_type; }
        int  getCoeffNumber()       const { return coeff_number; }
        bool getIsNorm()            const { return is_norm; }

        void setNumThreads(int n) {
            SimilaritySearchAlgorithm::num_threads = n;
            search_workers = n;
        }
        void setReadBlockLength(int n)   { read_block_length = n; }
        void setSearchWorkers(int n)     { search_workers = n; }
        void setIndexWorkers(int n)      { index_workers = n; }
        void setPaaSegments(int n)       { paa_segments = n; }
        void setSaxCardinality(int n)    { sax_cardinality = n; }
        void setLeafSize(int n)          { leaf_size = n; }
        void setMinLeafSize(int n)       { min_leaf_size = n; }
        void setInitialLblSize(int n)    { initial_lbl_size = n; }
        void setFlushLimit(int n)        { flush_limit = n; }
        void setInitialFblSize(int n)    { initial_fbl_size = n; }
        void setTotalLoadedLeaves(int n) { total_loaded_leaves = n; }
        void setTightBound(int n)        { tight_bound = n; }
        void setSampleSize(int n)        { sample_size = n; }
        void setHistogramType(int n)     { histogram_type = n; }
        void setCoeffNumber(int n)       { coeff_number = n; }
        void setIsNorm(bool v)           { is_norm = v; }

        ~Sofa();
    };

}  // namespace daisy

#endif  // SOFA_FFTW_ENABLED
#endif  // SOFA_HPP
