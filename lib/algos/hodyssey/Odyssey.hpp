#ifndef ODYSSEY_HPP
#define ODYSSEY_HPP

#include "SimilaritySearchAlgorithm.hpp"
#include "bsf_sharing.hpp"
#include "workstealing.hpp"
#include "replication.hpp"
#include "../../utils/TimerManager.hpp"
#include "query_answering.hpp"  

#include <queue>
#include <cfloat>
#include <string>
#include <omp.h> 

#if ODYSSEY_MPI
#include <mpi.h>
#endif

namespace daisy
{
    struct OdysseyConfig
    {
        int search_workers = 1;
        int index_threads = 1;
        int warping_window = 10;
        int leaf_size = 2000;
        int paa_segments = 16;
        int replication_groups = 0;
        int query_threads = 1;
        int pq_th_div_factor = 16;
    };

    using ws_func_type = query_result (*)(WsSearchFunctionParams);

    class Odyssey : public SimilaritySearchAlgorithm
    {
        
        friend void odyssey_optimize_params(Odyssey *odyssey);
        friend void odyssey_log_parameters(Odyssey *odyssey);
        friend void odyssey_prepare_structures(Odyssey *odyssey, const char *raw_filename);
        friend void print_index_stats(isax_index *index, int my_rank);
        friend void* index_creation_sequence_worker(void *transferdata);
        friend void tree_index_creation_from_pRecBuf_fai_blocking(void *transferdata);
        friend void odyssey_preprocess_and_sort_queries(Odyssey *odyssey, OdysseyQuery *queries, int q_num, bool apply_sort);
        friend void odyssey_perform_workstealing(Odyssey *odyssey, OdysseyQuery *queries, NodeList nodelist,
                                                  ws_func_type ws_func, double (*estimation_func)(double),
                                                  query_result *results, std::vector<BsfMessage> *shared_bsf_results);
        
    private:
        int search_workers = 1;
        int index_threads = 1;
        int replication_groups = 0;
        int query_threads = 1;

        idx_t dataset_size = 0;

        int min_leaf_size = 2000;
        int time_series_size = 256;
        int density_aware_prepro = 0;
        std::string dataset_type = "default"; 
        int pq_th_div_factor = 16;
        int read_block_length = 20000;
        int flush_limit = 1000000;
        std::string output_file;
        bool bsf_sharing = false;
        std::string replication_groups_file;
        int workstealing_mode = 1;       
        int ws_items_to_send = 0;
        std::string query_time_predictions_file;
        int mode = 1;      
        int query_scheduling = 3;
        int merge_offset = 0;
        float corr_threshold = 0.2f;
        bool verbose = false; 
        int dynamic_scheduling_mode = 1;  
        int top_k = 1;

        int my_rank = 0; 
        int comm_sz = 1;

        float *rawfile = nullptr;

        int query_counter = 0;

        int ts_group_length = 1;

        BsfSharingData bsf_sharing_data;

        WorkstealingData workstealing_data;

        ReplicationData replication_data;

        query_result *results = nullptr;

        void initializeMPI(int argc, char **argv);

        float *buildIndexSequence();  

        void searchIndexL2Squared(OdysseyQuery *queries, int q_num, int topk,
                                  query_result *results, std::vector<BsfMessage> *shared_bsf_results,
                                  NodeList &nodelist, idx_t *I, float *D,
                                  double (*basis_func)(double));
        void searchIndexDTW(OdysseyQuery *queries, int q_num, int topk,
                           query_result *results, std::vector<BsfMessage> *shared_bsf_results,
                           NodeList &nodelist, idx_t *I, float *D,
                           double (*basis_func)(double));

    protected:
        
        bool validateSearchParams(const idx_t k, const idx_t n_query) const;

    public:
        
        Odyssey(DistanceType distance_type);
        
        Odyssey(DistanceType distance_type, int argc, char **argv);
        
        Odyssey(const OdysseyConfig &config, DistanceType distance_type, int argc, char **argv);
        void setNumThreads(int num_threads);
        int getNumThreads() const;
        void setWarpingWindow(int w) { warping_window = w; }
        int getWarpingWindow() const { return warping_window; }

        int getMyRank() const { return my_rank; }
        int getCommSz() const { return comm_sz; }

        using SimilaritySearchAlgorithm::buildIndex;
        
        void buildIndex(DataSource *data_source) override;

        void buildIndex(float *database, idx_t n_database, idx_t dim) override {
            throw std::runtime_error("Odyssey requires file-based data. Use buildIndex(filename, dim, n_database) instead.");
        }
        
        void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) override;

        int getResultCompareRank() const override { return my_rank; }

        ~Odyssey();

    };

    #include "../isax/iSAXIndex.hpp"

    #include "query_answering.hpp"

    using index_func_type = ts_type* (*)(Odyssey*);  
    using qa_func_type = query_result (*)(SearchFunctionParams);  
    
    using qa_sched_func_type = query_result* (*)(Odyssey*, qa_func_type, ws_func_type);  

    OdysseyQuery* load_queries_from_buffer(const float *query_buf, int q_num, isax_index *index, int my_rank);
    void free_queries(OdysseyQuery *queries, int q_num);
    double predict_exec_time(float bsf, const char *dataset_type);  
    int cmp_query(const void *a, const void *b);

    void send_initial_queries_module_coordinator_async_chatzakis(int *q_loaded, int my_rank, int comm_sz,
                                                                 int distributed_queries_initial_burst, 
                                                                 int **process_buffer_initial,  
                                                                 int *rec_message, MPI_Request *request, MPI_Request *send_request,
                                                                 int q_num, int *termination_message_id,
                                                                 ReplicationData *replication_data);

    int send_queries_module_coordinator_async_chatzakis(int *q_loaded, int q_num, int *process_buffer, MPI_Request *request, int *rec_message,
                                                        MPI_Request *send_request, int *termination_message_id,
                                                        ReplicationData *replication_data, int my_rank, int comm_sz,
                                                         bool verbose);

    NodeList initialize_node_list(isax_index *index, int my_rank);

    void odyssey_optimize_params(Odyssey *odyssey);
    void odyssey_log_parameters(Odyssey *odyssey);
    void odyssey_prepare_structures(Odyssey *odyssey, const char *raw_filename);

    void odyssey_preprocess_and_sort_queries(Odyssey *odyssey, OdysseyQuery *queries, int q_num, bool apply_sort);
    void odyssey_perform_workstealing(Odyssey *odyssey, OdysseyQuery *queries, NodeList nodelist, 
                                      ws_func_type ws_func, double (*estimation_func)(double), 
                                      query_result *results, std::vector<BsfMessage> *shared_bsf_results);  

} 

#endif 
