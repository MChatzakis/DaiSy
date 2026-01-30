#ifndef ODYSSEY_HPP
#define ODYSSEY_HPP

#include "SimilaritySearchAlgorithm.hpp"
#include "bsf_sharing.hpp"
#include "workstealing.hpp"
#include "replication.hpp"
#include "../../utils/TimerManager.hpp"
#include "query_answering.hpp"  // brings SearchFunctionParams, WsSearchFunctionParams, OdysseyQuery, NodeList

#include <queue>
#include <cfloat>
#include <string>
#include <omp.h> 

#if ODYSSEY_MPI
#include <mpi.h>
#endif

namespace diNoLib
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
    };

    // Forward declare type used in friend declaration (full definition repeated later)
    using ws_func_type = query_result (*)(WsSearchFunctionParams);
    
    class Odyssey : public SimilaritySearchAlgorithm
    {
        // Friend declarations for helper functions that need access to private members
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
        // Configurable via OdysseyConfig
        int search_workers = 1;
        int index_threads = 1; //HERE
        int warping_window = 10;
        int leaf_size = 2000;
        int paa_segments = 16;
        int replication_groups = 0;
        int query_threads = 1;//HERE
      

        // Defaults for the remaining Odyssey parameters
        // std::string dataset; // COMMENTED: Not needed - filename obtained from FileDataSource::getFilename() in buildIndex()
        idx_t dataset_size = 0; // Optional: if 0, auto-calculated from FileDataSource::getTotalRecords(). Set to limit dataset subset.
        // std::string queries; // COMMENTED: Not needed - queries passed directly to searchIndex() as parameters
        // idx_t queries_size = 0; // COMMENTED: Not needed - n_query passed as parameter to searchIndex()
        int initial_lbl_size = 2000; //HERE  
        int initial_fbl_size = 100; //HERE  
        int min_leaf_size = 2000; //HERE  
        int time_series_size = 256; //HERE  
        int sax_cardinality = 8; //HERE  
        int density_aware_prepro = 0;//HERE
        std::string dataset_type = "default"; //HERE  
        int pq_th_div_factor = 16;//HERE
        int read_block_length = 20000;//HERE
        int flush_limit = 1000000;
        std::string output_file;//HERE
        bool bsf_sharing = false;
        std::string replication_groups_file;
        int workstealing_mode = 1;       // 0: disabled, 1: S-WS (only KNN + S-WS supported)
        int ws_items_to_send = 0;
        std::string query_time_predictions_file;
        int mode = 1;      //HERE                // 0: subsequence-similarity-search, 1: sequence-similarity-search
        int query_scheduling = 3;//HERE        // 0: single-node, 1: static, 2: static-pred-based, 3: dynamic-pred-based
        int merge_offset = 0;//HERE
        float corr_threshold = 0.2f;//HERE
        bool verbose = false; //HERE
        int dynamic_scheduling_mode = 2;//HERE // 0: coordinator-idle, 1: periodic-check, 2: standalone-thread
        int top_k = 1;//HERE

        // MPI parameters
        int my_rank = 0; //HERE
        int comm_sz = 1;//HERE

        // Raw data pointer
        float *rawfile = nullptr;//HERE

        // Query tracking
        int query_counter = 0;//HERE

        // Time series group length
        int ts_group_length = 1;//HERE

        // BSF sharing data
        BsfSharingData bsf_sharing_data;//HERE

        // Workstealing data
        WorkstealingData workstealing_data;//HERE

        // Replication data
        ReplicationData replication_data;//HERE

        // Timer manager
        // ::dinoLib::TimerManager timer_manager;//HERE  // Commented out - only for profiling

        // Query results
        query_result *results = nullptr;//HERE

        // Private initialization methods
        void initializeMPI(int argc, char **argv);
        
        // Private index building method (internal implementation)
        float *buildIndexSequence();  // Internal build index for sequence similarity

        // Query answering: L2 vs DTW (called from searchIndex after common setup)
        void searchIndexL2Squared(OdysseyQuery *queries, int q_num, int topk,
                                  query_result *results, std::vector<BsfMessage> *shared_bsf_results,
                                  NodeList &nodelist, idx_t *I, float *D,
                                  double (*basis_func)(double));
        void searchIndexDTW(OdysseyQuery *queries, int q_num, int topk,
                           query_result *results, std::vector<BsfMessage> *shared_bsf_results,
                           NodeList &nodelist, idx_t *I, float *D,
                           double (*basis_func)(double));

    protected:
        // Odyssey uses index/rawfile, not database; custom version checks getIndex() instead of database
        bool validateSearchParams(const idx_t k, const idx_t n_query) const;

    public:
        // Constructor for Python bindings (no MPI args needed)
        Odyssey(DistanceType distance_type);
        // Constructor with MPI args (for C++ usage)
        Odyssey(DistanceType distance_type, int argc, char **argv);
        // Constructor with config and MPI args
        Odyssey(const OdysseyConfig &config, DistanceType distance_type, int argc, char **argv);
        void setNumThreads(int num_threads);
        int getNumThreads() const;
        
        // MPI rank and size getters (for demos/tests)
        int getMyRank() const { return my_rank; }
        int getCommSz() const { return comm_sz; }
        
        // Bring base class buildIndex overloads into scope
        using SimilaritySearchAlgorithm::buildIndex;
        
        void buildIndex(DataSource *data_source) override;
        
        // Odyssey only supports file-based data - override to give clear error
        void buildIndex(float *database, idx_t n_database, idx_t dim) override {
            throw std::runtime_error("Odyssey requires file-based data. Use buildIndex(filename, dim, n_database) instead.");
        }
        
        void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) override;

        ~Odyssey();

    };
    
    // Include iSAXIndex.hpp first to get query_result definition (it's a typedef, not a struct)
    #include "../isax/iSAXIndex.hpp"
    
    // Include query_answering.hpp for SearchFunctionParams and WsSearchFunctionParams definitions
    // (included after Odyssey class definition to avoid circular dependency)
    // Note: query_answering.hpp includes indexing.hpp which defines NodeList, and iSAXIndex.hpp which defines query_result
    #include "query_answering.hpp"

    // Function pointer type definitions (using C++11 'using' syntax)
    // These match the original C typedefs from odyssey.h, but use C++ types:
    //   typedef ts_type *(*index_func_type)(Odyssey_t *odyssey);
    //   typedef query_result (*qa_func_type)(search_function_params);
    //   typedef query_result (*ws_func_type)(ws_search_function_params);
    //   typedef query_result* (*qa_sched_func_type)(Odyssey_t *odyssey, qa_func_type search_function, ws_func_type ws_search_function);
    // Note: query_result is a typedef struct defined in iSAXIndex.hpp which is included via query_answering.hpp
    // We're already in diNoLib namespace, so query_result should be available without qualification
    using index_func_type = ts_type* (*)(Odyssey*);  // Build index function: returns rawfile pointer
    using qa_func_type = query_result (*)(SearchFunctionParams);  // Query answering function
    // ws_func_type already declared before class for friend declaration
    using qa_sched_func_type = query_result* (*)(Odyssey*, qa_func_type, ws_func_type);  // Query scheduling function

    // Query management functions
    // Note: OdysseyQuery, ReplicationData, etc. are defined in query_answering.hpp which is included above
    OdysseyQuery* load_queries_from_buffer(const float *query_buf, int q_num, isax_index *index, int my_rank);
    void free_queries(OdysseyQuery *queries, int q_num);
    double predict_exec_time(float bsf, const char *dataset_type);  // Changed char* to const char*
    int cmp_query(const void *a, const void *b);

    void greedy_query_scheduling(OdysseyQuery *queries, OdysseyQuery *queries_to_ans, int *queries_counter, int q_num,
                                 ReplicationData *replication_data, int my_rank, int comm_sz);

    // Communication module functions
    // NOTE: VLA (Variable Length Array) in original C code: process_buffer_initial[comm_sz][distributed_queries_initial_burst]
    // In C++ we use pointer to pointer (int**) or flattened array with manual indexing
    void send_initial_queries_module_coordinator_async_chatzakis(int *q_loaded, int my_rank, int comm_sz,
                                                                 int distributed_queries_initial_burst, 
                                                                 int **process_buffer_initial,  // Changed from VLA to int**
                                                                 int *rec_message, MPI_Request *request, MPI_Request *send_request,
                                                                 int q_num, int *termination_message_id,
                                                                 ReplicationData *replication_data);

    int send_queries_module_coordinator_async_chatzakis(int *q_loaded, int q_num, int *process_buffer, MPI_Request *request, int *rec_message,
                                                        MPI_Request *send_request, int *termination_message_id,
                                                        ReplicationData *replication_data, int my_rank, int comm_sz,
                                                        /* ::dinoLib::TimerManager *timer_manager, */ bool verbose);

    // NodeList comes from indexing.hpp (included via query_answering.hpp)
    NodeList initialize_node_list(isax_index *index, int my_rank);

    // Odyssey API functions (converted from C Odyssey_t* to C++ Odyssey*)
    // NOTE: odyssey_init is replaced by Odyssey constructors
    // NOTE: odyssey_set_dataset/odyssey_set_queries not needed - handled via buildIndex/searchIndex API
    // Optimize parameters based on configuration
    // Simplified version: only disables unnecessary features, never changes scheduling (always dynamic)
    void odyssey_optimize_params(Odyssey *odyssey);
    void odyssey_log_parameters(Odyssey *odyssey);
    void odyssey_prepare_structures(Odyssey *odyssey, const char *raw_filename);
    // NOTE: odyssey_build_index_sequence moved to private method buildIndexSequence()
    // NOTE: odyssey_build_index removed - it coincides with buildIndex() method from SimilaritySearchAlgorithm
    void odyssey_preprocess_and_sort_queries(Odyssey *odyssey, OdysseyQuery *queries, int q_num, bool apply_sort);
    void odyssey_perform_workstealing(Odyssey *odyssey, OdysseyQuery *queries, NodeList nodelist, 
                                      ws_func_type ws_func, double (*estimation_func)(double), 
                                      query_result *results, std::vector<BsfMessage> *shared_bsf_results);  // Changed bsf_msg* to std::vector<BsfMessage>*
    
    // Query scheduling functions - only dynamic scheduling supported
    // COMMENTED: Single node, static, and static pred-based scheduling not supported
    // query_result *odyssey_sched_snode_qa(Odyssey *odyssey, qa_func_type qa_func, ws_func_type ws_func);
    // query_result *odyssey_sched_static_qa(Odyssey *odyssey, qa_func_type qa_func, ws_func_type ws_func);
    // query_result *odyssey_sched_static_pred_based_qa(Odyssey *odyssey, qa_func_type qa_func, ws_func_type ws_func);
    query_result *odyssey_sched_dynamic_pred_based_qa(Odyssey *odyssey, qa_func_type qa_func, ws_func_type ws_func);  // Only dynamic scheduling supported
    query_result *odyssey_query_answering(Odyssey *odyssey);
    void odyssey_collect_timers(Odyssey *odyssey);
    void odyssey_collect_results_knn(Odyssey *odyssey);
    void odyssey_collect(Odyssey *odyssey);
    void odyssey_destroy(Odyssey *odyssey);

} // namespace diNoLib


#endif // ODYSSEY_HPP
