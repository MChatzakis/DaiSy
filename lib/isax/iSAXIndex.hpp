#ifndef ISAXINDEX_HPP
#define ISAXINDEX_HPP

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <pthread.h>

#include "./iSAXTypes.hpp"
#include "./SAXBreakpoints.hpp"
#include "./SAX.hpp"

namespace daisy
{
    typedef struct
    {
        unsigned long mem_tree_structure;
        unsigned long mem_data;
        unsigned long mem_summaries;
        unsigned long disk_data_full;
        unsigned long disk_data_partial;
    } meminfo;

    typedef enum
    {
        BP_GAUSSIAN_HARDCODED = 0,
        BP_EQUIDEPTH = 1
    } breakpoint_mode;

    typedef struct
    {
        char new_index;

        char *raw_filename;
        const char *root_directory;
        int initial_fbl_buffer_size;
        sax_type *max_sax_cardinalities;

        // ALWAYS: TIMESERIES_SIZE = TS_VALUES_PER_PAA_SEGMENT * PAA_SEGMENTS
        int timeseries_size;
        int ts_values_per_paa_segment;
        int paa_segments;

        int tight_bound;
        int aggressive_check;

        int sax_byte_size;
        int position_byte_size;
        int ts_byte_size;

        int full_record_size;
        int partial_record_size;

        // ALWAYS: SAX_ALPHABET_CARDINALITY = 2^SAX_BIT_CARDINALITY
        int sax_bit_cardinality;
        root_mask_type *bit_masks;
        int sax_alphabet_cardinality;

        int max_leaf_size;
        int min_leaf_size;
        int initial_leaf_buffer_size;
        int max_total_buffer_size;
        int max_total_full_buffer_size;

        int max_filename_size;

        float mindist_sqrt;
        int root_nodes_size;

        int total_loaded_leaves;

        // Gaussian: alias the global tables. Equi-depth: per-index tables (the *_owned
        // pointers hold the heap allocations). breakpoints = triangular, _max = flat.
        breakpoint_mode bp_mode;
        const float *breakpoints;
        const float *breakpoints_max;
        float *breakpoints_owned;
        float *breakpoints_max_owned;

    } isax_index_settings;

    typedef struct isax_node_split_data
    {
        int splitpoint;
        sax_type *split_mask;
    } isax_node_split_data;

    typedef struct isax_node_buffer
    {
        int initial_buffer_size;

        // Full data:
        sax_type **full_sax_buffer;
        file_position_type **full_position_buffer;
        ts_type **full_ts_buffer;

        // Partial data:
        sax_type **partial_sax_buffer;
        file_position_type **partial_position_buffer;

        // TMP Buffers (stuff coming from splits and loads)
        // Full data:
        sax_type **tmp_full_sax_buffer;
        file_position_type **tmp_full_position_buffer;
        ts_type **tmp_full_ts_buffer;

        // Partial data:
        sax_type **tmp_partial_sax_buffer;
        file_position_type **tmp_partial_position_buffer;

        int full_buffer_size;
        int max_full_buffer_size;

        int partial_buffer_size;
        int max_partial_buffer_size;

        int tmp_full_buffer_size;
        int max_tmp_full_buffer_size;

        int tmp_partial_buffer_size;
        int max_tmp_partial_buffer_size;

        float *lbdarray;
        short *lbdarrayshort;
        unsigned long int arrayposition;
    } isax_node_buffer;

    typedef struct isax_node
    {
        // General
        int leaf_size;
        char has_partial_data_file;
        char has_full_data_file;

        sax_type *isax_values;
        sax_type *isax_cardinalities;

        struct isax_node *next;
        struct isax_node *previous;
        root_mask_type mask;
        struct isax_node *parent;

        // If is leaf
        unsigned char is_leaf;
        char *filename;
        isax_node_buffer *buffer;

        // If is intermediate
        struct isax_node_split_data *split_data;
        struct isax_node *left_child;
        struct isax_node *right_child;
        // Wedges
        ts_type *wedges;

    } isax_node;

    typedef struct fbl_soft_buffer
    {
        isax_node *node;
        sax_type **sax_records;
        file_position_type **pos_records;
        int initialized;
        int max_buffer_size;
        int buffer_size;
    } fbl_soft_buffer;

    typedef struct first_buffer_layer
    {
        int number_of_buffers;
        int initial_buffer_size;
        int max_total_size;
        int current_record_index;
        char *current_record;
        char *hard_buffer;
        fbl_soft_buffer *soft_buffers;

    } first_buffer_layer;

    typedef struct fbl_soft_buffer2
    {
        isax_node *node;
        sax_type *sax_records;
        file_position_type *pos_records;
        int initialized;
        int max_buffer_size;
        int buffer_size;
    } fbl_soft_buffer2;

    typedef struct first_buffer_layer2
    {
        int number_of_buffers;
        int initial_buffer_size;
        int max_total_size;
        int current_record_index;
        char *current_record;
        char *hard_buffer;
        fbl_soft_buffer2 *soft_buffers;
    } first_buffer_layer2;

    typedef struct parallel_fbl_soft_buffer
    {
        isax_node *node;
        sax_type **sax_records;
        file_position_type **pos_records;
        int initialized;
        int *max_buffer_size;
        int *buffer_size;
        int finished;
    } parallel_fbl_soft_buffer;

    typedef struct parallel_first_buffer_layer
    {
        int number_of_buffers;
        int initial_buffer_size;
        int max_total_size;
        int current_record_index;
        char *current_record;
        char *hard_buffer;
        parallel_fbl_soft_buffer *soft_buffers;
        int total_worker_number;
    } parallel_first_buffer_layer;

    // EKOSMAS version: simplified parallel_first_buffer_layer without hard_buffer and total_worker_number
    typedef struct parallel_fbl_soft_buffer_ekosmas
    {
        isax_node *volatile node;  // EKOSMAS: ADDED 'volatile' for thread-safety
        sax_type **sax_records;
        file_position_type **pos_records;
        volatile int initialized;  // EKOSMAS: ADDED 'volatile'
        int *max_buffer_size;
        int *buffer_size;
        volatile int finished;     // EKOSMAS: ADDED 'volatile'
    } parallel_fbl_soft_buffer_ekosmas;

    // Layout must match parallel_first_buffer_layer up to and including soft_buffers offset,
    // so that iSAXSearch.cpp's cast (parallel_first_buffer_layer *)(index->fbl) and ->soft_buffers access work.
    typedef struct parallel_first_buffer_layer_ekosmas
    {
        int number_of_buffers;
        int initial_buffer_size;
        int max_total_size;
        int current_record_index;
        char *current_record;   // padding for layout compatibility with parallel_first_buffer_layer
        char *hard_buffer;      // padding for layout compatibility with parallel_first_buffer_layer
        parallel_fbl_soft_buffer_ekosmas *soft_buffers;
    } parallel_first_buffer_layer_ekosmas;

    typedef struct
    {
        meminfo memory_info;

        FILE *sax_file;
        sax_type *sax_cache;
        unsigned long sax_cache_size;

        unsigned long long allocated_memory;
        unsigned long root_nodes;
        unsigned long long total_records;
        unsigned long long loaded_records;

        int *locations;
        isax_node *first_node;
        isax_index_settings *settings;

        char has_wedges;

        first_buffer_layer *fbl;

        float *answer;
    } isax_index;

    typedef struct buffer_data_inmemory
    {
        isax_index *index;
        int start_number, stop_number;
        ts_type *ts;
        pthread_mutex_t *lock_record;
        pthread_mutex_t *lock_fbl;
        pthread_mutex_t *lock_index;
        pthread_mutex_t *lock_cbl;
        pthread_mutex_t *lock_firstnode;
        pthread_mutex_t *lock_nodeconter;
        pthread_mutex_t *lock_disk;
        int workernumber;
        int total_workernumber;
        pthread_barrier_t *lock_barrier1, *lock_barrier2;
        int *node_counter;
        bool finished;
        int *nodeid;
        unsigned long *shared_start_number;

        int read_block_length;
    } buffer_data_inmemory;

    typedef struct trans_fbl_input
    {
        int start_number, stop_number, conternumber;
        isax_index *index;
        pthread_mutex_t *lock_index;
        pthread_mutex_t *lock_fbl_conter;
        pthread_mutex_t *lock_write;
        first_buffer_layer *fbl;
        int preworkernumber;
        int fbloffset;
        int *buffersize;
        pthread_barrier_t *lock_barrier1, *lock_barrier2, *lock_barrier3;
        bool finished;
    } trans_fbl_input;

    typedef struct index_buffer_data
    {
        //int ts_num,ts_loaded;
        file_position_type pos; //file offset (in bytes) where this worker’s first time series starts in the raw file.
        isax_index *index; //the shared isax_index* being built.
        ts_type * ts; //pointer to this worker’s slice of the current block of time-series data (already read into memory).
        sax_type *saxv; // pointer to the shared SAX output buffer for the current block (workers write their SAX words here).
        int fin_number; //how many time series this worker must process in this block (normally read_block_length, possibly fewer in the tail block).
        int blocid; //logical block id for this worker (used in some variants).
        //lock pointers for coordinating insertion into buffers, tree updates, and disk writes.
        pthread_mutex_t *lock_record; 
        pthread_mutex_t *lock_fbl;
        pthread_mutex_t *lock_index;
        pthread_mutex_t *lock_cbl;
        pthread_mutex_t *lock_firstnode;
        pthread_mutex_t *lock_nodeconter;
        pthread_mutex_t *lock_disk;
        file_position_type *fbl; //unused in this path (reserved in other variants).
        int *bufferpresize; //pre-size hints for buffers (used by “_new” variant).
        int workernumber; //worker ID (0 to total_workernumber-1).
        int total_workernumber; //total number of worker threads.
        int *nodecounter; //shared counter for tracking total nodes created (used in some variants).
        pthread_barrier_t *lock_barrier1, *lock_barrier2, *lock_barrier3; //barriers for synchronization between workers.
        bool finished; //flag to tell a worker that no more data will come (used to exit its loop).
    } index_buffer_data;

    typedef struct isax_node_record
    {
        sax_type *sax;
        ts_type *ts;
        file_position_type *position;
        enum insertion_mode insertion_mode;
        void *destination;
    } isax_node_record;

    typedef struct node_list
    {
        isax_node **nlist;
        int node_amount;
    } node_list;

    typedef struct localStack
    {
        isax_node **val;
        int top;
        int bottom;
    } localStack;

    struct pqueue_bsf;  // forward declaration (defined in iSAXPqueue.hpp)
    typedef struct query_result
    {
        float distance;
        isax_node *node;
        size_t pqueue_position;
        pqueue_bsf *pq_bsf;  // for K-NN results (top-k distances and positions)
        float total_time;    // for timing (microseconds or same unit as timer)
    } query_result;

    typedef struct deque
    {
        int *dq;
        int size, capacity;
        int f, r;
    } deque;

    isax_index_settings *isax_index_settings_init(const char *root_directory, int timeseries_size,
                                                  int paa_segments, int sax_bit_cardinality,
                                                  int max_leaf_size, int min_leaf_size,
                                                  int initial_leaf_buffer_size,
                                                  int max_total_buffer_size, int initial_fbl_buffer_size,
                                                  int total_loaded_leaves, int tight_bound, int aggressive_check, int new_index, char inmemory_flag,
                                                  breakpoint_mode bp_mode = BP_GAUSSIAN_HARDCODED);

    // Compute per-index equi-depth breakpoints from a sample of database (row-major
    // [n_series * timeseries_size]). No-op unless settings->bp_mode == BP_EQUIDEPTH.
    void compute_equidepth_breakpoints(isax_index_settings *settings,
                                       const float *database, size_t n_series);

    first_buffer_layer *initialize_fbl(int initial_buffer_size, int number_of_buffers, int max_total_buffers_size, isax_index *index);

    parallel_first_buffer_layer *initialize_pRecBuf(int initial_buffer_size, int number_of_buffers, int max_total_buffers_size, isax_index *index, int total_workers);

    // EKOSMAS-specific parallel buffer initialization (to be implemented)
    // Returns a parallel_first_buffer_layer_ekosmas used by Odyssey's EKOSMAS index build.
    parallel_first_buffer_layer_ekosmas *initialize_pRecBuf_ekosmas(int initial_buffer_size,
                                                                     int number_of_buffers,
                                                                     int max_total_buffers_size,
                                                                     isax_index *index,
                                                                     int total_workers);

    isax_index *isax_index_init_inmemory(isax_index_settings *settings);
    isax_index *isax_index_init_inmemory_ekosmas(isax_index_settings *settings);  // EKOSMAS version for Odyssey
    isax_index *isax_index_init(isax_index_settings *settings);

    isax_node_buffer *init_node_buffer(int initial_buffer_size);

    isax_node *isax_leaf_node_init(int initial_buffer_size);
    isax_node *isax_root_node_init(root_mask_type mask, int initial_buffer_size);
    isax_node *insert_to_pRecBuf(parallel_first_buffer_layer *fbl, sax_type *sax, file_position_type *pos, root_mask_type mask, isax_index *index, pthread_mutex_t *lock_firstnode, int workernumber, int total_workernumber);
    
    // EKOSMAS-specific versions for Odyssey
    isax_node *insert_to_pRecBuf_ekosmas(parallel_first_buffer_layer_ekosmas *fbl, sax_type *sax,
                                         file_position_type *pos, root_mask_type mask,
                                         isax_index *index, pthread_mutex_t *lock_firstnode, int workernumber, int total_workernumber);
    
    isax_node *add_record_to_node(isax_index *index, isax_node *tree_node, isax_node_record *record, const char leaf_size_check);
    
    // EKOSMAS-specific in-memory version (no disk files)
    isax_node *add_record_to_node_inmemory(isax_index *index,
                                            isax_node *tree_node,
                                            isax_node_record *record,
                                            const char leaf_size_check);
    
    enum response initialize_isax_values_and_cardinalities(isax_index *index,
                                                           isax_node *node,
                                                           sax_type *sax);

    root_mask_type isax_pRecBuf_index_insert_inmemory(isax_index *index, sax_type *sax, file_position_type *pos, pthread_mutex_t *lock_firstnode, int workernumber, int total_workernumber);
    
    // EKOSMAS-specific version for Odyssey
    root_mask_type isax_pRecBuf_index_insert_inmemory_ekosmas(isax_index *index,
                                                               sax_type *sax,
                                                               file_position_type *pos,
                                                               pthread_mutex_t *lock_firstnode,
                                                               int workernumber,
                                                               int total_workernumber);

    void destroy_fbl(first_buffer_layer *fbl);
    void destroy_fbl2(first_buffer_layer2 *fbl);
    void destroy_parallel_fbl(parallel_first_buffer_layer *fbl);
    void destroy_node_buffer(isax_node_buffer *node_buffer);
    void split_node(isax_index *index, isax_node *node);
    void split_node(isax_index *index, isax_node *node);

    float sax2paa_word(sax_type sax_word, int cardinality);
    int informed_split_decision(isax_node_split_data *split_data, isax_index_settings *settings, isax_node_record *records_buffer, int records_buffer_size);

    enum response create_node_filename(isax_index *index, isax_node *node, isax_node_record *record);
    enum response create_node_filename(isax_index *index, isax_node *node, isax_node_record *record);
    enum response add_to_node_buffer(isax_node_buffer *node_buffer, isax_node_record *record, int sax_segments, int ts_segments);
    enum response add_to_node_buffer(isax_node_buffer *node_buffer, isax_node_record *record, int sax_segments, int ts_segments);
    enum response flush_subtree_leaf_buffers_inmemory(isax_index *index, isax_node *node);

    int cmp_pri(double next, double curr);
    size_t get_pos(void *a);
    double get_pri(void *a);
    void set_pri(void *a, double pri);
    void set_pos(void *a, size_t pos);

    float calculate_minimum_distance_inmemory(isax_index *index, isax_node *node, ts_type *raw_query, ts_type *query);

    enum response flush_node_buffer(isax_node_buffer *node_buffer, int sax_segments, int ts_segments, const char *filename);
    enum response flush_subtree_leaf_buffers(isax_index *index, isax_node *node);
    enum response clear_node_buffer(isax_node_buffer *node_buffer, enum buffer_cleaning_mode clean_mode);
    void isax_index_clear_node_buffers(isax_index *index, isax_node *node, enum node_cleaning_mode node_cleaning_mode, enum buffer_cleaning_mode buffer_clean_mode);
    enum response flush_fbl(first_buffer_layer *fbl, isax_index *index);

    isax_node *insert_to_fbl(first_buffer_layer *fbl, sax_type *sax, file_position_type *pos, root_mask_type mask, isax_index *index);
    root_mask_type isax_fbl_index_insert(isax_index *index, sax_type *sax, file_position_type *pos);
    
    // Multi-threaded file-based indexing functions
    root_mask_type isax_fbl_index_insert_m(isax_index *index, sax_type *sax, file_position_type *pos,
                                           pthread_mutex_t *lock_record, pthread_mutex_t *lock_fbl,
                                           pthread_mutex_t *lock_cbl, pthread_mutex_t *lock_firstnode,
                                           pthread_mutex_t *lock_index, pthread_mutex_t *lock_disk);
    enum response flush_subtree_leaf_buffers_m(isax_index *index, isax_node *node, pthread_mutex_t *lock_index, pthread_mutex_t *lock_write);
    enum response indexconstruction(first_buffer_layer *fbl, isax_index *index, pthread_mutex_t *lock_index,
                          pthread_mutex_t *lock_disk, int calculate_thread);
    void *indexconstructionworker(void *input);
    void *indexbulkloadingworker(void *transferdata);
    void isax_index_binary_file_m(const char *ifilename, int ts_num, isax_index *index, int calculate_thread, int read_block_length = 100000);

    void lower_upper_lemire(float *t, int len, int r, float *l, float *u);

}

#endif