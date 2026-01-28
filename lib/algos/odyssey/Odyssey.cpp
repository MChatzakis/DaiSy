#include "../hodyssey/Odyssey.hpp"
#include "../hodyssey/indexing.hpp"  // for print_index_stats
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <cstring>

namespace diNoLib
{

    void Odyssey::initializeMPI(int argc, char **argv)
    {
#if ODYSSEY_MPI
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        if (provided < MPI_THREAD_MULTIPLE)
        {
            printf("The threading support level is lesser than that demanded.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        MPI_Comm_size(MPI_COMM_WORLD, &this->comm_sz);
        MPI_Comm_rank(MPI_COMM_WORLD, &this->my_rank);
#else
        // Non-MPI build: set default values
        this->comm_sz = 1;
        this->my_rank = 0;
#endif
    }

    Odyssey::Odyssey(DistanceType distance_type, int argc, char **argv)
        : SimilaritySearchAlgorithm(distance_type)
    {
        initializeMPI(argc, argv);
        // ::dinoLib::timer_init(&this->timer_manager);  // Commented out - only for profiling

        if (this->time_series_size % 8 != 0)
        {
            if (this->my_rank == 0)
            {
                std::cerr << "[Node " << this->my_rank
                          << "]: Error: SIMD calculations require query length to be a multiple of 8. Current value is "
                          << this->time_series_size << "\n";
            }
            std::exit(EXIT_FAILURE);
        }
    }

    Odyssey::Odyssey(const OdysseyConfig &config, DistanceType distance_type, int argc, char **argv)
        : SimilaritySearchAlgorithm(distance_type)
    {
        this->search_workers = config.search_workers;
        this->index_threads = config.index_threads;
        this->warping_window = config.warping_window;
        this->leaf_size = config.leaf_size;
        this->paa_segments = config.paa_segments;
        this->replication_groups = config.replication_groups;
        this->query_threads = config.query_threads;
        this->num_threads = config.search_workers;
        
        initializeMPI(argc, argv);
        // ::dinoLib::timer_init(&this->timer_manager);  // Commented out - only for profiling

        if (this->time_series_size % 8 != 0)
        {
            if (this->my_rank == 0)
            {
                std::cerr << "[Node " << this->my_rank
                          << "]: Error: SIMD calculations require query length to be a multiple of 8. Current value is "
                          << this->time_series_size << "\n";
            }
            std::exit(EXIT_FAILURE);
        }
    }

    void Odyssey::setNumThreads(int num_threads)
    {
        int max_threads = omp_get_max_threads();

        if (num_threads > max_threads) 
        {
            std::cerr << "[Warning] " << num_threads 
                    << " threads exceeds max available " << max_threads << " Using the max threads available.\n";
            this->num_threads = max_threads;
        } 
        else if (num_threads < 1) 
        {
            std::cerr << "[Warning] Thread count must be >= 1. Using 1.\n";
            this->num_threads = 1;
        } 
        else 
        {
            this->num_threads = num_threads;
        }
    } 

    int Odyssey::getNumThreads() const
    {
        return this->num_threads;
    } 

    void Odyssey::buildIndex(DataSource *data_source)
    {
        // Odyssey requires FileDataSource for disk-based distributed indexing
        FileDataSource *file_source = dynamic_cast<FileDataSource *>(data_source);
        if (file_source == nullptr)
        {
            fprintf(stderr, "Error: Odyssey::buildIndex requires FileDataSource\n");
            throw std::runtime_error("Odyssey::buildIndex requires FileDataSource");
        }

        const char *raw_filename = file_source->getFilename();
        if (raw_filename == nullptr)
        {
            fprintf(stderr, "Error: FileDataSource does not have a filename\n");
            throw std::runtime_error("FileDataSource does not have a filename");
        }

        this->dim = data_source->getDim();
        this->n_database = data_source->getTotalRecords();

        // If dataset_size is 0, use the total records from FileDataSource
        if (this->dataset_size == 0)
        {
            this->dataset_size = this->n_database;
        }

        // Step 1: Optimize parameters based on configuration
        odyssey_optimize_params(this);

        // Step 2: Prepare all data structures (index settings, BSF sharing, replication, workstealing)
        // NOTE: All index parameters (time_series_size, paa_segments, etc.) are read from 'this' object
        odyssey_prepare_structures(this, raw_filename);

        // Step 3: Log parameters (only if verbose and master node)
        odyssey_log_parameters(this);

        // Step 4: Build the index sequence
        this->rawfile = this->buildIndexSequence();

        // Step 5: Print index statistics (if verbose)
        if (this->verbose)
        {
            diNoLib::print_index_stats(this->index, this->my_rank);
        }

        // Step 6: Synchronize all MPI processes before moving to query answering
        #if ODYSSEY_MPI
                MPI_Barrier(MPI_COMM_WORLD);
        #endif
    }
    
    void Odyssey::searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        #pragma omp parallel num_threads(num_threads)
        {
            #pragma omp for 
            for (idx_t qi = 0; qi < n_query; qi++)
            {   
                std::priority_queue<std::pair<float, idx_t>> pq;
                const float *q_vec = query + qi * dim;

                float bound = FLT_MAX;  // initialize bound to max float

                for (idx_t dbi = 0; dbi < n_database; ++dbi)
                {
                    const float *db_vec = database + dbi * dim;
                    float dist = this->distance_computer->compute_dist(const_cast<float *>(q_vec), 
                                                                        const_cast<float *>(db_vec), 
                                                                        dim, 
                                                                        bound);
                    if ((idx_t)pq.size() < k) // maintain max-heap
                    {
                        pq.emplace(dist, dbi); // equivalent to pq.push(make_pair(dist, dbi));
                    }
                    else if (dist < pq.top().first) 
                    {
                        pq.pop();
                        pq.emplace(dist, dbi);                         
                        bound = pq.top().first; // update the bound variable for pruning
                    }

                }
                
                // store top-k results in reverse order
                for (idx_t j = k; j > 0; --j)
                {
                    D[qi * k + (j - 1)] = pq.top().first;
                    I[qi * k + (j - 1)] = pq.top().second;
                    pq.pop();
                }               
            }
        }
    }

    // ============================================================================
    // Odyssey::buildIndexSequence
    //
    // C++ port of:
    //   float *odyssey_build_index_sequence(Odyssey_t *odyssey)
    //
    // NOTE:
    // - Uses ReplicationData to compute how many time series this node owns
    //   and its starting offset in the global file.
    // - Relies on EKOSMAS-specific helpers that are DECLARED but not yet
    //   IMPLEMENTED in this project:
    //     * initialize_pRecBuf_ekosmas(...)
    //     * index_creation_sequence_worker(...)
    //   The calls are kept as in the original code so they can be
    //   implemented later without changing this method.
    // ============================================================================
    float *Odyssey::buildIndexSequence()
    {
        // --------------------------------------------------------------------
        // Local aliases to match the original C code
        // --------------------------------------------------------------------
        isax_index *index = this->index;
        int my_rank = this->my_rank;
        ReplicationData *replication_data = &this->replication_data;
        int index_threads = this->index_threads;
        // ::dinoLib::TimerManager *timer_manager = &this->timer_manager;  // Commented out - only for profiling

        // The original code uses odyssey->dataset (string path). In our port,
        // the filename is stored in index_settings->raw_filename (set in
        // odyssey_prepare_structures using FileDataSource::getFilename()).
        const char *ifilename = nullptr;
        if (this->index_settings && this->index_settings->raw_filename)
        {
            ifilename = this->index_settings->raw_filename;
        }
        else if (index && index->settings && index->settings->raw_filename)
        {
            ifilename = index->settings->raw_filename;
        }

        if (ifilename == nullptr)
        {
            fprintf(stderr, "[Node %d] Error: raw_filename not set in index settings.\n", my_rank);
            std::exit(EXIT_FAILURE);
        }

        // Total samples (time series) in the dataset as configured
        idx_t total_samples = this->dataset_size;

        // --------------------------------------------------------------------
        // Open input file and compute total records
        // --------------------------------------------------------------------
        // ::dinoLib::timer_start(timer_manager, "INPUT");  // Commented out - only for profiling
        FILE *ifile = std::fopen(ifilename, "rb");
        if (!ifile)
        {
            fprintf(stderr, "[Node %d] File %s not found!\n", my_rank, ifilename);
            std::exit(EXIT_FAILURE);
        }

        std::fseek(ifile, 0L, SEEK_END);
        file_position_type sz = static_cast<file_position_type>(std::ftell(ifile)); // size in bytes
        std::fseek(ifile, 0L, SEEK_SET);
        // ::dinoLib::timer_stop(timer_manager, "INPUT");  // Commented out - only for profiling

        // How many time series does this node own?
        idx_t my_time_series = rep_get_time_series_of_group(*replication_data, my_rank);

        // Total records in file (in time-series units)
        file_position_type total_records = sz / static_cast<file_position_type>(index->settings->ts_byte_size);

        if (total_records < static_cast<file_position_type>(total_samples))
        {
            fprintf(stderr,
                    "[Node %d] File %s has only %llu records (expected at least %llu)!\n",
                    my_rank,
                    ifilename,
                    static_cast<unsigned long long>(total_records),
                    static_cast<unsigned long long>(total_samples));
            std::fclose(ifile);
            std::exit(EXIT_FAILURE);
        }

        // --------------------------------------------------------------------
        // Allocate local rawfile buffer for this node
        // --------------------------------------------------------------------
        // Each time series has ts_byte_size bytes; my_time_series series total
        size_t rawfile_bytes =
            static_cast<size_t>(index->settings->ts_byte_size) *
            static_cast<size_t>(my_time_series);

        float *rawfile = static_cast<float *>(std::malloc(rawfile_bytes));
        if (rawfile == nullptr)
        {
            fprintf(stderr, "[Node %d] Error: Memory allocation failed for rawfile.\n", my_rank);
            std::fclose(ifile);
            std::exit(EXIT_FAILURE);
        }

        // --------------------------------------------------------------------
        // Compute byte offset in the global file based on replication groups
        // --------------------------------------------------------------------
        // rep_get_time_series_offset returns the starting time-series index
        // for the group this rank belongs to; convert to bytes.
        idx_t ts_offset = rep_get_time_series_offset(*replication_data, my_rank);
        file_position_type position_to_file =
            static_cast<file_position_type>(ts_offset) *
            static_cast<file_position_type>(index->settings->ts_byte_size);

        // ::dinoLib::timer_start(timer_manager, "INPUT");  // Commented out - only for profiling
        int returned_val = std::fseek(ifile, static_cast<long>(position_to_file), SEEK_SET);
        if (returned_val != 0)
        {
            fprintf(stderr, "[Node %d] Error on fseek() while positioning to %llu bytes.\n",
                    my_rank,
                    static_cast<unsigned long long>(position_to_file));
            std::fclose(ifile);
            std::exit(EXIT_FAILURE);
        }

        size_t elements_to_read =
            static_cast<size_t>(index->settings->timeseries_size) *
            static_cast<size_t>(my_time_series);

        size_t read_number = std::fread(
            rawfile,
            sizeof(ts_type),
            elements_to_read,
            ifile);

        std::fclose(ifile);
        // ::dinoLib::timer_stop(timer_manager, "INPUT");  // Commented out - only for profiling

        printf("[Node %d]: Loaded %zu data series starting from %llu.\n",
               my_rank,
               read_number / static_cast<size_t>(index->settings->timeseries_size),
               static_cast<unsigned long long>(position_to_file));

        if ((read_number / static_cast<size_t>(index->settings->timeseries_size)) !=
            static_cast<size_t>(my_time_series))
        {
            fprintf(stderr,
                    "[Node %d] Must read: %llu but node read %zu time series\n",
                    my_rank,
                    static_cast<unsigned long long>(my_time_series),
                    read_number / static_cast<size_t>(index->settings->timeseries_size));
            std::exit(EXIT_FAILURE);
        }

        // --------------------------------------------------------------------
        // Initialize FBL (parallel EKOSMAS version) and worker threads
        // --------------------------------------------------------------------
        // ::dinoLib::timer_start(timer_manager, "TOTAL_INDEX_BUFFER");  // Commented out - only for profiling

        // NOTE: In the original C code there was a branch on settings->znorm
        // allocating index->means and index->stds. Our C++ iSAXIndex no longer
        // has these fields and znorm is handled externally, so this block is
        // intentionally omitted.

        // ::dinoLib::timer_start(timer_manager, "BUFFER");  // Commented out - only for profiling
        // TODO: initialize_pRecBuf_ekosmas is DECLARED in iSAXIndex.hpp but not
        // yet IMPLEMENTED. It should allocate and initialize a
        // parallel_first_buffer_layer_ekosmas and assign it to index->fbl.
        index->fbl = reinterpret_cast<first_buffer_layer *>(
            initialize_pRecBuf_ekosmas(
                index->settings->initial_fbl_buffer_size,
                static_cast<int>(std::pow(2.0, index->settings->paa_segments)),
                index->settings->max_total_buffer_size +
                    DISK_BUFFER_SIZE * (PROGRESS_CALCULATE_THREAD_NUMBER - 1),
                index,
                index_threads));
        // ::dinoLib::timer_stop(timer_manager, "BUFFER");  // Commented out - only for profiling

        // Thread array for index creation
        std::vector<pthread_t> threadid(static_cast<size_t>(index_threads));

        // Allocate worker input structures (one per index thread)
        diNoLib::buffer_data_inmemory_ekosmas *input_data =
            static_cast<diNoLib::buffer_data_inmemory_ekosmas *>(
                std::malloc(sizeof(diNoLib::buffer_data_inmemory_ekosmas) *
                            static_cast<size_t>(index_threads)));
        if (input_data == nullptr)
        {
            fprintf(stderr,
                    "[Node %d] Error: Memory allocation failed for buffer_data_inmemory_ekosmas.\n",
                    my_rank);
            std::exit(EXIT_FAILURE);
        }

        unsigned long next_block_to_process = 0;
        int node_counter = 0; // required for tree construction using FAI

        // EKOSMAS bookkeeping array for iSAX groups
        volatile unsigned long *next_iSAX_group =
            static_cast<volatile unsigned long *>(
                std::calloc(static_cast<size_t>(index->fbl->max_total_size),
                            sizeof(unsigned long)));
        if (next_iSAX_group == nullptr)
        {
            fprintf(stderr,
                    "[Node %d] Error: Memory allocation failed for next_iSAX_group.\n",
                    my_rank);
            std::free(input_data);
            std::exit(EXIT_FAILURE);
        }

        // Barrier to ensure all summaries are computed before filling buffers
        pthread_barrier_t wait_summaries_to_compute;
        pthread_barrier_init(&wait_summaries_to_compute, nullptr,
                             static_cast<unsigned int>(index_threads));

        pthread_mutex_t lock_firstnode = PTHREAD_MUTEX_INITIALIZER;

        for (int i = 0; i < index_threads; i++)
        {
            diNoLib::buffer_data_inmemory_ekosmas &data = input_data[i];
            data.index = index;
            data.lock_firstnode = &lock_firstnode; // subtree root node initialization
            data.workernumber = i;
            data.shared_start_number = &next_block_to_process;
            data.ts_num = my_time_series;
            data.wait_summaries_to_compute = &wait_summaries_to_compute;
            data.node_counter = &node_counter; // required for tree construction using FAI
            data.parallelism_in_subtree = diNoLib::NO_PARALLELISM_IN_SUBTREE;
            data.next_iSAX_group = next_iSAX_group; // EKOSMAS bookkeeping
            data.rawfile = rawfile;
            data.deterministic_index = this->workstealing_data.deterministic_index;
            data.index_threads = index_threads;
            data.readblock = this->read_block_length;
            data.my_rank = my_rank;
            data.comm_sz = this->comm_sz;
            data.replication_data = replication_data;
            // data.timer_manager = reinterpret_cast<::dinoLib::TimerManager*>(timer_manager);  // Commented out - only for profiling
        }

        // Create worker threads
        for (int i = 0; i < index_threads; i++)
        {
            if (pthread_create(&threadid[static_cast<size_t>(i)],
                               nullptr,
                               diNoLib::index_creation_sequence_worker,
                               static_cast<void *>(&input_data[i])) != 0)
            {
                fprintf(stderr,
                        "[Node %d] Error: could not create index_creation_sequence_worker thread %d\n",
                        my_rank,
                        i);
                std::free(input_data);
                std::exit(EXIT_FAILURE);
            }
        }

        // Join worker threads
        for (int i = 0; i < index_threads; i++)
        {
            if (pthread_join(threadid[static_cast<size_t>(i)], nullptr) != 0)
            {
                fprintf(stderr,
                        "[Node %d] Error: could not join index_creation_sequence_worker thread %d\n",
                        my_rank,
                        i);
                std::free(input_data);
                std::exit(EXIT_FAILURE);
            }
        }

        std::free(input_data);
        std::free(const_cast<unsigned long *>(next_iSAX_group));

        // ::dinoLib::timer_stop(timer_manager, "TOTAL_INDEX_BUFFER");  // Commented out - only for profiling

        // Synchronize MPI processes before moving on
        #if ODYSSEY_MPI
                MPI_Barrier(MPI_COMM_WORLD);
        #endif

        return rawfile;
    }

    Odyssey::~Odyssey()
    {
        delete[] database;
    }

    // ============================================================================
    // odyssey_optimize_params - Simplified version for KNN + Dynamic Scheduling only
    // ============================================================================
    // 
    // PURPOSE: Optimize parameters by disabling unnecessary features based on configuration
    // 
    // CONSTRAINTS (simplified version):
    // - Only supports KNN queries (no threshold search)
    // - Only supports dynamic scheduling (never changes query_scheduling)
    // - Only supports S-WS workstealing (can disable if not needed)
    //
    // OPTIMIZATIONS PERFORMED:
    // 1. Single node (comm_sz == 1):
    //    - Disables BSF-sharing (not needed on single node)
    //    - Disables workstealing (not needed on single node)
    //    - Sets replication_groups to 1 (single node = single group)
    //
    // 2. Full replication (replication_groups == 1):
    //    - Disables BSF-sharing (doesn't make sense with full replication)
    //
    // REMOVED OPTIMIZATIONS (from original):
    // - Threshold search check (not supported - only KNN)
    // - No replication -> disable workstealing (we keep workstealing enabled if configured)
    // - No replication -> change scheduling to STATIC (we always use dynamic)
    // - Single node -> change scheduling to SINGLE_NODE (we always use dynamic)
    //
    void odyssey_optimize_params(Odyssey *odyssey)
    {
        // ========================================================================
        // OPTIMIZATION 1: Single Node Execution (comm_sz == 1)
        // ========================================================================
        // If running on a single node, many distributed features are unnecessary
        if (odyssey->comm_sz == 1)
        {
            // STEP 1.1: Disable BSF-sharing on single node
            // WHY: BSF-sharing is for sharing best-so-far values across nodes.
            //      On a single node, there's no one to share with.
            if (odyssey->bsf_sharing_data.bsf_sharing_enabled)
            {
                odyssey->bsf_sharing_data.bsf_sharing_enabled = false;
                if (odyssey->my_rank == 0)  // MASTER = 0
                {
                    printf("[Node %d, OptParams]: Single node execution. Disabling BSF-sharing\n", 
                           odyssey->my_rank);
                }
            }

            // STEP 1.2: Disable workstealing on single node
            // WHY: Workstealing distributes work across nodes. On a single node,
            //      there's no other node to steal from or give work to.
            if (odyssey->workstealing_data.ws_type != WorkstealingType::DISABLED)
            {
                odyssey->workstealing_data.ws_type = WorkstealingType::DISABLED;
                if (odyssey->my_rank == 0)
                {
                    printf("[Node %d, OptParams]: Single node execution. Disabling Workstealing\n", 
                           odyssey->my_rank);
                }
            }

            // STEP 1.3: Set replication groups to 1 on single node
            // WHY: A single node forms a single replication group by definition.
            //      This ensures consistency in the replication_data structure.
            if (odyssey->replication_data.total_groups != 1)
            {
                odyssey->replication_data.total_groups = 1;
                if (odyssey->my_rank == 0)
                {
                    printf("[Node %d, OptParams]: Single node execution. Replication groups set to 1\n", 
                           odyssey->my_rank);
                }
            }

            // NOTE: We do NOT change query_scheduling here (original code set it to SINGLE_NODE)
            // REASON: We only support dynamic scheduling, so we keep it as is.
            return;  // Early return: single node optimizations complete
        }

        // ========================================================================
        // OPTIMIZATION 2: Full Replication (replication_groups == 1)
        // ========================================================================
        // If all nodes have full replication (all data on all nodes),
        // BSF-sharing doesn't make sense because each node has the same data.
        if (odyssey->replication_data.total_groups == 1)
        {
            // STEP 2.1: Disable BSF-sharing with full replication
            // WHY: With full replication, every node has the same data.
            //      BSF-sharing is meant to share best values across nodes with different data.
            //      If all nodes have the same data, sharing BSF values is redundant.
            if (odyssey->bsf_sharing_data.bsf_sharing_enabled)
            {
                odyssey->bsf_sharing_data.bsf_sharing_enabled = false;
                if (odyssey->my_rank == 0)
                {
                    printf("[Node %d, OptParams]: Full replication selected. Disabling BSF-sharing\n", 
                           odyssey->my_rank);
                }
            }
        }

        // ========================================================================
        // OPTIMIZATION 3: No Replication (replication_groups == comm_sz)
        // ========================================================================
        // If each node has completely different data (no replication),
        // workstealing cannot work because nodes cannot process data they don't have.
        if (odyssey->replication_data.total_groups == odyssey->comm_sz)
        {
            // STEP 3.1: Disable workstealing with no replication
            // WHY: With no replication, each node has different data.
            //      Workstealing requires nodes to be able to process each other's work,
            //      but if Node A has data X and Node B has data Y, Node A cannot process
            //      work from Node B because it doesn't have data Y.
            if (odyssey->workstealing_data.ws_type != WorkstealingType::DISABLED)
            {
                odyssey->workstealing_data.ws_type = WorkstealingType::DISABLED;
                if (odyssey->my_rank == 0)
                {
                    printf("[Node %d, OptParams]: No replication selected. Disabling Workstealing\n", 
                           odyssey->my_rank);
                }
            }
        }

        // ========================================================================
        // REMOVED OPTIMIZATIONS (not applicable to our simplified version):
        // ========================================================================
        //
        // 1. Threshold search check:
        //    ORIGINAL: If query_answering_method == ODYSSEY_TH, disable BSF-sharing
        //    REMOVED: We only support KNN (ODYSSEY_KNN), so this check is unnecessary
        //
        // 2. No replication -> change scheduling to STATIC:
        //    ORIGINAL: If replication_groups == comm_sz, set query_scheduling = STATIC
        //    REMOVED: We only support dynamic scheduling, so we never change it
        //
    }

    // ============================================================================
    // odyssey_prepare_structures - Initialize index settings and data structures
    // ============================================================================
    // 
    // PURPOSE: Prepare all data structures needed for index building:
    //   - Initialize isax_index_settings and create the index
    //   - Initialize BSF sharing, replication, and workstealing structures
    //
    // PARAMETERS:
    //   - odyssey: Odyssey object containing all configuration parameters
    //   - raw_filename: Filename of the raw data file (from FileDataSource)
    //
    // NOTE: All index parameters (timeseries_size, paa_segments, etc.) are read
    //       from the odyssey object itself, not passed as parameters.
    //
    void odyssey_prepare_structures(Odyssey *odyssey, const char *raw_filename)
    {
        // Initialize isax_index_settings using parameters from odyssey object
        // Parameters: root_directory, timeseries_size, paa_segments, sax_cardinality,
        //            leaf_size, min_leaf_size, initial_lbl_size, flush_limit, initial_fbl_size,
        //            total_loaded_leaves, tight_bound, aggressive_check, new_index, inmemory_flag
        odyssey->index_settings = isax_index_settings_init(
            "",                              // root_directory (empty for in-memory)
            odyssey->time_series_size,       // timeseries_size
            odyssey->paa_segments,           // paa_segments
            odyssey->sax_cardinality,        // sax_cardinality
            odyssey->leaf_size,              // leaf_size
            odyssey->min_leaf_size,          // min_leaf_size
            odyssey->initial_lbl_size,       // initial_lbl_size
            odyssey->flush_limit,            // flush_limit
            odyssey->initial_fbl_size,      // initial_fbl_size
            1,                               // total_loaded_leaves (Leaves to load at each fetch)
            0,                               // tight_bound (Tightness of leaf bounds)
            0,                               // aggressive_check
            1,                               // new_index
            1                                // inmemory_flag (In memory)
        );

        // Set raw_filename after initialization (if provided)
        if (raw_filename != nullptr && odyssey->index_settings != nullptr)
        {
            // Allocate memory and copy the filename
            if (odyssey->index_settings->raw_filename != nullptr)
            {
                free(odyssey->index_settings->raw_filename);
            }
            odyssey->index_settings->raw_filename = (char *)malloc(strlen(raw_filename) + 1);
            if (odyssey->index_settings->raw_filename != nullptr)
            {
                strcpy(odyssey->index_settings->raw_filename, raw_filename);
            }
        }

        // Initialize the in-memory index (ekosmas version - to be implemented)
        odyssey->index = isax_index_init_inmemory_ekosmas(odyssey->index_settings);

        // Initialize BSF sharing data structure
        bsf_sharing_init(odyssey->bsf_sharing_data, odyssey->my_rank, odyssey->comm_sz);

        // Initialize replication data structure
        // Note: index_threads and query_threads are passed by reference and may be modified
        // if a configuration file specifies different values for the current node
        rep_init(odyssey->replication_data, odyssey->dataset_size, odyssey->my_rank, 
                 odyssey->comm_sz, odyssey->index_threads, odyssey->query_threads);

        // Initialize workstealing data structure
        ws_init(odyssey->workstealing_data, odyssey->comm_sz);
    }

    void odyssey_log_parameters(Odyssey *odyssey)
    {
        if (odyssey->my_rank == 0 && odyssey->verbose)  // MASTER = 0
        {
            const char *scheduling_methods[] = {"Single Node", "Static", "Round Robin", "Dynamic"};
            const char *odyssey_modes[] = {"Subsequence Similarity Search", "Sequence Similarity Search"};
            const char *dynamic_scheduling_modes[] = {"Coordinator Idle", "Periodic Check", "Standalone Thread"};
            const char *qa_methods[] = {"Odyssey-knn", "Odyssey-threshold"};
            const char *dis_enab[] = {"Disabled", "Enabled"};
            const char *workstealing_types[] = {"Disabled", "S-WS", "P-WS"};

            printf("================ Odyssey Settings ================\n");
            printf("Total Processes: [%d]\n", odyssey->comm_sz);
            
            // Dataset and queries are now handled via DataSource interface, not stored as strings
            printf("Dataset, Size: [%s, %llu]\n", 
                   "From FileDataSource", 
                   (unsigned long long)odyssey->dataset_size);
            printf("Queries, Size: [%s, %llu]\n", 
                   "Passed to searchIndex()", 
                   0ULL);  // queries_size is not stored, passed as parameter to searchIndex()
            
            printf("PAA Segments, SAX Cardinality: [%d, %d]\n", 
                   odyssey->index->settings->paa_segments, 
                   odyssey->index->settings->sax_bit_cardinality);
            printf("Time-series Size: [%d]\n", odyssey->index->settings->timeseries_size);
            printf("Leaf Size, Min Leaf Size, Read Block, Flush Limit: [%d, %d, %d, %d]\n", 
                   odyssey->index->settings->max_leaf_size, 
                   odyssey->index->settings->min_leaf_size, 
                   odyssey->read_block_length, 
                   odyssey->index->settings->max_total_full_buffer_size);
            
            // Convert mode, scheduling, qa_method to indices for array access
            int mode_idx = odyssey->mode;
            int scheduling_idx = odyssey->query_scheduling;
            int qa_method_idx = odyssey->qa_method;
            
            // Bounds checking for array access
            if (mode_idx < 0 || mode_idx >= 2) mode_idx = 1;  // Default to sequence search
            if (scheduling_idx < 0 || scheduling_idx >= 4) scheduling_idx = 3;  // Default to dynamic
            if (qa_method_idx < 0 || qa_method_idx >= 2) qa_method_idx = 0;  // Default to knn
            
            printf("Mode, Scheduling, Method: [%s, %s, %s]\n", 
                   odyssey_modes[mode_idx], 
                   scheduling_methods[scheduling_idx], 
                   qa_methods[qa_method_idx]);

            if (odyssey->query_scheduling == 3)  // DYNAMIC_PRED_BASED = 3
            {
                int dyn_sched_idx = odyssey->dynamic_scheduling_mode;
                if (dyn_sched_idx < 0 || dyn_sched_idx >= 3) dyn_sched_idx = 2;  // Default to standalone thread
                printf("Dynamic Scheduling: [%s]\n", dynamic_scheduling_modes[dyn_sched_idx]);
            }

            if (odyssey->mode == 0)  // SUBSEQUENCE_SIMILARITY_SEARCH = 0 (not supported, but log if set)
            {
                printf("Merge Offset: [%d]\n", odyssey->merge_offset);
                if (odyssey->qa_method == 1)  // ODYSSEY_TH = 1 (threshold, not supported)
                {
                    printf("Corr Threshold: [%f]\n", odyssey->corr_threshold);
                }
            }

            printf("TH Division Factor: [%d]\n", odyssey->pq_th_div_factor);
            printf("Dataset-Type: [%s]\n", odyssey->dataset_type.c_str());
            
            // Convert WorkstealingType enum to int for array access
            int ws_type_idx = static_cast<int>(odyssey->workstealing_data.ws_type);
            if (ws_type_idx < 0 || ws_type_idx >= 3) ws_type_idx = 0;  // Default to disabled
            printf("Workstealing: [%s]\n", workstealing_types[ws_type_idx]);
            
            printf("BSF-Sharing: [%s]\n", 
                   dis_enab[odyssey->bsf_sharing_data.bsf_sharing_enabled ? 1 : 0]);
            printf("Density-Aware Distribution: [%s]\n", 
                   dis_enab[odyssey->density_aware_prepro ? 1 : 0]);
            printf("Output File Name: [%s]\n", 
                   odyssey->output_file.empty() ? "Not Provided" : odyssey->output_file.c_str());
            
            // znorm removed - normalization is handled externally before buildIndex/searchIndex
            printf("Online znorm: [%s]\n", "Not Supported (handle externally)");
            
            printf("KNN k-size: [%d]\n", odyssey->top_k);
            printf("Verbose: [%s]\n", odyssey->verbose ? "Enabled" : "Disabled");

            rep_log_info(odyssey->replication_data, odyssey->index_threads, odyssey->query_threads);

            printf("==================================================\n");
        }
    }

} // namespace diNoLib
