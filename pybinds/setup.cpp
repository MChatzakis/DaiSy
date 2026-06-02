#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdio>
#include <cstring>
#include <string>

#if ODYSSEY_MPI
#include <mpi.h>
#endif

#include "../lib/distance_computers/DistanceComputer.hpp"
#include "../lib/algos/Bruteforce.hpp"
#include "../lib/algos/LbBruteforce.hpp"
#include "../lib/algos/Messi.hpp"
#if ODYSSEY_MPI
#include "../lib/algos/hodyssey/Odyssey.hpp"
#endif
#include "../lib/algos/ParIS.hpp"
#ifdef SING_CUDA_ENABLED
    #if SING_CUDA_ENABLED != 0
    #include "../lib/algos/Sing.hpp"
    #endif
#endif
#include "../lib/algos/DataSource.hpp"
#ifdef SOFA_FFTW_ENABLED
    #if SOFA_FFTW_ENABLED != 0
    #include "../lib/algos/Sofa.hpp"
    #endif
#endif

// Define a Python module named 'daisy._core'
PYBIND11_MODULE(_core, m)
{
    m.doc() = "daisy C++ backend";

    ////// DISTANCETYPE //////
    pybind11::enum_<daisy::DistanceType>(m, "DistanceType")
        .value("L2_SQUARED", daisy::DistanceType::L2_SQUARED)
        .value("DTW", daisy::DistanceType::DTW)
        .export_values();

    ////// BRUTEFORCE //////
    pybind11::class_<daisy::BruteForceSearch>(m, "BruteForceSearch", "Brute force similarity search")
        // Constructor
        .def(pybind11::init<daisy::DistanceType>(), "Create a new BruteForceSearch with the given distance metric")

        // Setter
        .def("setNumThreads", &daisy::BruteForceSearch::setNumThreads, "Set the number of threads to use")

        // Getter
        .def("getNumThreads", &daisy::BruteForceSearch::getNumThreads, "Get the number of threads")

        // Bind method to build the index from a NumPy array
        .def("buildIndex", [](daisy::BruteForceSearch &self, pybind11::array_t<float> db)
             {
            pybind11::buffer_info buf = db.request();
            if (buf.ndim != 2)
                throw std::runtime_error("Database array must be 2D");

            daisy::idx_t n = buf.shape[0];
            daisy::idx_t d = buf.shape[1];

            // Create InMemoryDataSource from numpy array
            daisy::InMemoryDataSource data_source(static_cast<float *>(buf.ptr), n, d);
            self.buildIndex(&data_source); }, "Build the index from a 2D float32 numpy array")

        // Bind method to perform similarity search
        .def("searchIndex", [](daisy::BruteForceSearch &self, pybind11::array_t<float> query, daisy::idx_t k)
             {
            pybind11::buffer_info query_buf = query.request(); // Get query buffer info

            // Check for valid input shape
            if (query_buf.ndim != 2)
                throw std::runtime_error("Query array must be 2D");
            if (k <= 0)
                throw std::runtime_error("k must be positive");            

            // Number of query vectors and their dimension
            daisy::idx_t n_query = query_buf.shape[0];
            daisy::idx_t dim = query_buf.shape[1];

            // Allocate output arrays for indices and distances
            std::vector<daisy::idx_t> indices(n_query * k);
            std::vector<float> distances(n_query * k);
            
            // Perform the search
            self.searchIndex(static_cast<float *>(query_buf.ptr), n_query, k,
                             indices.data(), distances.data());

            // Return results as a tuple of NumPy arrays
            return pybind11::make_tuple(
                pybind11::array_t<daisy::idx_t>({n_query, k}, indices.data()),
                pybind11::array_t<float>({n_query, k}, distances.data())
            ); }, "Search the index with queries and return (indices, distances)");

    ////// LBBRUTEFORCE //////
    pybind11::class_<daisy::LbBruteforce>(m, "LbBruteforce", "Lower Bound brute-force similarity search")
        // Constructor
        .def(pybind11::init<daisy::DistanceType>(), "Create a new LbBruteforce with the given distance metric")

        // Getters
        .def("getPaaSegments", &daisy::LbBruteforce::getPaaSegments, "Get the number of PAA segments")
        .def("getSaxCardinality", &daisy::LbBruteforce::getSaxCardinality, "Get the SAX cardinality used for dimensionality reduction")
        .def("getLeafSize", &daisy::LbBruteforce::getLeafSize, "Get the configured leaf size")
        .def("getMinLeafSize", &daisy::LbBruteforce::getMinLeafSize, "Get the minimum leaf size for search optimization")
        .def("getInitialLblSize", &daisy::LbBruteforce::getInitialLblSize, "Get the initial size of the lower-bound leaf buffer (LBL)")
        .def("getFlushLimit", &daisy::LbBruteforce::getFlushLimit, "Get the flush limit for internal buffers")
        .def("getInitialFblSize", &daisy::LbBruteforce::getInitialFblSize, "Get the initial size of the full buffer list (FBL)")
        .def("getTotalLoadedLeaves", &daisy::LbBruteforce::getTotalLoadedLeaves, "Get the number of total loaded leaves")
        .def("getTightBound", &daisy::LbBruteforce::getTightBound, "Get whether tight lower bounds are used")

        // Setters
        .def("setNumThreads", &daisy::LbBruteforce::setNumThreads, "Set the number of threads to use")
        .def("setPaaSegments", &daisy::LbBruteforce::setPaaSegments, "Set the number of PAA segments")
        .def("setSaxCardinality", &daisy::LbBruteforce::setSaxCardinality, "Set the SAX cardinality")
        .def("setLeafSize", &daisy::LbBruteforce::setLeafSize, "Set the leaf size")
        .def("setMinLeafSize", &daisy::LbBruteforce::setMinLeafSize, "Set the minimum leaf size")
        .def("setInitialLblSize", &daisy::LbBruteforce::setInitialLblSize, "Set the initial size of the lower-bound leaf buffer (LBL)")
        .def("setFlushLimit", &daisy::LbBruteforce::setFlushLimit, "Set the flush limit for internal buffers")
        .def("setInitialFblSize", &daisy::LbBruteforce::setInitialFblSize, "Set the initial size of the full buffer list (FBL)")
        .def("setTotalLoadedLeaves", &daisy::LbBruteforce::setTotalLoadedLeaves, "Set the number of total loaded leaves")
        .def("setTightBound", &daisy::LbBruteforce::setTightBound, "Enable or disable tight lower bounds")

        // Build the index from a NumPy array
        .def("buildIndex", [](daisy::LbBruteforce &self, pybind11::array_t<float> db)
             {
            pybind11::buffer_info buf = db.request();
            if (buf.ndim != 2) 
                throw std::runtime_error("Database array must be 2D");
            
            daisy::idx_t n = buf.shape[0];
            daisy::idx_t d = buf.shape[1];
            
            // Create InMemoryDataSource from numpy array
            daisy::InMemoryDataSource data_source(static_cast<float *>(buf.ptr), n, d);
            self.buildIndex(&data_source); }, "Build the index from a 2D float32 numpy array")

        // Search the index using a query array and return (indices, distances)
        .def("searchIndex", [](daisy::LbBruteforce &self, pybind11::array_t<float> query, daisy::idx_t k)
             {
            pybind11::buffer_info buf = query.request();
            if (buf.ndim != 2) 
                throw std::runtime_error("Query array must be 2D");

            std::vector<daisy::idx_t> indices(buf.shape[0] * k);
            std::vector<float> distances(buf.shape[0] * k);

            self.searchIndex(static_cast<float *>(buf.ptr), buf.shape[0], k, indices.data(), distances.data());

            return pybind11::make_tuple(
                pybind11::array_t<daisy::idx_t>(pybind11::buffer_info(
                    indices.data(),
                    sizeof(daisy::idx_t),
                    pybind11::format_descriptor<daisy::idx_t>::format(),
                    2,
                    std::vector<pybind11::ssize_t>{static_cast<pybind11::ssize_t>(buf.shape[0]), static_cast<pybind11::ssize_t>(k)},
                    std::vector<pybind11::ssize_t>{
                        static_cast<pybind11::ssize_t>(sizeof(daisy::idx_t) * k),
                        static_cast<pybind11::ssize_t>(sizeof(daisy::idx_t))
                    }
                )),
                pybind11::array_t<float>(pybind11::buffer_info(
                    distances.data(),
                    sizeof(float),
                    pybind11::format_descriptor<float>::format(),
                    2,
                    std::vector<pybind11::ssize_t>{static_cast<pybind11::ssize_t>(buf.shape[0]), static_cast<pybind11::ssize_t>(k)},
                    std::vector<pybind11::ssize_t>{
                        static_cast<pybind11::ssize_t>(sizeof(float) * k),
                        static_cast<pybind11::ssize_t>(sizeof(float))
                    }
                ))
            ); }, "Search the index with queries and return (indices, distances)")

        // Search the index using DTW distance specifically
        .def("searchIndexDTW", [](daisy::LbBruteforce &self, pybind11::array_t<float> query, daisy::idx_t k)
             {
            pybind11::buffer_info buf = query.request();
            if (buf.ndim != 2) 
                throw std::runtime_error("Query array must be 2D");

            std::vector<daisy::idx_t> indices(buf.shape[0] * k);
            std::vector<float> distances(buf.shape[0] * k);

            self.searchIndexDTW(static_cast<float *>(buf.ptr), buf.shape[0], k, indices.data(), distances.data());

            return pybind11::make_tuple(
                pybind11::array_t<daisy::idx_t>(
                    std::vector<pybind11::ssize_t>{static_cast<pybind11::ssize_t>(buf.shape[0]), static_cast<pybind11::ssize_t>(k)}, 
                    indices.data()
                ),
                pybind11::array_t<float>(
                    std::vector<pybind11::ssize_t>{static_cast<pybind11::ssize_t>(buf.shape[0]), static_cast<pybind11::ssize_t>(k)}, 
                    distances.data()
                )
            ); }, "Search the index using DTW distance and return (indices, distances)")

        // Search the index using L2 squared distance specifically
        .def("searchIndexL2Squared", [](daisy::LbBruteforce &self, pybind11::array_t<float> query, daisy::idx_t k)
             {
            pybind11::buffer_info buf = query.request();
            if (buf.ndim != 2) 
                throw std::runtime_error("Query array must be 2D");

            std::vector<daisy::idx_t> indices(buf.shape[0] * k);
            std::vector<float> distances(buf.shape[0] * k);

            self.searchIndexL2Squared(static_cast<float *>(buf.ptr), buf.shape[0], k, indices.data(), distances.data());

            return pybind11::make_tuple(
                pybind11::array_t<daisy::idx_t>(
                    std::vector<pybind11::ssize_t>{static_cast<pybind11::ssize_t>(buf.shape[0]), static_cast<pybind11::ssize_t>(k)}, 
                    indices.data()
                ),
                pybind11::array_t<float>(
                    std::vector<pybind11::ssize_t>{static_cast<pybind11::ssize_t>(buf.shape[0]), static_cast<pybind11::ssize_t>(k)}, 
                    distances.data()
                )
            ); }, "Search the index using L2 squared distance and return (indices, distances)");

    ////// MESSI //////
    pybind11::class_<daisy::Messi>(m, "Messi", "MESSI (Multi-Queue Efficient SAX Similarity Index) algorithm for time series similarity search")
        // Constructor
        .def(pybind11::init<daisy::DistanceType>(), "Create a new Messi instance with the given distance metric")

        // Getters
        .def("getNumThreads", &daisy::Messi::getNumThreads, "Get the number of search threads")
        .def("getPaaSegments", &daisy::Messi::getPaaSegments, "Get the number of PAA segments used in SAX transformation")
        .def("getSaxCardinality", &daisy::Messi::getSaxCardinality, "Get the cardinality of SAX symbols")
        .def("getLeafSize", &daisy::Messi::getLeafSize, "Get the maximum leaf size in the index tree")
        .def("getMinLeafSize", &daisy::Messi::getMinLeafSize, "Get the minimum number of entries per leaf")
        .def("getInitialLblSize", &daisy::Messi::getInitialLblSize, "Get the initial size of the lower-bound buffer")
        .def("getFlushLimit", &daisy::Messi::getFlushLimit, "Get the flush limit before writing to disk")
        .def("getInitialFblSize", &daisy::Messi::getInitialFblSize, "Get the initial full-buffer size")
        .def("getTotalLoadedLeaves", &daisy::Messi::getTotalLoadedLeaves, "Get the total number of leaves loaded")
        .def("getTightBound", &daisy::Messi::getTightBound, "Check whether tight bounds are enabled")

        .def("getSearchWorkers", &daisy::Messi::getSearchWorkers, "Get number of worker threads used for search")
        .def("getIndexWorkers", &daisy::Messi::getIndexWorkers, "Get number of worker threads used for indexing")
        .def("getReadBlockLength", &daisy::Messi::getReadBlockLength, "Get block size for reading the time series data")
        .def("getWarpingWindow", &daisy::Messi::getWarpingWindow, "Get the DTW warping window constraint")

        // Setters
        .def("setNumThreads", &daisy::Messi::setNumThreads, "Set the number of threads to use for both indexing and search")
        .def("setPaaSegments", &daisy::Messi::setPaaSegments, "Set the number of PAA segments")
        .def("setSaxCardinality", &daisy::Messi::setSaxCardinality, "Set the SAX cardinality")
        .def("setLeafSize", &daisy::Messi::setLeafSize, "Set the leaf size of the index tree")
        .def("setMinLeafSize", &daisy::Messi::setMinLeafSize, "Set the minimum size of a leaf")
        .def("setInitialLblSize", &daisy::Messi::setInitialLblSize, "Set the initial LBL size")
        .def("setFlushLimit", &daisy::Messi::setFlushLimit, "Set the flush limit")
        .def("setInitialFblSize", &daisy::Messi::setInitialFblSize, "Set the initial FBL size")
        .def("setTotalLoadedLeaves", &daisy::Messi::setTotalLoadedLeaves, "Set the number of total loaded leaves")
        .def("setTightBound", &daisy::Messi::setTightBound, "Enable or disable tight bounds")
        .def("setSearchWorkers", &daisy::Messi::setSearchWorkers, "Set the number of worker threads for search")
        .def("setIndexWorkers", &daisy::Messi::setIndexWorkers, "Set the number of worker threads for indexing")
        .def("setReadBlockLength", &daisy::Messi::setReadBlockLength, "Set the length of each read block")
        .def("setWarpingWindow", &daisy::Messi::setWarpingWindow, "Set the warping window size for DTW")

        // Build the index from a 2D NumPy array
        .def("buildIndex", [](daisy::Messi &self, pybind11::array_t<float> db)
             {
            pybind11::buffer_info buf = db.request();
            if (buf.ndim != 2)
                throw std::runtime_error("Database array must be 2D");
            
            daisy::idx_t n = buf.shape[0];
            daisy::idx_t d = buf.shape[1];
            
            // Create InMemoryDataSource from numpy array
            daisy::InMemoryDataSource data_source(static_cast<float *>(buf.ptr), n, d);
            self.buildIndex(&data_source); }, "Build the MESSI index from a 2D float32 NumPy array")

        // Search the index with query array and return top-k results
        .def("searchIndex", [](daisy::Messi &self, pybind11::array_t<float> query, daisy::idx_t k)
             {
            pybind11::buffer_info query_buf = query.request();
            if (query_buf.ndim != 2)
                throw std::runtime_error("Query array must be 2D");
            if (k <= 0)
                throw std::runtime_error("k must be positive");

            const daisy::idx_t n_query = query_buf.shape[0];
            const daisy::idx_t dim = query_buf.shape[1];

            std::vector<daisy::idx_t> indices(n_query * k);
            std::vector<float> distances(n_query * k);

            self.searchIndex(static_cast<float *>(query_buf.ptr), n_query, k, indices.data(), distances.data());

            return pybind11::make_tuple(
                pybind11::array_t<daisy::idx_t>({n_query, k}, indices.data()),
                pybind11::array_t<float>({n_query, k}, distances.data())
            ); }, "Search the MESSI index using queries and return (indices, distances)");

#if ODYSSEY_MPI
    ////// ODYSSEY //////
    pybind11::class_<daisy::Odyssey>(m, "Odyssey", "Odyssey similarity search with MPI")
        // Constructor
        .def(pybind11::init<daisy::DistanceType>(), "Create a new Odyssey with the given distance metric")

        // Setters
        .def("setNumThreads", &daisy::Odyssey::setNumThreads, "Set the number of threads to use (for OpenMP parts)")
        .def("setWarpingWindow", &daisy::Odyssey::setWarpingWindow, "Set the warping window size for DTW (typically 10% of time series length)")

        // Getters
        .def("getNumThreads", &daisy::Odyssey::getNumThreads, "Get the number of threads (for OpenMP parts)")
        .def("getWarpingWindow", &daisy::Odyssey::getWarpingWindow, "Get the warping window size for DTW")

        // Bind method to build the index from a NumPy array
        // Odyssey requires FileDataSource (disk-based indexing). We write the array to a temp file.
        .def("buildIndex", [](daisy::Odyssey &self, pybind11::array_t<float> db)
             {
            pybind11::buffer_info buf = db.request();
            if (buf.ndim != 2)
                throw std::runtime_error("Database array must be 2D");

            daisy::idx_t n = buf.shape[0];
            daisy::idx_t d = buf.shape[1];

#if ODYSSEY_MPI
            int my_rank = self.getMyRank();
            if (my_rank == 0) {
#endif
            // Write array to temp file (rank 0 only with MPI)
            const char *tmp_path = "/tmp/odyssey_pybind_db.bin";
            FILE *fp = std::fopen(tmp_path, "wb");
            if (!fp)
                throw std::runtime_error("Odyssey buildIndex: could not create temp file " + std::string(tmp_path));
            size_t written = std::fwrite(buf.ptr, sizeof(float), static_cast<size_t>(n) * static_cast<size_t>(d), fp);
            std::fclose(fp);
            if (written != static_cast<size_t>(n) * static_cast<size_t>(d))
                throw std::runtime_error("Odyssey buildIndex: failed to write temp file");
#if ODYSSEY_MPI
            }
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            daisy::FileDataSource data_source("/tmp/odyssey_pybind_db.bin", d, n);
            self.buildIndex(&data_source); }, "Build the index from a 2D float32 numpy array")

        // Bind method to perform similarity search
        .def("searchIndex", [](daisy::Odyssey &self, pybind11::array_t<float> query, daisy::idx_t k)
             {
            pybind11::buffer_info query_buf = query.request(); // Get query buffer info

            // Check for valid input shape
            if (query_buf.ndim != 2)
                throw std::runtime_error("Query array must be 2D");
            if (k <= 0)
                throw std::runtime_error("k must be positive");            

            // Number of query vectors and their dimension
            daisy::idx_t n_query = query_buf.shape[0];
            daisy::idx_t dim = query_buf.shape[1];

            // Allocate output arrays for indices and distances
            std::vector<daisy::idx_t> indices(n_query * k);
            std::vector<float> distances(n_query * k);
            
            // Perform the search
            self.searchIndex(static_cast<float *>(query_buf.ptr), n_query, k,
                             indices.data(), distances.data());

            // Return results as a tuple of NumPy arrays
            return pybind11::make_tuple(
                pybind11::array_t<daisy::idx_t>({n_query, k}, indices.data()),
                pybind11::array_t<float>({n_query, k}, distances.data())
            ); }, "Search the index with queries and return (indices, distances)");
#endif  // ODYSSEY_MPI

    ////// PARIS //////
    pybind11::class_<daisy::ParIS>(m, "ParIS", "ParIS similarity search (file-based)")
        // Constructor
        .def(pybind11::init<daisy::DistanceType>(), "Create a new ParIS instance with the given distance metric")

        // Setters
        .def("setNumThreads", &daisy::ParIS::setNumThreads, "Set the number of threads to use")
        .def("setWarpingWindow", &daisy::ParIS::setWarpingWindow, "Set the warping window size for DTW (typically 10% of time series length)")

        // Getters
        .def("getNumThreads", &daisy::ParIS::getNumThreads, "Get the number of threads")
        .def("getWarpingWindow", &daisy::ParIS::getWarpingWindow, "Get the warping window size for DTW")

        // Build the index from a file (ParIS requires file-based data)
        .def("buildIndex", [](daisy::ParIS &self, const std::string &filename, daisy::idx_t dim, daisy::idx_t n_database = 0)
             {
            self.buildIndex(filename, dim, n_database); }, 
            pybind11::arg("filename"), pybind11::arg("dim"), pybind11::arg("n_database") = 0,
            "Build the ParIS index from a binary file. filename: path to binary data file, dim: dimension of each time series, n_database: number of time series (0 = auto-detect from file size)")

        // Search the index with query array and return top-k results
        .def("searchIndex", [](daisy::ParIS &self, pybind11::array_t<float> query, daisy::idx_t k)
             {
            pybind11::buffer_info query_buf = query.request();
            if (query_buf.ndim != 2)
                throw std::runtime_error("Query array must be 2D");
            if (k <= 0)
                throw std::runtime_error("k must be positive");

            const daisy::idx_t n_query = query_buf.shape[0];
            const daisy::idx_t dim = query_buf.shape[1];

            std::vector<daisy::idx_t> indices(n_query * k);
            std::vector<float> distances(n_query * k);

            self.searchIndex(static_cast<float *>(query_buf.ptr), n_query, k, indices.data(), distances.data());

            return pybind11::make_tuple(
                pybind11::array_t<daisy::idx_t>({n_query, k}, indices.data()),
                pybind11::array_t<float>({n_query, k}, distances.data())
            ); }, "Search the ParIS index using queries and return (indices, distances)");

    ////// SING //////
    // Only include Sing bindings if Sing is built (requires CUDA)
#ifdef SING_CUDA_ENABLED
    #if SING_CUDA_ENABLED != 0
    pybind11::class_<daisy::Sing>(m, "Sing", "Sing similarity search algorithm")
        // Constructor
        .def(pybind11::init<daisy::DistanceType>(), "Create a new Sing instance with the given distance metric")

        // Setters
        .def("setNumThreads", &daisy::Sing::setNumThreads, "Set the number of threads to use")

        // Getters
        .def("getNumThreads", &daisy::Sing::getNumThreads, "Get the number of threads")

        // Bind method to build the index from a NumPy array
        .def("buildIndex", [](daisy::Sing &self, pybind11::array_t<float> db)
             {
            pybind11::buffer_info buf = db.request();
            if (buf.ndim != 2)
                throw std::runtime_error("Database array must be 2D");

            daisy::idx_t n = buf.shape[0];
            daisy::idx_t d = buf.shape[1];

            // Create InMemoryDataSource from numpy array
            daisy::InMemoryDataSource data_source(static_cast<float *>(buf.ptr), n, d);
            self.buildIndex(&data_source); }, "Build the index from a 2D float32 numpy array")

        // Bind method to perform similarity search
        .def("searchIndex", [](daisy::Sing &self, pybind11::array_t<float> query, daisy::idx_t k)
             {
            pybind11::buffer_info query_buf = query.request(); // Get query buffer info

            // Check for valid input shape
            if (query_buf.ndim != 2)
                throw std::runtime_error("Query array must be 2D");
            if (k <= 0)
                throw std::runtime_error("k must be positive");

            // Number of query vectors and their dimension
            daisy::idx_t n_query = query_buf.shape[0];
            daisy::idx_t dim = query_buf.shape[1];

            // Allocate output arrays for indices and distances
            std::vector<daisy::idx_t> indices(n_query * k);
            std::vector<float> distances(n_query * k);

            // Perform the search
            self.searchIndex(static_cast<float *>(query_buf.ptr), n_query, k,
                             indices.data(), distances.data());

            // Return results as a tuple of NumPy arrays
            return pybind11::make_tuple(
                pybind11::array_t<daisy::idx_t>({n_query, k}, indices.data()),
                pybind11::array_t<float>({n_query, k}, distances.data())
            ); }, "Search the index with queries and return (indices, distances)");
    #endif
#endif

#ifdef SOFA_FFTW_ENABLED
    #if SOFA_FFTW_ENABLED != 0
    ////// SOFA //////
    pybind11::class_<daisy::Sofa>(m, "Sofa", "SOFA (SFA + iSAX) time series similarity index")
        // Constructor
        .def(pybind11::init<daisy::DistanceType>(), "Create a new Sofa with the given distance metric")

        // Getters
        .def("getNumThreads", &daisy::Sofa::getNumThreads, "Get the number of search threads")
        .def("getWordLength", &daisy::Sofa::getWordLength, "Get the SFA word length")
        .def("getAlphabetSize", &daisy::Sofa::getAlphabetSize, "Get the alphabet size")
        .def("getSearchWorkers", &daisy::Sofa::getSearchWorkers, "Get the number of search worker threads")
        .def("getIndexWorkers", &daisy::Sofa::getIndexWorkers, "Get the number of index worker threads")
        .def("getReadBlockLength", &daisy::Sofa::getReadBlockLength, "Get the read block length")
        .def("getLeafSize", &daisy::Sofa::getLeafSize, "Get the leaf size")
        .def("getMinLeafSize", &daisy::Sofa::getMinLeafSize, "Get the minimum leaf size")
        .def("getInitialLblSize", &daisy::Sofa::getInitialLblSize, "Get the initial LBL size")
        .def("getFlushLimit", &daisy::Sofa::getFlushLimit, "Get the flush limit")
        .def("getInitialFblSize", &daisy::Sofa::getInitialFblSize, "Get the initial FBL size")
        .def("getTotalLoadedLeaves", &daisy::Sofa::getTotalLoadedLeaves, "Get the total number of loaded leaves")
        .def("getTightBound", &daisy::Sofa::getTightBound, "Get whether tight bounds are enabled")
        .def("getSampleSize", &daisy::Sofa::getSampleSize, "Get the MCB training sample size")
        .def("getHistogramType", &daisy::Sofa::getHistogramType, "Get the histogram type (1=equi-depth, 2=equi-width)")
        .def("getCoeffNumber", &daisy::Sofa::getCoeffNumber, "Get the number of DFT coefficients (0 = all)")
        .def("getIsNorm", &daisy::Sofa::getIsNorm, "Get whether z-normalization is applied")

        // Setters
        .def("setNumThreads", &daisy::Sofa::setNumThreads, "Set the number of threads to use for both indexing and search")
        .def("setWordLength", &daisy::Sofa::setWordLength, "Set the SFA word length")
        .def("setAlphabetSize", &daisy::Sofa::setAlphabetSize, "Set the alphabet size")
        .def("setSearchWorkers", &daisy::Sofa::setSearchWorkers, "Set the number of search worker threads")
        .def("setIndexWorkers", &daisy::Sofa::setIndexWorkers, "Set the number of index worker threads")
        .def("setReadBlockLength", &daisy::Sofa::setReadBlockLength, "Set the read block length")
        .def("setLeafSize", &daisy::Sofa::setLeafSize, "Set the leaf size")
        .def("setMinLeafSize", &daisy::Sofa::setMinLeafSize, "Set the minimum leaf size")
        .def("setInitialLblSize", &daisy::Sofa::setInitialLblSize, "Set the initial LBL size")
        .def("setFlushLimit", &daisy::Sofa::setFlushLimit, "Set the flush limit")
        .def("setInitialFblSize", &daisy::Sofa::setInitialFblSize, "Set the initial FBL size")
        .def("setTotalLoadedLeaves", &daisy::Sofa::setTotalLoadedLeaves, "Set the total number of loaded leaves")
        .def("setTightBound", &daisy::Sofa::setTightBound, "Enable or disable tight bounds")
        .def("setSampleSize", &daisy::Sofa::setSampleSize, "Set the MCB training sample size")
        .def("setHistogramType", &daisy::Sofa::setHistogramType, "Set the histogram type (1=equi-depth, 2=equi-width)")
        .def("setCoeffNumber", &daisy::Sofa::setCoeffNumber, "Set the number of DFT coefficients (0 = all)")
        .def("setIsNorm", &daisy::Sofa::setIsNorm, "Set whether z-normalization is applied")

        // Build the index from a 2D NumPy array
        .def("buildIndex", [](daisy::Sofa &self, pybind11::array_t<float> db)
             {
            pybind11::buffer_info buf = db.request();
            if (buf.ndim != 2)
                throw std::runtime_error("Database array must be 2D");

            daisy::idx_t n = buf.shape[0];
            daisy::idx_t d = buf.shape[1];

            daisy::InMemoryDataSource data_source(static_cast<float *>(buf.ptr), n, d);
            self.buildIndex(&data_source); }, "Build the SOFA index from a 2D float32 NumPy array")

        // Search the index with query array and return top-k results
        .def("searchIndex", [](daisy::Sofa &self, pybind11::array_t<float> query, daisy::idx_t k)
             {
            pybind11::buffer_info query_buf = query.request();
            if (query_buf.ndim != 2)
                throw std::runtime_error("Query array must be 2D");
            if (k <= 0)
                throw std::runtime_error("k must be positive");

            const daisy::idx_t n_query = query_buf.shape[0];

            std::vector<daisy::idx_t> indices(n_query * k);
            std::vector<float> distances(n_query * k);

            self.searchIndex(static_cast<float *>(query_buf.ptr), n_query, k,
                             indices.data(), distances.data());

            return pybind11::make_tuple(
                pybind11::array_t<daisy::idx_t>({n_query, k}, indices.data()),
                pybind11::array_t<float>({n_query, k}, distances.data())
            ); }, "Search the SOFA index using queries and return (indices, distances)");
    #endif
#endif
}
