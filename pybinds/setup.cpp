#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../lib/distance_computers/DistanceComputer.hpp" 
#include "../lib/algos/Bruteforce.hpp"
#include "../lib/algos/LbBruteforce.hpp"
#include "../lib/algos/Messi.hpp"
#include "../lib/algos/Odyssey.hpp"
// #include "../lib/algos/ParIS.hpp"
#include "../lib/algos/Sing.hpp"

// Define a Python module named 'diNoSimilaritySearch'
PYBIND11_MODULE(diNoSimilaritySearch, m) {
    m.doc() = "diNo::diNoSimilaritySearch Python bindings";

    ////// DISTANCETYPE //////
    pybind11::enum_<diNoLib::DistanceType>(m, "DistanceType")
        .value("L2_SQUARED", diNoLib::DistanceType::L2_SQUARED)
        .export_values();

    ////// BRUTEFORCE //////
    pybind11::class_<diNoLib::BruteForceSearch>(m, "BruteForceSearch", "Brute force similarity search")
        // Constructor
        .def(pybind11::init<diNoLib::DistanceType>(), "Create a new BruteForceSearch with the given distance metric")

        // Setter
        .def("setNumThreads", &diNoLib::BruteForceSearch::setNumThreads, "Set the number of threads to use")

        // Getter
        .def("getNumThreads", &diNoLib::BruteForceSearch::getNumThreads, "Get the number of threads")

        // Bind method to build the index from a NumPy array
        .def("buildIndex", [](diNoLib::BruteForceSearch &self, pybind11::array_t<float> db) {
            pybind11::buffer_info buf = db.request();
            if (buf.ndim != 2)
                throw std::runtime_error("Database array must be 2D");

            diNoLib::idx_t n = buf.shape[0];
            diNoLib::idx_t d = buf.shape[1];

            self.buildIndex(static_cast<float *>(buf.ptr), n, d);
        }, "Build the index from a 2D float32 numpy array")

        // Bind method to perform similarity search
        .def("searchIndex", [](diNoLib::BruteForceSearch &self,
                               pybind11::array_t<float> query,
                               diNoLib::idx_t k) {
            pybind11::buffer_info query_buf = query.request(); // Get query buffer info

            // Check for valid input shape
            if (query_buf.ndim != 2)
                throw std::runtime_error("Query array must be 2D");
            if (k <= 0)
                throw std::runtime_error("k must be positive");            

            // Number of query vectors and their dimension
            diNoLib::idx_t n_query = query_buf.shape[0];
            diNoLib::idx_t dim = query_buf.shape[1];

            // Allocate output arrays for indices and distances
            std::vector<diNoLib::idx_t> indices(n_query * k);
            std::vector<float> distances(n_query * k);
            
            // Perform the search
            self.searchIndex(static_cast<float *>(query_buf.ptr), n_query, k,
                             indices.data(), distances.data());

            // Return results as a tuple of NumPy arrays
            return pybind11::make_tuple(
                pybind11::array_t<diNoLib::idx_t>({n_query, k}, indices.data()),
                pybind11::array_t<float>({n_query, k}, distances.data())
            );
        }, "Search the index with queries and return (indices, distances)");

    ////// LBBRUTEFORCE //////
    pybind11::class_<diNoLib::LbBruteforce>(m, "LbBruteforce", "Lower Bound brute-force similarity search")
        // Constructor
        .def(pybind11::init<diNoLib::DistanceType>(), "Create a new LbBruteforce with the given distance metric")

        // Getters
        .def("getPaaSegments", &diNoLib::LbBruteforce::getPaaSegments, "Get the number of PAA segments")
        .def("getSaxCardinality", &diNoLib::LbBruteforce::getSaxCardinality, "Get the SAX cardinality used for dimensionality reduction")
        .def("getLeafSize", &diNoLib::LbBruteforce::getLeafSize, "Get the configured leaf size")
        .def("getMinLeafSize", &diNoLib::LbBruteforce::getMinLeafSize, "Get the minimum leaf size for search optimization")
        .def("getInitialLblSize", &diNoLib::LbBruteforce::getInitialLblSize, "Get the initial size of the lower-bound leaf buffer (LBL)")
        .def("getFlushLimit", &diNoLib::LbBruteforce::getFlushLimit, "Get the flush limit for internal buffers")
        .def("getInitialFblSize", &diNoLib::LbBruteforce::getInitialFblSize, "Get the initial size of the full buffer list (FBL)")
        .def("getTotalLoadedLeaves", &diNoLib::LbBruteforce::getTotalLoadedLeaves, "Get the number of total loaded leaves")
        .def("getTightBound", &diNoLib::LbBruteforce::getTightBound, "Get whether tight lower bounds are used")

        // Setters
        .def("setNumThreads", &diNoLib::LbBruteforce::setNumThreads, "Set the number of threads to use")
        .def("setPaaSegments", &diNoLib::LbBruteforce::setPaaSegments, "Set the number of PAA segments")
        .def("setSaxCardinality", &diNoLib::LbBruteforce::setSaxCardinality, "Set the SAX cardinality")
        .def("setLeafSize", &diNoLib::LbBruteforce::setLeafSize, "Set the leaf size")
        .def("setMinLeafSize", &diNoLib::LbBruteforce::setMinLeafSize, "Set the minimum leaf size")
        .def("setInitialLblSize", &diNoLib::LbBruteforce::setInitialLblSize, "Set the initial size of the lower-bound leaf buffer (LBL)")
        .def("setFlushLimit", &diNoLib::LbBruteforce::setFlushLimit, "Set the flush limit for internal buffers")
        .def("setInitialFblSize", &diNoLib::LbBruteforce::setInitialFblSize, "Set the initial size of the full buffer list (FBL)")
        .def("setTotalLoadedLeaves", &diNoLib::LbBruteforce::setTotalLoadedLeaves, "Set the number of total loaded leaves")
        .def("setTightBound", &diNoLib::LbBruteforce::setTightBound, "Enable or disable tight lower bounds")

        // Build the index from a NumPy array
        .def("buildIndex", [](diNoLib::LbBruteforce &self, pybind11::array_t<float> db) {
            pybind11::buffer_info buf = db.request();
            if (buf.ndim != 2) 
                throw std::runtime_error("Database array must be 2D");
            self.buildIndex(static_cast<float *>(buf.ptr), buf.shape[0], buf.shape[1]);
        }, "Build the index from a 2D float32 numpy array")

        // Search the index using a query array and return (indices, distances)
        .def("searchIndex", [](diNoLib::LbBruteforce &self, pybind11::array_t<float> query, diNoLib::idx_t k) {
            pybind11::buffer_info buf = query.request();
            if (buf.ndim != 2) 
                throw std::runtime_error("Query array must be 2D");

            std::vector<diNoLib::idx_t> indices(buf.shape[0] * k);
            std::vector<float> distances(buf.shape[0] * k);

            self.searchIndex(static_cast<float *>(buf.ptr), buf.shape[0], k, indices.data(), distances.data());

            return pybind11::make_tuple(
                pybind11::array_t<diNoLib::idx_t>(pybind11::buffer_info(
                    indices.data(),
                    sizeof(diNoLib::idx_t),
                    pybind11::format_descriptor<diNoLib::idx_t>::format(),
                    2,
                    std::vector<pybind11::ssize_t>{static_cast<pybind11::ssize_t>(buf.shape[0]), static_cast<pybind11::ssize_t>(k)},
                    std::vector<pybind11::ssize_t>{
                        static_cast<pybind11::ssize_t>(sizeof(diNoLib::idx_t) * k),
                        static_cast<pybind11::ssize_t>(sizeof(diNoLib::idx_t))
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
            );
        }, "Search the index with queries and return (indices, distances)");    

    ////// MESSI //////
    pybind11::class_<diNoLib::Messi>(m, "Messi", "MESSI (Multi-Queue Efficient SAX Similarity Index) algorithm for time series similarity search")
        // Constructor
        .def(pybind11::init<diNoLib::DistanceType>(), "Create a new Messi instance with the given distance metric")

        // Getters 
        .def("getNumThreads", &diNoLib::Messi::getNumThreads, "Get the number of search threads")
        .def("getPaaSegments", &diNoLib::Messi::getPaaSegments, "Get the number of PAA segments used in SAX transformation")
        .def("getSaxCardinality", &diNoLib::Messi::getSaxCardinality, "Get the cardinality of SAX symbols")
        .def("getLeafSize", &diNoLib::Messi::getLeafSize, "Get the maximum leaf size in the index tree")
        .def("getMinLeafSize", &diNoLib::Messi::getMinLeafSize, "Get the minimum number of entries per leaf")
        .def("getInitialLblSize", &diNoLib::Messi::getInitialLblSize, "Get the initial size of the lower-bound buffer")
        .def("getFlushLimit", &diNoLib::Messi::getFlushLimit, "Get the flush limit before writing to disk")
        .def("getInitialFblSize", &diNoLib::Messi::getInitialFblSize, "Get the initial full-buffer size")
        .def("getTotalLoadedLeaves", &diNoLib::Messi::getTotalLoadedLeaves, "Get the total number of leaves loaded")
        .def("getTightBound", &diNoLib::Messi::getTightBound, "Check whether tight bounds are enabled")

        .def("getSearchWorkers", &diNoLib::Messi::getSearchWorkers, "Get number of worker threads used for search")
        .def("getIndexWorkers", &diNoLib::Messi::getIndexWorkers, "Get number of worker threads used for indexing")
        .def("getReadBlockLength", &diNoLib::Messi::getReadBlockLength, "Get block size for reading the time series data")
        .def("getWarpingWindow", &diNoLib::Messi::getWarpingWindow, "Get the DTW warping window constraint")

        // Setters 
        .def("setNumThreads", &diNoLib::Messi::setNumThreads, "Set the number of threads to use for both indexing and search")
        .def("setPaaSegments", &diNoLib::Messi::setPaaSegments, "Set the number of PAA segments")
        .def("setSaxCardinality", &diNoLib::Messi::setSaxCardinality, "Set the SAX cardinality")
        .def("setLeafSize", &diNoLib::Messi::setLeafSize, "Set the leaf size of the index tree")
        .def("setMinLeafSize", &diNoLib::Messi::setMinLeafSize, "Set the minimum size of a leaf")
        .def("setInitialLblSize", &diNoLib::Messi::setInitialLblSize, "Set the initial LBL size")
        .def("setFlushLimit", &diNoLib::Messi::setFlushLimit, "Set the flush limit")
        .def("setInitialFblSize", &diNoLib::Messi::setInitialFblSize, "Set the initial FBL size")
        .def("setTotalLoadedLeaves", &diNoLib::Messi::setTotalLoadedLeaves, "Set the number of total loaded leaves")
        .def("setTightBound", &diNoLib::Messi::setTightBound, "Enable or disable tight bounds")
        .def("setSearchWorkers", &diNoLib::Messi::setSearchWorkers, "Set the number of worker threads for search")
        .def("setIndexWorkers", &diNoLib::Messi::setIndexWorkers, "Set the number of worker threads for indexing")
        .def("setReadBlockLength", &diNoLib::Messi::setReadBlockLength, "Set the length of each read block")
        .def("setWarpingWindow", &diNoLib::Messi::setWarpingWindow, "Set the warping window size for DTW")

        // Build the index from a 2D NumPy array
        .def("buildIndex", [](diNoLib::Messi &self, pybind11::array_t<float> db) {
            pybind11::buffer_info buf = db.request();
            if (buf.ndim != 2)
                throw std::runtime_error("Database array must be 2D");
            self.buildIndex(static_cast<float *>(buf.ptr), buf.shape[0], buf.shape[1]);
        }, "Build the MESSI index from a 2D float32 NumPy array")

        // Search the index with query array and return top-k results
        .def("searchIndex", [](diNoLib::Messi &self, pybind11::array_t<float> query, diNoLib::idx_t k) {
            pybind11::buffer_info query_buf = query.request();
            if (query_buf.ndim != 2)
                throw std::runtime_error("Query array must be 2D");
            if (k <= 0)
                throw std::runtime_error("k must be positive");

            const diNoLib::idx_t n_query = query_buf.shape[0];
            const diNoLib::idx_t dim = query_buf.shape[1];

            std::vector<diNoLib::idx_t> indices(n_query * k);
            std::vector<float> distances(n_query * k);

            self.searchIndex(static_cast<float *>(query_buf.ptr), n_query, k, indices.data(), distances.data());

            return pybind11::make_tuple(
                pybind11::array_t<diNoLib::idx_t>({n_query, k}, indices.data()),
                pybind11::array_t<float>({n_query, k}, distances.data())
            );
        }, "Search the MESSI index using queries and return (indices, distances)");

    ////// ODYSSEY //////
    pybind11::class_<diNoLib::Odyssey>(m, "Odyssey", "Odyssey similarity search with MPI")
        // Constructor
        .def(pybind11::init<diNoLib::DistanceType>(), "Create a new Odyssey with the given distance metric")

        // Setters
        .def("setNumThreads", &diNoLib::Odyssey::setNumThreads, "Set the number of threads to use (for OpenMP parts)")

        // Getters
        .def("getNumThreads", &diNoLib::Odyssey::getNumThreads, "Get the number of threads (for OpenMP parts)")

        // Bind method to build the index from a NumPy array
        .def("buildIndex", [](diNoLib::Odyssey &self, pybind11::array_t<float> db) {
            pybind11::buffer_info buf = db.request();
            if (buf.ndim != 2)
                throw std::runtime_error("Database array must be 2D");

            diNoLib::idx_t n = buf.shape[0];
            diNoLib::idx_t d = buf.shape[1];

            self.buildIndex(static_cast<float *>(buf.ptr), n, d);
        }, "Build the index from a 2D float32 numpy array")

        // Bind method to perform similarity search
        .def("searchIndex", [](diNoLib::Odyssey &self,
                               pybind11::array_t<float> query,
                               diNoLib::idx_t k) {
            pybind11::buffer_info query_buf = query.request(); // Get query buffer info

            // Check for valid input shape
            if (query_buf.ndim != 2)
                throw std::runtime_error("Query array must be 2D");
            if (k <= 0)
                throw std::runtime_error("k must be positive");            

            // Number of query vectors and their dimension
            diNoLib::idx_t n_query = query_buf.shape[0];
            diNoLib::idx_t dim = query_buf.shape[1];

            // Allocate output arrays for indices and distances
            std::vector<diNoLib::idx_t> indices(n_query * k);
            std::vector<float> distances(n_query * k);
            
            // Perform the search
            self.searchIndex(static_cast<float *>(query_buf.ptr), n_query, k,
                             indices.data(), distances.data());

            // Return results as a tuple of NumPy arrays
            return pybind11::make_tuple(
                pybind11::array_t<diNoLib::idx_t>({n_query, k}, indices.data()),
                pybind11::array_t<float>({n_query, k}, distances.data())
            );
        }, "Search the index with queries and return (indices, distances)");    

    ////// PARIS //////

    ////// SING //////
    pybind11::class_<diNoLib::Sing>(m, "Sing", "Sing similarity search algorithm")
        // Constructor
        .def(pybind11::init<diNoLib::DistanceType>(), "Create a new Sing instance with the given distance metric")

        // Setters
        .def("setNumThreads", &diNoLib::Sing::setNumThreads, "Set the number of threads to use")

        // Getters
        .def("getNumThreads", &diNoLib::Sing::getNumThreads, "Get the number of threads")

        // Bind method to build the index from a NumPy array
        .def("buildIndex", [](diNoLib::Sing &self, pybind11::array_t<float> db) {
            pybind11::buffer_info buf = db.request();
            if (buf.ndim != 2)
                throw std::runtime_error("Database array must be 2D");

            diNoLib::idx_t n = buf.shape[0];
            diNoLib::idx_t d = buf.shape[1];

            self.buildIndex(static_cast<float *>(buf.ptr), n, d);
        }, "Build the index from a 2D float32 numpy array")

        // Bind method to perform similarity search
        .def("searchIndex", [](diNoLib::Sing &self,
                               pybind11::array_t<float> query,
                               diNoLib::idx_t k) {
            pybind11::buffer_info query_buf = query.request(); // Get query buffer info

            // Check for valid input shape
            if (query_buf.ndim != 2)
                throw std::runtime_error("Query array must be 2D");
            if (k <= 0)
                throw std::runtime_error("k must be positive");

            // Number of query vectors and their dimension
            diNoLib::idx_t n_query = query_buf.shape[0];
            diNoLib::idx_t dim = query_buf.shape[1];

            // Allocate output arrays for indices and distances
            std::vector<diNoLib::idx_t> indices(n_query * k);
            std::vector<float> distances(n_query * k);

            // Perform the search
            self.searchIndex(static_cast<float *>(query_buf.ptr), n_query, k,
                             indices.data(), distances.data());

            // Return results as a tuple of NumPy arrays
            return pybind11::make_tuple(
                pybind11::array_t<diNoLib::idx_t>({n_query, k}, indices.data()),
                pybind11::array_t<float>({n_query, k}, distances.data())
            );
        }, "Search the index with queries and return (indices, distances)");    
}
