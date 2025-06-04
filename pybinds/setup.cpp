#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../lib/distance_computers/DistanceComputer.hpp" 
#include "../lib/algos/BruteforceSearch.hpp"
#include "../lib/algos/LbBruteforceSearch.hpp"

// Define a Python module named 'diNoSimilaritySearch'
PYBIND11_MODULE(diNoSimilaritySearch, m) {
    m.doc() = "diNo::diNoSimilaritySearch Python bindings";

    // Expose the DistanceType enum to Python
    pybind11::enum_<diNoLib::DistanceType>(m, "DistanceType")
        .value("L2_SQUARED", diNoLib::DistanceType::L2_SQUARED)
        .export_values();

    // Bind the BruteForceSearch class to Python
    pybind11::class_<diNoLib::BruteForceSearch>(m, "BruteForceSearch", "Brute force similarity search")
        // Constructor: create with specified distance metric
        .def(pybind11::init<diNoLib::DistanceType>(), "Create a new BruteForceSearch with the given distance metric")

        // Set the number of threads for parallel processing
        .def("setNumThreads", &diNoLib::BruteForceSearch::setNumThreads, "Set the number of threads to use")

        // Get the number of threads
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

    // Bind the LbBruteforceSearch class to Python
    pybind11::class_<diNoLib::LbBruteforceSearch>(m, "LbBruteforceSearch", "Lower Bound brute-force similarity search")
        // Constructor: create with specified distance metric
        .def(pybind11::init<diNoLib::DistanceType>(), "Create a new LbBruteforceSearch with the given distance metric")

        // Set the number of threads for parallel processing
        .def("setNumThreads", &diNoLib::LbBruteforceSearch::setNumThreads, "Set the number of threads to use")

        // Accessor methods for internal parameters
        .def("getPaaSegments", &diNoLib::LbBruteforceSearch::getPaaSegments, "Get the number of PAA segments")
        .def("getSaxCardinality", &diNoLib::LbBruteforceSearch::getSaxCardinality, "Get the SAX cardinality used for dimensionality reduction")
        .def("getLeafSize", &diNoLib::LbBruteforceSearch::getLeafSize, "Get the configured leaf size")
        .def("getMinLeafSize", &diNoLib::LbBruteforceSearch::getMinLeafSize, "Get the minimum leaf size for search optimization")
        .def("getInitialLblSize", &diNoLib::LbBruteforceSearch::getInitialLblSize, "Get the initial size of the lower-bound leaf buffer (LBL)")
        .def("getFlushLimit", &diNoLib::LbBruteforceSearch::getFlushLimit, "Get the flush limit for internal buffers")
        .def("getInitialFblSize", &diNoLib::LbBruteforceSearch::getInitialFblSize, "Get the initial size of the full buffer list (FBL)")
        .def("getTotalLoadedLeaves", &diNoLib::LbBruteforceSearch::getTotalLoadedLeaves, "Get the number of total loaded leaves")
        .def("getTightBound", &diNoLib::LbBruteforceSearch::getTightBound, "Get whether tight lower bounds are used")

        // Mutator methods for internal parameters
        .def("setPaaSegments", &diNoLib::LbBruteforceSearch::setPaaSegments, "Set the number of PAA segments")
        .def("setSaxCardinality", &diNoLib::LbBruteforceSearch::setSaxCardinality, "Set the SAX cardinality")
        .def("setLeafSize", &diNoLib::LbBruteforceSearch::setLeafSize, "Set the leaf size")
        .def("setMinLeafSize", &diNoLib::LbBruteforceSearch::setMinLeafSize, "Set the minimum leaf size")
        .def("setInitialLblSize", &diNoLib::LbBruteforceSearch::setInitialLblSize, "Set the initial size of the lower-bound leaf buffer (LBL)")
        .def("setFlushLimit", &diNoLib::LbBruteforceSearch::setFlushLimit, "Set the flush limit for internal buffers")
        .def("setInitialFblSize", &diNoLib::LbBruteforceSearch::setInitialFblSize, "Set the initial size of the full buffer list (FBL)")
        .def("setTotalLoadedLeaves", &diNoLib::LbBruteforceSearch::setTotalLoadedLeaves, "Set the number of total loaded leaves")
        .def("setTightBound", &diNoLib::LbBruteforceSearch::setTightBound, "Enable or disable tight lower bounds")

        // Build the index from a NumPy array
        .def("buildIndex", [](diNoLib::LbBruteforceSearch &self, pybind11::array_t<float> db) {
            pybind11::buffer_info buf = db.request();
            if (buf.ndim != 2) 
                throw std::runtime_error("Database array must be 2D");
            self.buildIndex(static_cast<float *>(buf.ptr), buf.shape[0], buf.shape[1]);
        }, "Build the index from a 2D float32 numpy array")

        // Search the index using a query array and return (indices, distances)
        .def("searchIndex", [](diNoLib::LbBruteforceSearch &self, pybind11::array_t<float> query, diNoLib::idx_t k) {
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
}
