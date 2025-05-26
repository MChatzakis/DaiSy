#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../lib/algos/BruteforceSearch.hpp"
#include "../lib/distance_computers/DistanceComputer.hpp" 

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
}
