#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../lib/algos/BruteforceSearch.hpp"
#include "../lib/distance_computers/DistanceComputer.hpp" 

using namespace diNoLib;

PYBIND11_MODULE(diNo_lib, m) {
    m.doc() = "diNo::BruteForceSearch_lib Python bindings";

    pybind11::enum_<DistanceType>(m, "DistanceType")
        .value("L2_SQUARED", DistanceType::L2_SQUARED)
        .export_values();

    pybind11::class_<BruteForceSearch>(m, "BruteForceSearch")
        .def(pybind11::init<DistanceType>())
        .def("setNumThreads", &BruteForceSearch::setNumThreads)
        .def("getNumThreads", &BruteForceSearch::getNumThreads)
        .def("buildIndex", [](BruteForceSearch &self, pybind11::array_t<float> db) {
            pybind11::buffer_info buf = db.request();
            if (buf.ndim != 2)
                throw std::runtime_error("Database array must be 2D");

            idx_t n = buf.shape[0];
            idx_t d = buf.shape[1];

            self.buildIndex(static_cast<float *>(buf.ptr), n, d);
        })
        .def("searchIndex", [](BruteForceSearch &self,
                               pybind11::array_t<float> query,
                               idx_t k) {
            pybind11::buffer_info query_buf = query.request();

            if (query_buf.ndim != 2)
                throw std::runtime_error("Query array must be 2D");

            idx_t n_query = query_buf.shape[0];
            idx_t dim = query_buf.shape[1];

            std::vector<idx_t> indices(n_query * k);
            std::vector<float> distances(n_query * k);

            self.searchIndex(static_cast<float *>(query_buf.ptr), n_query, k,
                             indices.data(), distances.data());

            return pybind11::make_tuple(
                pybind11::array_t<idx_t>({n_query, k}, indices.data()),
                pybind11::array_t<float>({n_query, k}, distances.data())
            );
        });
}
