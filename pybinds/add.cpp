#include <pybind11/pybind11.h>
#include "../lib/algos/tmpAdd.cpp"

PYBIND11_MODULE(Addexample, m) {
    m.doc() = "diNo"; 

    m.def("add", &add, "A function that adds two numbers",
          pybind11::arg("a"), pybind11::arg("b"));

    m.def("subtract", &subtract, "A function that subtracts two numbers",
          pybind11::arg("a"), pybind11::arg("b"));
}