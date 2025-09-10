#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "abcluster.hpp"

namespace py = pybind11;

PYBIND11_MODULE(abcluster, m) {
    m.doc() = "Robust average-linkage antibody clustering (Python wrapper)";

    py::class_<ClusterOptions>(m, "ClusterOptions")
        .def(py::init<>())
        .def_readwrite("cutoff", &ClusterOptions::cutoff)
        .def_readwrite("mut_value", &ClusterOptions::mut_value)
        .def_readwrite("epsilon", &ClusterOptions::epsilon)
        .def_readwrite("len_penalty", &ClusterOptions::len_penalty)
        .def_readwrite("canonical_samples", &ClusterOptions::canonical_samples);

    py::class_<Record>(m, "Record")
        .def(py::init<>())
        .def_readwrite("junc", &Record::junc)
        .def_readwrite("v_gene", &Record::v_gene)
        .def_readwrite("j_gene", &Record::j_gene)
        .def_readwrite("mutations", &Record::mutations);

    m.def("cluster",
          [](const std::vector<Record>& recs, const ClusterOptions& opt) {
              return cluster_records(recs, opt).labels;
          },
          py::arg("records"), py::arg("options") = ClusterOptions(),
          "Cluster records and return a list of string cluster labels.");
}
