#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <thread>

#include <omp.h>

#include "pyanns/algorithms/clustering.hpp"
#include "pyanns/builder.hpp"
#include "pyanns/hnsw/hnsw.hpp"
#include "pyanns/nsg/nsg.hpp"
#include "pyanns/searcher/graph_searcher.hpp"
#include "pyanns/searcher/sparse_searcher.hpp"

namespace py = pybind11;
using namespace pybind11::literals; // needed to bring in _a literal

inline void get_input_array_shapes(const py::buffer_info &buffer, size_t *rows,
                                   size_t *features) {
  if (buffer.ndim != 2 && buffer.ndim != 1) {
    char msg[256];
    snprintf(msg, sizeof(msg),
             "Input vector data wrong shape. Number of dimensions %d. Data "
             "must be a 1D or 2D array.",
             buffer.ndim);
  }
  if (buffer.ndim == 2) {
    *rows = buffer.shape[0];
    *features = buffer.shape[1];
  } else {
    *rows = 1;
    *features = buffer.shape[0];
  }
}

void set_num_threads(int num_threads) { omp_set_num_threads(num_threads); }

struct Graph {
  pyanns::Graph<int> graph;

  Graph() = default;

  explicit Graph(Graph &&rhs) : graph(std::move(rhs.graph)) {}

  Graph(const std::string &filename, const std::string &format = "pyanns") {
    graph.load(filename, format);
  }

  explicit Graph(pyanns::Graph<int> graph) : graph(std::move(graph)) {}

  void save(const std::string &filename) { graph.save(filename); }

  void load(const std::string &filename, const std::string &format = "pyanns") {
    graph.load(filename, format);
  }
};

struct IndexSparse {
  pyanns::IndexSparse index;

  IndexSparse() = default;

  void add(const std::string &filename, float drop_ratio) {
    index.add(filename, drop_ratio);
  }

  py::object search_batch(int32_t nq, py::object indptr, py::object indices,
                          py::object data, int32_t topk, float budget,
                          int32_t refine_mul) {
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> indptr_x(
        indptr);
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> indices_x(
        indices);
    py::array_t<float, py::array::c_style | py::array::forcecast> data_x(data);
    int32_t *ids = new int32_t[nq * topk];
    index.search_batch(nq, (int32_t *)indptr_x.data(0),
                       (int32_t *)indices_x.data(0), (float *)data_x.data(0),
                       topk, ids, budget, refine_mul);
    py::capsule free_when_done(ids, [](void *f) { delete[] f; });
    return py::array_t<int>({nq * topk}, {sizeof(int)}, ids, free_when_done);
  }

  auto inverted_index() { return index.inverted_index; }
};


struct Index {
  std::unique_ptr<pyanns::Builder> index = nullptr;

  Index(const std::string &index_type, int dim, const std::string &metric,
        const std::string &quant = "FP32", int R = 32, int L = 200) {
    if (index_type == "NSG") {
      index = std::unique_ptr<pyanns::Builder>(
          (pyanns::Builder *)new pyanns::NSG(dim, metric, R, L));
    } else if (index_type == "HNSW") {
      index = pyanns::create_hnsw(metric, quant, dim, R, L);
    } else {
      printf("Index type [%s] not supported\n", index_type.c_str());
    }
  }

  Graph build(py::object input) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(input);
    auto buffer = items.request();
    size_t rows, features;
    get_input_array_shapes(buffer, &rows, &features);
    float *vector_data = (float *)items.data(0);
    index->Build(vector_data, rows);
    return Graph(index->GetGraph());
  }
};

struct Searcher {

  std::unique_ptr<pyanns::GraphSearcherBase> searcher;

  Searcher(Graph &graph, py::object input, const std::string &metric,
           const std::string &quantizer)
      : searcher(
            std::unique_ptr<pyanns::GraphSearcherBase>(pyanns::create_searcher(
                std::move(graph.graph), metric, quantizer))) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(input);
    auto buffer = items.request();
    size_t rows, features;
    get_input_array_shapes(buffer, &rows, &features);
    float *vector_data = (float *)items.data(0);
    searcher->SetData(vector_data, rows, features);
  }

  py::object search(py::object query, int k) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(query);
    int *ids;
    {
      py::gil_scoped_release l;
      ids = new int[k];
      searcher->Search(items.data(0), k, ids);
    }
    py::capsule free_when_done(ids, [](void *f) { delete[] f; });
    return py::array_t<int>({k}, {sizeof(int)}, ids, free_when_done);
  }

  py::object batch_search(py::object query, int k, int num_threads = 0) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(query);
    auto buffer = items.request();
    int32_t *ids;
    size_t nq, dim;
    {
      py::gil_scoped_release l;
      get_input_array_shapes(buffer, &nq, &dim);
      ids = new int[nq * k];
      if (num_threads != 0) {
        omp_set_num_threads(num_threads);
      }
      searcher->SearchBatch(items.data(0), nq, k, ids);
    }
    py::capsule free_when_done(ids, [](void *f) { delete[] f; });
    return py::array_t<int>({nq * k}, {sizeof(int)}, ids, free_when_done);
  }

  void set_ef(int ef) { searcher->SetEf(ef); }

  void optimize(int num_threads = 0) { searcher->Optimize(num_threads); }
};

PYBIND11_PLUGIN(pyanns) {
  py::module m("pyanns");

  m.def("set_num_threads", &set_num_threads, py::arg("num_threads"));

  py::class_<Graph>(m, "Graph")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &>(),
           py::arg("filename"), py::arg("format") = "pyanns")
      .def("save", &Graph::save, py::arg("filename"))
      .def("load", &Graph::load, py::arg("filename"),
           py::arg("format") = "pyanns");

  py::class_<Index>(m, "Index")
      .def(py::init<const std::string &, int, const std::string &,
                    const std::string &, int, int>(),
           py::arg("index_type"), py::arg("dim"), py::arg("metric"),
           py::arg("quant") = "FP32", py::arg("R") = 32, py::arg("L") = 0)
      .def("build", &Index::build, py::arg("data"));

  py::class_<Searcher>(m, "Searcher")
      .def(py::init<Graph &, py::object, const std::string &,
                    const std::string &>(),
           py::arg("graph"), py::arg("data"), py::arg("metric"),
           py::arg("quantizer"))
      .def("set_ef", &Searcher::set_ef, py::arg("ef"))
      .def("search", &Searcher::search, py::arg("query"), py::arg("k"))
      .def("batch_search", &Searcher::batch_search, py::arg("query"),
           py::arg("k"), py::arg("num_threads") = 0)
      .def("optimize", &Searcher::optimize, py::arg("num_threads") = 0);

  py::class_<IndexSparse>(m, "IndexSparse")
      .def(py::init<>())
      .def("add", &IndexSparse::add, py::arg("filename"), py::arg("drop_ratio"))
      .def("search_batch", &IndexSparse::search_batch, py::arg("nq"),
           py::arg("indptr"), py::arg("indices"), py::arg("data"),
           py::arg("topk"), py::arg("budget"), py::arg("refine_mul"))
      .def_property_readonly("inverted_index", &IndexSparse::inverted_index);

  return m.ptr();
}