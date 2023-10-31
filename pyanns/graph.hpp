#pragma once

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

#include "pyanns/hnsw/HNSWInitializer.hpp"
#include "pyanns/memory.hpp"
#include "pyanns/neighbor.hpp"
#include "pyanns/quant/computer.hpp"
#include "pyanns/simd/prefetch.hpp"

namespace pyanns {

constexpr int EMPTY_ID = -1;

template <typename node_t> struct Graph {
  int32_t N;
  int32_t K;

  node_t *data = nullptr;

  std::unique_ptr<HNSWInitializer> initializer = nullptr;

  std::vector<int> eps;

  Graph() = default;

  Graph(node_t *edges, int32_t N, int32_t K) : N(N), K(K), data(edges) {}

  Graph(int32_t N, int32_t K)
      : N(N), K(K),
        data((node_t *)align_alloc((size_t)N * K * sizeof(node_t))) {}

  Graph(const Graph &g) = delete;

  Graph(Graph &&g) { swap(*this, g); }

  Graph &operator=(const Graph &rhs) = delete;

  Graph &operator=(Graph &&rhs) {
    swap(*this, rhs);
    return *this;
  }

  friend void swap(Graph &lhs, Graph &rhs) {
    using std::swap;
    swap(lhs.N, rhs.N);
    swap(lhs.K, rhs.K);
    swap(lhs.data, rhs.data);
    swap(lhs.initializer, rhs.initializer);
    swap(lhs.eps, rhs.eps);
  }

  void init(int32_t N, int K) {
    data = (node_t *)align_alloc((size_t)N * K * sizeof(node_t));
    std::memset(data, -1, N * K * sizeof(node_t));
    this->K = K;
    this->N = N;
  }

  ~Graph() { free(data); }

  const int *edges(int32_t u) const { return data + (int64_t)K * u; }

  int *edges(int32_t u) { return data + (int64_t)K * u; }

  int32_t degree(int32_t u) const {
    int32_t d = 0;
    while (d < K && at(u, d) >= 0) {
      d++;
    }
    return d;
  }

  bool add_edge(int32_t u, int32_t v) {
    auto d = degree(u);
    if (d == K) {
      return false;
    }
    at(u, d) = v;
    return true;
  }

  node_t at(int32_t i, int32_t j) const { return data[(int64_t)i * K + j]; }

  node_t &at(int32_t i, int32_t j) { return data[(int64_t)i * K + j]; }

  void prefetch(int32_t u, int32_t lines) const {
    mem_prefetch((char *)edges(u), lines);
  }

  void initialize_search(inference::NeighborPoolConcept auto &pool,
                         const ComputerConcept auto &computer) const {
    if (initializer) {
      initializer->initialize(pool, computer);
    } else {
      for (auto ep : eps) {
        pool.insert(ep, computer(ep));
        pool.set_visited(ep);
      }
    }
  }

  void save(const std::string &filename) const {
    static_assert(std::is_same_v<node_t, int32_t>);
    std::ofstream writer(filename.c_str(), std::ios::binary);
    int nep = eps.size();
    writer.write((char *)&nep, 4);
    writer.write((char *)eps.data(), nep * 4);
    writer.write((char *)&N, 4);
    writer.write((char *)&K, 4);
    writer.write((char *)data, (int64_t)N * K * 4);
    if (initializer) {
      initializer->save(writer);
    }
    printf("Graph Saving done\n");
  }

  void load(const std::string &filename, const std::string &format) {
    if (format == "pyanns") {
      load(filename);
    } else if (format == "diskann") {
      load_diskann(filename);
    }
  }

  void load(const std::string &filename) {
    static_assert(std::is_same_v<node_t, int32_t>);
    free(data);
    std::ifstream reader(filename.c_str(), std::ios::binary);
    int nep;
    reader.read((char *)&nep, 4);
    eps.resize(nep);
    reader.read((char *)eps.data(), nep * 4);
    reader.read((char *)&N, 4);
    reader.read((char *)&K, 4);
    data = (node_t *)align_alloc((int64_t)N * K * 4);
    reader.read((char *)data, N * K * 4);
    if (reader.peek() != EOF) {
      initializer = std::make_unique<HNSWInitializer>(N);
      initializer->load(reader);
    }
    printf("Graph Loding done\n");
  }

  void load_diskann(const std::string &filename) {
    static_assert(std::is_same_v<node_t, int32_t>);
    free(data);
    std::ifstream reader(filename.c_str(), std::ios::binary);
    size_t size;
    reader.read((char *)&size, 8);
    reader.read((char *)&K, 4);
    eps.resize(1);

    reader.read((char *)&eps[0], 4);
    size_t x;
    reader.read((char *)&x, 8);
    N = 0;
    while (reader.tellg() < size) {
      N++;
      int32_t cur_k;
      reader.read((char *)&cur_k, 4);
      reader.seekg(cur_k * 4, reader.cur);
    }
    reader.seekg(24, reader.beg);
    data = (node_t *)align_alloc((int64_t)N * K * 4);
    memset(data, -1, (int64_t)N * K * 4);
    for (int i = 0; i < N; ++i) {
      int cur_k;
      reader.read((char *)&cur_k, 4);
      reader.read((char *)edges(i), 4 * cur_k);
    }
  }
};

} // namespace pyanns
