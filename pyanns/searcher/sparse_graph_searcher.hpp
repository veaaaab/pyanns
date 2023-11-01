
#pragma once

#include "pyanns/common.hpp"
#include "pyanns/graph.hpp"
#include "pyanns/hnsw/hnsw.hpp"
#include "pyanns/neighbor.hpp"
#include "pyanns/quant/product_quant.hpp"
#include "pyanns/quant/quant.hpp"
#include "pyanns/quant/quant_base.hpp"
#include "pyanns/quant/sparse_quant.hpp"
#include "pyanns/searcher/refiner.hpp"
#include "pyanns/searcher/searcher_base.hpp"
#include "pyanns/utils.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <omp.h>
#include <random>

namespace pyanns {

struct SparseGraphSearcher {

  int32_t nb;
  Graph<int32_t> graph;
  SparseQuant quant;

  // Search parameters
  int32_t ef = 32;

  SparseGraphSearcher(const std::string &filename,
                      const std::string &graphfile) {
    if (!std::filesystem::exists(graphfile)) {
      SparseHNSW hnsw(32, 500);
      hnsw.Build(filename);
      graph = hnsw.GetGraph();
      graph.save(graphfile);
    }
    graph.load(graphfile);
    quant.add(filename);
    nb = quant.n;
  }

  void SetEf(int32_t ef) { this->ef = ef; }

  int32_t GetEf() const { return ef; }

  void Search(int32_t nnz, const int32_t *ids, const float *vals, int32_t k,
              int32_t *res, float *dis = nullptr) const {
    auto computer = quant.get_computer(nnz, ids, vals);
    inference::LinearPool<float> pool(nb, std::max(k, ef), k);
    graph.initialize_search(pool, computer);
    SearchImpl(pool, computer);
    for (int32_t i = 0; i < k; ++i) {
      res[i] = pool.id(i);
      if (dis != nullptr) {
        dis[i] = pool.dist(i);
      }
    }
  }

  void SearchBatch(int32_t nq, const int32_t *indptr, const int32_t *indices,
                   const float *data, int32_t topk, int32_t *res,
                   float budget) const {
    float max = 0.0f;
    for (int i = 0; i < indptr[nq]; ++i) {
      max = std::max(max, data[i]);
    }

    std::vector<int32_t> nnz(nq);
    std::vector<std::vector<int>> ids(nq);
    std::vector<std::vector<float>> vals(nq);

    for (int i = 0; i < nq; ++i) {
      for (int j = indptr[i]; j < indptr[i + 1]; ++j) {
        if (data[j] > max * budget) {
          nnz[i]++;
          ids[i].push_back(indices[j]);
          vals[i].push_back(data[j]);
        }
      }
    }
    int32_t refine_k = ef;
    std::vector<int> refine_ids(refine_k * nq);
    {
#pragma omp parallel for schedule(dynamic)
      for (int i = 0; i < nq; ++i) {
        Search(nnz[i], ids[i].data(), vals[i].data(), refine_k,
               refine_ids.data() + i * refine_k, nullptr);
      }
    }
    {
#pragma omp parallel for schedule(dynamic)
      for (int i = 0; i < nq; ++i) {
        auto computer = quant.get_computer(
            indptr[i + 1] - indptr[i], indices + indptr[i], data + indptr[i]);
        inference::MaxHeap<float> heap(topk);
        for (int j = 0; j < refine_k; ++j) {
          int32_t u = refine_ids[i * refine_k + j];
          auto dist = computer(u);
          heap.push(u, dist);
        }
        for (int j = 0; j < topk; ++j) {
          res[i * topk + j] = heap.pool[j].id;
        }
      }
    }
  }

  void SearchImpl(inference::NeighborPoolConcept auto &pool,
                  ComputerConcept auto &computer) const {
    alignas(64) int32_t edge_buf[graph.K];
    int po = 1, pl = 1;
    while (pool.has_next()) {
      auto u = pool.pop();
      graph.prefetch(u, 1);
      int32_t edge_size = 0;
      for (int32_t i = 0; i < graph.K; ++i) {
        int32_t v = graph.at(u, i);
        if (v == -1) {
          break;
        }
        if (pool.is_visited(v)) {
          continue;
        }
        pool.set_visited(v);
        edge_buf[edge_size++] = v;
      }
      for (int i = 0; i < std::min(po, edge_size); ++i) {
        computer.prefetch(edge_buf[i], pl);
      }
      for (int i = 0; i < edge_size; ++i) {
        if (i + po < edge_size) {
          computer.prefetch(edge_buf[i + po], pl);
        }
        auto v = edge_buf[i];
        auto cur_dist = computer(v);
        pool.insert(v, cur_dist);
      }
    }
  }
};

} // namespace pyanns
