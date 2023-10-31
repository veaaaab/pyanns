#pragma once

#include "pyanns/builder.hpp"
#include "pyanns/common.hpp"
#include "pyanns/graph.hpp"
#include "pyanns/hnsw/HNSWInitializer.hpp"
#include "pyanns/hnswlib/hnswalg.h"
#include "pyanns/memory.hpp"
#include "pyanns/quant/quant.hpp"
#include "pyanns/quant/quant_base.hpp"
#include <chrono>
#include <memory>

namespace pyanns {

template <SymComputableQuantConcept QuantType> struct HNSW : public Builder {
  int64_t nb;
  int32_t R, efConstruction;
  std::unique_ptr<HierarchicalNSW<QuantType>> hnsw = nullptr;

  QuantType quant;

  Graph<int32_t> final_graph;

  HNSW(int dim, int32_t R = 32, int32_t L = 200)
      : R(R), efConstruction(L), quant(dim) {}

  void Build(float *data, int32_t N) override {
    nb = N;
    quant.train(data, N);
    quant.add(data, N);
    hnsw = std::make_unique<HierarchicalNSW<QuantType>>(quant, N, R / 2,
                                                        efConstruction);
    std::atomic<int32_t> cnt{0};
    auto st = std::chrono::high_resolution_clock::now();
    hnsw->addPoint(0);
#pragma omp parallel for schedule(dynamic)
    for (int32_t i = 1; i < nb; ++i) {
      hnsw->addPoint(i);
      int32_t cur = cnt += 1;
      if (cur % 10000 == 0) {
        printf("HNSW building progress: [%d/%ld]\n", cur, nb);
      }
    }
    auto ed = std::chrono::high_resolution_clock::now();
    auto ela = std::chrono::duration<double>(ed - st).count();
    printf("HNSW building cost: %.2lfs\n", ela);
    final_graph.init(nb, R);
#pragma omp parallel for
    for (int64_t i = 0; i < nb; ++i) {
      int32_t *edges = (int32_t *)hnsw->get_linklist0(i);
      for (int j = 1; j <= edges[0]; ++j) {
        final_graph.at(i, j - 1) = edges[j];
      }
    }
    auto initializer = std::make_unique<HNSWInitializer>(nb, R / 2);
    initializer->ep = hnsw->enterpoint_node_;
    for (int64_t i = 0; i < nb; ++i) {
      int32_t level = hnsw->element_levels_[i];
      initializer->levels[i] = level;
      if (level > 0) {
        initializer->lists[i] = (int *)align_alloc(level * R * 2, -1);
        for (int32_t j = 1; j <= level; ++j) {
          int32_t *edges = (int32_t *)hnsw->get_linklist(i, j);
          for (int32_t k = 1; k <= edges[0]; ++k) {
            initializer->at(j, i, k - 1) = edges[k];
          }
        }
      }
    }
    final_graph.initializer = std::move(initializer);
  }

  Graph<int32_t> GetGraph() override { return std::move(final_graph); }
};

inline std::unique_ptr<Builder>
create_hnsw(const std::string &metric, const std::string &quantizer = "BF16",
            int32_t dim = 0, int32_t R = 32, int32_t L = 200) {
  auto m = metric_map[metric];
  auto qua = quantizer_map[quantizer];
  if (qua == QuantizerType::FP32) {
    if (m == Metric::L2) {
      return std::make_unique<HNSW<FP32Quantizer<Metric::L2>>>(dim, R, L);
    }
    if (m == Metric::IP) {
      return std::make_unique<HNSW<FP32Quantizer<Metric::IP>>>(dim, R, L);
    }
  }
  if (qua == QuantizerType::FP16) {
    if (m == Metric::L2) {
      return std::make_unique<HNSW<FP16Quantizer<Metric::L2>>>(dim, R, L);
    }
    if (m == Metric::IP) {
      return std::make_unique<HNSW<FP16Quantizer<Metric::IP>>>(dim, R, L);
    }
  }

  printf("Quantizer type %s not supported\n", quantizer.c_str());
  return nullptr;
}

} // namespace pyanns
