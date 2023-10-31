#pragma once

#include "pyanns/neighbor.hpp"
#include "pyanns/quant/quant_base.hpp"
#include "pyanns/searcher/searcher_base.hpp"
#include "pyanns/utils.hpp"
#include <memory>
#include <vector>

namespace pyanns {

template <QuantConcept QuantType> struct Refiner : GraphSearcherBase {
  int32_t dim;
  std::unique_ptr<GraphSearcherBase> inner;
  QuantType quant;

  int32_t reorder_mul = 0;

  Refiner(std::unique_ptr<GraphSearcherBase> inner, int32_t reorder_mul = 0)
      : inner(std::move(inner)), reorder_mul(reorder_mul) {}

  void SetData(const float *data, int32_t n, int32_t dim) override {
    this->dim = dim;
    quant = QuantType(dim);
    quant.train(data, n);
    quant.add(data, n);
    inner->SetData(data, n, dim);
  }

  void SetEf(int32_t ef) override { inner->SetEf(ef); }

  void Optimize(int32_t num_threads = 0) override {
    inner->Optimize(num_threads);
  }

  void Search(const float *q, int32_t k, int32_t *ids,
              float *dis = nullptr) const override {
    int32_t reorder_k;
    if (reorder_mul == 0) {
      reorder_k = std::max(k, inner->GetEf());
    } else {
      reorder_k = std::max(k, std::min(k * reorder_mul, inner->GetEf()));
    }
    if (reorder_k == k) {
      inner->Search(q, k, ids, dis);
      return;
    }
    std::vector<int32_t> ret(reorder_k);
    inner->Search(q, reorder_k, ret.data());
    auto computer = quant.get_computer(q);
    inference::MaxHeap<typename decltype(computer)::dist_type> heap(k);
    for (int i = 0; i < reorder_k; ++i) {
      if (i + 1 < reorder_k) {
        computer.prefetch(ret[i + 1], 1);
      }
      int id = ret[i];
      float dist = computer(id);
      heap.push(id, dist);
    }
    for (int i = 0; i < k; ++i) {
      ids[i] = heap.pop();
    }
  }

  void SearchBatch(const float *q, int32_t nq, int32_t k, int32_t *ids,
                   float *dis = nullptr) const override {
    int32_t reorder_k;
    if (reorder_mul == 0) {
      reorder_k = std::max(k, inner->GetEf());
    } else {
      reorder_k = std::max(k, std::min(k * reorder_mul, inner->GetEf()));
    }
    if (reorder_k == k) {
      inner->SearchBatch(q, nq, k, ids);
      return;
    }
    std::vector<int32_t> ret(nq * reorder_k);
    inner->SearchBatch(q, nq, reorder_k, ret.data());
    {
#pragma omp parallel for schedule(dynamic)
      for (int32_t i = 0; i < nq; ++i) {
        const float *cur_q = q + i * dim;
        const int32_t *cur_ret = &ret[i * reorder_k];
        int32_t *cur_dst = ids + i * k;
        auto computer = quant.get_computer(cur_q);
        inference::MaxHeap<typename decltype(computer)::dist_type> heap(k);
        for (int32_t j = 0; j < reorder_k; ++j) {
          if (j + 1 < reorder_k) {
            computer.prefetch(cur_ret[j + 1], 1);
          }
          int id = cur_ret[j];
          float dist = computer(id);
          heap.push(id, dist);
        }
        for (int j = 0; j < k; ++j) {
          cur_dst[j] = heap.pop();
        }
      }
    }
  }
};

} // namespace pyanns
