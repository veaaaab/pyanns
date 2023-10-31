#pragma once

#include "pyanns/memory.hpp"
#include "pyanns/neighbor.hpp"
#include "helpa/types.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <immintrin.h>
#include <queue>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define BENCHMARK false

namespace pyanns {

constexpr size_t size_1G = 1024 * 1024 * 1024;
constexpr size_t size_2M = 2 * 1024 * 1024;

using bf16 = helpa::bf16;
using fp16 = helpa::fp16;
using dtype = float;

struct IndexSparse {
  int64_t n = 0, m = 0;
  std::vector<std::vector<int32_t>> inverted_index;
  std::vector<std::vector<dtype>> inverted_value;
  std::vector<float> U;
  dtype *data = nullptr;
  int32_t *indices = nullptr;
  int64_t *indptr = nullptr;

  IndexSparse() = default;

  ~IndexSparse() {
    free(data);
    free(indices);
    free(indptr);
  }

  void add(const std::string &filename, float drop_ratio) {
    auto fd = open(filename.c_str(), O_RDONLY);
    struct stat sb;
    fstat(fd, &sb);
    char *ptr = (char *)mmap(nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    char *cur = ptr;
    n = *(int64_t *)cur;
    cur += 8;
    m = *(int64_t *)cur;
    cur += 16;
    this->inverted_index.resize(m);
    this->inverted_value.resize(m);
    this->U.resize(m);
    this->indptr = (int64_t *)align_alloc((n + 1) * sizeof(int64_t));
    memcpy(this->indptr, cur, (n + 1) * sizeof(int64_t));
    cur += (n + 1) * sizeof(int64_t);
    this->indices = (int32_t *)align_alloc(indptr[n] * sizeof(int32_t));
    memcpy(this->indices, cur, indptr[n] * sizeof(int32_t));
    cur += indptr[n] * sizeof(int32_t);
    this->data = (dtype *)align_alloc(indptr[n] * sizeof(dtype));
    float *data_tmp = (float *)cur;
    float mx = 0.0f;
    for (int i = 0; i < indptr[n]; ++i) {
      this->data[i] = dtype(data_tmp[i]);
      mx = std::max(data_tmp[i], mx);
    }
    for (int32_t i = 0; i < n; ++i) {
      for (int32_t j = indptr[i]; j < indptr[i + 1]; ++j) {
        auto u = indices[j];
        auto v = data[j];
        if (float(v) / mx > drop_ratio) {
          inverted_index[u].push_back(i);
          inverted_value[u].push_back(v);
          U[u] = std::max(U[u], float(v));
        }
      }
    }
  }

  void search_batch(int32_t nq, const int32_t *indptr, const int32_t *indices,
                    const float *data, int32_t topk, int32_t *res, float budget,
                    int32_t refine_mul) {
    int32_t topk_refine = topk * refine_mul;
    std::vector<int32_t> res_refine(nq * topk_refine);

    search_wand_batch(nq, indptr, indices, data, topk_refine, res_refine.data(),
                      budget);

    refine_batch(nq, indptr, indices, data, topk_refine, res_refine.data(),
                 topk, res);
  }

  void refine_batch(int32_t nq, const int32_t *indptr, const int32_t *indices,
                    const float *data, int32_t refine_topk, int32_t *refine_ids,
                    int32_t topk, int32_t *res_ids) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nq; ++i) {
      int32_t nnz = indptr[i + 1] - indptr[i];
      const int32_t *cur_indices = indices + indptr[i];
      const float *cur_data = data + indptr[i];
      int32_t *cur_res = res_ids + i * topk;
      pyanns::inference::MaxHeap<float> heap(topk);

      auto prefetch = [&](int32_t u) {
        int32_t ind = this->indptr[u];
        _mm_prefetch(this->data + ind, _MM_HINT_T0);
        _mm_prefetch(this->indices + ind, _MM_HINT_T0);
      };
      constexpr static int32_t po = 1;
      for (int j = 0; j < std::min(refine_topk, po); ++j) {
        prefetch(refine_ids[i * refine_topk + j]);
      }
      for (int j = 0; j < refine_topk; ++j) {
        if (j + po < refine_topk) {
          prefetch(refine_ids[i * refine_topk + j + po]);
        }
        auto u = refine_ids[i * refine_topk + j];
        auto dist_acc = dot_bf(nnz, cur_indices, cur_data, u);
        heap.push(u, -dist_acc);
      }
      for (int i = 0; i < heap.size(); ++i) {
        cur_res[i] = heap.pool[i].id;
      }
    }
  }

  void search_wand_batch(int32_t nq, const int32_t *indptr,
                         const int32_t *indices, const float *data,
                         int32_t topk, int32_t *res, float budget) {

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nq; ++i) {
      search_wand(indptr[i + 1] - indptr[i], indices + indptr[i],
                  data + indptr[i], topk, res + i * topk, budget);
    }
  }

  struct Cursor {
    std::vector<int32_t> *idx = nullptr;
    std::vector<dtype> *val = nullptr;
    int32_t loc = 0;
    int32_t num_doc = 0;
    float max_score = 0.0f;
    float qval = 0.0f;

    Cursor() = default;
    Cursor(std::vector<int32_t> *idx, std::vector<dtype> *val, int32_t num_doc,
           float max_score, float qval)
        : idx(idx), val(val), num_doc(num_doc), max_score(max_score),
          qval(qval) {}
    Cursor(const Cursor &rhs) = delete;
    Cursor(Cursor &&rhs) { swap(rhs, *this); }
    Cursor &operator=(const Cursor &rhs) = delete;
    Cursor &operator=(Cursor &&rhs) {
      Cursor tmp(std::move(rhs));
      swap(*this, tmp);
      return *this;
    }

    friend void swap(Cursor &lhs, Cursor &rhs) {
      using std::swap;
      swap(lhs.idx, rhs.idx);
      swap(lhs.val, rhs.val);
      swap(lhs.loc, rhs.loc);
      swap(lhs.num_doc, rhs.num_doc);
      swap(lhs.max_score, rhs.max_score);
      swap(lhs.qval, rhs.qval);
    }

    void next() { loc++; }
    void seek(int32_t d) {
      while (loc < idx->size() && cur_idx() < d) {
        loc++;
      }
    }
    int32_t cur_idx() {
      if (is_end()) {
        return num_doc;
      }
      return (*idx)[loc];
    }
    dtype cur_val() { return (*val)[loc]; }

    bool is_end() { return loc >= size(); }

    float qvalue() { return qval; }
    int32_t size() { return idx->size(); }
  };

  void search_wand(int32_t nnz, const int32_t *ids, const float *vals,
                   int32_t topk, int32_t *res, float budget) {
    std::vector<std::pair<int32_t, float>> pairs;
    pairs.reserve(nnz);
    for (int i = 0; i < nnz; ++i) {
      pairs.emplace_back(ids[i], vals[i]);
    }
    std::sort(pairs.begin(), pairs.end(), [&](const auto &p1, const auto &p2) {
      return std::abs(p1.second) > std::abs(p2.second);
    });
    while (pairs.size() > 5 ||
           (pairs.size() && pairs[0].second * budget > pairs.back().second)) {
      pairs.pop_back();
    }
    int32_t D = pairs.size();
    std::vector<Cursor> cursors_(D);
    for (int i = 0; i < D; ++i) {
      cursors_[i] = Cursor(
          &inverted_index[pairs[i].first], &inverted_value[pairs[i].first], n,
          U[pairs[i].first] * pairs[i].second, pairs[i].second);
    }
    std::vector<Cursor *> cursors(D);
    for (int i = 0; i < D; ++i) {
      cursors[i] = &cursors_[i];
    }
    pyanns::inference::MaxHeap<float> heap(topk);
    auto sort_cursors = [&] {
      std::sort(cursors.begin(), cursors.end(),
                [](auto &x, auto &y) { return x->cur_idx() < y->cur_idx(); });
    };
    sort_cursors();

    auto would_enter = [&](float x) {
      return heap.size() < heap.capacity || x > -heap.top_dist();
    };
    while (true) {
      float upper_bound = 0;
      size_t pivot;
      bool found_pivot = false;
      for (pivot = 0; pivot < cursors.size(); ++pivot) {
        if (cursors[pivot]->is_end()) {
          break;
        }
        upper_bound += cursors[pivot]->max_score;
        if (would_enter(upper_bound)) {
          found_pivot = true;
          break;
        }
      }
      if (!found_pivot) {
        break;
      }
      int32_t pivot_id = cursors[pivot]->cur_idx();
      if (pivot_id == cursors[0]->cur_idx()) {
        float score = 0;
        for (auto &cursor : cursors) {
          if (cursor->cur_idx() != pivot_id) {
            break;
          }
          score += cursor->cur_val() * cursor->qvalue();
          cursor->next();
        }
        heap.push(pivot_id, -score);
        sort_cursors();
      } else {
        uint64_t next_list = pivot;
        for (; cursors[next_list]->cur_idx() == pivot_id; --next_list) {
        }
        cursors[next_list]->seek(pivot_id);
        for (size_t i = next_list + 1; i < cursors.size(); ++i) {
          if (cursors[i]->cur_idx() < cursors[i - 1]->cur_idx()) {
            std::swap(cursors[i], cursors[i - 1]);
          } else {
            break;
          }
        }
      }
    }

    for (int i = 0; i < topk; ++i) {
      res[i] = heap.pool[i].id;
    }
  }

  float dot_bf(int32_t nnz, const int32_t *ids, const float *vals, int32_t u) {
    float sum = 0.0f;
    for (int i = indptr[u], j = 0; j < nnz; ++j) {
      while (i < indptr[u + 1] && indices[i] < ids[j]) {
        i++;
      }
      if (i >= indptr[u + 1]) {
        break;
      }
      if (indices[i] == ids[j]) {
        sum += data[i] * vals[j];
      }
    }
    return sum;
  }
};

} // namespace pyanns
