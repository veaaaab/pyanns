#pragma once

#include "pyanns/quant/calibrator.hpp"
#include "pyanns/quant/computer.hpp"
#include "pyanns/quant/quant_base.hpp"
#include "pyanns/quant/utils.hpp"
#include "pyanns/simd/prefetch.hpp"

#include <cmath>
#include <fcntl.h>
#include <sys/stat.h>
#include <thread>

namespace pyanns {

struct SparseQuant {
  using type = SparseQuant;

  using dtype = fp16;

  int64_t n = 0, m = 0;
  int64_t *indptr = nullptr;
  int32_t *indices = nullptr;
  dtype *data = nullptr;

  SparseQuant() = default;

  ~SparseQuant() {
    free(this->indptr);
    free(this->indices);
    free(this->data);
  }

  void swap(SparseQuant &lhs, SparseQuant &rhs) {
    using std::swap;
    swap(lhs.n, rhs.n);
    swap(lhs.m, rhs.m);
    swap(lhs.indptr, rhs.indptr);
    swap(lhs.indices, rhs.indices);
    swap(lhs.data, rhs.data);
  }

  SparseQuant(const SparseQuant &rhs) = delete;
  SparseQuant(SparseQuant &&rhs) { swap(rhs, *this); }
  SparseQuant &operator=(const SparseQuant &rhs) = delete;
  SparseQuant &operator=(SparseQuant &&rhs) {
    SparseQuant tmp(std::move(rhs));
    swap(*this, tmp);
    return *this;
  }

  void add(const std::string &filename) {
    auto fd = open(filename.c_str(), O_RDONLY);
    struct stat sb;
    fstat(fd, &sb);
    char *ptr = (char *)mmap(nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    char *cur = ptr;
    n = *(int64_t *)cur;
    cur += 8;
    m = *(int64_t *)cur;
    cur += 16;
    this->indptr = (int64_t *)align_alloc((n + 1) * sizeof(int64_t));
    memcpy(this->indptr, cur, (n + 1) * sizeof(int64_t));
    cur += (n + 1) * sizeof(int64_t);
    this->indices = (int32_t *)align_alloc(indptr[n] * sizeof(int32_t));
    memcpy(this->indices, cur, indptr[n] * sizeof(int32_t));
    cur += indptr[n] * sizeof(int32_t);
    this->data = (dtype *)align_alloc(indptr[n] * sizeof(dtype));
    const float *fcur = (const float *)cur;
    for (int i = 0; i < indptr[n]; ++i) {
      this->data[i] = dtype(fcur[i]);
    }
  }

  struct Computer {
    using dist_type = float;
    const type &quant;
    int32_t nnz = 0;
    const int32_t *ids = nullptr;
    const float *vals = nullptr;
    Computer(const type &quant) : quant(quant) {}
    Computer(const type &quant, int32_t nnz, const int32_t *ids,
             const float *vals)
        : quant(quant), nnz(nnz), ids(ids), vals(vals) {}
    void prefetch(int32_t u, int32_t lines) const {
      mem_prefetch((const char *)(quant.indices + quant.indptr[u]), 1);
      mem_prefetch((const char *)(quant.data + quant.indptr[u]), 1);
    }

    dist_type operator()(int32_t u) const {
      float sum = 0.0f;
      for (int i = quant.indptr[u], j = 0; j < nnz; ++j) {
        while (i < quant.indptr[u + 1] && quant.indices[i] < ids[j]) {
          i++;
        }
        if (i >= quant.indptr[u + 1]) {
          break;
        }
        if (quant.indices[i] == ids[j]) {
          sum += quant.data[i] * vals[j];
        }
      }
      return -sum;
    }

    dist_type operator()(int32_t u, int32_t v) const {
      float sum = 0.0f;
      for (int i = quant.indptr[u], j = quant.indptr[v];
           j < quant.indptr[v + 1]; ++j) {
        while (i < quant.indptr[u + 1] && quant.indices[i] < quant.indices[j]) {
          i++;
        }
        if (i >= quant.indptr[u + 1]) {
          break;
        }
        if (quant.indices[i] == quant.indices[j]) {
          sum += (float)quant.data[i] * (float)quant.data[j];
        }
      }
      return -sum;
    }
  };

  using ComputerType = Computer;
  using SymComputerType = Computer;

  auto get_computer(int32_t nnz, const int32_t *ids, const float *vals) const {
    return Computer(*this, nnz, ids, vals);
  }

  auto get_sym_computer() const { return Computer(*this); }
};

} // namespace pyanns
