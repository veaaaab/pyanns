#pragma once

#include "pyanns/storage/tensor.hpp"

#undef FINTEGER
#define FINTEGER long

extern "C" {

int sgemm_(const char *transa, const char *transb, FINTEGER *m, FINTEGER *n,
           FINTEGER *k, const float *alpha, const float *a, FINTEGER *lda,
           const float *b, FINTEGER *ldb, float *beta, float *c, FINTEGER *ldc);
}

namespace pyanns::linalg {

// X @ Y = Z
template <typename Tensor> Tensor matmul(const Tensor &X, const Tensor &Y) {
  FINTEGER m = X.size(), k = X.dim(), n = Y.dim();
  Tensor Z(m, n);
  float one = 1.0f, zero = 0.0f;
  sgemm_("Not transposed", "Not transposed", &m, &n, &k, &one,
         (const float *)X.get(0), &m, (const float *)Y.get(0), &k, &zero,
         (float *)Z.get(0), &m);
  return Z;
}

} // namespace pyanns::linalg