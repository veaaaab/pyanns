#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
namespace pyanns {

template <typename QuantType> struct Evaluator {
  const QuantType &quant;

  explicit Evaluator(const QuantType &quant) : quant(quant) {}

  double evaluate(const float *data, int32_t n) {
    std::vector<typename QuantType::data_type> code(quant.code_size());
    std::vector<float> fvec(quant.dim());
    double err = 0.0;
    for (int32_t i = 0; i < n; ++i) {
      const float *cur_vec = data + (int64_t)i * quant.dim();
      quant.encode(cur_vec, code.data());
      quant.decode(code.data(), fvec.data());
      for (int32_t j = 0; j < quant.dim(); ++j) {
        fvec[j] -= cur_vec[j];
      }
      double d0 = 0.0, d1 = 0.0;
      for (int32_t j = 0; j < quant.dim(); ++j) {
        d0 += cur_vec[j] * cur_vec[j];
        d1 += fvec[j] * fvec[j];
      }
      err += std::sqrt(d1) / std::sqrt(d0 + 1e-9);
    }
    err /= n;
    return err;
  }
};

} // namespace pyanns