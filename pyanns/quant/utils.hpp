#pragma once

#include "pyanns/neighbor.hpp"
#include <cmath>
#include <queue>

namespace pyanns {

inline float limit_range(float x) {
  if (x < 0.0f) {
    x = 0.0f;
  }
  if (x > 1.0f) {
    x = 1.0f;
  }
  return x;
}

inline float limit_range_sym(float x) {
  if (x < -1.0f) {
    x = -1.0f;
  }
  if (x > 1.0f) {
    x = 1.0f;
  }
  return x;
}

inline std::pair<float, float> find_minmax(const float *data, int64_t nitems,
                                           float ratio = 0.0f) {
  size_t top = int64_t(nitems * ratio) + 1;
  std::priority_queue<float> mx_heap;
  std::priority_queue<float, std::vector<float>, std::greater<float>> mi_heap;
  for (int64_t i = 0; i < nitems; ++i) {
    if (mx_heap.size() < top) {
      mx_heap.push(data[i]);
    } else if (data[i] < mx_heap.top()) {
      mx_heap.pop();
      mx_heap.push(data[i]);
    }
    if (mi_heap.size() < top) {
      mi_heap.push(data[i]);
    } else if (data[i] > mi_heap.top()) {
      mi_heap.pop();
      mi_heap.push(data[i]);
    }
  }
  return {mx_heap.top(), mi_heap.top()};
}

inline float find_absmax_without_drop(const float *data, int64_t nitems) {
  float ans = 0.0f;
  for (int i = 0; i < nitems; ++i) {
    ans = std::max(ans, std::abs(data[i]));
  }
  return ans;
}

inline float find_absmax(const float *data, int64_t nitems,
                         float ratio = 0.0f) {
  size_t top = int64_t(nitems * ratio) + 1;
  std::priority_queue<float, std::vector<float>, std::greater<float>> heap;
  for (int64_t i = 0; i < nitems; ++i) {
    float x = std::abs(data[i]);
    if (heap.size() < top) {
      heap.push(x);
    } else if (x > heap.top()) {
      heap.pop();
      heap.push(x);
    }
  }
  return heap.top();
}

inline void find_minmax_perdim(std::vector<float> &mins,
                               std::vector<float> &maxs, const float *data,
                               int32_t n, int32_t d, float ratio = 0.0f) {
  int64_t top = (int64_t)n * ratio + 1;
  std::vector<std::priority_queue<float>> mx_heaps(d);
  std::vector<
      std::priority_queue<float, std::vector<float>, std::greater<float>>>
      mi_heaps(d);
  for (int64_t i = 0; i < (int64_t)n * d; ++i) {
    auto &mx_heap = mx_heaps[i / n];
    auto &mi_heap = mi_heaps[i / n];
    if ((int64_t)mx_heap.size() < top) {
      mx_heap.push(data[i]);
    } else if (data[i] < mx_heap.top()) {
      mx_heap.pop();
      mx_heap.push(data[i]);
    }
    if ((int64_t)mi_heap.size() < top) {
      mi_heap.push(data[i]);
    } else if (data[i] > mi_heap.top()) {
      mi_heap.pop();
      mi_heap.push(data[i]);
    }
  }
  mins.resize(d);
  maxs.resize(d);
  for (int32_t i = 0; i < d; ++i) {
    mins[i] = mx_heaps[i].top();
    maxs[i] = mi_heaps[i].top();
  }
}

} // namespace pyanns