#pragma once

#if defined(__AVX2__)

#include <immintrin.h>

#include "pyanns/simd/avx2_utils.hpp"
#include "pyanns/types.hpp"

namespace pyanns {

inline float L2SqrE5M2(const float *x, const e5m2 *y, int d) {
  __m256 sum = _mm256_setzero_ps();
  for (int i = 0; i < d; i += 16) {
    auto xx0 = _mm256_loadu_ps(x + i);
    auto xx1 = _mm256_loadu_ps(x + i + 8);
    auto zz = _mm_loadu_si128((__m128i *)(y + i));
    auto zzz = _mm256_slli_epi16(_mm256_cvtepi8_epi16(zz), 8);
    auto zzz0 = _mm256_extracti128_si256(zzz, 0);
    auto zzz1 = _mm256_extracti128_si256(zzz, 1);
    auto zzzz0 = _mm256_cvtph_ps(zzz0);
    auto zzzz1 = _mm256_cvtph_ps(zzz1);
    auto t0 = _mm256_sub_ps(xx0, zzzz0);
    auto t1 = _mm256_sub_ps(xx1, zzzz1);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(t0, t0));
    sum = _mm256_add_ps(sum, _mm256_mul_ps(t1, t1));
  }
  return reduce_add_f32x8(sum);
}

inline float IPE5M2(const float *x, const e5m2 *y, int d) {
  __m256 sum = _mm256_setzero_ps();
  for (int i = 0; i < d; i += 16) {
    auto xx0 = _mm256_loadu_ps(x + i);
    auto xx1 = _mm256_loadu_ps(x + i + 8);
    auto zz = _mm_loadu_si128((__m128i *)(y + i));
    auto zzz = _mm256_slli_epi16(_mm256_cvtepi8_epi16(zz), 8);
    auto zzz0 = _mm256_extracti128_si256(zzz, 0);
    auto zzz1 = _mm256_extracti128_si256(zzz, 1);
    auto zzzz0 = _mm256_cvtph_ps(zzz0);
    auto zzzz1 = _mm256_cvtph_ps(zzz1);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(xx0, zzzz0));
    sum = _mm256_add_ps(sum, _mm256_mul_ps(xx1, zzzz1));
  }
  return -reduce_add_f32x8(sum);
}

FAST_BEGIN
inline float L2SqrSQ8_ext(const float *x, const uint8_t *y, int d,
                          const float *mi, const float *dif) {
  float sum = 0.0;
  for (int i = 0; i < d; ++i) {
    float yy = (y[i] + 0.5f);
    yy = yy * dif[i] + mi[i] * 256.0f;
    auto dif = x[i] * 256.0f - yy;
    sum += dif * dif;
  }
  return sum;
}
FAST_END

FAST_BEGIN
inline float IPSQ8_ext(const float *x, const uint8_t *y, int d, const float *mi,
                       const float *dif) {
  float sum = 0.0;
  for (int i = 0; i < d; ++i) {
    float yy = y[i] + 0.5f;
    yy = yy * dif[i] + mi[i] * 256.0f;
    sum += x[i] * yy;
  }
  return -sum;
}
FAST_END

inline float L2SqrSQ6_ext(const float *x, const uint8_t *y, int d,
                          const float *mi, const float *dif) {
  auto sum = _mm256_setzero_ps();
  for (int i = 0; i < d; i += 8) {
    auto xx = _mm256_loadu_ps(x + i);
    auto yy = SQ6::decode_8_components(y, i);
    auto mi_256 = _mm256_load_ps(mi + i);
    auto dif_256 = _mm256_load_ps(dif + i);
    yy = _mm256_add_ps(_mm256_mul_ps(yy, dif_256), mi_256);
    auto t = _mm256_sub_ps(xx, yy);
    sum = _mm256_fmadd_ps(t, t, sum);
  }
  return reduce_add_f32x8(sum);
}

inline float IPSQ6_ext(const float *x, const uint8_t *y, int d, const float *mi,
                       const float *dif) {
  auto sum = _mm256_setzero_ps();
  for (int i = 0; i < d; i += 8) {
    auto xx = _mm256_loadu_ps(x + i);
    auto yy = SQ6::decode_8_components(y, i);
    auto mi_256 = _mm256_load_ps(mi + i);
    auto dif_256 = _mm256_load_ps(dif + i);
    yy = _mm256_add_ps(_mm256_mul_ps(yy, dif_256), mi_256);
    sum = _mm256_fmadd_ps(xx, yy, sum);
  }
  return -reduce_add_f32x8(sum);
}

inline float L2SqrSQ4_ext(const float *x, const uint8_t *y, int d,
                          const float *mi, const float *dif) {
  float sum = 0.0f;
  for (int i = 0; i < d; i += 2) {
    {
      float yy = (y[i / 2] & 15) + 0.5f;
      yy = yy * dif[i] + mi[i] * 16.0f;
      auto dif = x[i] * 16.0f - yy;
      sum += dif * dif;
    }
    {
      float yy = (y[i / 2] >> 4 & 15) + 0.5f;
      yy = yy * dif[i + 1] + mi[i + 1] * 16.0f;
      auto dif = x[i + 1] * 16.0f - yy;
      sum += dif * dif;
    }
  }
  return sum;
}

inline float IPSQ4_ext(const float *x, const uint8_t *y, int d, const float *mi,
                       const float *dif) {
  float sum = 0.0f;
  for (int i = 0; i < d; i += 2) {
    {
      float yy = ((y[i / 2] & 15) + 0.5f) / 16.0f;
      yy = yy * dif[i] + mi[i];
      sum += x[i] * yy;
    }
    {
      float yy = ((y[i / 2] >> 4 & 15) + 0.5f) / 16.0f;
      yy = yy * dif[i + 1] + mi[i + 1];
      sum += x[i + 1] * yy;
    }
  }
  return -sum;
}

inline int32_t L2SqrSQ8SQ4(const uint8_t *x, const uint8_t *y, int d) {
  auto sum = _mm256_setzero_si256();
  for (int i = 0; i < d; i += 32) {
    auto xx = _mm256_loadu_si256((__m256i *)(x + i));
    auto yy = _mm_loadu_si128((__m128i *)(y + i / 2));
    auto yyy = cvti4x32_i8x32(yy);
    yyy = _mm256_add_epi8(yyy, _mm256_set1_epi8(4));
    auto d = _mm256_sub_epi8(xx, yyy);
    d = _mm256_abs_epi8(d);
    auto tmp = _mm256_maddubs_epi16(d, d);
    sum = _mm256_add_epi32(
        sum, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(tmp, 0)));
    sum = _mm256_add_epi32(
        sum, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(tmp, 1)));
  }
  return reduce_add_i32x8(sum);
}

inline int32_t L2SqrSQ2(const uint8_t *x, const uint8_t *y, int d) {
  const uint8_t *end = x + d / 4;
  __m256i sum = _mm256_setzero_si256();
  __m256i mask = _mm256_set1_epi8(0x3);
  while (x < end) {
    auto xx = _mm256_loadu_si256((__m256i *)x);
    x += 32;
    auto yy = _mm256_loadu_si256((__m256i *)y);
    y += 32;
    auto xx1 = _mm256_and_si256(xx, mask);
    auto xx2 = _mm256_and_si256(_mm256_srli_epi16(xx, 2), mask);
    auto xx3 = _mm256_and_si256(_mm256_srli_epi16(xx, 4), mask);
    auto xx4 = _mm256_and_si256(_mm256_srli_epi16(xx, 6), mask);
    auto yy1 = _mm256_and_si256(yy, mask);
    auto yy2 = _mm256_and_si256(_mm256_srli_epi16(yy, 2), mask);
    auto yy3 = _mm256_and_si256(_mm256_srli_epi16(yy, 4), mask);
    auto yy4 = _mm256_and_si256(_mm256_srli_epi16(yy, 6), mask);
    auto d1 = _mm256_sub_epi8(xx1, yy1);
    auto d2 = _mm256_sub_epi8(xx2, yy2);
    auto d3 = _mm256_sub_epi8(xx3, yy3);
    auto d4 = _mm256_sub_epi8(xx4, yy4);
    d1 = _mm256_abs_epi8(d1);
    d2 = _mm256_abs_epi8(d2);
    d3 = _mm256_abs_epi8(d3);
    d4 = _mm256_abs_epi8(d4);
    auto m1 = _mm256_maddubs_epi16(d1, d1);
    auto m2 = _mm256_maddubs_epi16(d2, d2);
    auto m3 = _mm256_maddubs_epi16(d3, d3);
    auto m4 = _mm256_maddubs_epi16(d4, d4);
    m1 = _mm256_add_epi16(m1, m2);
    m1 = _mm256_add_epi16(m1, m3);
    m1 = _mm256_add_epi16(m1, m4);
    sum = _mm256_add_epi16(sum, m1);
  }
  return reduce_add_i16x16(sum);
}

inline int32_t L2SqrSQ1(const uint8_t *x, const uint8_t *y, int d) {
  auto xx = (const uint64_t *)x;
  auto yy = (const uint64_t *)y;
  const uint64_t *end = xx + d / 64;
  int32_t sum = 0;
  while (xx < end) {
    sum += __builtin_popcountll(*xx ^ *yy);
    xx += 1;
    yy += 1;
  }
  return sum;
}

} // namespace pyanns

#endif
