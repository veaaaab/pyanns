#pragma once

#include <cstdint>

#include "helpa/types.hpp"

namespace helpa {

inline float l2_fp32_fp32(const float *x, const float *y, const int32_t d);
inline float l2_fp32_fp16(const float *x, const fp16 *y, const int32_t d);
inline float l2_fp16_fp16(const fp16 *x, const fp16 *y, const int32_t d);
inline float l2_fp32_bf16(const float *x, const bf16 *y, const int32_t d);
inline float l2_bf16_bf16(const bf16 *x, const bf16 *y, const int32_t d);
inline int32_t l2_u8_s8(const uint8_t *x, const int8_t *y, const int32_t d);
inline int32_t l2_s8_s8(const int8_t *x, const int8_t *y, const int32_t d);
inline float l2a_fp32_fp32(const float *x, const float *y, const int32_t d);
inline float l2a_fp32_fp16(const float *x, const fp16 *y, const int32_t d);
inline float l2a_fp16_fp16(const fp16 *x, const fp16 *y, const int32_t d);
inline float l2a_fp32_bf16(const float *x, const bf16 *y, const int32_t d);
inline float l2a_bf16_bf16(const bf16 *x, const bf16 *y, const int32_t d);
inline int32_t l2a_u8_s8(const uint8_t *x, const int8_t *y, const int32_t d);
inline int32_t l2a_s8_s8(const int8_t *x, const int8_t *y, const int32_t d);
inline int32_t l2a_u4_u4(const uint8_t *x, const uint8_t *y, const int32_t d);

} // namespace helpa

#if defined(__AVX2__)
#include "helpa/platform/x86/l2_impl.hpp"
#elif defined(__aarch64__)
#include "helpa/platform/neon/l2_impl.hpp"
#else
#include "helpa/platform/scalar/l2_impl.hpp"
#endif
