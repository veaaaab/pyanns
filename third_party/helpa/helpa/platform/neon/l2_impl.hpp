#pragma once

#include "helpa/platform/neon/neon_utils.hpp"
#if defined(__aarch64__)

#include <arm_neon.h>

#include "helpa/l2.hpp"
#include "helpa/ref/l2_ref.hpp"

#include "helpa/types.hpp"

namespace helpa {

inline float l2_fp32_fp32(const float *x, const float *y, const int32_t d) {
  int32_t da = d / 16 * 16;
  return l2a_fp32_fp32(x, y, da) + l2_fp32_fp32_ref(x + da, y + da, d - da);
}

inline float l2_fp32_fp16(const float *x, const fp16 *y, const int32_t d) {
  int32_t da = d / 16 * 16;
  return l2a_fp32_fp16(x, y, da) + l2_fp32_fp16_ref(x + da, y + da, d - da);
}

inline float l2_fp16_fp16(const fp16 *x, const fp16 *y, const int32_t d) {
  int32_t da = d / 16 * 16;
  return l2a_fp16_fp16(x, y, da) + l2_fp16_fp16_ref(x + da, y + da, d - da);
}

inline float l2_fp32_bf16(const float *x, const bf16 *y, const int32_t d) {
  int32_t da = d / 16 * 16;
  return l2a_fp32_bf16(x, y, da) + l2_fp32_bf16_ref(x + da, y + da, d - da);
}

inline float l2_bf16_bf16(const bf16 *x, const bf16 *y, const int32_t d) {
  int32_t da = d / 16 * 16;
  return l2a_bf16_bf16(x, y, da) + l2_bf16_bf16_ref(x + da, y + da, d - da);
}

inline int32_t l2_u8_s8(const uint8_t *x, const int8_t *y, const int32_t d) {
  int32_t da = d / 64 * 64;
  return l2a_u8_s8(x, y, da) + l2_u8_s8_ref(x + da, y + da, d - da);
}

inline int32_t l2_s8_s8(const int8_t *x, const int8_t *y, const int32_t d) {
  int32_t da = d / 64 * 64;
  return l2a_s8_s8(x, y, da) + l2_s8_s8_ref(x + da, y + da, d - da);
}

inline float l2a_fp32_fp32(const float *x, const float *y, const int32_t d) {
  float32x4x4_t sum = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                       vdupq_n_f32(0.0f)};
  for (int32_t i = 0; i < d; i += 16) {
    auto xx0 = vld1q_f32(x + i);
    auto xx1 = vld1q_f32(x + i + 4);
    auto xx2 = vld1q_f32(x + i + 8);
    auto xx3 = vld1q_f32(x + i + 12);
    auto yy0 = vld1q_f32(y + i);
    auto yy1 = vld1q_f32(y + i + 4);
    auto yy2 = vld1q_f32(y + i + 8);
    auto yy3 = vld1q_f32(y + i + 12);
    auto t0 = vsubq_f32(xx0, yy0);
    auto t1 = vsubq_f32(xx1, yy1);
    auto t2 = vsubq_f32(xx2, yy2);
    auto t3 = vsubq_f32(xx3, yy3);
    sum.val[0] = vmlaq_f32(sum.val[0], t0, t0);
    sum.val[1] = vmlaq_f32(sum.val[1], t1, t1);
    sum.val[2] = vmlaq_f32(sum.val[2], t2, t2);
    sum.val[3] = vmlaq_f32(sum.val[3], t3, t3);
  }
  return reduce_f32x4x4(sum);
}

inline float l2a_fp32_fp16(const float *x, const fp16 *y, const int32_t d) {
  float32x4x4_t sum = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                       vdupq_n_f32(0.0f)};
  for (int32_t i = 0; i < d; i += 16) {
    auto xx0 = vld1q_f32(x + i);
    auto xx1 = vld1q_f32(x + i + 4);
    auto xx2 = vld1q_f32(x + i + 8);
    auto xx3 = vld1q_f32(x + i + 12);
    auto zz0 = vld1q_f16((const __fp16 *)(y + i));
    auto zz1 = vld1q_f16((const __fp16 *)(y + i + 8));
    auto yy0 = vcvt_f32_f16(vget_low_f16(zz0));
    auto yy1 = vcvt_f32_f16(vget_high_f16(zz0));
    auto yy2 = vcvt_f32_f16(vget_low_f16(zz1));
    auto yy3 = vcvt_f32_f16(vget_high_f16(zz1));
    auto t0 = vsubq_f32(xx0, yy0);
    auto t1 = vsubq_f32(xx1, yy1);
    auto t2 = vsubq_f32(xx2, yy2);
    auto t3 = vsubq_f32(xx3, yy3);
    sum.val[0] = vmlaq_f32(sum.val[0], t0, t0);
    sum.val[1] = vmlaq_f32(sum.val[1], t1, t1);
    sum.val[2] = vmlaq_f32(sum.val[2], t2, t2);
    sum.val[3] = vmlaq_f32(sum.val[3], t3, t3);
  }
  return reduce_f32x4x4(sum);
}

inline float l2a_fp16_fp16(const fp16 *x, const fp16 *y, const int32_t d) {
  float32x4x4_t sum = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                       vdupq_n_f32(0.0f)};
  for (int32_t i = 0; i < d; i += 16) {
    auto uu0 = vld1q_f16((const __fp16 *)(x + i));
    auto uu1 = vld1q_f16((const __fp16 *)(x + i + 8));
    auto zz0 = vld1q_f16((const __fp16 *)(y + i));
    auto zz1 = vld1q_f16((const __fp16 *)(y + i + 8));
    auto xx0 = vcvt_f32_f16(vget_low_f16(uu0));
    auto xx1 = vcvt_f32_f16(vget_high_f16(uu0));
    auto xx2 = vcvt_f32_f16(vget_low_f16(uu1));
    auto xx3 = vcvt_f32_f16(vget_high_f16(uu1));
    auto yy0 = vcvt_f32_f16(vget_low_f16(zz0));
    auto yy1 = vcvt_f32_f16(vget_high_f16(zz0));
    auto yy2 = vcvt_f32_f16(vget_low_f16(zz1));
    auto yy3 = vcvt_f32_f16(vget_high_f16(zz1));
    auto t0 = vsubq_f32(xx0, yy0);
    auto t1 = vsubq_f32(xx1, yy1);
    auto t2 = vsubq_f32(xx2, yy2);
    auto t3 = vsubq_f32(xx3, yy3);
    sum.val[0] = vmlaq_f32(sum.val[0], t0, t0);
    sum.val[1] = vmlaq_f32(sum.val[1], t1, t1);
    sum.val[2] = vmlaq_f32(sum.val[2], t2, t2);
    sum.val[3] = vmlaq_f32(sum.val[3], t3, t3);
  }
  return reduce_f32x4x4(sum);
}

inline float l2a_fp32_bf16(const float *x, const bf16 *y, const int32_t d) {
  float32x4x4_t sum = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                       vdupq_n_f32(0.0f)};
  for (int32_t i = 0; i < d; i += 16) {
    auto xx0 = vld1q_f32(x + i);
    auto xx1 = vld1q_f32(x + i + 4);
    auto xx2 = vld1q_f32(x + i + 8);
    auto xx3 = vld1q_f32(x + i + 12);
    auto zz0 = vld1q_u16((const uint16_t *)(y + i));
    auto zz1 = vld1q_u16((const uint16_t *)(y + i + 8));
    auto yy0 = bf16_to_fp32(vget_low_u16(zz0));
    auto yy1 = bf16_to_fp32(vget_high_u16(zz0));
    auto yy2 = bf16_to_fp32(vget_low_u16(zz1));
    auto yy3 = bf16_to_fp32(vget_high_u16(zz1));
    auto t0 = vsubq_f32(xx0, yy0);
    auto t1 = vsubq_f32(xx1, yy1);
    auto t2 = vsubq_f32(xx2, yy2);
    auto t3 = vsubq_f32(xx3, yy3);
    sum.val[0] = vmlaq_f32(sum.val[0], t0, t0);
    sum.val[1] = vmlaq_f32(sum.val[1], t1, t1);
    sum.val[2] = vmlaq_f32(sum.val[2], t2, t2);
    sum.val[3] = vmlaq_f32(sum.val[3], t3, t3);
  }
  return reduce_f32x4x4(sum);
}

inline float l2a_bf16_bf16(const bf16 *x, const bf16 *y, const int32_t d) {
  float32x4x4_t sum = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                       vdupq_n_f32(0.0f)};
  for (int32_t i = 0; i < d; i += 16) {
    auto uu0 = vld1q_u16((const uint16_t *)(x + i));
    auto uu1 = vld1q_u16((const uint16_t *)(x + i + 8));
    auto zz0 = vld1q_u16((const uint16_t *)(y + i));
    auto zz1 = vld1q_u16((const uint16_t *)(y + i + 8));
    auto xx0 = bf16_to_fp32(vget_low_u16(uu0));
    auto xx1 = bf16_to_fp32(vget_high_u16(uu0));
    auto xx2 = bf16_to_fp32(vget_low_u16(uu1));
    auto xx3 = bf16_to_fp32(vget_high_u16(uu1));
    auto yy0 = bf16_to_fp32(vget_low_u16(zz0));
    auto yy1 = bf16_to_fp32(vget_high_u16(zz0));
    auto yy2 = bf16_to_fp32(vget_low_u16(zz1));
    auto yy3 = bf16_to_fp32(vget_high_u16(zz1));
    auto t0 = vsubq_f32(xx0, yy0);
    auto t1 = vsubq_f32(xx1, yy1);
    auto t2 = vsubq_f32(xx2, yy2);
    auto t3 = vsubq_f32(xx3, yy3);
    sum.val[0] = vmlaq_f32(sum.val[0], t0, t0);
    sum.val[1] = vmlaq_f32(sum.val[1], t1, t1);
    sum.val[2] = vmlaq_f32(sum.val[2], t2, t2);
    sum.val[3] = vmlaq_f32(sum.val[3], t3, t3);
  }
  return reduce_f32x4x4(sum);
}

inline int32_t l2a_u8_s8(const uint8_t *x, const int8_t *y, const int32_t d) {
  int32_t sum = 0;
  for (int32_t i = 0; i < d; ++i) {
    auto d = int32_t(x[i]) - int32_t(y[i]);
    sum += d * d;
  }
  return sum;
}

inline int32_t l2a_s8_s8(const int8_t *x, const int8_t *y, const int32_t d) {
  // int32x4_t sum = vdupq_n_s32(0);
  // for (int32_t i = 0; i < d; i += 16) {
  //   auto uu = vld1q_s8(x + i);
  //   auto zz = vld1q_s8(y + i);
  //   auto xx0 = vreinterpretq_s16_u16(vmovl_s8(vget_low_s8(uu)));
  //   auto xx1 = vreinterpretq_s16_u16(vmovl_s8(vget_high_s8(uu)));
  //   auto yy0 = vreinterpretq_s16_u16(vmovl_s8(vget_low_s8(zz)));
  //   auto yy1 = vreinterpretq_s16_u16(vmovl_s8(vget_high_s8(zz)));
  //   auto t0 = vsubq_s16(xx0, yy0);
  //   auto t1 = vsubq_s16(xx1, yy1);
  //   t0 = vmulq_s16(t0, t0);
  //   t1 = vmulq_s16(t1, t1);
  //   sum = vaddw_s16(sum, vget_low_s16(t0));
  //   sum = vaddw_s16(sum, vget_high_s16(t0));
  //   sum = vaddw_s16(sum, vget_low_s16(t1));
  //   sum = vaddw_s16(sum, vget_high_s16(t1));
  // }
  // return vaddvq_s32(sum);
  int32_t sum = 0;
  for (int32_t i = 0; i < d; ++i) {
    auto d = int32_t(x[i]) - int32_t(y[i]);
    sum += d * d;
  }
  return sum;
}

inline int32_t l2a_u4_u4(const uint8_t *x, const uint8_t *y, const int32_t d) {
  return l2_u4_u4_ref(x, y, d);
}

} // namespace helpa

#endif
