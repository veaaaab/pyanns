#pragma once

#include "helpa/platform/neon/neon_utils.hpp"
#if defined(__aarch64__)

#include <arm_neon.h>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

#include "helpa/dot.hpp"
#include "helpa/ref/dot_ref.hpp"
#include "helpa/types.hpp"

namespace helpa {

inline float
dot_fp32_fp32(const float* x, const float* y, const int32_t d) {
    int32_t da = d / 16 * 16;
    return dota_fp32_fp32(x, y, da) + dot_fp32_fp32_ref(x + da, y + da, d - da);
}

inline float
dot_fp32_fp16(const float* x, const fp16* y, const int32_t d) {
    int32_t da = d / 16 * 16;
    return dota_fp32_fp16(x, y, da) + dot_fp32_fp16_ref(x + da, y + da, d - da);
}

inline float
dot_fp16_fp16(const fp16* x, const fp16* y, const int32_t d) {
    int32_t da = d / 32 * 32;
    return dota_fp16_fp16(x, y, da) + dot_fp16_fp16_ref(x + da, y + da, d - da);
}

inline float
dot_fp32_bf16(const float* x, const bf16* y, const int32_t d) {
    int32_t da = d / 16 * 16;
    return dota_fp32_bf16(x, y, da) + dot_fp32_bf16_ref(x + da, y + da, d - da);
}

inline float
dot_bf16_bf16(const bf16* x, const bf16* y, const int32_t d) {
    int32_t da = d / 32 * 32;
    return dota_bf16_bf16(x, y, da) + dot_bf16_bf16_ref(x + da, y + da, d - da);
}

inline int32_t
dot_u8_s8(const uint8_t* x, const int8_t* y, const int32_t d) {
    int32_t da = d / 64 * 64;
    return dota_u8_s8(x, y, da) + dot_u8_s8_ref(x + da, y + da, d - da);
}

inline int32_t
dot_s8_s8(const int8_t* x, const int8_t* y, const int32_t d) {
    int32_t da = d / 64 * 64;
    return dota_s8_s8(x, y, da) + dot_s8_s8_ref(x + da, y + da, d - da);
}

inline float
dota_fp32_fp32(const float* x, const float* y, const int32_t d) {
    float32x4x4_t sum = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    for (int32_t i = 0; i < d; i += 16) {
        auto xx0 = vld1q_f32(x + i);
        auto xx1 = vld1q_f32(x + i + 4);
        auto xx2 = vld1q_f32(x + i + 8);
        auto xx3 = vld1q_f32(x + i + 12);
        auto yy0 = vld1q_f32(y + i);
        auto yy1 = vld1q_f32(y + i + 4);
        auto yy2 = vld1q_f32(y + i + 8);
        auto yy3 = vld1q_f32(y + i + 12);
        sum.val[0] = vmlaq_f32(sum.val[0], xx0, yy0);
        sum.val[1] = vmlaq_f32(sum.val[1], xx1, yy1);
        sum.val[2] = vmlaq_f32(sum.val[2], xx2, yy2);
        sum.val[3] = vmlaq_f32(sum.val[3], xx3, yy3);
    }
    return -reduce_f32x4x4(sum);
}

inline float
dota_fp32_fp16(const float* x, const fp16* y, const int32_t d) {
    float32x4x4_t sum = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    for (int32_t i = 0; i < d; i += 16) {
        auto xx0 = vld1q_f32(x + i);
        auto xx1 = vld1q_f32(x + i + 4);
        auto xx2 = vld1q_f32(x + i + 8);
        auto xx3 = vld1q_f32(x + i + 12);
        auto zz0 = vld1q_f16((const __fp16*)(y + i));
        auto zz1 = vld1q_f16((const __fp16*)(y + i + 8));
        auto yy0 = vcvt_f32_f16(vget_low_f16(zz0));
        auto yy1 = vcvt_f32_f16(vget_high_f16(zz0));
        auto yy2 = vcvt_f32_f16(vget_low_f16(zz1));
        auto yy3 = vcvt_f32_f16(vget_high_f16(zz1));
        sum.val[0] = vmlaq_f32(sum.val[0], xx0, yy0);
        sum.val[1] = vmlaq_f32(sum.val[1], xx1, yy1);
        sum.val[2] = vmlaq_f32(sum.val[2], xx2, yy2);
        sum.val[3] = vmlaq_f32(sum.val[3], xx3, yy3);
    }
    return -reduce_f32x4x4(sum);
}

inline float
dota_fp16_fp16(const fp16* x, const fp16* y, const int32_t d) {
    float32x4x4_t sum = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    for (int32_t i = 0; i < d; i += 16) {
        auto uu0 = vld1q_f16((const __fp16*)(x + i));
        auto uu1 = vld1q_f16((const __fp16*)(x + i + 8));
        auto zz0 = vld1q_f16((const __fp16*)(y + i));
        auto zz1 = vld1q_f16((const __fp16*)(y + i + 8));
        auto xx0 = vcvt_f32_f16(vget_low_f16(uu0));
        auto xx1 = vcvt_f32_f16(vget_high_f16(uu0));
        auto xx2 = vcvt_f32_f16(vget_low_f16(uu1));
        auto xx3 = vcvt_f32_f16(vget_high_f16(uu1));
        auto yy0 = vcvt_f32_f16(vget_low_f16(zz0));
        auto yy1 = vcvt_f32_f16(vget_high_f16(zz0));
        auto yy2 = vcvt_f32_f16(vget_low_f16(zz1));
        auto yy3 = vcvt_f32_f16(vget_high_f16(zz1));
        sum.val[0] = vmlaq_f32(sum.val[0], xx0, yy0);
        sum.val[1] = vmlaq_f32(sum.val[1], xx1, yy1);
        sum.val[2] = vmlaq_f32(sum.val[2], xx2, yy2);
        sum.val[3] = vmlaq_f32(sum.val[3], xx3, yy3);
    }
    return -reduce_f32x4x4(sum);
}

inline float
dota_fp32_bf16(const float* x, const bf16* y, const int32_t d) {
    float32x4x4_t sum = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    for (int32_t i = 0; i < d; i += 16) {
        auto xx0 = vld1q_f32(x + i);
        auto xx1 = vld1q_f32(x + i + 4);
        auto xx2 = vld1q_f32(x + i + 8);
        auto xx3 = vld1q_f32(x + i + 12);
        auto zz0 = vld1q_u16((const uint16_t*)(y + i));
        auto zz1 = vld1q_u16((const uint16_t*)(y + i + 8));
        auto yy0 = bf16_to_fp32(vget_low_u16(zz0));
        auto yy1 = bf16_to_fp32(vget_high_u16(zz0));
        auto yy2 = bf16_to_fp32(vget_low_u16(zz1));
        auto yy3 = bf16_to_fp32(vget_high_u16(zz1));
        sum.val[0] = vmlaq_f32(sum.val[0], xx0, yy0);
        sum.val[1] = vmlaq_f32(sum.val[1], xx1, yy1);
        sum.val[2] = vmlaq_f32(sum.val[2], xx2, yy2);
        sum.val[3] = vmlaq_f32(sum.val[3], xx3, yy3);
    }
    return -reduce_f32x4x4(sum);
}

inline float
dota_bf16_bf16(const bf16* x, const bf16* y, const int32_t d) {
    float32x4x4_t sum = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    for (int32_t i = 0; i < d; i += 16) {
        auto uu0 = vld1q_u16((const uint16_t*)(x + i));
        auto uu1 = vld1q_u16((const uint16_t*)(x + i + 8));
        auto zz0 = vld1q_u16((const uint16_t*)(y + i));
        auto zz1 = vld1q_u16((const uint16_t*)(y + i + 8));
        auto xx0 = bf16_to_fp32(vget_low_u16(uu0));
        auto xx1 = bf16_to_fp32(vget_high_u16(uu0));
        auto xx2 = bf16_to_fp32(vget_low_u16(uu1));
        auto xx3 = bf16_to_fp32(vget_high_u16(uu1));
        auto yy0 = bf16_to_fp32(vget_low_u16(zz0));
        auto yy1 = bf16_to_fp32(vget_high_u16(zz0));
        auto yy2 = bf16_to_fp32(vget_low_u16(zz1));
        auto yy3 = bf16_to_fp32(vget_high_u16(zz1));
        sum.val[0] = vmlaq_f32(sum.val[0], xx0, yy0);
        sum.val[1] = vmlaq_f32(sum.val[1], xx1, yy1);
        sum.val[2] = vmlaq_f32(sum.val[2], xx2, yy2);
        sum.val[3] = vmlaq_f32(sum.val[3], xx3, yy3);
    }
    return -reduce_f32x4x4(sum);
}

inline int32_t
dota_u8_s8(const uint8_t* x, const int8_t* y, const int32_t d) {
    int32_t sum = 0;
    for (int32_t i = 0; i < d; ++i) {
        sum += int32_t(x[i]) * int32_t(y[i]);
    }
    return -sum;
}

inline int32_t
dota_s8_s8(const int8_t* x, const int8_t* y, const int32_t d) {
    // int32x4_t sum = vdupq_n_s32(0);
    // for (int32_t i = 0; i < d; i += 8) {
    //   auto xx = vld1_s8(x + i);
    //   auto yy = vld1_s8(y + i);
    //   auto xxx = vreinterpretq_s16_u16(vmovl_s8(xx));
    //   auto yyy = vreinterpretq_s16_u16(vmovl_s8(yy));
    //   auto t = vsubq_s16(xxx, yyy);
    //   sum = vaddw_s16(sum, vget_low_s16(t));
    //   sum = vaddw_s16(sum, vget_high_s16(t));
    // }
    // return -vaddvq_s32(sum);
    int32_t sum = 0;
    for (int32_t i = 0; i < d; ++i) {
        sum += int32_t(x[i]) * int32_t(y[i]);
    }
    return -sum;
}

inline int32_t
dota_u4_u4(const uint8_t* x, const uint8_t* y, const int32_t d) {
    return -dot_u4_u4_ref(x, y, d);
}

}  // namespace helpa

#endif
