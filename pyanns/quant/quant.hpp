#pragma once

#include <string>
#include <unordered_map>

#include "pyanns/quant/bf16_quant.hpp"
#include "pyanns/quant/e5m2_quant.hpp"
#include "pyanns/quant/fp16_quant.hpp"
#include "pyanns/quant/fp32_quant.hpp"
#include "pyanns/quant/product_quant.hpp"
#include "pyanns/quant/sq4_quant.hpp"
#include "pyanns/quant/sq8p_quant.hpp"
#include "pyanns/quant/sq4u_quant.hpp"
#include "pyanns/quant/sq4ua_quant.hpp"
#include "pyanns/quant/sq6_quant.hpp"
#include "pyanns/quant/sq8_quant.hpp"
#include "pyanns/quant/sq8u_quant.hpp"

namespace pyanns {

enum class QuantizerType {
  FP32,
  FP16,
  BF16,
  E5M2,
  SQ8U,
  SQ8,
  SQ6,
  SQ4U,
  SQ4UA,
  SQ4,
  SQ8P,
  PQ8
};

inline std::unordered_map<std::string, QuantizerType> quantizer_map;

inline int quantizer_map_init = [] {
  quantizer_map["FP32"] = QuantizerType::FP32;
  quantizer_map["FP16"] = QuantizerType::FP16;
  quantizer_map["BF16"] = QuantizerType::BF16;
  quantizer_map["E5M2"] = QuantizerType::E5M2;
  quantizer_map["SQ8U"] = QuantizerType::SQ8U;
  quantizer_map["SQ8"] = QuantizerType::SQ8;
  quantizer_map["SQ6"] = QuantizerType::SQ6;
  quantizer_map["SQ4U"] = QuantizerType::SQ4U;
  quantizer_map["SQ4UA"] = QuantizerType::SQ4UA;
  quantizer_map["SQ4"] = QuantizerType::SQ4;
  quantizer_map["SQ8P"] = QuantizerType::SQ8P;
  quantizer_map["PQ8"] = QuantizerType::PQ8;
  return 42;
}();

} // namespace pyanns
