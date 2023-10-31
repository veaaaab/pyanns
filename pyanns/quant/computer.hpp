#pragma once

#include "pyanns/memory.hpp"
#include "pyanns/simd/common.hpp"
#include "pyanns/storage/tensor.hpp"
#include <concepts>
#include <cstdint>
#include <cstring>
#include <functional>
#include <tuple>
#include <type_traits>

namespace pyanns {

template <typename Computer>
concept ComputerBaseConcept =
    requires(Computer computer, int32_t u, int32_t lines) {
      { computer.prefetch(u, lines) };
    };

template <typename Computer>
concept ComputerConcept = ComputerBaseConcept<Computer> &&
                          requires(Computer computer, int32_t u) {
                            {
                              computer.operator()(u)
                              } -> std::same_as<typename Computer::dist_type>;
                          };

template <typename Computer>
concept SymComputerConcept =
    ComputerBaseConcept<Computer> &&
    requires(Computer computer, int32_t u, int32_t v) {
      {
        computer.operator()(u, v)
        } -> std::same_as<typename Computer::dist_type>;
    };

template <StorageConcept Storage> struct Computer {
  const Storage &tensor;

  explicit Computer(const Tensor &tensor) : tensor(tensor) {}

  void prefetch(int32_t u, int32_t lines) const { tensor.prefetch(u, lines); }
};

struct MemCpyTag {};

template <StorageConcept Storage, auto dist_func, typename U, typename T,
          typename T1, typename T2, typename... Args>
struct ComputerImpl : Computer<Storage> {
  using dist_type = U;
  using S = T;
  using X = T1;
  using Y = T2;
  static_assert(
      std::is_convertible_v<
          decltype(dist_func),
          std::function<dist_type(const X *, const Y *, int32_t, Args...)>>);
  X *q = nullptr;
  std::tuple<Args...> args;

  ComputerImpl(const Storage &tensor, const S *query, const auto &encoder,
               Args &&...args)
      : Computer<Storage>(tensor), args(std::forward<Args>(args)...) {
    if constexpr (std::is_same_v<std::decay_t<decltype(encoder)>, MemCpyTag>) {
      static_assert(std::is_same_v<S, X>);
      q = (X *)align_alloc(this->tensor.dim_align() * sizeof(X));
      memcpy(q, query, this->tensor.dim() * sizeof(X));
    } else {
      encoder((const S *)query, q);
    }
  }

  ~ComputerImpl() { free(q); }

  PYANNS_INLINE dist_type operator()(int32_t u) const {
    return std::apply(
        [&](auto &&...args) {
          return dist_func(q, (const Y *)this->tensor.get(u),
                           this->tensor.dim_align(), args...);
        },
        args);
  }
};

template <StorageConcept Storage, auto dist_func, typename U, typename T,
          typename... Args>
struct SymComputerImpl : Computer<Storage> {
  using dist_type = U;
  using X = T;
  static_assert(
      std::is_convertible_v<
          decltype(dist_func),
          std::function<dist_type(const X *, const X *, int32_t, Args...)>>);

  std::tuple<Args...> args;

  SymComputerImpl(const Storage &tensor, Args &&...args)
      : Computer<Storage>(tensor), args(std::forward<Args>(args)...) {}

  PYANNS_INLINE dist_type operator()(int32_t u, int32_t v) const {
    return std::apply(
        [&](auto &&...args) {
          return dist_func((const X *)this->tensor.get(u),
                           (const X *)this->tensor.get(v),
                           this->tensor.dim_align(), args...);
        },
        args);
  }
};

// template <typename Tensor, Metric metric, typename Decoder>
// struct ReconsComputer : Computer<Tensor> {
//   using dist_type = float;
//   using DistFunc = Dist<void, metric, float, float, float>;
//   constexpr static DistFunc dist{};
//   float *q = nullptr;
//   float *x = nullptr;

//   const Decoder &decoder;

//   ReconsComputer() = default;

//   ReconsComputer(const Tensor &tensor, const float *query,
//                  const Decoder &decoder)
//       : Computer<Tensor>(tensor),
//         q((float *)align_alloc(this->tensor.dim_align() * sizeof(float))),
//         x((float *)align_alloc(this->tensor.dim_align() * sizeof(float))),
//         decoder(decoder) {
//     std::memcpy(q, query, this->tensor.dim() * sizeof(float));
//   }
//   ~ReconsComputer() { free(q); }
//   auto operator()(int u) const {
//     decoder.decode(this->tensor.get(u), x);
//     return dist(q, x, this->tensor.dim_align());
//   }
// };

// // f(T1, T2) -> U
// template <typename Tag, typename Tensor, Metric metric, typename U, typename
// T1,
//           typename T2, typename... Args>
// struct TaggedAsymComputer : Computer<Tensor> {
//   using dist_type = U;
//   using X = T1;
//   using Y = T2;
//   using DistFunc = Dist<Tag, metric, dist_type, X, Y, Args...>;
//   using DistFuncSym = Dist<Tag, metric, dist_type, Y, Y, Args...>;
//   constexpr static DistFunc dist{};
//   constexpr static DistFuncSym dist_sym{};
//   X *q = nullptr;
//   std::tuple<Args...> args;

//   TaggedAsymComputer() = default;

//   TaggedAsymComputer(const Tensor &tensor, Args &&...args)
//       : Computer<Tensor>(tensor), args(args...) {}

//   TaggedAsymComputer(const Tensor &tensor, const X *query, Args &&...args)
//       : Computer<Tensor>(tensor),
//         q((X *)align_alloc(this->tensor.dim_align() * sizeof(X))),
//         args(args...) {
//     std::memcpy(q, query, this->tensor.dim() * sizeof(X));
//   }

//   ~TaggedAsymComputer() { free(q); }

//   auto operator()(int u) const {
//     return std::apply(
//         [&](auto &&...args) {
//           return dist(q, (const Y *)this->tensor.get(u),
//                       this->tensor.dim_align(), args...);
//         },
//         args);
//   }

//   auto operator()(int32_t u, int32_t v) const {
//     return std::apply(
//         [&](auto &&...args) {
//           return dist_sym((const Y *)this->tensor.get(u),
//                           (const Y *)this->tensor.get(v),
//                           this->tensor.dim_align(), args...);
//         },
//         args);
//   }
// };

// template <typename Tensor, Metric metric, typename U, typename T1, typename
// T2,
//           typename... Args>
// using AsymComputer =
//     TaggedAsymComputer<void, Tensor, metric, U, T1, T2, Args...>;

// // encode(T1) -> T2
// // f(T2, T2) -> U
// template <typename Tag, typename Tensor, Metric metric, typename U, typename
// T1,
//           typename T2, typename... Args>
// struct TaggedSymComputer : Computer<Tensor> {
//   using dist_type = U;
//   using X = T1;
//   using Y = T2;
//   using DistFunc = Dist<Tag, metric, dist_type, Y, Y, Args...>;
//   constexpr static DistFunc dist{};
//   Y *q = nullptr;
//   std::tuple<Args...> args;

//   TaggedSymComputer() = default;

//   TaggedSymComputer(const Tensor &tensor, Args &&...args)
//       : Computer<Tensor>(tensor), args(args...) {}

//   TaggedSymComputer(const Tensor &tensor, const X *query, const auto
//   &encoder,
//                     Args &&...args)
//       : Computer<Tensor>(tensor),
//         q((Y *)align_alloc(this->tensor.dim_align() * sizeof(Y))),
//         args(args...) {
//     encoder(query, (char *)q);
//   }

//   ~TaggedSymComputer() { free(q); }

//   auto operator()(int32_t u) const {
//     return std::apply(
//         [&](auto &&...args) {
//           return this->dist(this->q, (const Y *)this->tensor.get(u),
//                             this->tensor.dim_align(), args...);
//         },
//         args);
//   }

//   auto operator()(int32_t u, int32_t v) {
//     return std::apply(
//         [&](auto &&...args) {
//           return this->dist((const Y *)this->tensor.get(u),
//                             (const Y *)this->tensor.get(v),
//                             this->tensor.dim_align(), args...);
//         },
//         args);
//   }
// };

// template <typename Tensor, Metric metric, typename U, typename T1, typename
// T2,
//           typename... Args>
// using SymComputer = TaggedSymComputer<void, Tensor, metric, U, T1, T2,
// Args...>;

// get_lut(T1) -> T2
// f(T2, T3) -> U
// template <typename Tensor, typename U, typename T1, typename T2, typename T3,
//           typename... Args>
// struct LUTComputer : Computer<Tensor> {
//   using dist_type = U;
//   using X = T1;
//   using LUT = T2;
//   using Y = T2;

//   LUT lut;
//   std::tuple<Args...> args;

//   LUTComputer() = default;

//   LUTComputer(const Tensor &tensor, const X *query, const auto &encoder,
//               Args &&...args)
//       : Computer<Tensor>(tensor), lut(encoder(query)), args(args...) {}

//   ~LUTComputer() = default;

//   auto operator()(int32_t u) const {
//     return std::apply(
//         [&](auto &&...args) {
//           return lut.dist((const Y *)this->tensor.code(u), args...);
//         },
//         args);
//   }
// };

} // namespace pyanns
