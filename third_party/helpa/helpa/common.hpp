#pragma once

namespace helpa {

#if defined(__clang__)

#define HELPA_FAST_BEGIN
#define HELPA_FAST_END
#define HELPA_INLINE __attribute__((always_inline))

#elif defined(__GNUC__)

#define HELPA_FAST_BEGIN                                                       \
  _Pragma("GCC push_options") _Pragma(                                         \
      "GCC optimize (\"unroll-loops,associative-math,no-signed-zeros\")")
#define HELPA_FAST_END _Pragma("GCC pop_options")
#define HELPA_INLINE [[gnu::always_inline]]

#endif

} // namespace helpa
