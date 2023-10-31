#pragma once

#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <sys/mman.h>

#include "pyanns/common.hpp"

namespace pyanns {

inline void *align1G(size_t nbytes, uint8_t x = 0) {
  size_t len = (nbytes + (1 << 30) - 1) >> 30 << 30;
  auto p = std::aligned_alloc(1 << 30, len);
  madvise(p, len, MADV_HUGEPAGE);
  std::memset(p, x, len);
  return p;
}

inline void *align2M(size_t nbytes, uint8_t x = 0) {
  size_t len = (nbytes + (1 << 21) - 1) >> 21 << 21;
  auto p = std::aligned_alloc(1 << 21, len);
  madvise(p, len, MADV_HUGEPAGE);
  std::memset(p, x, len);
  return p;
}

inline void *alloc64B(size_t nbytes, uint8_t x = 0) {
  size_t len = (nbytes + (1 << 6) - 1) >> 6 << 6;
  auto p = std::aligned_alloc(1 << 6, len);
  std::memset(p, x, len);
  return p;
}

inline void *align_alloc(size_t nbytes, uint8_t x = 0) {
  if (nbytes >= 1 * 1024 * 1024 * 1024) {
    return align1G(nbytes, x);
  } else if (nbytes >= 2 * 1024 * 1024) {
    return align2M(nbytes, x);
  } else {
    return alloc64B(nbytes, x);
  }
}

} // namespace pyanns
