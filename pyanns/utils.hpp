#pragma once

#include <algorithm>
#include <mutex>
#include <random>
#include <unordered_set>

namespace pyanns {

struct Timer {
#define CUR_TIME std::chrono::high_resolution_clock::now()
  Timer(const std::string &msg) : msg(msg), start(CUR_TIME) {}

  ~Timer() {
    auto ed = CUR_TIME;
    auto ela = std::chrono::duration<double>(ed - start).count();
    printf("FUCK!!!! [%s] time %lfs\n", msg.c_str(), ela);
  }
  std::string msg;
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

using LockGuard = std::lock_guard<std::mutex>;

inline void GenRandom(std::mt19937 &rng, int *addr, const int size,
                      const int N) {
  for (int i = 0; i < size; ++i) {
    addr[i] = rng() % (N - size);
  }
  std::sort(addr, addr + size);
  for (int i = 1; i < size; ++i) {
    if (addr[i] <= addr[i - 1]) {
      addr[i] = addr[i - 1] + 1;
    }
  }
  int off = rng() % N;
  for (int i = 0; i < size; ++i) {
    addr[i] = (addr[i] + off) % N;
  }
}

struct RandomGenerator {
  std::mt19937 mt;

  explicit RandomGenerator(int64_t seed = 1234) : mt((unsigned int)seed) {}

  /// random positive integer
  int rand_int() { return mt() & 0x7fffffff; }

  /// random int64_t
  int64_t rand_int64() {
    return int64_t(rand_int()) | int64_t(rand_int()) << 31;
  }

  /// generate random integer between 0 and max-1
  int rand_int(int max) { return mt() % max; }

  /// between 0 and 1
  float rand_float() { return mt() / float(mt.max()); }

  double rand_double() { return mt() / double(mt.max()); }
};

} // namespace pyanns