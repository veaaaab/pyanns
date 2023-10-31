#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
#include <vector>

#include "pyanns/memory.hpp"

namespace pyanns {

namespace inference {

template <typename Pool>
concept NeighborPoolConcept = requires(Pool pool, int32_t u,
                                       typename Pool::dist_type dist) {
  { pool.insert(u, dist) } -> std::same_as<bool>;
  { pool.pop() } -> std::same_as<int32_t>;
  { pool.has_next() } -> std::same_as<bool>;
};

struct BitsetStl {
  std::vector<bool> data;

  BitsetStl() = default;

  explicit BitsetStl(int n) : data(n) {}

  void reset() { std::fill(data.begin(), data.end(), false); }

  void set(int32_t i) { data[i] = true; }

  bool get(int32_t i) { return data[i]; }
};

template <typename Block = uint64_t> struct Bitset {
  constexpr static int block_size = sizeof(Block) * 8;
  int nbytes;
  Block *data;

  Bitset() = default;

  explicit Bitset(int n)
      : nbytes((n + block_size - 1) / block_size * sizeof(Block)),
        data((uint64_t *)align_alloc(nbytes)) {}

  ~Bitset() { free(data); }

  void reset() { memset(data, 0, nbytes); }

  void set(int i) {
    data[i / block_size] |= (Block(1) << (i & (block_size - 1)));
  }

  bool get(int i) {
    return (data[i / block_size] >> (i & (block_size - 1))) & 1;
  }
};

template <typename dist_t = float> struct Neighbor {
  int id;
  dist_t distance;

  Neighbor() = default;
  Neighbor(int id, dist_t distance) : id(id), distance(distance) {}

  inline friend bool operator<(const Neighbor &lhs, const Neighbor &rhs) {
    return lhs.distance < rhs.distance ||
           (lhs.distance == rhs.distance && lhs.id < rhs.id);
  }
  inline friend bool operator>(const Neighbor &lhs, const Neighbor &rhs) {
    return !(lhs < rhs);
  }
};

template <typename dist_t> struct MaxHeap {
  explicit MaxHeap(int capacity) : capacity(capacity), pool(capacity) {}
  void push(int u, dist_t dist) {
    if (sz < capacity) {
      pool[sz] = {u, dist};
      std::push_heap(pool.begin(), pool.begin() + ++sz);
    } else if (dist < pool[0].distance) {
      sift_down(0, u, dist);
    }
  }
  int pop() {
    std::pop_heap(pool.begin(), pool.begin() + sz--);
    return pool[sz].id;
  }
  void sift_down(int i, int u, dist_t dist) {
    pool[0] = {u, dist};
    for (; 2 * i + 1 < sz;) {
      int j = i;
      int l = 2 * i + 1, r = 2 * i + 2;
      if (pool[l].distance > dist) {
        j = l;
      }
      if (r < sz && pool[r].distance > std::max(pool[l].distance, dist)) {
        j = r;
      }
      if (i == j) {
        break;
      }
      pool[i] = pool[j];
      i = j;
    }
    pool[i] = {u, dist};
  }
  int32_t size() const { return sz; }
  bool empty() const { return size() == 0; }
  dist_t top_dist() const { return pool[0].distance; }
  int sz = 0, capacity;
  std::vector<Neighbor<dist_t>> pool;
};

template <typename dist_t> struct MinMaxHeap {

  explicit MinMaxHeap(int capacity) : capacity(capacity), pool(capacity) {}

  bool push(int u, dist_t dist) {
    if (cur == capacity) {
      if (dist >= pool[0].distance) {
        return false;
      }
      if (pool[0].id != -1) {
        sz--;
      }
      std::pop_heap(pool.begin(), pool.begin() + cur--);
    }
    pool[cur] = {u, dist};
    std::push_heap(pool.begin(), pool.begin() + ++cur);
    sz++;
    return true;
  }

  int32_t size() const { return sz; }

  dist_t max() const { return pool[0].distance; }

  void clear() { sz = cur = 0; }

  int pop_min() {
    int i = cur - 1;
    for (; i >= 0 && pool[i].id == -1; --i)
      ;
    if (i == -1) {
      return -1;
    }
    int imin = i;
    dist_t vmin = pool[i].distance;
    for (; --i >= 0;) {
      if (pool[i].id != -1 && pool[i].distance < vmin) {
        vmin = pool[i].distance;
        imin = i;
      }
    }
    int ret = pool[imin].id;
    pool[imin].id = -1;
    --sz;
    return ret;
  }

  int32_t count_below(float thresh) const {
    int32_t n_below = 0;
    for (int32_t i = 0; i < cur; ++i) {
      if (pool[i].distance < thresh) {
        n_below++;
      }
    }
    return n_below;
  }

  int sz = 0, cur = 0, capacity;
  std::vector<Neighbor<dist_t>> pool;
};

template <typename dist_t> struct LinearPool {
  using dist_type = dist_t;

  LinearPool() = default;

  LinearPool(int n, int capacity, int = 0)
      : nb(n), capacity_(capacity), data_(capacity_ + 1), vis(n) {}

  void reset() {
    size_ = cur_ = 0;
    vis.reset();
  }

  PYANNS_INLINE int find_bsearch(dist_t dist) {
    int lo = 0, hi = size_;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (data_[mid].distance > dist) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    return lo;
  }

  PYANNS_INLINE bool insert(int u, dist_t dist) {
    if (size_ == capacity_ && dist >= data_[size_ - 1].distance) {
      return false;
    }
    int lo = find_bsearch(dist);
    std::memmove(&data_[lo + 1], &data_[lo],
                 (size_ - lo) * sizeof(Neighbor<dist_t>));
    data_[lo] = {u, dist};
    if (size_ < capacity_) {
      size_++;
    }
    if (lo < cur_) {
      cur_ = lo;
    }
    return true;
  }

  int pop() {
    set_checked(data_[cur_].id);
    int pre = cur_;
    while (cur_ < size_ && is_checked(data_[cur_].id)) {
      cur_++;
    }
    return get_id(data_[pre].id);
  }

  bool has_next() const { return cur_ < size_; }
  int id(int i) const { return get_id(data_[i].id); }
  dist_type dist(int i) const { return data_[i].distance; }
  int size() const { return size_; }
  int capacity() const { return capacity_; }

  void set_visited(int32_t u) { return vis.set(u); }
  bool is_visited(int32_t u) { return vis.get(u); }

  constexpr static int kMask = 2147483647;
  int get_id(int id) const { return id & kMask; }
  void set_checked(int &id) { id |= 1 << 31; }
  bool is_checked(int id) { return id >> 31 & 1; }

  int nb, size_ = 0, cur_ = 0, capacity_;
  std::vector<Neighbor<dist_t>> data_;
  Bitset<uint64_t> vis;
};

template <typename dist_t> struct HeapPool {
  HeapPool(int n, int capacity, int topk)
      : nb(n), capacity_(capacity), candidates(capacity), retset(topk), vis(n) {
  }
  bool insert(int u, dist_t dist) {
    retset.push(u, dist);
    return candidates.push(u, dist);
  }
  int pop() { return candidates.pop_min(); }
  bool has_next() const { return candidates.size() > 0; }
  int32_t id(int i) const { return retset.pool[i].id; }
  int32_t size() const { return retset.size(); }
  int32_t capacity() const { return capacity_; }
  int nb, size_ = 0, capacity_;
  MinMaxHeap<dist_t> candidates;
  MaxHeap<dist_t> retset;
  Bitset<uint64_t> vis;
};

} // namespace inference

struct Neighbor {
  int id;
  float distance;
  bool flag;

  Neighbor() = default;
  Neighbor(int id, float distance, bool f)
      : id(id), distance(distance), flag(f) {}

  inline bool operator<(const Neighbor &other) const {
    return distance < other.distance;
  }
};

struct Node {
  int id;
  float distance;

  Node() = default;
  Node(int id, float distance) : id(id), distance(distance) {}

  inline bool operator<(const Node &other) const {
    return distance < other.distance;
  }
};

inline int insert_into_pool(Neighbor *addr, int K, Neighbor nn) {
  // find the location to insert
  int left = 0, right = K - 1;
  if (addr[left].distance > nn.distance) {
    memmove(&addr[left + 1], &addr[left], K * sizeof(Neighbor));
    addr[left] = nn;
    return left;
  }
  if (addr[right].distance < nn.distance) {
    addr[K] = nn;
    return K;
  }
  while (left < right - 1) {
    int mid = (left + right) / 2;
    if (addr[mid].distance > nn.distance) {
      right = mid;
    } else {
      left = mid;
    }
  }
  // check equal ID

  while (left > 0) {
    if (addr[left].distance < nn.distance) {
      break;
    }
    if (addr[left].id == nn.id) {
      return K + 1;
    }
    left--;
  }
  if (addr[left].id == nn.id || addr[right].id == nn.id) {
    return K + 1;
  }
  memmove(&addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
  addr[right] = nn;
  return right;
}

} // namespace pyanns
