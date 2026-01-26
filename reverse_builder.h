#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>
#include "splitmix64.h"

class ReverseBuilder {
public:
    using id_t = uint32_t;

    ReverseBuilder(int n, int cap, uint64_t seed)
            : n_(n), cap_(cap),
              old_ids_((size_t)n * (size_t)cap),
              new_ids_((size_t)n * (size_t)cap),
              old_sz_(n, 0), new_sz_(n, 0),
              old_seen_(n, 0), new_seen_(n, 0),
              rng_(seed)
    {}

    int n() const { return n_; }
    int cap() const { return cap_; }

    void reset() {
        std::fill(old_sz_.begin(), old_sz_.end(), 0);
        std::fill(new_sz_.begin(), new_sz_.end(), 0);
        std::fill(old_seen_.begin(), old_seen_.end(), 0);
        std::fill(new_seen_.begin(), new_seen_.end(), 0);
    }

    void push_old(id_t to, id_t from) { push_impl(old_ids_, old_sz_, old_seen_, to, from); }
    void push_new(id_t to, id_t from) { push_impl(new_ids_, new_sz_, new_seen_, to, from); }

    const id_t* old_ptr(int v) const { return old_ids_.data() + (size_t)v * (size_t)cap_; }
    const id_t* new_ptr(int v) const { return new_ids_.data() + (size_t)v * (size_t)cap_; }
    uint16_t old_size(int v) const { return old_sz_[v]; }
    uint16_t new_size(int v) const { return new_sz_[v]; }

private:
    int n_, cap_;
    std::vector<id_t> old_ids_, new_ids_;
    std::vector<uint16_t> old_sz_, new_sz_;
    std::vector<uint32_t> old_seen_, new_seen_;
    SplitMix64 rng_;

    void push_impl(std::vector<id_t>& ids,
                   std::vector<uint16_t>& sz,
                   std::vector<uint32_t>& seen,
                   id_t to, id_t from) {
        uint32_t s = seen[to]++; // これまで来た数
        uint16_t &z = sz[to];
        id_t* base = ids.data() + (size_t)to * (size_t)cap_;

        if (z < (uint16_t)cap_) {
            base[z++] = from;
            return;
        }
        // reservoir sampling
        uint32_t r = rng_.uniform_u32(s + 1); // [0, s]
        if (r < (uint32_t)cap_) base[r] = from;
    }
};
