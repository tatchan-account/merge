#pragma once
#include <cstdint>

struct SplitMix64 {
    uint64_t x;
    explicit SplitMix64(uint64_t seed = 0x123456789abcdefULL) : x(seed) {}

    uint64_t next_u64() {
        uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    // [0, hi) 一様（hi>0）
    uint32_t uniform_u32(uint32_t hi) {
        return (uint32_t)(next_u64() % hi);
    }
};
