// gt_io.hpp
#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>

struct GroundTruth {
    int n = 0;
    int k = 0; // k_gt
    std::vector<uint32_t> ids; // size = n*k, row-major

    const uint32_t* row(int i) const { return ids.data() + (size_t)i * (size_t)k; }
};

inline GroundTruth load_gt(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("Failed to open GT: " + path);

    GroundTruth gt;
    ifs >> gt.n >> gt.k;
    if (!ifs || gt.n <= 0 || gt.k <= 0) throw std::runtime_error("Invalid GT header (n k).");

    gt.ids.resize((size_t)gt.n * (size_t)gt.k);
    for (int i = 0; i < gt.n; ++i) {
        for (int t = 0; t < gt.k; ++t) {
            uint64_t tmp;
            ifs >> tmp;
            if (!ifs) throw std::runtime_error("Invalid GT body.");
            gt.ids[(size_t)i * (size_t)gt.k + (size_t)t] = (uint32_t)tmp;
        }
    }
    return gt;
}
