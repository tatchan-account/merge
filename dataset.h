#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

struct Dataset {
    int n = 0;
    int d = 0;
    std::vector<float> x;

    const float* ptr(int i) const { return x.data() + (size_t)i * (size_t)d; }

    static Dataset load_text(const std::string& path) {
        std::ifstream ifs(path);
        if (!ifs) throw std::runtime_error("Failed to open : " + path);

        Dataset ds;
        ifs >> ds.n >> ds.d;
        if (!ifs || ds.n <= 0 || ds.d <= 0) throw std::runtime_error("Invalid data count n or dimension d.");

        ds.x.resize((size_t)ds.n * (size_t)ds.d);
        for (int i = 0; i < ds.n; ++i) {
            for (int t = 0; t < ds.d; ++t) {
                ifs >> ds.x[(size_t)i * (size_t)ds.d + (size_t)t];
                if (!ifs) throw std::runtime_error("Unexpected data format was detected.");
            }
        }
        return ds;
    }
};

struct dist_func {
    const Dataset& ds;

    float operator()(int i, int j) const {
        const float* a = ds.ptr(i);
        const float* b = ds.ptr(j);
        float s = 0;
        for (int t = 0; t < ds.d; ++t) {
            float diff = a[t] - b[t];
            s += diff * diff;
        }
        return s;
    }
};