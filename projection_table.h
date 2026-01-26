#pragma once
#include <random>
#include "dataset.h"

struct ProjTableParams {
    int P = 8;
    uint64_t seed = 777;   // RNG seed
};

class ProjectionTable {
public:
    ProjectionTable() = default;

    ProjectionTable(const Dataset& ds, const ProjTableParams& pp) : ds_(&ds), p_(pp) {
        build();
    }

    // max_p (|a_p·(x_i-x_j)| / ||a_p||)^2
    inline float lower_bound_sq(int i, int j) const {
        const size_t n = (size_t)ds_->n;
        const size_t si = (size_t)i, sj = (size_t)j;
        float best = 0.0f;
        for (int p = 0; p < p_.P; ++p) {
            float di = proj_[(size_t)p * n + si];
            float dj = proj_[(size_t)p * n + sj];
            float diff = std::fabs(di - dj) * inv_norm_[(size_t)p];
            float sq = diff * diff;
            if (sq > best) best = sq;
        }
        return best;
    }

    // 射影空間での距離（必ずしも正確でない）
    inline float proj_l2_sq(int i, int j) const {
        const size_t n = (size_t)ds_->n;
        const size_t si = (size_t)i, sj = (size_t)j;
        float sum = 0.0f;
        for (int p = 0; p < p_.P; ++p) {
            float di = proj_[(size_t)p * n + si];
            float dj = proj_[(size_t)p * n + sj];
            float diff = (di - dj) * inv_norm_[(size_t)p];
            sum += diff * diff;
        }
        return sum;
    }

private:
    const Dataset* ds_ = nullptr;
    ProjTableParams p_{};

    // a_: [P][d]
    std::vector<float> a_;
    // inv_norm_: [P] (1/||a_p||)
    std::vector<float> inv_norm_;
    // proj_: [P][n] packed as proj_[p*n + i] = (a_p x_i) 標準ん相席
    std::vector<float> proj_;

    void build() {
        if (!ds_) throw std::runtime_error("ProjectionTable: dataset not set");
        const int n = ds_->n;
        const int d = ds_->d;
        if (n <= 0 || d <= 0) throw std::runtime_error("ProjectionTable: invalid dataset");

        // random gaussian projections
        std::mt19937_64 gen(p_.seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);

        a_.resize((size_t)p_.P * (size_t)d);
        inv_norm_.resize((size_t)p_.P);
        for (int p = 0; p < p_.P; ++p) {
            double ss = 0.0;
            for (int t = 0; t < d; ++t) {
                float v = nd(gen);
                a_[(size_t)p * (size_t)d + (size_t)t] = v;
                ss += (double)v * (double)v;
            }
            double norm = std::sqrt(ss);
            if (norm > 0.0) {} else norm = 1.0;
            inv_norm_[(size_t)p] = (float)(1.0 / norm);
        }

        // precompute projections
        proj_.assign((size_t)p_.P * (size_t)n, 0.0f);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n; ++i) {
            const float* x = ds_->ptr(i);
            for (int p = 0; p < p_.P; ++p) {
                const float* a = a_.data() + (size_t)p * (size_t)d;
                float dot = 0.0f;
                for (int t = 0; t < d; ++t) dot += a[t] * x[t];
                proj_[(size_t)p * (size_t)n + (size_t)i] = dot;
            }
        }
    }
};