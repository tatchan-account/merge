#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

#include "dataset.h"

// ------------------------------------------------------------
// Projection-distance pruning for squared L2 distance.
//
// For any vector a and points x,y:
//   |a·(x-y)| <= ||a|| * ||x-y||   (Cauchy-Schwarz)
// so
//   (|a·(x-y)| / ||a||)^2 <= ||x-y||^2.
//
// If we precompute s_p(i) = a_p · x_i for P random vectors a_p,
// then
//   LB(i,j)^2 = max_p (|s_p(i)-s_p(j)| / ||a_p||)^2
// is a *lower bound* on true squared L2 distance.
//
// Therefore, when keeping the best (smallest) distances and we have
// a current worst among accepted candidates (worst_sq), we can safely
// skip the expensive true distance computation if:
//   LB(i,j)^2 >= worst_sq.
// ------------------------------------------------------------

struct ProjPruneParams {
    int P = 4;               // number of projection vectors
    uint64_t seed = 12345;   // RNG seed

    void validate() const {
        if (P <= 0) throw std::runtime_error("ProjPrune: P must be positive");
    }
};

class ProjectionPruner {
public:
    ProjectionPruner() = default;

    ProjectionPruner(const Dataset& ds, const ProjPruneParams& pp)
            : ds_(&ds), p_(pp) {
        build();
    }

    int P() const { return p_.P; }

    // Lower bound on squared L2 distance between global ids i and j.
    inline float lower_bound_sq(int i, int j) const {
        const size_t n = (size_t)ds_->n;
        const int P = p_.P;
        const size_t si = (size_t)i;
        const size_t sj = (size_t)j;

        float best = 0.0f;
        const float* invn = inv_norm_.data();

        // layout: proj_[p*n + i]
        for (int p = 0; p < P; ++p) {
            float di = proj_[(size_t)p * n + si];
            float dj = proj_[(size_t)p * n + sj];
            float diff = std::fabs(di - dj) * invn[(size_t)p];
            float sq = diff * diff;
            if (sq > best) best = sq;
        }
        return best;
    }

    // ------------------------------------------------------------
    // Approximate squared distance in the P-dimensional projection
    // space (normalized by ||a_p||):
    //
    //   approx_sq(i,j) = sum_p ( |a_p·x_i - a_p·x_j| / ||a_p|| )^2
    //
    // This is NOT a strict bound of the true distance; it is intended
    // only for *ranking / prefiltering* candidates cheaply.
    // ------------------------------------------------------------
    inline float approx_dist_sq(int i, int j) const {
        const size_t n = (size_t)ds_->n;
        const int P = p_.P;
        const size_t si = (size_t)i;
        const size_t sj = (size_t)j;

        float sum = 0.0f;
        const float* invn = inv_norm_.data();

        for (int p = 0; p < P; ++p) {
            float di = proj_[(size_t)p * n + si];
            float dj = proj_[(size_t)p * n + sj];
            float diff = (di - dj) * invn[(size_t)p];
            sum += diff * diff;
        }
        return sum;
    }

private:
    const Dataset* ds_ = nullptr;
    ProjPruneParams p_{};

    // a_: [P][d]
    std::vector<float> a_;
    // inv_norm_: [P] (1/||a_p||)
    std::vector<float> inv_norm_;
    // proj_: [P][n] packed as proj_[p*n + i] = a_p·x_i
    std::vector<float> proj_;

    void build() {
        p_.validate();
        if (!ds_) throw std::runtime_error("ProjPrune: dataset missing");
        const int n = ds_->n;
        const int d = ds_->d;
        if (n <= 0 || d <= 0) throw std::runtime_error("ProjPrune: invalid dataset");

        // random normal projection vectors
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
            if (!(norm > 0.0)) norm = 1.0;
            inv_norm_[(size_t)p] = (float)(1.0 / norm);
        }

        // precompute projections
        proj_.assign((size_t)p_.P * (size_t)n, 0.0f);
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
