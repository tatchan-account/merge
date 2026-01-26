#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>

#include "dataset.h"
#include "splitmix64.h"

/*
 * お互いのサブグラフのseedに対するLSHインデックス作成
 *
 * L／ハッシュテーブル数
 * w／ハッシュの底
 * K／ハッシュの次元
 */
struct PStableLSHParams {
    int L = 2;
    int K = 4;
    float w = 4;
    int bucket_cap = 64;
    uint64_t seed = 12345;

    void validate() const {
        if (L <= 0 || K <= 0) throw std::runtime_error("LSH: L and K must be positive.");
        if (w <= 0.0f || !std::isfinite(w)) throw std::runtime_error("LSH: w must be finite and > 0.");
        if (bucket_cap <= 0) throw std::runtime_error("LSH: bucket_cap must be positive.");
    }
};

class PStableLSHIndex {
public:
    struct Entry {
        uint64_t key;
        // global id
        uint32_t id;
    };

    PStableLSHIndex() = default;

    // [start, end)でのLSHインデックスを作成
    PStableLSHIndex(const Dataset& ds, int start, int end, const PStableLSHParams& pp)
        : ds_(&ds), start_(start), end_(end), p_(pp) {
        build();
    }

    int start() const { return start_; }
    int end()   const { return end_; }
    int L()     const { return p_.L; }
    int K()     const { return p_.K; }

    // Query by global id, returns candidate ids (global) in out.
    // out will be overwritten.
    void query(int q_global, int cand_cap, std::vector<uint32_t>& out, SplitMix64& rng) const {
        if (!ds_) throw std::runtime_error("LSHIndex not initialized");
        if (q_global < 0 || q_global >= ds_->n) throw std::runtime_error("LSH: q out of range");
        out.clear();
        out.reserve((size_t)cand_cap);

        std::vector<uint32_t> tmp;
        tmp.reserve((size_t)std::min(cand_cap, p_.bucket_cap) * (size_t)p_.L);

        const float* q = ds_->ptr(q_global);
        for (int l = 0; l < p_.L; ++l) {
            uint64_t key = compute_key(l, q);
            const auto& tab = tables_[(size_t)l];
            if (tab.empty()) continue;

            auto lo = std::lower_bound(tab.begin(), tab.end(), key,
                                       [](const Entry& e, uint64_t k){ return e.key < k; });
            auto hi = std::upper_bound(tab.begin(), tab.end(), key,
                                       [](uint64_t k, const Entry& e){ return k < e.key; });
            for (auto it = lo; it != hi; ++it) tmp.push_back(it->id);
        }

        if (tmp.empty()) return;

        // Dedup (tmp is small: O(L*bucket_cap))
        std::sort(tmp.begin(), tmp.end());
        tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());

        // Randomly subsample if too many
        if ((int)tmp.size() > cand_cap) {
            // Fisher–Yates partial shuffle using rng
            for (int i = 0; i < cand_cap; ++i) {
                int j = i + (int)rng.uniform_u32((uint32_t)(tmp.size() - (size_t)i));
                std::swap(tmp[(size_t)i], tmp[(size_t)j]);
            }
            tmp.resize((size_t)cand_cap);
        }

        out.swap(tmp);
    }

private:
    const Dataset* ds_ = nullptr;
    int start_ = 0;
    int end_ = 0;
    PStableLSHParams p_{};

    // Random projections and offsets
    // a_: [L][K][d]
    std::vector<float> a_;
    // b_: [L][K]
    std::vector<float> b_;

    // tables_[l] sorted by key
    std::vector<std::vector<Entry>> tables_;

    static inline uint64_t mix_u64(uint64_t x) {
        // splitmix64 finalizer
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    const float* a_ptr(int l, int k) const {
        const int d = ds_->d;
        return a_.data() + ((size_t)l * (size_t)p_.K + (size_t)k) * (size_t)d;
    }

    float b_val(int l, int k) const {
        return b_[(size_t)l * (size_t)p_.K + (size_t)k];
    }

    uint64_t compute_key(int l, const float* x) const {
        const int d = ds_->d;
        uint64_t key = 0xCBF29CE484222325ULL; // FNV-ish seed
        for (int kk = 0; kk < p_.K; ++kk) {
            const float* a = a_ptr(l, kk);
            float dot = 0.0f;
            for (int t = 0; t < d; ++t) dot += a[t] * x[t];
            float v = (dot + b_val(l, kk)) / p_.w;
            int32_t h = (int32_t)std::floor(v);
            key ^= mix_u64((uint64_t)(uint32_t)h + 0x9e3779b97f4a7c15ULL * (uint64_t)(kk + 1));
            key *= 0x100000001B3ULL;
        }
        return key;
    }

    void build() {
        p_.validate();
        if (!ds_) throw std::runtime_error("LSH: dataset missing");
        if (start_ < 0 || end_ > ds_->n || start_ >= end_) throw std::runtime_error("LSH: invalid [start,end)");

        const int d = ds_->d;
        // Generate random a and b
        {
            std::mt19937_64 gen(p_.seed);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            std::uniform_real_distribution<float> ud(0.0f, p_.w);

            a_.resize((size_t)p_.L * (size_t)p_.K * (size_t)d);
            b_.resize((size_t)p_.L * (size_t)p_.K);

            for (size_t i = 0; i < a_.size(); ++i) a_[i] = nd(gen);
            for (size_t i = 0; i < b_.size(); ++i) b_[i] = ud(gen);
        }

        tables_.assign((size_t)p_.L, {});

        // Build each table
        for (int l = 0; l < p_.L; ++l) {
            std::vector<Entry> ent;
            ent.reserve((size_t)(end_ - start_));

            for (int id = start_; id < end_; ++id) {
                uint64_t key = compute_key(l, ds_->ptr(id));
                ent.push_back(Entry{key, (uint32_t)id});
            }

            std::sort(ent.begin(), ent.end(), [](const Entry& a, const Entry& b){
                if (a.key != b.key) return a.key < b.key;
                return a.id < b.id;
            });

            // Cap each bucket using reservoir sampling
            std::vector<Entry> capped;
            capped.reserve(ent.size());

            SplitMix64 rng(p_.seed ^ (uint64_t)l * 0x9e3779b97f4a7c15ULL);

            size_t i = 0;
            while (i < ent.size()) {
                size_t j = i + 1;
                while (j < ent.size() && ent[j].key == ent[i].key) ++j;
                size_t len = j - i;

                if ((int)len <= p_.bucket_cap) {
                    capped.insert(capped.end(), ent.begin() + (ptrdiff_t)i, ent.begin() + (ptrdiff_t)j);
                } else {
                    // reservoir sample bucket_cap items from [i, j)
                    std::vector<Entry> res;
                    res.reserve((size_t)p_.bucket_cap);
                    for (int t = 0; t < p_.bucket_cap; ++t) res.push_back(ent[i + (size_t)t]);
                    for (size_t t = (size_t)p_.bucket_cap; t < len; ++t) {
                        // random int in [0, t]
                        uint32_t r = rng.uniform_u32((uint32_t)(t + 1));
                        if ((int)r < p_.bucket_cap) res[(size_t)r] = ent[i + t];
                    }
                    capped.insert(capped.end(), res.begin(), res.end());
                }

                i = j;
            }

            std::sort(capped.begin(), capped.end(), [](const Entry& a, const Entry& b){
                if (a.key != b.key) return a.key < b.key;
                return a.id < b.id;
            });

            tables_[(size_t)l].swap(capped);
        }
    }
};
