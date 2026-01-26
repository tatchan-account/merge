#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>
#include "knngraph.h"
#include "lsh_pstable.h"
#include "splitmix64.h"

/*
 * これはNN-DescentでLSHを使う時のみ使用される
 * mainから呼び出されはするが、あまり使っていない
 */

struct NNDLSHInitParams {
    PStableLSHParams lsh;
    int cand_cap = 256;        // candidates per point from LSH (before filtering/self-removal)

    // Fill policy
    // If seed_k >= 0: take min(seed_k, k) items from LSH, rest random.
    // Else: take (k - rand_min) items from LSH when possible.
    int seed_k = -1;
    int rand_min = 2;

    // If true, rank LSH candidates by exact distance and take top items.
    // If false, take candidates as-is (after dedup) which is faster but noisier.
    bool rank_by_distance = true;

    void validate() const {
        lsh.validate();
        if (cand_cap <= 0) throw std::runtime_error("LSHInit: cand_cap must be positive");
        if (seed_k < -1) throw std::runtime_error("LSHInit: seed_k must be >= -1");
        if (rand_min < 0) throw std::runtime_error("LSHInit: rand_min must be >= 0");
    }
};

class LSHInit {
public:
    // Build an index over [start_global, end_global)
    LSHInit(const Dataset& ds, int start_global, int end_global, const NNDLSHInitParams& ip)
        : ds_(&ds), start_(start_global), end_(end_global), ip_(ip), idx_(ds, start_global, end_global, ip.lsh) {
        ip_.validate();
        if (start_ < 0 || end_ > ds.n || start_ >= end_) {
            throw std::runtime_error("LSHInit: invalid [start,end)");
        }
    }

    // Build an index over the whole dataset [0, n)
    LSHInit(const Dataset& ds, const NNDLSHInitParams& ip)
        : LSHInit(ds, 0, ds.n, ip) {}

    // Use an externally built index (avoid rebuilding when you already have it).
    // NOTE: the provided index must cover [start_global, end_global).
    LSHInit(const Dataset& ds, int start_global, int end_global, const NNDLSHInitParams& ip, const PStableLSHIndex* external)
        : ds_(&ds), start_(start_global), end_(end_global), ip_(ip), external_(external) {
        ip_.validate();
        if (!external_) throw std::runtime_error("LSHInit: external index is null");
    }

    template<class Dist>
    void operator()(kNNGraph& g, const Dist& dist, SplitMix64& rng) const {
        if (!ds_) throw std::runtime_error("LSHInit: dataset null");
        const int n = g.n();
        const int k = g.k();
        if (n <= 0 || k <= 0) return;

        const PStableLSHIndex& index = external_ ? *external_ : idx_;

        std::vector<uint32_t> cands;
        std::vector<std::pair<float,uint32_t>> scored;

        for (int i = 0; i < n; ++i) {
            auto* nbr = g.nbr_ptr(i);
            auto* ds  = g.dist_ptr(i);
            auto* flg = g.flag_ptr(i);

            const int gi = start_ + i; // global id
            if (gi < start_ || gi >= end_) throw std::runtime_error("LSHInit: global id out of [start,end)");

            index.query(gi, ip_.cand_cap, cands, rng);

            // Filter to local ids, drop self, dedup (query already dedups, but keep safe)
            for (uint32_t& id : cands) {
                if ((int)id < start_ || (int)id >= end_) id = kNNGraph::invalid_id();
            }
            cands.erase(std::remove(cands.begin(), cands.end(), kNNGraph::invalid_id()), cands.end());
            cands.erase(std::remove(cands.begin(), cands.end(), (uint32_t)gi), cands.end());
            std::sort(cands.begin(), cands.end());
            cands.erase(std::unique(cands.begin(), cands.end()), cands.end());

            const int want_seed = (ip_.seed_k >= 0) ? std::min(ip_.seed_k, k)
                                                    : std::max(0, k - ip_.rand_min);

            int used_seed = 0;

            if (!cands.empty() && want_seed > 0) {
                const int m = std::min(want_seed, (int)cands.size());

                if (ip_.rank_by_distance) {
                    scored.clear();
                    scored.reserve(cands.size());
                    for (uint32_t gid : cands) {
                        const uint32_t lj = (uint32_t)((int)gid - start_);
                        float d2 = dist(i, (int)lj);
                        scored.push_back({d2, lj});
                    }
                    if ((int)scored.size() > m) {
                        std::nth_element(scored.begin(), scored.begin() + m, scored.end(),
                                         [](const auto& a, const auto& b){ return a.first < b.first; });
                        scored.resize((size_t)m);
                    }
                    std::sort(scored.begin(), scored.end(),
                              [](const auto& a, const auto& b){ return a.first < b.first; });
                    for (int t = 0; t < (int)scored.size(); ++t) {
                        nbr[t] = (uint32_t)scored[t].second;
                        ds[t]  = scored[t].first;
                        flg[t] = kNNGraph::IS_NEW;
                    }
                    used_seed = (int)scored.size();
                } else {
                    // take first m after a quick partial shuffle
                    for (int t = 0; t < m; ++t) {
                        int j = t + (int)rng.uniform_u32((uint32_t)((int)cands.size() - t));
                        std::swap(cands[(size_t)t], cands[(size_t)j]);
                        uint32_t lj = (uint32_t)((int)cands[(size_t)t] - start_);
                        nbr[t] = lj;
                        ds[t]  = dist(i, (int)lj);
                        flg[t] = kNNGraph::IS_NEW;
                    }
                    used_seed = m;
                }
            }

            // Fill the rest uniformly at random within [0,n) (local ids)
            auto already = [&](uint32_t id, int upto) -> bool {
                for (int t = 0; t < upto; ++t) if (nbr[t] == id) return true;
                return false;
            };

            for (int t = used_seed; t < k; ++t) {
                // sample from [0,n) excluding i
                for (int tries = 0; tries < 200; ++tries) {
                    uint32_t r = rng.uniform_u32((uint32_t)(n - 1));
                    uint32_t id = (r >= (uint32_t)i) ? (r + 1) : r;
                    if (already(id, t)) continue;
                    nbr[t] = id;
                    ds[t]  = dist(i, (int)id);
                    flg[t] = kNNGraph::IS_NEW;
                    break;
                }
            }

            // In case used_seed==0 and random fill didn't assign (very unlikely), sanitize
            for (int t = 0; t < k; ++t) {
                if (nbr[t] == kNNGraph::invalid_id() || (int)nbr[t] == i) {
                    // fallback to something
                    uint32_t r = rng.uniform_u32((uint32_t)(n - 1));
                    uint32_t id = (r >= (uint32_t)i) ? (r + 1) : r;
                    nbr[t] = id;
                    ds[t]  = dist(i, (int)id);
                    flg[t] = kNNGraph::IS_NEW;
                }
            }

            g.recompute_worst_row(i);
        }
    }

private:
    const Dataset* ds_ = nullptr;
    int start_ = 0;
    int end_ = 0;
    NNDLSHInitParams ip_{};

    PStableLSHIndex idx_{};
    const PStableLSHIndex* external_ = nullptr;
};
