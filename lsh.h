#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include "smerge.h"
#include "lsh_pstable.h"

struct SMergeLSHSeedParams {
    bool enable = false;
    int cand_cap = 200;     // max LSH candidates per row
    int rand_min = 2;       // minimum random seeds when seed_k < 0
    int seed_k = -1;        // if >=0: exact #LSH seeds, else slots-rand_min
};

template<class Dist>
static inline void build_half_baked_lsh(const kNNGraph& g1, const kNNGraph& g2,
                                        int n1, const Dist& dist_global, int keep_k, kNNGraph& out,
                                        TailLists& tail, SplitMix64& rng,
                                        const PStableLSHIndex* idx_s1, const PStableLSHIndex* idx_s2,
                                        const SMergeLSHSeedParams& sp) {
    const int k  = out.k();
    const int n  = out.n();
    const int n2 = n - n1;
    const int tail_k = k - keep_k;

    auto fill_from_lsh_then_random = [&](int v_global, bool v_in_s1,
                                         kNNGraph::id_t* nbr, kNNGraph::dist_t* ds, uint8_t* flg) {
        const int slots = k - keep_k;
        if (slots <= 0) return;

        int seed_fill = 0;
        if (!sp.enable) {
            seed_fill = 0;
        } else if (sp.seed_k >= 0) {
            seed_fill = std::min(sp.seed_k, slots);
        } else {
            seed_fill = std::max(0, slots - sp.rand_min);
        }

        int pos = keep_k;

        // LSH seeding
        if (seed_fill > 0 && sp.enable) {
            const PStableLSHIndex* other = v_in_s1 ? idx_s2 : idx_s1;
            if (other) {
                std::vector<uint32_t> cand;
                other->query(v_global, sp.cand_cap, cand, rng);

                std::vector<std::pair<float, uint32_t>> scored;
                scored.reserve(cand.size());

                for (uint32_t id : cand) {
                    if ((uint32_t)v_global == id) continue;
                    // ensure cross-subset
                    if (v_in_s1) { if (id < (uint32_t)n1) continue; }
                    else         { if (id >= (uint32_t)n1) continue; }

                    bool dup = false;
                    for (int u = 0; u < keep_k; ++u) {
                        if (nbr[u] == id) { dup = true; break; }
                    }
                    if (dup) continue;

                    float d = dist_global(v_global, (int)id);
                    scored.emplace_back(d, id);
                }

                std::sort(scored.begin(), scored.end(), [](auto& a, auto& b){
                    if (a.first != b.first) return a.first < b.first;
                    return a.second < b.second;
                });

                int take = std::min(seed_fill, (int)scored.size());
                for (int i = 0; i < take; ++i) {
                    nbr[pos] = scored[(size_t)i].second;
                    ds[pos]  = scored[(size_t)i].first;
                    flg[pos] = kNNGraph::IS_NEW;
                    ++pos;
                }
            }
        }

        // 足りない分ランダムで埋める
        const int other_n   = v_in_s1 ? n2 : n1;
        const int other_off = v_in_s1 ? n1 : 0;

        for (; pos < k; ++pos) {
            bool placed = false;
            for (int tries = 0; tries < 2000; ++tries) {
                if (other_n <= 0) break;
                uint32_t r = rng.uniform_u32((uint32_t)other_n);
                uint32_t id = (uint32_t)other_off + r;
                if ((uint32_t)v_global == id) continue;

                bool dup = false;
                for (int u = 0; u < pos; ++u) {
                    if (nbr[u] == id) { dup = true; break; }
                }
                if (dup) continue;

                nbr[pos] = id;
                ds[pos]  = dist_global(v_global, (int)id);
                flg[pos] = kNNGraph::IS_NEW;
                placed = true;
                break;
            }
            if (!placed) {
                nbr[pos] = kNNGraph::invalid_id();
                ds[pos]  = kNNGraph::inf();
                flg[pos] = 0;
            }
        }
    };

    // S1
    for (int i = 0; i < n1; ++i) {
        int v_global = i;
        auto* out_nbr = out.nbr_ptr(v_global);
        auto* out_ds  = out.dist_ptr(v_global);
        auto* out_fl  = out.flag_ptr(v_global);

        const auto* src_nbr = g1.nbr_ptr(i);
        const auto* src_ds  = g1.dist_ptr(i);
        auto idx = sorted_indices_by_dist(src_ds, k);

        for (int t = 0; t < keep_k; ++t) {
            int j = idx[t];
            out_nbr[t] = src_nbr[j];
            out_ds[t]  = src_ds[j];
            out_fl[t]  = 0;
        }
        if (tail_k > 0) {
            auto* tn = tail.nbr_ptr(v_global);
            auto* td = tail.dist_ptr(v_global);
            for (int t = 0; t < tail_k; ++t) {
                int j = idx[keep_k + t];
                tn[t] = src_nbr[j];
                td[t] = src_ds[j];
            }
        }

        fill_from_lsh_then_random(v_global, true, out_nbr, out_ds, out_fl);
        out.recompute_worst_row(v_global);
    }

    // S2
    for (int i = 0; i < n2; ++i) {
        int v_global = n1 + i;
        auto* out_nbr = out.nbr_ptr(v_global);
        auto* out_ds  = out.dist_ptr(v_global);
        auto* out_fl  = out.flag_ptr(v_global);

        const auto* src_nbr = g2.nbr_ptr(i);
        const auto* src_ds  = g2.dist_ptr(i);
        auto idx = sorted_indices_by_dist(src_ds, k);

        for (int t = 0; t < keep_k; ++t) {
            int j = idx[t];
            out_nbr[t] = (uint32_t)n1 + src_nbr[j];
            out_ds[t]  = src_ds[j];
            out_fl[t]  = 0;
        }
        if (tail_k > 0) {
            auto* tn = tail.nbr_ptr(v_global);
            auto* td = tail.dist_ptr(v_global);
            for (int t = 0; t < tail_k; ++t) {
                int j = idx[keep_k + t];
                tn[t] = (uint32_t)n1 + src_nbr[j];
                td[t] = src_ds[j];
            }
        }

        fill_from_lsh_then_random(v_global, false, out_nbr, out_ds, out_fl);
        out.recompute_worst_row(v_global);
    }
}

template<class Dist>
kNNGraph s_merge_lsh(const kNNGraph& g1, const kNNGraph& g2, int n1, const Dist& dist_global,
                     const NNDParams& p, float keep_ratio,
                     const PStableLSHIndex& idx_s1, const PStableLSHIndex& idx_s2,
                     const SMergeLSHSeedParams& sp, bool verbose = false) {
    const int k = g1.k();
    const int n = g1.n() + g2.n();
    const int n2 = n - n1;

    if (g2.k() != k) throw std::runtime_error("S-Merge(LSH): k mismatch between g1 and g2.");
    if (g1.n() != n1) throw std::runtime_error("S-Merge(LSH): n1 mismatch.");
    if (n1 <= k || n2 <= k) throw std::runtime_error("S-Merge(LSH): each subset must satisfy subset_size > k (for now).");

    int keep_k = (int)std::floor(keep_ratio * (float)k);
    if (keep_k < 1) keep_k = 1;
    if (keep_k > k - 1) keep_k = k - 1;

    int lo = std::max(1, k / 5);
    int hi = std::max(1, k / 2);
    if (keep_k < lo) keep_k = lo;
    if (keep_k > hi) keep_k = hi;

    const int tail_k = k - keep_k;

    kNNGraph g(n, k);
    TailLists tail(n, tail_k);

    SplitMix64 rng(p.seed ^ 0xD1B54A32D192ED03ULL);

    build_half_baked_lsh(g1, g2, n1, dist_global, keep_k, g, tail, rng, &idx_s1, &idx_s2, sp);

    int m = (int)std::ceil(p.rho * (float)k);
    if (m < 1) m = 1;
    if (k >= 2 && m < 2) m = 2;
    if (m > k) m = k;

    ReverseBuilder rb(n, m, p.seed ^ 0x9e3779b97f4a7c15ULL);

    uint64_t threshold = (uint64_t)(p.delta * (double)n * (double)k);
    if (threshold < 1) threshold = 1;

    for (int it = 0; it < p.max_iter; ++it) {
        sample_and_build_reverse(g, rb, m, rng);
        uint64_t c = join_and_update_cross(g, rb, m, dist_global, rng, n1);

        if (verbose) {
            std::cerr << "[smerge-lsh iter " << it << "] changes=" << c
                      << " threshold=" << threshold << " m=" << m
                      << " keep_k=" << keep_k << " tail_k=" << tail_k
                      << " L=" << idx_s1.L() << " K=" << idx_s1.K() << "\n";
        }
        if (c < threshold) break;
    }

    finalize_with_tail(g, tail);
    return g;
}
