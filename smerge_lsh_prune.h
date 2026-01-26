#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <utility>

#include "smerge.h"        // TailLists / join_and_update_cross / finalize_with_tail
#include "lsh_pstable.h"  // PStableLSHIndex
#include "proj_prune.h"   // ProjectionPruner

// ------------------------------------------------------------
// LSH-enhanced S-Merge with projection-distance pruning.
//
// Pruning is *safe* for squared L2, because ProjectionPruner provides
// a lower bound LB^2 <= true_dist^2 (Cauchy-Schwarz).
//
// During LSH seed selection for each row, when we already have
// `seed_fill` accepted candidates with current worst squared distance
// `worst_sq`, we skip computing the true distance for a candidate (v,u)
// if LB(v,u)^2 >= worst_sq.
// ------------------------------------------------------------

struct SMergeLSHSeedPruneParams {
    bool enable_lsh = true;

    // LSH candidate controls
    int cand_cap = 200;     // max candidates per row gathered from LSH buckets
    int rand_min = 2;       // minimum random seeds when seed_k < 0
    int seed_k = -1;        // if >=0: exact #LSH seeds, else slots-rand_min

    // pruning
    bool enable_prune = true;

    // ------------------------------------------------------------
    // A案: "低次元(射影)距離" で LSH 候補を先に絞り込んでから
    // 真の距離(dist_global)を計算する。
    //
    // - enable_prefilter=true かつ pruner!=nullptr のとき有効
    // - prefilter_mult * seed_fill 個だけ残して真の距離を計算する
    //   (0/負の値なら無効)
    // - prefilter_min / prefilter_max は M の下限/上限
    // ------------------------------------------------------------
    bool enable_prefilter = false;
    int  prefilter_mult = 4;   // M = prefilter_mult * seed_fill
    int  prefilter_min  = 0;   // M >= prefilter_min
    int  prefilter_max  = 0;   // if >0: M <= prefilter_max
};

struct LSHSeedStats {
    uint64_t cand_total = 0;
    uint64_t prefilter_total = 0; // candidates entering prefilter ranking
    uint64_t prefilter_kept  = 0; // candidates kept after prefilter
    uint64_t pruned_by_lb = 0;
    uint64_t full_dist = 0;
};

namespace smerge_detail {

    static inline bool contains_id_prefix(const uint32_t* nbr, int upto, uint32_t id) {
        for (int i = 0; i < upto; ++i) if (nbr[i] == id) return true;
        return false;
    }

// Keep the best `cap` by distance using a max-heap.
    static inline void heap_push_best(std::vector<std::pair<float, uint32_t>>& heap,
                                      int cap,
                                      float dist,
                                      uint32_t id) {
        auto cmp = [](const auto& a, const auto& b){ return a.first < b.first; }; // max-heap by dist
        if ((int)heap.size() < cap) {
            heap.emplace_back(dist, id);
            std::push_heap(heap.begin(), heap.end(), cmp);
        } else {
            if (dist >= heap.front().first) return;
            std::pop_heap(heap.begin(), heap.end(), cmp);
            heap.back() = {dist, id};
            std::push_heap(heap.begin(), heap.end(), cmp);
        }
    }

} // namespace smerge_detail

template<class Dist>
static inline void build_half_baked_lsh_pruned(const KNNGraph& g1,
                                               const KNNGraph& g2,
                                               int n1,
                                               const Dist& dist_global,          // returns squared L2 (float)
                                               int keep_k,
                                               KNNGraph& out,
                                               TailLists& tail,
                                               SplitMix64& rng,
                                               const PStableLSHIndex* idx_s1,
                                               const PStableLSHIndex* idx_s2,
                                               const ProjectionPruner* pruner,   // may be nullptr
                                               const SMergeLSHSeedPruneParams& sp,
                                               LSHSeedStats* stats) {
    const int n  = out.n();
    const int k  = out.k();
    const int n2 = n - n1;
    const int tail_k = k - keep_k;

    auto fill_lsh_then_random = [&](int v_global, bool v_in_s1,
                                    uint32_t* nbr, float* ds, uint8_t* flg) {
        const int slots = k - keep_k;
        if (slots <= 0) return;

        int seed_fill = 0;
        if (!sp.enable_lsh) {
            seed_fill = 0;
        } else if (sp.seed_k >= 0) {
            seed_fill = std::min(sp.seed_k, slots);
        } else {
            seed_fill = std::max(0, slots - sp.rand_min);
        }

        int pos = keep_k;

        // 1) LSH seeds (best seed_fill by true distance)
        //
        // A案(推奨): LSH候補 cand から "射影空間の距離" で上位M個に絞り、
        // 真の距離(dist_global)はそのM個だけ計算する。
        if (seed_fill > 0 && sp.enable_lsh) {
            const PStableLSHIndex* other = v_in_s1 ? idx_s2 : idx_s1;
            if (other) {
                std::vector<uint32_t> cand;
                other->query(v_global, sp.cand_cap, cand, rng);

                // ---- 1-a) filter candidates (cross only, no duplicates vs keep)
                std::vector<uint32_t> valid;
                valid.reserve(cand.size());
                for (uint32_t id : cand) {
                    if ((uint32_t)v_global == id) continue;
                    if (v_in_s1) { if (id < (uint32_t)n1) continue; }
                    else         { if (id >= (uint32_t)n1) continue; }
                    if (smerge_detail::contains_id_prefix(nbr, keep_k, id)) continue;

                    bool dup = false;
                    for (uint32_t u : valid) { if (u == id) { dup = true; break; } }
                    if (dup) continue;

                    valid.push_back(id);
                    if (stats) stats->cand_total++;
                }

                // ---- 1-b) optional prefilter by projection-space distance
                // Keep only top-M candidates by approx_dist_sq.
                // This reduces expensive dist_global computations.
                std::vector<uint32_t> shortlist;
                shortlist = valid;
                if (sp.enable_prefilter && pruner && sp.prefilter_mult > 0 && seed_fill > 0) {
                    int M = sp.prefilter_mult * seed_fill;
                    if (sp.prefilter_min > 0) M = std::max(M, sp.prefilter_min);
                    if (sp.prefilter_max > 0) M = std::min(M, sp.prefilter_max);
                    if (M < 1) M = 1;
                    if (M > (int)shortlist.size()) M = (int)shortlist.size();

                    if (stats) stats->prefilter_total += (uint64_t)shortlist.size();

                    // compute approx distance
                    std::vector<std::pair<float, uint32_t>> approx;
                    approx.reserve(shortlist.size());
                    for (uint32_t id : shortlist) {
                        float ad = pruner->approx_dist_sq(v_global, (int)id);
                        approx.emplace_back(ad, id);
                    }

                    auto cmp = [](const auto& a, const auto& b){
                        if (a.first != b.first) return a.first < b.first;
                        return a.second < b.second;
                    };

                    if ((int)approx.size() > M) {
                        std::nth_element(approx.begin(), approx.begin() + M, approx.end(), cmp);
                        approx.resize((size_t)M);
                    }
                    std::sort(approx.begin(), approx.end(), cmp);

                    shortlist.clear();
                    shortlist.reserve(approx.size());
                    for (auto& e : approx) shortlist.push_back(e.second);
                    if (stats) stats->prefilter_kept += (uint64_t)shortlist.size();
                }

                // ---- 1-c) exact scoring (true distance) with optional safe pruning
                std::vector<std::pair<float, uint32_t>> heap; // max-heap by true dist
                heap.reserve((size_t)seed_fill);

                for (uint32_t id : shortlist) {
                    // safe prune using lower bound if heap already full
                    if (sp.enable_prune && pruner && (int)heap.size() >= seed_fill) {
                        float worst_sq = heap.front().first;
                        float lb_sq = pruner->lower_bound_sq(v_global, (int)id);
                        if (lb_sq >= worst_sq) {
                            if (stats) stats->pruned_by_lb++;
                            continue;
                        }
                    }

                    float d = dist_global(v_global, (int)id);
                    if (stats) stats->full_dist++;
                    smerge_detail::heap_push_best(heap, seed_fill, d, id);
                }

                std::sort(heap.begin(), heap.end(), [](auto& a, auto& b){
                    if (a.first != b.first) return a.first < b.first;
                    return a.second < b.second;
                });

                for (auto& e : heap) {
                    if (pos >= k) break;
                    uint32_t id = e.second;
                    if (smerge_detail::contains_id_prefix(nbr, pos, id)) continue;
                    nbr[pos] = id;
                    ds[pos]  = e.first;
                    flg[pos] = KNNGraph::IS_NEW;
                    ++pos;
                }
            }
        }

        // 2) Random fill remaining
        const int other_n   = v_in_s1 ? n2 : n1;
        const int other_off = v_in_s1 ? n1 : 0;

        for (; pos < k; ++pos) {
            bool placed = false;
            for (int tries = 0; tries < 2000; ++tries) {
                if (other_n <= 0) break;
                uint32_t r = rng.uniform_u32((uint32_t)other_n);
                uint32_t id = (uint32_t)other_off + r;
                if ((uint32_t)v_global == id) continue;

                if (smerge_detail::contains_id_prefix(nbr, pos, id)) continue;

                nbr[pos] = id;
                ds[pos]  = dist_global(v_global, (int)id);
                flg[pos] = KNNGraph::IS_NEW;
                placed = true;
                break;
            }
            if (!placed) {
                nbr[pos] = KNNGraph::invalid_id();
                ds[pos]  = KNNGraph::inf();
                flg[pos] = 0;
            }
        }
    };

    // ---- S1 ----
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

        fill_lsh_then_random(v_global, true, out_nbr, out_ds, out_fl);
        out.recompute_worst_row(v_global);
    }

    // ---- S2 ----
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

        fill_lsh_then_random(v_global, false, out_nbr, out_ds, out_fl);
        out.recompute_worst_row(v_global);
    }
}

template<class Dist>
KNNGraph s_merge_lsh_pruned(const KNNGraph& g1,
                            const KNNGraph& g2,
                            int n1,
                            const Dist& dist_global,                 // squared L2
                            const NNDParams& p,
                            float keep_ratio,
                            const PStableLSHIndex& idx_s1,
                            const PStableLSHIndex& idx_s2,
                            const ProjectionPruner* pruner,          // may be nullptr
                            const SMergeLSHSeedPruneParams& sp,
                            LSHSeedStats* stats_out = nullptr,
                            bool verbose = false) {
    const int k = g1.k();
    const int n = g1.n() + g2.n();
    const int n2 = n - n1;

    if (g2.k() != k) throw std::runtime_error("S-Merge(LSH+Prune): k mismatch between g1 and g2.");
    if (g1.n() != n1) throw std::runtime_error("S-Merge(LSH+Prune): n1 mismatch.");
    if (n1 <= k || n2 <= k) throw std::runtime_error("S-Merge(LSH+Prune): each subset must satisfy subset_size > k.");

    int keep_k = (int)std::floor(keep_ratio * (float)k);
    if (keep_k < 1) keep_k = 1;
    if (keep_k > k - 1) keep_k = k - 1;

    int lo = std::max(1, k / 5);
    int hi = std::max(1, k / 2);
    if (keep_k < lo) keep_k = lo;
    if (keep_k > hi) keep_k = hi;

    const int tail_k = k - keep_k;

    KNNGraph g(n, k);
    TailLists tail(n, tail_k);

    SplitMix64 rng(p.seed ^ 0xD1B54A32D192ED03ULL);

    LSHSeedStats local_stats{};
    build_half_baked_lsh_pruned(g1, g2, n1, dist_global, keep_k, g, tail, rng,
                                &idx_s1, &idx_s2, pruner, sp, &local_stats);

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
            std::cerr << "[smerge-lsh-prune iter " << it << "] changes=" << c
                      << " threshold=" << threshold << " m=" << m
                      << " keep_k=" << keep_k << " tail_k=" << tail_k
                      << " L=" << idx_s1.L() << " K=" << idx_s1.K()
                      << " prune=" << (sp.enable_prune && pruner ? "on" : "off")
                      << "\n";
        }
        if (c < threshold) break;
    }

    finalize_with_tail(g, tail);

    if (stats_out) *stats_out = local_stats;
    return g;
}
