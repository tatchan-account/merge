#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>

#include "knngraph.h"
#include "reverse_builder.h"
#include "splitmix64.h"
#include "nndescent.h"

struct TailLists {
    using id_t   = kNNGraph::id_t;
    using dist_t = kNNGraph::dist_t;

    int n = 0;
    int tail_k = 0;
    std::vector<id_t>   nbr;   // n * tail_k
    std::vector<dist_t> dist;  // n * tail_k

    TailLists() = default;
    TailLists(int n_, int tail_k_)
            : n(n_), tail_k(tail_k_),
              nbr((size_t)n_ * (size_t)tail_k_, kNNGraph::invalid_id()),
              dist((size_t)n_ * (size_t)tail_k_, kNNGraph::inf()) {}

    size_t base(int i) const { return (size_t)i * (size_t)tail_k; }

    id_t* nbr_ptr(int i) { return nbr.data() + base(i); }
    dist_t* dist_ptr(int i) { return dist.data() + base(i); }
    const id_t* nbr_ptr(int i) const { return nbr.data() + base(i); }
    const dist_t* dist_ptr(int i) const { return dist.data() + base(i); }
};

// S-Merge: cross-only join_and_update
// (NN-Descent の join_and_update をほぼコピーして、cross 判定だけ追加)
template<class Dist>
uint64_t join_and_update_cross(kNNGraph& g,
                               const ReverseBuilder& rb,
                               int m,
                               const Dist& dist,
                               SplitMix64& rng,
                               int n1,
                               bool sort_unique = true) {
    const int n = g.n();
    const int k = g.k();

    std::vector<uint32_t> new_list;
    std::vector<uint32_t> old_list;
    new_list.reserve((size_t)(k + 2*m + rb.cap() + 8));
    old_list.reserve((size_t)(k + 2*m + rb.cap() + 8));

    std::vector<uint32_t> old_buf((size_t)m);

    uint64_t changes = 0;

    auto in_s1 = [&](uint32_t id) -> bool {
        return id < (uint32_t)n1;
    };

    for (int v = 0; v < n; ++v) {
        new_list.clear();
        old_list.clear();

        const auto* nbr  = g.nbr_ptr(v);
        const auto* flag = g.flag_ptr(v);

        // new_list: 行内の SAMPLED
        for (int t = 0; t < k; ++t) {
            if (flag[t] & kNNGraph::SAMPLED) {
                uint32_t id = nbr[t];
                if (id != kNNGraph::invalid_id()) new_list.push_back(id);
            }
        }

        // old_list: 行内 old から m 個 reservoir
        int osz = 0;
        uint32_t seen = 0;
        for (int t = 0; t < k; ++t) {
            if ((flag[t] & (kNNGraph::IS_NEW | kNNGraph::SAMPLED)) == 0) {
                uint32_t id = nbr[t];
                if (id == kNNGraph::invalid_id()) continue;

                ++seen;
                if (osz < m) old_buf[osz++] = id;
                else {
                    uint32_t r = rng.uniform_u32(seen);
                    if (r < (uint32_t)m) old_buf[r] = id;
                }
            }
        }
        for (int i = 0; i < osz; ++i) old_list.push_back(old_buf[i]);

        // reverse 由来を追加
        {
            const uint32_t* p = rb.new_ptr(v);
            const int sz = (int)rb.new_size(v);
            for (int i = 0; i < sz; ++i) new_list.push_back(p[i]);
        }
        {
            const uint32_t* p = rb.old_ptr(v);
            const int sz = (int)rb.old_size(v);
            for (int i = 0; i < sz; ++i) old_list.push_back(p[i]);
        }

        if (sort_unique) {
            std::sort(new_list.begin(), new_list.end());
            new_list.erase(std::unique(new_list.begin(), new_list.end()), new_list.end());
            std::sort(old_list.begin(), old_list.end());
            old_list.erase(std::unique(old_list.begin(), old_list.end()), old_list.end());
        }

        // join: new-new & new-old (ただし cross-only)
        const int nn = (int)new_list.size();
        const int on = (int)old_list.size();

        for (int a = 0; a < nn; ++a) {
            uint32_t u1 = new_list[a];
            if (u1 == kNNGraph::invalid_id()) continue;

            for (int b = a + 1; b < nn; ++b) {
                uint32_t u2 = new_list[b];
                if (u2 == kNNGraph::invalid_id() || u2 == u1) continue;

                // ★ cross-only
                if (in_s1(u1) == in_s1(u2)) continue;

                float d = dist((int)u1, (int)u2);
                if (g.update((int)u1, u2, d)) ++changes;
                if (g.update((int)u2, u1, d)) ++changes;
            }

            for (int b = 0; b < on; ++b) {
                uint32_t u2 = old_list[b];
                if (u2 == kNNGraph::invalid_id() || u2 == u1) continue;

                // ★ cross-only
                if (in_s1(u1) == in_s1(u2)) continue;

                float d = dist((int)u1, (int)u2);
                if (g.update((int)u1, u2, d)) ++changes;
                if (g.update((int)u2, u1, d)) ++changes;
            }
        }
    }

    return changes;
}

// 内部: row を dist 昇順に並べ替えた index を作る（k が小さいので O(k log k)）
inline std::vector<int> sorted_indices_by_dist(const kNNGraph::dist_t* ds, int k) {
    std::vector<int> idx(k);
    for (int i = 0; i < k; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return ds[a] < ds[b];
    });
    return idx;
}

// half-baked graph を作り、tail を保存する
//  - keep_k 個残す（old 扱い）
//  - 残りを別集合からランダムで埋める（IS_NEW）
template<class Dist>
static inline void build_half_baked(const kNNGraph& g1,
                                    const kNNGraph& g2,
                                    int n1,
                                    const Dist& dist_global,
                                    int keep_k,
                                    kNNGraph& out,
                                    TailLists& tail,
                                    SplitMix64& rng) {
    const int n  = out.n();
    const int k  = out.k();
    const int n2 = n - n1;
    const int tail_k = k - keep_k;

    auto fill_random_from_other = [&](int v_global, bool v_in_s1,
                                      kNNGraph::id_t* nbr, kNNGraph::dist_t* ds, uint8_t* flg,
                                      int start_pos) {
        // v_in_s1=true なら相手集合は S2 [n1, n)
        // v_in_s1=false なら相手集合は S1 [0, n1)
        const int other_n = v_in_s1 ? n2 : n1;
        const int other_off = v_in_s1 ? n1 : 0;

        for (int t = start_pos; t < k; ++t) {
            // 小規模データで other_n が小さいと unique を作れないので、試行回数で打ち切る
            bool placed = false;
            for (int tries = 0; tries < 2000; ++tries) {
                if (other_n <= 0) break;
                uint32_t r = rng.uniform_u32((uint32_t)other_n);
                uint32_t id = (uint32_t)other_off + r;

                if ((uint32_t)v_global == id) continue;

                // duplicate check in row
                bool dup = false;
                for (int u = 0; u < t; ++u) {
                    if (nbr[u] == id) { dup = true; break; }
                }
                if (dup) continue;

                nbr[t] = id;
                ds[t]  = dist_global(v_global, (int)id);
                flg[t] = kNNGraph::IS_NEW; // ★ new edge
                placed = true;
                break;
            }
            if (!placed) {
                nbr[t] = kNNGraph::invalid_id();
                ds[t]  = kNNGraph::inf();
                flg[t] = 0;
            }
        }
    };

    // S1側 (g1)
    for (int i = 0; i < n1; ++i) {
        int v_global = i;

        auto* out_nbr = out.nbr_ptr(v_global);
        auto* out_ds  = out.dist_ptr(v_global);
        auto* out_fl  = out.flag_ptr(v_global);

        // dist 昇順 index
        const auto* src_nbr = g1.nbr_ptr(i);
        const auto* src_ds  = g1.dist_ptr(i);
        auto idx = sorted_indices_by_dist(src_ds, k);

        // keep
        for (int t = 0; t < keep_k; ++t) {
            int j = idx[t];
            uint32_t nb_local = src_nbr[j];
            out_nbr[t] = nb_local;          // offset 0
            out_ds[t]  = src_ds[j];
            out_fl[t]  = 0;                 // old
        }
        // tail 保存
        if (tail_k > 0) {
            auto* tn = tail.nbr_ptr(v_global);
            auto* td = tail.dist_ptr(v_global);
            for (int t = 0; t < tail_k; ++t) {
                int j = idx[keep_k + t];
                uint32_t nb_local = src_nbr[j];
                tn[t] = nb_local;
                td[t] = src_ds[j];
            }
        }

        // random from S2
        fill_random_from_other(v_global, true, out_nbr, out_ds, out_fl, keep_k);

        out.recompute_worst_row(v_global);
    }

    // S2側 (g2)
    for (int i = 0; i < n2; ++i) {
        int v_global = n1 + i;

        auto* out_nbr = out.nbr_ptr(v_global);
        auto* out_ds  = out.dist_ptr(v_global);
        auto* out_fl  = out.flag_ptr(v_global);

        const auto* src_nbr = g2.nbr_ptr(i);
        const auto* src_ds  = g2.dist_ptr(i);
        auto idx = sorted_indices_by_dist(src_ds, k);

        // keep
        for (int t = 0; t < keep_k; ++t) {
            int j = idx[t];
            uint32_t nb_local = src_nbr[j];
            out_nbr[t] = (uint32_t)n1 + nb_local; // offset n1
            out_ds[t]  = src_ds[j];
            out_fl[t]  = 0;                       // old
        }
        // tail 保存
        if (tail_k > 0) {
            auto* tn = tail.nbr_ptr(v_global);
            auto* td = tail.dist_ptr(v_global);
            for (int t = 0; t < tail_k; ++t) {
                int j = idx[keep_k + t];
                uint32_t nb_local = src_nbr[j];
                tn[t] = (uint32_t)n1 + nb_local;
                td[t] = src_ds[j];
            }
        }

        // random from S1
        fill_random_from_other(v_global, false, out_nbr, out_ds, out_fl, keep_k);

        out.recompute_worst_row(v_global);
    }
}

// 最後に tail を戻して top-k を作り直す（merge sort の代替：k が小さいので sort でOK）
static inline void finalize_with_tail(kNNGraph& g, const TailLists& tail) {
    const int n = g.n();
    const int k = g.k();
    const int tail_k = tail.tail_k;

    std::vector<std::pair<kNNGraph::id_t, kNNGraph::dist_t>> cand;
    cand.reserve((size_t)k + (size_t)tail_k);

    for (int v = 0; v < n; ++v) {
        cand.clear();

        // 現在の g
        {
            const auto* nb = g.nbr_ptr(v);
            const auto* ds = g.dist_ptr(v);
            for (int t = 0; t < k; ++t) {
                if (nb[t] == kNNGraph::invalid_id()) continue;
                cand.emplace_back(nb[t], ds[t]);
            }
        }
        // tail
        if (tail_k > 0) {
            const auto* nb = tail.nbr_ptr(v);
            const auto* ds = tail.dist_ptr(v);
            for (int t = 0; t < tail_k; ++t) {
                if (nb[t] == kNNGraph::invalid_id()) continue;
                cand.emplace_back(nb[t], ds[t]);
            }
        }

        // id でまとめて最小 dist を残す（小さいので sort）
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) {
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        });
        // unique by id (keep min dist)
        size_t w = 0;
        for (size_t r = 0; r < cand.size(); ++r) {
            if (w == 0 || cand[r].first != cand[w-1].first) {
                cand[w++] = cand[r];
            } else {
                // same id, keep smaller dist
                if (cand[r].second < cand[w-1].second) cand[w-1].second = cand[r].second;
            }
        }
        cand.resize(w);

        // sort by dist
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) {
            if (a.second != b.second) return a.second < b.second;
            return a.first < b.first;
        });

        auto* out_nb = g.nbr_ptr(v);
        auto* out_ds = g.dist_ptr(v);
        auto* out_fl = g.flag_ptr(v);

        // fill top-k
        int sz = (int)cand.size();
        int lim = (sz < k) ? sz : k;
        for (int t = 0; t < lim; ++t) {
            out_nb[t] = cand[t].first;
            out_ds[t] = cand[t].second;
            out_fl[t] = 0;
        }
        for (int t = lim; t < k; ++t) {
            out_nb[t] = kNNGraph::invalid_id();
            out_ds[t] = kNNGraph::inf();
            out_fl[t] = 0;
        }
        g.recompute_worst_row(v);
    }
}

// S-Merge 本体
//  - keep_ratio は論文だと [k/5, k/2] が良いとされています（経験的）
template<class Dist>
kNNGraph s_merge(const kNNGraph& g1,
                 const kNNGraph& g2,
                 int n1,
                 const Dist& dist_global,
                 const NNDParams& p,
                 float keep_ratio = 0.5f,
                 bool verbose = false) {
    const int k = g1.k();
    const int n = g1.n() + g2.n();
    const int n2 = n - n1;

    if (g2.k() != k) throw std::runtime_error("S-Merge: k mismatch between g1 and g2.");
    if (g1.n() != n1) throw std::runtime_error("S-Merge: n1 mismatch.");
    if (n1 <= k || n2 <= k) {
        throw std::runtime_error("S-Merge: each subset must satisfy subset_size > k (for now).");
    }

    // keep_k を決める
    int keep_k = (int)std::floor(keep_ratio * (float)k);
    if (keep_k < 1) keep_k = 1;
    if (keep_k > k - 1) keep_k = k - 1;

    // 論文では keep の経験範囲を [k/5, k/2]
    // 「うっかり keep が大きすぎ/小さすぎ」を防ぐだけのガード
    int lo = std::max(1, k / 5);
    int hi = std::max(1, k / 2);
    if (keep_k < lo) keep_k = lo;
    if (keep_k > hi) keep_k = hi;

    const int tail_k = k - keep_k;

    kNNGraph g(n, k);
    TailLists tail(n, tail_k);

    SplitMix64 rng(p.seed ^ 0xD1B54A32D192ED03ULL);

    // half-baked graph 構築
    build_half_baked(g1, g2, n1, dist_global, keep_k, g, tail, rng);

    // NN-Descent のパラメータ（m, threshold）は既存の nndescent_fullと同じ
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
            std::cerr << "[smerge iter " << it << "] changes=" << c
                      << " threshold=" << threshold << " m=" << m
                      << " keep_k=" << keep_k << " tail_k=" << tail_k << "\n";
        }
        if (c < threshold) break;
    }

    // 最後に tail を戻して top-k を作り直す
    finalize_with_tail(g, tail);

    return g;
}
