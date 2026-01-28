#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include "knngraph.h"
#include "reverse_builder.h"
#include "qgt_io.h"
#include "splitmix64.h"
#define RECALL_LEN_MAX 20

struct NNDParams {
    int max_iter = 20;
    float rho = 0.5;
    float delta = 0.001;
    uint64_t seed = 12345;
};

// 反復1回分 sampling + reverse構築
inline void sample_and_build_reverse(kNNGraph& g, ReverseBuilder& rb, int m, SplitMix64& rng) {
    const int k = g.k();
    const int n = g.n();
    rb.reset();

    std::vector<int> sample_pos((size_t)m);

    for (int v = 0; v < n; ++v) {
        auto* nbr  = g.nbr_ptr(v);
        auto* flag = g.flag_ptr(v);

        // SAMPLEDクリア
        for (int t = 0; t < k; ++t) flag[t] = (uint8_t)(flag[t] & ~kNNGraph::SAMPLED);

        // IS_NEW の位置 t を reservoir で m 個選ぶ
        int ssz = 0;
        uint32_t seen = 0;
        for (int t = 0; t < k; ++t) {
            if (flag[t] & kNNGraph::IS_NEW) {
                ++seen;
                if (ssz < m) sample_pos[ssz++] = t;
                else {
                    uint32_t r = rng.uniform_u32(seen); // [0, seen)
                    if (r < (uint32_t)m) sample_pos[r] = t;
                }
            }
        }

        // sampled: SAMPLED=1, IS_NEW=0
        for (int i = 0; i < ssz; ++i) {
            int t = sample_pos[i];
            flag[t] = (uint8_t)((flag[t] | kNNGraph::SAMPLED) & ~kNNGraph::IS_NEW);
        }

        // reverse構築（old/new分離）
        for (int t = 0; t < k; ++t) {
            const auto to = nbr[t];
            if (to == kNNGraph::invalid_id()) continue;

            if (flag[t] & kNNGraph::SAMPLED) {
                rb.push_new(to, (uint32_t)v);
            } else if ((flag[t] & kNNGraph::IS_NEW) == 0) {
                rb.push_old(to, (uint32_t)v);
            } else {
                // IS_NEW=1 だが今回サンプルされなかった → 次反復へ持ち越し
            }
        }
    }
}

// join → update（new-new と new-old）
// ここでは old の行内サンプルも reservoir で m 個に抑える（乱数は rng を使う）
template<class Dist>
uint64_t join_and_update(kNNGraph& g, const ReverseBuilder& rb,int m, const Dist& dist, SplitMix64& rng) {
    const int k = g.k();
    const int n = g.n();

    std::vector<uint32_t> new_list;
    std::vector<uint32_t> old_list;
    new_list.reserve((size_t)(k + 2*m + rb.cap() + 8));
    old_list.reserve((size_t)(k + 2*m + rb.cap() + 8));

    std::vector<uint32_t> old_buf((size_t)m);

    uint64_t changes = 0;

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

        // old_list: 行内の old から m 個を reservoir でサンプル
        int osz = 0;
        uint32_t seen = 0;
        for (int t = 0; t < k; ++t) {
            if ((flag[t] & (kNNGraph::IS_NEW | kNNGraph::SAMPLED)) == 0) {
                uint32_t id = nbr[t];
                if (id == kNNGraph::invalid_id()) continue;

                ++seen;
                if (osz < m) old_buf[osz++] = id;
                else {
                    uint32_t r = rng.uniform_u32(seen); // [0,seen)
                    if (r < (uint32_t)m) old_buf[r] = id;
                }
            }
        }
        for (int i = 0; i < osz; ++i) old_list.push_back(old_buf[i]);

        // reverse 由来を追加（rb 自体が cap で抑えられている）
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

        // 小さいリストなので sort+unique で重複を軽く除去（安定化）
        std::sort(new_list.begin(), new_list.end());
        new_list.erase(std::unique(new_list.begin(), new_list.end()), new_list.end());
        std::sort(old_list.begin(), old_list.end());
        old_list.erase(std::unique(old_list.begin(), old_list.end()), old_list.end());

        // join: new-new & new-old
        const int nn = (int)new_list.size();
        const int on = (int)old_list.size();

        for (int a = 0; a < nn; ++a) {
            uint32_t u1 = new_list[a];
            if (u1 == kNNGraph::invalid_id()) continue;

            for (int b = a + 1; b < nn; ++b) {
                uint32_t u2 = new_list[b];
                if (u2 == kNNGraph::invalid_id() || u2 == u1) continue;

                float d = dist((int)u1, (int)u2);
                if (g.update((int)u1, u2, d)) ++changes;
                if (g.update((int)u2, u1, d)) ++changes;
            }

            for (int b = 0; b < on; ++b) {
                uint32_t u2 = old_list[b];
                if (u2 == kNNGraph::invalid_id() || u2 == u1) continue;

                float d = dist((int)u1, (int)u2);
                if (g.update((int)u1, u2, d)) ++changes;
                if (g.update((int)u2, u1, d)) ++changes;
            }
        }
    }

    return changes;
}

// NN-Descent full（initializer差し替え可能）
template<class Dist, class Initializer>
kNNGraph nndescent_full(int k, int n, const Dist& dist, const Initializer& init, const NNDParams& p) {
    kNNGraph g(k, n);
    SplitMix64 rng(p.seed);

    // 初期化（ランダム／後でLSHに差し替え）
    init(g, dist, rng);

    // m = ceil(rho*k)
    int m = (int)std::ceil(p.rho * (float)k);
    if (m < 1) m = 1;
    // 小さいkで退化しないように下限（k>=2ならm>=2）
    if (k >= 2 && m < 2) m = 2;
    if (m > k) m = k;

    ReverseBuilder rb(n, m, p.seed ^ 0x9e3779b97f4a7c15ULL);

    // 停止閾値：delta * n * k
    uint64_t threshold = (uint64_t)(p.delta * (double)n * (double)k);
    if (threshold < 1) threshold = 1;

    for (int it = 0; it < p.max_iter; ++it) {
        sample_and_build_reverse(g, rb, m, rng);
        uint64_t c = join_and_update(g, rb, m, dist, rng);

        if (c < threshold) break;
    }

    return g;
}

template<class Dist, class Initializer>
kNNGraph nndescent_full(int k, int n, const Dist& dist, const Initializer& init, const NNDParams& p, const recallParams& rp) {
    kNNGraph g(k, n);
    SplitMix64 rng(p.seed);
    // Recall vs iter times 用
    std::vector<double> recall(RECALL_LEN_MAX, -1);
    int iter_times = 0;

    // 初期化（ランダム／後でLSHに差し替え）
    init(g, dist, rng);

    // m = ceil(rho*k)
    int m = (int) std::ceil(p.rho * (float) k);
    if (m < 1) m = 1;
    // 小さいkで退化しないように下限（k>=2ならm>=2）
    if (k >= 2 && m < 2) m = 2;
    if (m > k) m = k;

    ReverseBuilder rb(n, m, p.seed ^ 0x9e3779b97f4a7c15ULL);

    // 停止閾値：delta * n * k
    uint64_t threshold = (uint64_t)(p.delta * (double) n * (double) k);
    if (threshold < 1) threshold = 1;

    for (iter_times = 0; iter_times < p.max_iter; ++iter_times) {
        sample_and_build_reverse(g, rb, m, rng);
        uint64_t c = join_and_update(g, rb, m, dist, rng);

        // 各回recallを計算して表示
        if (iter_times >= RECALL_LEN_MAX) recall.resize(2 * RECALL_LEN_MAX);
        recall[iter_times] = calc_recall(g, k, rp.eval_n, rp.eval_q, rp.gt_path, rp.qgt_path);
        // std::cerr << "Im " << iter_times << " of repetition\n";

        if (c < threshold) {
            iter_times++;
            break;
        }
    }

    // std::cerr << "iter_times is " << iter_times << "\n";

    std::cout
        << "iter times vs recall : iter_times = "
        << iter_times
        << "\n"
        << "iter-recall: ";
    for (int i = 0; i < iter_times; ++i) {
        std::cout << recall[i];
        if (i < iter_times - 1) std::cout << ", ";
    }
    std::cout << std::endl;

    return g;
}