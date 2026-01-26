// sample_and_build_reverse.hpp
#pragma once
#include <vector>
#include "knngraph.h"
#include "reverse_builder.h"
#include "splitmix64.h"

// 1反復ぶんの「サンプル選択 + reverse(old/new)構築」
// - graph.flags の SAMPLED を更新
// - sampled は IS_NEW を落とす（次の反復では old 扱い）
// - reverse は old/new に分けて詰める
inline void sample_and_build_reverse(KNNGraph& g,
                                     ReverseBuilder& rb,
                                     int m,           // m = ceil(rho*k) = rev_cap 推奨
                                     SplitMix64& rng) {
    const int n = g.n();
    const int k = g.k();
    rb.reset();

    std::vector<int> sample_pos;
    sample_pos.resize((size_t)m);

    for (int v = 0; v < n; ++v) {
        const size_t b = g.base(v);
        auto* nbr  = g.nbr_ptr(v);
        auto* flag = g.flag_ptr(v);

        // まず SAMPLED をクリア
        for (int t = 0; t < k; ++t) flag[t] = (uint8_t)(flag[t] & ~KNNGraph::SAMPLED);

        // IS_NEW のスロットから m 個を reservoir で選ぶ（位置 t をサンプル）
        int ssz = 0;
        uint32_t seen = 0;
        for (int t = 0; t < k; ++t) {
            if (flag[t] & KNNGraph::IS_NEW) {
                ++seen;
                if (ssz < m) sample_pos[ssz++] = t;
                else {
                    uint32_t r = rng.uniform_u32(seen); // [0, seen)
                    if (r < (uint32_t)m) sample_pos[r] = t;
                }
            }
        }

        // サンプルされたスロットに SAMPLED を立て、IS_NEW を落とす（論文の動き）
        for (int i = 0; i < ssz; ++i) {
            int t = sample_pos[i];
            flag[t] = (uint8_t)((flag[t] | KNNGraph::SAMPLED) & ~KNNGraph::IS_NEW);
        }

        // reverse を構築（old/new 分離）
        for (int t = 0; t < k; ++t) {
            const auto to = nbr[t];
            if (to == std::numeric_limits<KNNGraph::id_t>::max()) continue;

            if (flag[t] & KNNGraph::SAMPLED) {
                rb.push_new(to, (uint32_t)v);
            } else if ((flag[t] & KNNGraph::IS_NEW) == 0) {
                rb.push_old(to, (uint32_t)v);
            } else {
                // new だが今回サンプルされなかったもの：この反復では使わない（次の反復へ）
            }
        }
    }
}
