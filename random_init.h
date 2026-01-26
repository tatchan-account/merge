#pragma once
#include "knngraph.h"
#include "splitmix64.h"

// initは: init(graph, dist, rng) で呼べるようにする
struct RandomInit {
    template<class Dist>
    void operator()(kNNGraph& g, const Dist& dist, SplitMix64& rng) const {
        const int n = g.n();
        const int k = g.k();

        for (int i = 0; i < n; ++i) {
            auto* nbr  = g.nbr_ptr(i);
            auto* ds   = g.dist_ptr(i);
            auto* flg  = g.flag_ptr(i);

            // まずランダムに埋める（重複は極稀なので後で軽く修正）
            for (int t = 0; t < k; ++t) {
                // i を避けて [0,n) をサンプル
                uint32_t r = rng.uniform_u32((uint32_t)(n - 1));
                uint32_t id = (r >= (uint32_t)i) ? (r + 1) : r;

                nbr[t] = id;
                ds[t]  = dist(i, (int)id);
                flg[t] = kNNGraph::IS_NEW;
            }

            // 重複があったらその場で差し替え（通常ほぼ発生しない想定）
            for (int t = 0; t < k; ++t) {
                for (int u = 0; u < t; ++u) {
                    if (nbr[t] == nbr[u]) {
                        // リサンプル（重複がなくなるまで）
                        while (true) {
                            uint32_t r = rng.uniform_u32((uint32_t)(n - 1));
                            uint32_t id = (r >= (uint32_t)i) ? (r + 1) : r;
                            bool ok = true;
                            for (int z = 0; z < t; ++z) if (nbr[z] == id) { ok = false; break; }
                            if (ok) {
                                nbr[t] = id;
                                ds[t]  = dist(i, (int)id);
                                flg[t] = kNNGraph::IS_NEW;
                                break;
                            }
                        }
                    }
                }
            }

            g.recompute_worst_row(i);
        }
    }
};
