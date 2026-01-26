#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

#include "dataset.h"
#include "knngraph.h"
#include "splitmix64.h"
#include "lsh_pstable.h"
#include "proj_prune.h"

// LSH-based NN-Descent initializer with optional projection lower-bound pruning.
//
// Drop-in replacement for RandomInit:
//   init(graph, dist, rng)
//
// Two modes:
//   - rank_by_distance=false (default):
//        take `seed_fill` ids uniformly from the LSH candidate set and compute
//        true distances only for those selected ids. (cheap)
//   - rank_by_distance=true:
//        scan the whole candidate set and keep the best `seed_fill` by true distance.
//        If pruner!=nullptr, skip true distance when LB^2 >= current worst among kept.

struct NNDLSHInitPrunedParams {
    bool enable = true;

    int cand_cap = 128;        // query cap after union+dedup

    int seed_k = -1;           // >=0 exact LSH seeds, <0 use (k - rand_min)
    int rand_min = 2;          // minimum random fill count when seed_k < 0

    bool rank_by_distance = false;

    bool enable_prune = false; // only effective when rank_by_distance=true
    ProjPruneParams prune{};

    PStableLSHParams lsh{};
};

struct LSHInitSeedStats {
    uint64_t cand_total = 0;      // total candidates observed (after query cap)
    uint64_t pruned_by_lb = 0;    // skipped by LB
    uint64_t full_dist = 0;       // true distance computations
};

class LSHInitPruned {
public:
    LSHInitPruned(const Dataset& ds, const NNDLSHInitPrunedParams& ip)
        : ds_(&ds), idx_(ds, 0, ds.n, ip.lsh), ip_(ip) {
        if (ip_.enable_prune) {
            pr_ = ProjectionPruner(ds, ip_.prune);
        }
    }

    template<class Dist>
    void operator()(KNNGraph& g, const Dist& dist, SplitMix64& rng) const {
        if (!ds_) throw std::runtime_error("LSHInitPruned: dataset missing");
        if (g.n() != ds_->n) throw std::runtime_error("LSHInitPruned: graph.n != dataset.n");

        const int n = g.n();
        const int k = g.k();

        std::vector<uint32_t> cand;
        cand.reserve((size_t)std::max(8, ip_.cand_cap));

        // heap of (dist, id) where top is the worst among kept
        std::vector<std::pair<float, uint32_t>> heap;
        heap.reserve((size_t)k);

        for (int i = 0; i < n; ++i) {
            auto* nbr = g.nbr_ptr(i);
            auto* ds  = g.dist_ptr(i);
            auto* flg = g.flag_ptr(i);

            int seed_fill = 0;
            if (!ip_.enable) {
                seed_fill = 0;
            } else if (ip_.seed_k >= 0) {
                seed_fill = std::min(ip_.seed_k, k);
            } else {
                seed_fill = std::max(0, k - std::max(0, ip_.rand_min));
            }

            int pos = 0;

            // 1) LSH-based seeds
            if (seed_fill > 0 && ip_.enable) {
                idx_.query(i, ip_.cand_cap, cand, rng);
                cand.erase(std::remove(cand.begin(), cand.end(), (uint32_t)i), cand.end());

                const int take = std::min(seed_fill, (int)cand.size());
                if (take > 0) {
                    if (!ip_.rank_by_distance) {
                        // partial shuffle
                        for (int t = 0; t < take; ++t) {
                            int j = t + (int)rng.uniform_u32((uint32_t)((int)cand.size() - t));
                            std::swap(cand[(size_t)t], cand[(size_t)j]);
                        }
                        for (int t = 0; t < take; ++t) {
                            uint32_t id = cand[(size_t)t];
                            nbr[pos] = id;
                            ds[pos]  = dist(i, (int)id);
                            flg[pos] = KNNGraph::IS_NEW;
                            ++pos;
                        }
                    } else {
                        heap.clear();
                        float worst_sq = KNNGraph::inf();

                        auto heap_cmp = [](const std::pair<float,uint32_t>& a,
                                           const std::pair<float,uint32_t>& b){
                            // max-heap by dist
                            if (a.first != b.first) return a.first < b.first;
                            return a.second < b.second;
                        };

                        for (uint32_t id : cand) {
                            // prune only meaningful if heap is full
                            if (ip_.enable_prune && (int)heap.size() >= take) {
                                float lb = pr_.lower_bound_sq(i, (int)id);
                                if (lb >= worst_sq) continue;
                            }

                            float d = dist(i, (int)id);

                            if ((int)heap.size() < take) {
                                heap.emplace_back(d, id);
                                std::push_heap(heap.begin(), heap.end(), heap_cmp);
                                if ((int)heap.size() == take) worst_sq = heap.front().first;
                            } else if (d < heap.front().first) {
                                std::pop_heap(heap.begin(), heap.end(), heap_cmp);
                                heap.back() = {d, id};
                                std::push_heap(heap.begin(), heap.end(), heap_cmp);
                                worst_sq = heap.front().first;
                            }
                        }

                        std::sort(heap.begin(), heap.end(), [](auto& a, auto& b){
                            if (a.first != b.first) return a.first < b.first;
                            return a.second < b.second;
                        });

                        for (auto& e : heap) {
                            nbr[pos] = e.second;
                            ds[pos]  = e.first;
                            flg[pos] = KNNGraph::IS_NEW;
                            ++pos;
                        }
                    }
                }
            }

            // 2) Random fill remaining
            for (; pos < k; ++pos) {
                bool placed = false;
                for (int tries = 0; tries < 2000; ++tries) {
                    uint32_t r = rng.uniform_u32((uint32_t)(n - 1));
                    uint32_t id = (r >= (uint32_t)i) ? (r + 1) : r;

                    bool dup = false;
                    for (int u = 0; u < pos; ++u) {
                        if (nbr[u] == id) { dup = true; break; }
                    }
                    if (dup) continue;

                    nbr[pos] = id;
                    ds[pos]  = dist(i, (int)id);
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

            g.recompute_worst_row(i);
        }
    }

private:
    const Dataset* ds_ = nullptr;
    PStableLSHIndex idx_;
    NNDLSHInitPrunedParams ip_;

    // built only if enable_prune
    mutable ProjectionPruner pr_;
};
