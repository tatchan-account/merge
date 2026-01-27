#include <algorithm>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include "dataset.h"
#include "knngraph.h"
#include "nndescent.h"
#include "random_init.h"
#include "splitmix64.h"
#include "lsh_pstable.h"
#include "lsh_init.h"
#include "lsh.h"
#include "qgt_io.h"
#include "projection_table.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

static fs::path dataset_path_for_name(const fs::path& name_file) {
    fs::path p = name_file;
    if (!p.has_parent_path()) p = fs::path("dataset") / p;
    return p;
}

static fs::path ans_path_for_qgt(const fs::path& name_file) {
    fs::path p = name_file.filename();
    fs::path qgt = fs::path("ans") / (p.string() + ".qgt");
    if (fs::exists(qgt)) return qgt;
    fs::path andd = fs::path("and") / (p.string() + ".qgt");
    if (fs::exists(andd)) return andd;
    return qgt;
}

static fs::path ans_path_for_gt(const fs::path& name_file) {
    fs::path p = name_file.filename();
    fs::path gt = fs::path("ans") / p;
    if (fs::exists(gt)) return gt;
    fs::path andd = fs::path("and") / p;
    if (fs::exists(andd)) return andd;
    return gt;
}

// サブグラフはローカルインデックスなので、global indexに直したdistanceを計算（offsetを足す）
struct OffsetDist {
    const dist_func& base;
    int offset = 0;
    float operator()(int i, int j) const { return base(i + offset, j + offset); }
};

struct Timer {
    using clk = std::chrono::high_resolution_clock;
    clk::time_point t0;
    void tic() { t0 = clk::now(); }
    double toc_ms() const {
        auto t1 = clk::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// 既存グラフから始める時用のInit
struct CopyInit {
    const kNNGraph* src = nullptr;

    template<class Dist>
    void operator()(kNNGraph& g, const Dist&, SplitMix64&) const {
        if (!src) throw std::runtime_error("CopyInit : src null");
        if (g.n() != src->n() || g.k() != src->k()) throw std::runtime_error("CopyInit: size mismatch");

        const int k = g.k();
        const int n = g.n();
        for (int i = 0; i < n; ++i) {
            auto* nbr = g.nbr_ptr(i);
            auto* ds  = g.dist_ptr(i);
            auto* flg = g.flag_ptr(i);
            const auto* s_nbr = src->nbr_ptr(i);
            const auto* s_ds  = src->dist_ptr(i);
            const auto* s_flg = src->flag_ptr(i);
            for (int t = 0; t < k; ++t) {
                nbr[t] = s_nbr[t];
                ds[t]  = s_ds[t];
                flg[t] = s_flg[t];
            }
            g.recompute_worst_row(i);
        }
    }
};

// LSHなどの候補数の確認
struct SeedStats {
    uint64_t cand_total = 0;
    uint64_t prefilter_total = 0;
    uint64_t prefilter_kept = 0;
    uint64_t pruned_lb = 0;
    uint64_t full_dist = 0;
    uint64_t rand_added = 0;
};

static inline float row_worst_dist(const kNNGraph& g, int v) {
    const int k = g.k();
    const uint32_t* nbr = g.nbr_ptr(v);
    const float* ds = g.dist_ptr(v);
    float worst = 0.0f;
    bool any = false;
    for (int t = 0; t < k; ++t) {
        if (nbr[t] == kNNGraph::invalid_id()) continue;
        const float d = ds[t];
        if (!std::isfinite(d)) continue;
        if (!any || d > worst) { worst = d; any = true; }
    }
    return any ? worst : std::numeric_limits<float>::infinity();
}

static inline bool in_subset0(uint32_t id, int n1) { return (int)id < n1; }
static inline bool in_subset1(uint32_t id, int n1) { return (int)id >= n1; }

static void fill_random_cross_tail(kNNGraph& g, int v, int k_target, int k_cap, int n1, const dist_func& dist,
                                  SplitMix64& rng, bool mark_new, SeedStats* st = nullptr) {
    if (k_cap <= k_target) return;

    auto* nbr = g.nbr_ptr(v);
    auto* ds  = g.dist_ptr(v);
    auto* flg = g.flag_ptr(v);

    const bool v_in_s0 = (v < n1);
    const int other_lo = v_in_s0 ? n1 : 0;
    const int other_hi = v_in_s0 ? g.n() : n1;
    const int other_sz = other_hi - other_lo;

    auto already = [&](uint32_t id, int upto_exclusive) -> bool {
        for (int t = 0; t < upto_exclusive; ++t) if (nbr[t] == id) return true;
        return false;
    };

    for (int pos = k_target; pos < k_cap; ++pos) {
        // try a few times to avoid duplicates
        for (int tries = 0; tries < 200; ++tries) {
            uint32_t id = (uint32_t)(other_lo + (int)rng.uniform_u32((uint32_t)other_sz));
            if ((int)id == v) continue;
            if (already(id, pos)) continue;
            nbr[pos] = id;
            ds[pos]  = dist(v, (int)id);
            flg[pos] = mark_new ? kNNGraph::IS_NEW : 0;
            if (st) st->rand_added++;
            break;
        }
    }
    g.recompute_worst_row(v);
}

static void inject_lsh_seeds(kNNGraph& g, int v, int k_target, int k_cap, int n1, const dist_func& dist,
                            SplitMix64& rng, const PStableLSHIndex& other, const SMergeLSHSeedParams& sp,
                            const ProjectionTable* proj, // optional
                            bool use_prune, bool use_prefilter, int prefilter_mult, int prefilter_min,
                            int prefilter_max, SeedStats& st) {
    if (k_cap <= k_target) return;

    const bool v_in_s0 = (v < n1);
    const auto is_cross = [&](uint32_t id) -> bool {
        return v_in_s0 ? in_subset1(id, n1) : in_subset0(id, n1);
    };

    // ext slots count
    const int ext_slots = k_cap - k_target;

    int seed_fill = 0;
    if (sp.seed_k >= 0) seed_fill = std::min(sp.seed_k, ext_slots);
    else seed_fill = std::max(0, ext_slots - sp.rand_min);
    if (seed_fill <= 0) return;

    std::vector<uint32_t> cands;
    other.query(v, sp.cand_cap, cands, rng);
    st.cand_total += (uint64_t)cands.size();

    // filter to cross-subset and valid
    std::vector<uint32_t> xc;
    xc.reserve(cands.size());
    for (uint32_t id : cands) {
        if (id == (uint32_t)v) continue;
        if (!is_cross(id)) continue;
        xc.push_back(id);
    }

    if (xc.empty()) return;

    // prefilter stage (projection heuristic): keep M candidates by projection distance
    std::vector<uint32_t> kept;
    kept = std::move(xc);

    if (use_prefilter) {
        if (!proj) throw std::runtime_error("--prefilter requires projection table");
        st.prefilter_total += (uint64_t)kept.size();

        int M = prefilter_mult * seed_fill;
        M = std::max(M, prefilter_min);
        M = std::min(M, prefilter_max);
        M = std::min(M, (int)kept.size());
        if (M < (int)kept.size()) {
            std::nth_element(kept.begin(), kept.begin() + M, kept.end(), [&](uint32_t a, uint32_t b){
                return proj->proj_l2_sq(v, (int)a) < proj->proj_l2_sq(v, (int)b);
            });
            kept.resize((size_t)M);
        }
        st.prefilter_kept += (uint64_t)kept.size();
    }

    // compute exact distances for kept, optionally prune by LB against current worst
    std::vector<std::pair<float,uint32_t>> scored;
    scored.reserve(kept.size());

    float worst = row_worst_dist(g, v);
    for (uint32_t id : kept) {
        if (use_prune) {
            if (!proj) throw std::runtime_error("--prune requires projection table");
            float lb = proj->lower_bound_sq(v, (int)id);
            if (lb >= worst) { st.pruned_lb++; continue; }
        }
        float d2 = dist(v, (int)id);
        st.full_dist++;
        scored.push_back({d2, id});
    }
    if (scored.empty()) return;

    // take best seed_fill by exact distance
    if ((int)scored.size() > seed_fill) {
        std::nth_element(scored.begin(), scored.begin() + seed_fill, scored.end(),
                         [](auto& a, auto& b){ return a.first < b.first; });
        scored.resize((size_t)seed_fill);
    }
    std::sort(scored.begin(), scored.end(), [](auto& a, auto& b){ return a.first < b.first; });

    for (auto& pr : scored) {
        g.update(v, pr.second, pr.first); // marks IS_NEW
    }
}


/*
 * recallを計算する関数
 */
double calc_recall(const kNNGraph& g,int k_target, int eval_n, int eval_q, const fs::path& gt_path, const fs::path& qgt_path){
    if (fs::exists(gt_path)) {
        try {
            FullGT gt = load_full_gt(gt_path);
            if ((int)gt.n != g.n()) {
                std::cerr << "GT n mismatch: gt.n=" << gt.n << " vs data.n=" << g.n() << "\n";
                return -1;
            }
            std::cout << "GT loaded: " << gt_path.string() << " (k_gt=" << gt.k << ")\n";
            const int en = (eval_n <= 0) ? g.n() : std::min(eval_n, g.n());
            double rec = recall_from_gt_topk(g, gt, k_target, en);
            std::cout << "recall@" << k_target << " over first " << en << " points = " << rec << "\n";
            return rec;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load GT: " << e.what() << "\n";
            return -1;
        }
    } else if (fs::exists(qgt_path)) {
        try {
            QueryGT qgt = load_qgt(qgt_path.string());
            std::cout << "QGT loaded: " << qgt_path.string() << " (Q=" << (uint64_t)qgt.Q << ", k_gt=" << (uint64_t)qgt.k << ")\n";
            double rec = recall_from_qgt_topk(g, qgt, k_target, eval_q);
            const int qe = (eval_q <= 0) ? (int)qgt.Q : std::min(eval_q, (int)qgt.Q);
            std::cout << "recall@" << k_target << " over Q=" << qe << " queries = " << rec << "\n";
            return rec;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load QGT: " << e.what() << "\n";
            return -1;
        }
    } else {
        std::cerr << "GT/QGT not found under ./ans or ./and(skip recall)\n";
        return -1;
    }
}

static void usage(const char* proc) {
    std::cerr
        << "Usage: " << proc << " <name.txt> <k_target> <split_ratio>\n"
        << "  dataset is loaded from ./dataset/<name.txt>\n"
        << "  QGT is loaded from ./ans/<name.txt>.qgt if present\n\n"
        << "Options:\n"
        << "  --seed <u64>          RNG seed (default 12345)\n"
        << "  --iter <int>          NN-Descent max_iter (default 20)\n"
        << "  --rho <float>         NN-Descent rho (default 0.5)\n"
        << "  --delta <float>       NN-Descent delta (default 0.001)\n"
        << "  --keep <float>        keep_ratio for S-merge-style mixing (default 0.5). Used when k_cap==k_target to make room for cross edges.\n"
        << "  --k-cap <int>         internal capacity (>=k_target). default k_target\n"
        << "  --k-cap-mult <float>  set k_cap = ceil(mult*k_target)\n"
        << "  --no-refine           skip NN-Descent refinement on merged graph\n"
        << "  --refine-iter <int>   override refine iterations (default: --iter)\n"
        << "  --rand-new            mark random cross-injected edges as NEW (default: NEW)\n"
        << "  --rand-old            mark random cross-injected edges as OLD (faster but can hurt refinement if LSH is sparse)\n"
        << "  --lsh                 enable LSH seeding\n"
        << "    --lsh-L <int>       (default 10)\n"
        << "    --lsh-K <int>       (default 4)\n"
        << "    --lsh-w <float>     (default 4.0)\n"
        << "    --lsh-bucket <int>  bucket cap (default 256)\n"
        << "    --lsh-cand <int>    candidate cap per query (default 128)\n"
        << "    --lsh-seedk <int>   #LSH seeds inserted per node into extension part (default 16)\n"
        << "    --lsh-randmin <int> min random tail kept (default 0)\n"
        << "  --prune               enable SAFE LB pruning (projection)\n"
        << "  --prefilter           enable heuristic prefilter by projection distance\n"
        << "    --proj-P <int>      #projections (default 8)\n"
        << "    --proj-seed <u64>   projection seed (default 777)\n"
        << "    --pf-mult <int>     M = mult*seedk (default 8)\n"
        << "    --pf-min <int>      (default 64)\n"
        << "    --pf-max <int>      (default 1024)\n"
        << "  --sub-init <random|lsh> build subset graphs with init (default random)\n"
        << "    --sub-lsh-cand <int> candidates per point for LSH init (default 256)\n"
        << "    --sub-lsh-seedk <int> #LSH neighbors used in init (-1 auto)\n"
        << "    --sub-lsh-randmin <int> min random neighbors in init (default 2)\n"
        << "    --sub-lsh-norank      do not rank candidates by exact distance\n"
        << "  --eval-q <int>        evaluate first Q queries (default: all)\n"
        << "  --eval-n <int>        evaluate first N points when GT is used (default: all)\n"
        << "  --require-gt          exit with code 2 if GT/QGT is missing\n"
        << "  --refine-delta <int>  Threshold for refinemen\n"
        << "  --full-NND  Create graph fully by NN-Descent\n";
}

int main(int argc, char** argv) {
    if (argc < 4) { usage(argv[0]); return 1; }

    fs::path name_file = argv[1];
    const int k_target = std::stoi(argv[2]);
    const double split_ratio = std::stod(argv[3]);

    if (k_target <= 0) {
        std::cerr << "k_target (output k) has to be > 0\n";
        return 1;
    }
    if (!(split_ratio > 0.0 && split_ratio < 1.0)) {
        std::cerr << "split_ratio must be in (0,1)\n";
        return 1;
    }

    NNDParams p, p_refine;
    int refine_iter = -1;
    // もしLSHがスカスカになったら追加でランダム追加が必要になるため注意しておく
    bool rand_new = true;

    // capacity
    int k_cap = k_target;
    double k_cap_mult = -1.0;

    // LSH enable + params
    bool use_lsh = false;
    std::string sub_init = "random";
    NNDLSHInitParams sub_ip;
    sub_ip.cand_cap = 256;
    sub_ip.seed_k = -1;
    sub_ip.rand_min = 2;
    sub_ip.rank_by_distance = true;

    PStableLSHParams lshp;
    lshp.L = 10;
    lshp.K = 4;
    lshp.w = 4.0f;
    lshp.bucket_cap = 256;

    SMergeLSHSeedParams sp;
    sp.cand_cap = 128;
    sp.seed_k = 16;
    sp.rand_min = 0;

    // pruning / prefilter
    bool use_prune = false;
    bool use_prefilter = false;
    ProjTableParams pp;
    int pf_mult = 8;
    int pf_min = 64;
    int pf_max = 1024;
    int eval_q = -1;
    int eval_n = -1;
    // bool require_gt = false;
    float keep_ratio = 0.5;
    bool onlyNND = false;

    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* opt) {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + opt);
            return std::string(argv[++i]);
        };
        try {
            if (a == "--seed") p.seed = std::stoull(need("--seed"));
            else if (a == "--iter") p.max_iter = std::stoi(need("--iter"));
            else if (a == "--rho") p.rho = std::stof(need("--rho"));
            else if (a == "--delta") p.delta = std::stof(need("--delta"));
            else if (a == "--keep") keep_ratio = std::stof(need("--keep"));
            else if (a == "--eval-n") eval_n = std::stoi(need("--eval-n"));
            // else if (a == "--require-gt") require_gt = true;
            else if (a == "--k-cap") k_cap = std::stoi(need("--k-cap"));
            else if (a == "--k-cap-mult") k_cap_mult = std::stod(need("--k-cap-mult"));
            else if (a == "--refine-iter") refine_iter = std::stoi(need("--refine-iter"));
            else if (a == "--refine-delta") p_refine.delta = std::stoi(need("--refine-delta"));
            else if (a == "--rand-new") rand_new = true;
            else if (a == "--rand-old") rand_new = false;
            else if (a == "--lsh") use_lsh = true;
            else if (a == "--lsh-L") lshp.L = std::stoi(need("--lsh-L"));
            else if (a == "--lsh-K") lshp.K = std::stoi(need("--lsh-K"));
            else if (a == "--lsh-w") lshp.w = std::stof(need("--lsh-w"));
            else if (a == "--lsh-bucket") lshp.bucket_cap = std::stoi(need("--lsh-bucket"));
            else if (a == "--lsh-cand") sp.cand_cap = std::stoi(need("--lsh-cand"));
            else if (a == "--lsh-seedk") sp.seed_k = std::stoi(need("--lsh-seedk"));
            else if (a == "--lsh-randmin") sp.rand_min = std::stoi(need("--lsh-randmin"));
            else if (a == "--sub-init") sub_init = need("--sub-init");
            else if (a == "--sub-lsh-cand") sub_ip.cand_cap = std::stoi(need("--sub-lsh-cand"));
            else if (a == "--sub-lsh-seedk") sub_ip.seed_k = std::stoi(need("--sub-lsh-seedk"));
            else if (a == "--sub-lsh-randmin") sub_ip.rand_min = std::stoi(need("--sub-lsh-randmin"));
            else if (a == "--sub-lsh-norank") sub_ip.rank_by_distance = false;
            else if (a == "--prune") use_prune = true;
            else if (a == "--prefilter") use_prefilter = true;
            else if (a == "--proj-P") pp.P = std::stoi(need("--proj-P"));
            else if (a == "--proj-seed") pp.seed = std::stoull(need("--proj-seed"));
            else if (a == "--pf-mult") pf_mult = std::stoi(need("--pf-mult"));
            else if (a == "--pf-min") pf_min = std::stoi(need("--pf-min"));
            else if (a == "--pf-max") pf_max = std::stoi(need("--pf-max"));
            else if (a == "--eval-q") eval_q = std::stoi(need("--eval-q"));
            else if (a == "--full-NND") onlyNND = true;
            else {
                std::cerr << "Unknown option: " << a << "\n";
                usage(argv[0]);
                return 1;
            }
        } catch (const std::exception& e) {
            std::cerr << "Option parse error at " << a << ": " << e.what() << "\n";
            return 1;
        }
    }

    if (k_cap_mult > 0.0) {
        int kk = (int)std::ceil(k_cap_mult * (double)k_target);
        if (kk > k_cap) k_cap = kk;
    }
    if (k_cap < k_target) k_cap = k_target;

    // validate sub-init
    if (sub_init != "random" && sub_init != "lsh") {
        std::cerr << "Invalid --sub-init: " << sub_init << " (use random|lsh)\n";
        return 1;
    }

    // seedに関してlsh候補が一定になるように
    lshp.seed = p.seed;
    sub_ip.lsh = lshp;

    const fs::path data_path = dataset_path_for_name(name_file);
    Dataset ds;
    try {
        ds = Dataset::load_text(data_path.string());
    } catch (const std::exception& e) {
        std::cerr << "Failed to load dataset: " << e.what() << "\n";
        return 1;
    }
    std::cout << "Loaded: n=" << ds.n << " d=" << ds.d << "\n";

    if (ds.n <= k_target + 1) { std::cerr << "Dataset too small.\n"; return 1; }

    const int n = ds.n;
    const int n1 = (int)std::llround((double)n * split_ratio);
    const int n2 = n - n1;
    if (n1 <= k_target + 1 || n2 <= k_target + 1) {
        std::cerr << "Split too small: n1=" << n1 << " n2=" << n2 << " (k_target=" << k_target << ")\n";
        return 1;
    }
    std::cout << "Split: n1=" << n1 << " n2=" << n2 << " (k_target=" << k_target << ", k_cap=" << k_cap << ")\n";

    if (!(keep_ratio >= 0.0f && keep_ratio <= 1.0f)) {
        std::cerr << "keep_ratio must be in [0,1].\n";
        return 1;
    }
    int keep_k = k_target;
    if (k_cap == k_target) {
        keep_k = (int)std::llround((double)keep_ratio * (double)k_target);
        if (keep_k < 0) keep_k = 0;
        if (keep_k > k_target) keep_k = k_target;
    } else {
        int k_tmp = (int)std::llround((double)keep_ratio * (double)k_cap);
        if (keep_ratio == 1) {}
        else if (k_tmp <= k_target) keep_k = k_tmp;
    }
    const int boundary_k = keep_k; // slots [boundary_k, k_cap) will be used for cross seeds
    std::cout << "Merge : keep_k=" << keep_k << " (keep_ratio=" << keep_ratio << ")\n";

    dist_func dist_global{ds};

    Timer tt;
    double t_seed = 0.0, t_refine = 0.0, t_merge = 0.0,t_total = 0.0;
    // double t_build;

    const bool sub_use_lsh = (sub_init == "lsh");
    const bool need_lsh_index = use_lsh || sub_use_lsh;

    PStableLSHIndex idx_s1;
    PStableLSHIndex idx_s2;

    tt.tic();

    const fs::path gt_path  = ans_path_for_gt(name_file);
    const fs::path qgt_path = ans_path_for_qgt(name_file);

    // NN-Descentオンリーで実行
    if (onlyNND){
        kNNGraph g = nndescent_full(k_target, n, dist_global, RandomInit{}, p);
        double t_onlyNND = tt.toc_ms();
        double rec_val = calc_recall(g, k_target, eval_n, eval_q, gt_path, qgt_path);
        if (rec_val < 0) {
            std::cerr << "recall@k was not calculated";
        }

        std::cout
            << "k, k_cap, time, recall@\n"
            << "result: "
            << k_target << ", "
            << k_cap << ", "
            << t_onlyNND << ", "
            << rec_val << std::endl;

        return 0;
    }

    double t_subg1 = 0, t_subg2 = 0, t_subglsh1 = 0, t_subglsh2 = 0;
    tt.tic();

    if (need_lsh_index) {
        std::cout << "Build LSH index: L=" << lshp.L << " K=" << lshp.K << " w=" << lshp.w
                  << " bucket_cap=" << lshp.bucket_cap << " seed=" << lshp.seed << "\n";
        idx_s1 = PStableLSHIndex(ds, 0, n1, lshp);
        t_subglsh1 = tt.toc_ms();
        idx_s2 = PStableLSHIndex(ds, n1, n, lshp);
        t_subglsh2 = tt.toc_ms() - t_subglsh1;
    }


    tt.tic();
    OffsetDist dist1{dist_global, 0};
    OffsetDist dist2{dist_global, n1};

    NNDParams p1 = p;
    NNDParams p2 = p;
    // 異なるseedを設定し、サンプリングパターンが異なるように
    p2.seed = p.seed ^ 0x9e3779b97f4a7c15ULL;

    kNNGraph g1 = (!sub_use_lsh) ? nndescent_full(k_target, n1, dist1, RandomInit{}, p1)
            : [&](){
                // LSHを使ったNN-Descentの初期候補注入（現状不使用使っていない）
                LSHInit init1(ds, 0, n1, sub_ip, &idx_s1);
                return nndescent_full(k_target, n1, dist1, init1, p1);
            }();
    t_subg1 = tt.toc_ms();

    tt.tic();

    kNNGraph g2 = (!sub_use_lsh)? nndescent_full(k_target, n2, dist2, RandomInit{}, p2)
            : [&](){
                LSHInit init2(ds, n1, n, sub_ip, &idx_s2);
                return nndescent_full(k_target, n2, dist2, init2, p2);
            }();

    t_subg2 = tt.toc_ms();

    /*
     * lsh使ったNNDにしても良いように、インデクシングで使った時間subglshとNNDによるサブグラフ構築でかかった時間subgを足しておく
     * いずれも0で初期化されているので、falseなら0で計上
     */
    t_subg1 += t_subglsh1;
    t_subg2 += t_subglsh2;

    // 並列に構築していると仮定して、構築時間の大きい方をサブグラフ構築時間とする
    double t_subgraph = (t_subg1 > t_subg2)? t_subg1 : t_subg2;
    // t_build = t_subg1 + t_subg2;


    // 射影軸を準備
    ProjectionTable proj;
    ProjectionTable* proj_ptr = nullptr;
    if (use_prune || use_prefilter) {
        std::cout << "Build projection table: P=" << pp.P << " seed=" << pp.seed << "\n";
        proj = ProjectionTable(ds, pp);
        proj_ptr = &proj;
    }

    // （keep_k == k_targetならば、ただのコピーになる）
    tt.tic();
    kNNGraph g_init(k_cap, n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n1; ++i) {
        auto* nbr = g_init.nbr_ptr(i);
        auto* dsr = g_init.dist_ptr(i);
        auto* flg = g_init.flag_ptr(i);

        const auto* s_nbr = g1.nbr_ptr(i);
        const auto* s_ds  = g1.dist_ptr(i);

        if (keep_k >= k_target) {
            for (int t = 0; t < k_target; ++t) { nbr[t] = s_nbr[t]; dsr[t] = s_ds[t]; flg[t] = 0; }
        } else {
            // 最小のkeep_kを選ぶ
            std::vector<int> idx((size_t)k_target);
            for (int t = 0; t < k_target; ++t) idx[(size_t)t] = t;
            std::nth_element(idx.begin(), idx.begin() + keep_k, idx.end(),[&](int a, int b){ return s_ds[a] < s_ds[b]; });
            idx.resize((size_t)keep_k);
            std::sort(idx.begin(), idx.end(), [&](int a, int b){ return s_ds[a] < s_ds[b]; });
            for (int t = 0; t < keep_k; ++t) {
                int p = idx[(size_t)t];
                nbr[t] = s_nbr[p];
                dsr[t] = s_ds[p];
                flg[t] = 0;
            }
        }
        // 末尾の[keep_k, k_cap)はinvalidな値でコンストラクタで初期化済み
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n2; ++i) {
        const int v = n1 + i;
        auto* nbr = g_init.nbr_ptr(v);
        auto* dsr = g_init.dist_ptr(v);
        auto* flg = g_init.flag_ptr(v);

        const auto* s_nbr = g2.nbr_ptr(i);
        const auto* s_ds  = g2.dist_ptr(i);

        if (keep_k >= k_target) {
            for (int t = 0; t < k_target; ++t) {
                uint32_t id = s_nbr[t];
                nbr[t] = (id == kNNGraph::invalid_id()) ? id : (uint32_t)(id + (uint32_t)n1);
                dsr[t] = s_ds[t];
                flg[t] = 0;
            }
        } else {
            std::vector<int> idx((size_t)k_target);
            for (int t = 0; t < k_target; ++t) idx[(size_t)t] = t;
            std::nth_element(idx.begin(), idx.begin() + keep_k, idx.end(),
                             [&](int a, int b){ return s_ds[a] < s_ds[b]; });
            idx.resize((size_t)keep_k);
            std::sort(idx.begin(), idx.end(), [&](int a, int b){ return s_ds[a] < s_ds[b]; });
            for (int t = 0; t < keep_k; ++t) {
                int p = idx[(size_t)t];
                uint32_t id = s_nbr[p];
                nbr[t] = (id == kNNGraph::invalid_id()) ? id : (uint32_t)(id + (uint32_t)n1);
                dsr[t] = s_ds[p];
                flg[t] = 0;
            }
        }
        // tail remains invalid/inf
    }

    // random generators per thread
// random generators per thread
    SeedStats st;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        SplitMix64 rng(p.seed ^ 0x1234567890abcdefULL
#ifdef _OPENMP
                       ^ (uint64_t)omp_get_thread_num() * 0x9e3779b97f4a7c15ULL
#endif
        );
        SeedStats local{};
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int v = 0; v < n; ++v) {
            // Fill extension with random cross nodes first.
            // These should be NEW by default so that NN-Descent refinement can propagate across subsets,
            // even when LSH returns few (or zero) collided candidates.
            fill_random_cross_tail(g_init, v, boundary_k, k_cap, n1, dist_global, rng, rand_new, &local);

            // LSH seeds to replace/augment (marked NEW)
            if (use_lsh) {
                const PStableLSHIndex& other = (v < n1) ? idx_s2 : idx_s1;
                inject_lsh_seeds(g_init, v, boundary_k, k_cap, n1, dist_global, rng,
                                 other, sp, proj_ptr,
                                 use_prune, use_prefilter,
                                 pf_mult, pf_min, pf_max,
                                 local);
            }
        }
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            st.cand_total += local.cand_total;
            st.prefilter_total += local.prefilter_total;
            st.prefilter_kept += local.prefilter_kept;
            st.pruned_lb += local.pruned_lb;
            st.full_dist += local.full_dist;
            st.rand_added += local.rand_added;
        }
    }

    t_seed = tt.toc_ms();

    // refine (full NN-Descentをg_initに対して行う)
    // parameterはデフォルトか引数指定になっている
    kNNGraph g = g_init;
    tt.tic();
    if (refine_iter > 0) p_refine.max_iter = refine_iter;
    // refine deltaについては引数指定があれば、そこで変えている（上で変えている）
    // サブグラフと同じサンプリングにならないように、少しseedを帰る
    p_refine.seed = p.seed ^ 0xfeedfacecafebeefULL;

    g = nndescent_full(k_cap, n, dist_global, CopyInit{&g_init}, p_refine);
    t_refine = tt.toc_ms();

    // 合計時間
    t_total = t_subgraph + t_seed + t_refine;
	t_merge = t_seed + t_refine;

    double rec = -1;
    rec = calc_recall(g, k_target, eval_n, eval_q, gt_path, qgt_path);

    if (rec < 0) std::cerr << "Recall calculation was skipped.\n";

    std::cerr << "subgraph1 time: " << t_subg1 << std::endl;
    std::cerr << "subgraph2 time: " << t_subg2 << std::endl;
    std::cerr << "subgraph time: " << t_subgraph << std::endl;

    std::cout << "seed_stats.cand_total=" << st.cand_total
        << " prefilter_total=" << st.prefilter_total
        << " prefilter_kept=" << st.prefilter_kept
        << " pruned_lb=" << st.pruned_lb
        << " full_dist=" << st.full_dist
        << " rand_added=" << st.rand_added
        << "\n";

    std::cout << "k, k_cap, time_seed, time_refine, time_merge, time_total, recall@k\n";
    std::cout << "result: "
        << k_target << ", "
        << k_cap << ", "
        << t_seed << ", "
        << t_refine << ", "
        << t_merge << ", "
        << t_total << ", "
        << rec << std::endl;
    return 0;
}
