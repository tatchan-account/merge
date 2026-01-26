#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dataset.h"
#include "l2_squared.hpp"
#include "nnd_params.hpp"
#include "nndescent.h"
#include "random_init.h"
#include "lsh_init.h"              // existing cheap init
#include "lsh_init_pruned.h"       // optional pruned init

static int to_int(const char* s) { return (int)std::strtol(s, nullptr, 10); }
static uint64_t to_u64(const char* s) { return (uint64_t)std::strtoull(s, nullptr, 10); }
static float to_float(const char* s) { return (float)std::strtod(s, nullptr); }

static bool has_slash(const std::string& s) {
    return s.find('/') != std::string::npos || s.find('\\') != std::string::npos;
}

static std::string basename_only(const std::string& path) {
    size_t p1 = path.find_last_of('/');
    size_t p2 = path.find_last_of('\\');
    size_t p = std::string::npos;
    if (p1 != std::string::npos) p = p1;
    if (p2 != std::string::npos) p = (p == std::string::npos) ? p2 : std::max(p, p2);
    if (p == std::string::npos) return path;
    return path.substr(p + 1);
}

static std::string join_path(const std::string& dir, const std::string& file) {
    if (dir.empty()) return file;
    if (!dir.empty() && (dir.back() == '/' || dir.back() == '\\')) return dir + file;
    return dir + "/" + file;
}

// GT format (recommended):
//   first line: n k_gt
//   next n lines: k_gt ids (space-separated)
// We only load first eval_n rows to save memory.
struct GTData {
    int n = 0;
    int k = 0;
    std::vector<uint32_t> ids; // size = eval_n * k
    int eval_n = 0;
};

static GTData load_gt_prefix(const std::string& path, int expected_n, int eval_n) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("Failed to open GT: " + path);

    GTData gt;
    ifs >> gt.n >> gt.k;
    if (!ifs || gt.n <= 0 || gt.k <= 0) throw std::runtime_error("GT header parse failed: " + path);
    if (expected_n > 0 && gt.n != expected_n) {
        throw std::runtime_error("GT n mismatch: gt.n=" + std::to_string(gt.n) + " expected=" + std::to_string(expected_n));
    }

    gt.eval_n = std::min(eval_n, gt.n);
    gt.ids.resize((size_t)gt.eval_n * (size_t)gt.k);

    for (int i = 0; i < gt.eval_n; ++i) {
        uint32_t* row = gt.ids.data() + (size_t)i * (size_t)gt.k;
        for (int t = 0; t < gt.k; ++t) {
            uint64_t v;
            ifs >> v;
            if (!ifs) throw std::runtime_error("GT row parse failed at i=" + std::to_string(i));
            row[t] = (uint32_t)v;
        }
        std::sort(row, row + gt.k);
        // (optional) unique; assumes GT is unique already
    }

    return gt;
}

static double recall_at_k(const KNNGraph& g, const GTData& gt, int k_eval) {
    const int eval_n = gt.eval_n;
    if (eval_n <= 0) return 0.0;

    const int kk = k_eval;
    std::vector<uint32_t> pred;
    pred.reserve((size_t)kk);

    uint64_t hit = 0;
    uint64_t denom = (uint64_t)eval_n * (uint64_t)kk;

    for (int i = 0; i < eval_n; ++i) {
        pred.clear();
        const auto* nbr = g.nbr_ptr(i);
        for (int t = 0; t < g.k(); ++t) {
            uint32_t id = nbr[t];
            if (id == KNNGraph::invalid_id()) continue;
            pred.push_back(id);
        }
        std::sort(pred.begin(), pred.end());
        pred.erase(std::unique(pred.begin(), pred.end()), pred.end());
        if ((int)pred.size() > kk) pred.resize((size_t)kk);

        const uint32_t* gtrow = gt.ids.data() + (size_t)i * (size_t)gt.k;
        // gtrow is sorted; take first kk
        int a = 0, b = 0;
        while (a < (int)pred.size() && b < kk) {
            uint32_t pa = pred[(size_t)a];
            uint32_t gb = gtrow[b];
            if (pa == gb) { ++hit; ++a; ++b; }
            else if (pa < gb) ++a;
            else ++b;
        }
    }

    if (denom == 0) return 0.0;
    return (double)hit / (double)denom;
}

static void usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " <name_or_path> <k> [options]\n"
        << "  Input dataset: dataset/<name_or_path> if no slash in arg (default data-dir=dataset)\n"
        << "  Ground truth:  ans/<basename(name_or_path)> (default ans-dir=ans)\n\n"
        << "Options:\n"
        << "  --data-dir <dir>      (default dataset)\n"
        << "  --ans-dir <dir>       (default ans)\n"
        << "  --require-gt          fail if GT file missing\n"
        << "  --eval-n <N>          evaluate recall for first N points (default 10000; -1 => all)\n"
        << "  --print-n <N>         print first N rows (id and distances)\n"
        << "  --time                print timing\n"
        << "\nNN-Descent params:\n"
        << "  --iter <N>            max iterations (default 20)\n"
        << "  --rho <R>             (default 0.5)\n"
        << "  --delta <D>           (default 0.001)\n"
        << "  --seed <S>            (default 12345)\n"
        << "\nInit mode:\n"
        << "  --init random|lsh     (default random)\n"
        << "\nLSH init params (p-stable for L2):\n"
        << "  --lsh-L <int>         tables (default 2)\n"
        << "  --lsh-K <int>         hashes/table (default 4)\n"
        << "  --lsh-w <float>       width (default 4.0)\n"
        << "  --lsh-bcap <int>      bucket cap (default 64)\n"
        << "  --lsh-cand <int>      query candidate cap (default 128)\n"
        << "  --lsh-seedk <int>     LSH seeds per row (>=0 fixed, <0: k-randmin)\n"
        << "  --lsh-randmin <int>   minimum random fill (default 2)\n"
        << "  --lsh-rank 0|1        rank candidates by true distance (default 0)\n"
        << "  --lsh-seed <u64>      LSH RNG seed (default: derived from --seed)\n"
        << "\nProjection pruning (only when --lsh-rank=1):\n"
        << "  --prune               enable LB pruning in init\n"
        << "  --proj-P <int>        #projections (default 4)\n"
        << "  --proj-seed <u64>     projection seed (default: derived from --seed)\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }

    std::string name_or_path = argv[1];
    int k = to_int(argv[2]);

    std::string data_dir = "dataset";
    std::string ans_dir = "ans";

    bool require_gt = false;
    int eval_n = 10000;
    int print_n = 0;
    bool do_time = false;

    std::string init_mode = "random";

    NNDParams p;

    // LSH init params
    NNDLSHInitPrunedParams lip;
    lip.lsh.L = 2;
    lip.lsh.K = 4;
    lip.lsh.w = 4.0f;
    lip.lsh.bucket_cap = 64;
    lip.cand_cap = 128;
    lip.seed_k = -1;
    lip.rand_min = 2;
    lip.rank_by_distance = false;
    lip.enable_prune = false;
    lip.prune.P = 4;

    uint64_t lsh_seed_override = 0;
    uint64_t proj_seed_override = 0;

    auto arg_need = [&](int& i, const char* opt) -> const char* {
        if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value after ") + opt);
        return argv[++i];
    };

    try {
        for (int i = 3; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--data-dir") data_dir = arg_need(i, "--data-dir");
            else if (a == "--ans-dir") ans_dir = arg_need(i, "--ans-dir");
            else if (a == "--require-gt") require_gt = true;
            else if (a == "--eval-n") eval_n = to_int(arg_need(i, "--eval-n"));
            else if (a == "--print-n") print_n = to_int(arg_need(i, "--print-n"));
            else if (a == "--time") do_time = true;

            else if (a == "--iter") p.max_iter = to_int(arg_need(i, "--iter"));
            else if (a == "--rho") p.rho = to_float(arg_need(i, "--rho"));
            else if (a == "--delta") p.delta = to_float(arg_need(i, "--delta"));
            else if (a == "--seed") p.seed = to_u64(arg_need(i, "--seed"));

            else if (a == "--init") init_mode = arg_need(i, "--init");

            else if (a == "--lsh-L") lip.lsh.L = to_int(arg_need(i, "--lsh-L"));
            else if (a == "--lsh-K") lip.lsh.K = to_int(arg_need(i, "--lsh-K"));
            else if (a == "--lsh-w") lip.lsh.w = to_float(arg_need(i, "--lsh-w"));
            else if (a == "--lsh-bcap") lip.lsh.bucket_cap = to_int(arg_need(i, "--lsh-bcap"));
            else if (a == "--lsh-cand") lip.cand_cap = to_int(arg_need(i, "--lsh-cand"));
            else if (a == "--lsh-seedk") lip.seed_k = to_int(arg_need(i, "--lsh-seedk"));
            else if (a == "--lsh-randmin") lip.rand_min = to_int(arg_need(i, "--lsh-randmin"));
            else if (a == "--lsh-rank") lip.rank_by_distance = (to_int(arg_need(i, "--lsh-rank")) != 0);
            else if (a == "--lsh-seed") lsh_seed_override = to_u64(arg_need(i, "--lsh-seed"));

            else if (a == "--prune") lip.enable_prune = true;
            else if (a == "--proj-P") lip.prune.P = to_int(arg_need(i, "--proj-P"));
            else if (a == "--proj-seed") proj_seed_override = to_u64(arg_need(i, "--proj-seed"));

            else throw std::runtime_error(std::string("Unknown option: ") + a);
        }
    } catch (const std::exception& e) {
        std::cerr << "[argparse] " << e.what() << "\n";
        usage(argv[0]);
        return 2;
    }

    if (k <= 0) {
        std::cerr << "Error: k must be positive.\n";
        return 3;
    }

    // Resolve dataset path
    std::string data_path = name_or_path;
    if (!has_slash(name_or_path)) {
        data_path = join_path(data_dir, name_or_path);
    }

    const std::string base = basename_only(name_or_path);
    const std::string gt_path = join_path(ans_dir, base);

    auto t0 = std::chrono::steady_clock::now();

    Dataset ds;
    try {
        ds = Dataset::load_text(data_path);
    } catch (const std::exception& e) {
        std::cerr << "[load] " << e.what() << "\n";
        return 4;
    }

    if (k >= ds.n) {
        std::cerr << "Error: k must satisfy 1 <= k < n. (k=" << k << ", n=" << ds.n << ")\n";
        return 5;
    }

    if (eval_n < 0) eval_n = ds.n;
    if (eval_n > ds.n) eval_n = ds.n;

    auto t_load = std::chrono::steady_clock::now();

    L2Squared dist{ds};

    // seeds: derive unless overridden
    if (lsh_seed_override != 0) lip.lsh.seed = lsh_seed_override;
    else lip.lsh.seed = p.seed ^ 0xD1B54A32D192ED03ULL;

    if (proj_seed_override != 0) lip.prune.seed = proj_seed_override;
    else lip.prune.seed = p.seed ^ 0x9e3779b97f4a7c15ULL;

    std::cerr << "[run] data=" << data_path << " n=" << ds.n << " d=" << ds.d << " k=" << k
              << " init=" << init_mode
              << " iter=" << p.max_iter << " rho=" << p.rho << " delta=" << p.delta << " seed=" << p.seed << "\n";

    KNNGraph g(0,0);

    auto t_alg0 = std::chrono::steady_clock::now();

    if (init_mode == "random") {
        RandomInit init;
        g = nndescent_full(ds.n, k, dist, init, p, /*verbose=*/true);
    } else if (init_mode == "lsh") {
        // choose initializer
        if (lip.enable_prune && lip.rank_by_distance) {
            LSHInitPruned init(ds, lip);
            g = nndescent_full(ds.n, k, dist, init, p, /*verbose=*/true);
        } else {
            // use existing (cheaper) initializer
            NNDLSHInitParams ip;
            ip.enable = lip.enable;
            ip.cand_cap = lip.cand_cap;
            ip.seed_k = lip.seed_k;
            ip.rand_min = lip.rand_min;
            ip.rank_by_distance = lip.rank_by_distance;
            ip.lsh = lip.lsh;
            LSHInit init(ds, ip);
            g = nndescent_full(ds.n, k, dist, init, p, /*verbose=*/true);
        }
    } else {
        std::cerr << "Unknown --init: " << init_mode << "\n";
        return 6;
    }

    auto t_alg1 = std::chrono::steady_clock::now();

    // Evaluate recall if GT exists
    bool gt_ok = false;
    double recall = -1.0;
    int k_gt = -1;

    {
        std::ifstream test(gt_path);
        if (test.good()) {
            try {
                GTData gt = load_gt_prefix(gt_path, ds.n, eval_n);
                gt_ok = true;
                k_gt = gt.k;
                int k_eval = std::min(k, gt.k);
                recall = recall_at_k(g, gt, k_eval);
                std::cerr << "[eval] gt=" << gt_path << " eval_n=" << eval_n << " k_eval=" << k_eval
                          << " recall=" << recall << "\n";
            } catch (const std::exception& e) {
                std::cerr << "[eval] failed: " << e.what() << "\n";
                if (require_gt) return 7;
            }
        } else {
            if (require_gt) {
                std::cerr << "[eval] GT missing: " << gt_path << "\n";
                return 7;
            }
            std::cerr << "[eval] GT not found (skipped): " << gt_path << "\n";
        }
    }

    if (print_n > 0) {
        int pn = std::min(print_n, ds.n);
        for (int i = 0; i < pn; ++i) {
            std::cerr << "id:" << i << "\n";
            const auto* nbr = g.nbr_ptr(i);
            const auto* di  = g.dist_ptr(i);
            for (int t = 0; t < k; ++t) {
                std::cerr << "  " << t << ": " << nbr[t] << " dist=" << di[t] << "\n";
            }
        }
    }

    auto t1 = std::chrono::steady_clock::now();

    if (do_time) {
        auto ms = [](auto a, auto b) -> double {
            return (double)std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
        };
        std::cerr << "[time] load_ms=" << ms(t0, t_load)
                  << " alg_ms=" << ms(t_alg0, t_alg1)
                  << " total_ms=" << ms(t0, t1) << "\n";
    }

    std::cout << "Done.\n";
    return 0;
}
