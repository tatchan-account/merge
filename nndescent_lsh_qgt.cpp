// nndescent_lsh_qgt.cpp
// NN-Descent executable with optional LSH-based initialization and optional (GT/QGT) evaluation.
//
// Policy:
//  - dataset is in ./dataset/<name>
//  - Prefer full GT:   ./ans/<name> (or ./and/<name>)
//  - Else prefer QGT:  ./ans/<name>.qgt (or ./and/<name>.qgt)
//
// Build (Ubuntu):
//   g++ -O3 -std=c++17 -march=native -fopenmp -Wall -Wextra nndescent_lsh_qgt.cpp -o qnndescent_lsh
//
// Run:
//   ./qnndescent_lsh sift1m.txt 20 --init lsh --lsh-L 10 --lsh-K 4 --lsh-w 4.0 --lsh-cand 256 --time
//
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>

#include "dataset.h"
#include "l2_squared.hpp"
#include "knngraph.h"
#include "nnd_params.hpp"
#include "nndescent.h"
#include "random_init.h"
#include "splitmix64.h"
#include "lsh_pstable.h"
#include "lsh_init.h"
#include "gt_io.h"
#include "qgt_io.h"
#include "qgt_eval.h"

namespace fs = std::filesystem;

static void usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " <name_file> <k> [options]\n"
        << "  name_file: dataset filename under ./dataset/ (e.g., sift1m.txt)\n"
        << "  k:         target k for k-NN graph\n"
        << "\nOptions:\n"
        << "  --seed <u64>       RNG seed (default 12345)\n"
        << "  --iter <int>       max NN-Descent iterations (default 10)\n"
        << "  --rho <float>      rho (default 0.5)\n"
        << "  --delta <float>    delta (default 0.001)\n"
        << "\nInit:\n"
        << "  --init <random|lsh>   initializer (default random)\n"
        << "\nLSH init params (used when --init lsh):\n"
        << "  --lsh-L <int>      number of hash tables (default 10)\n"
        << "  --lsh-K <int>      hashes per table (default 4)\n"
        << "  --lsh-w <float>    bucket width (default 4.0)\n"
        << "  --lsh-bucket <int> bucket cap per key (default 64)\n"
        << "  --lsh-cand <int>   candidates per point from LSH (default 256)\n"
        << "  --lsh-seedk <int>  #LSH neighbors used in init (-1 auto)\n"
        << "  --lsh-randmin <int> min random neighbors in init (default 2)\n"
        << "  --lsh-norank       do not rank candidates by exact distance\n"
        << "\nEval:\n"
        << "  --eval-n <int>     eval limit (GT: first N points; QGT: first N queries; default all)\n"
        << "  --eval-q <int>     alias of --eval-n\n"
        << "  --require-gt       if neither GT nor QGT exists, exit with error\n"
        << "  --time             print timing (ms)\n";
}

static fs::path dataset_path_for_name(const fs::path& name_file) {
    return fs::path("dataset") / name_file;
}

// Evaluation path policy (in this order):
//   1) ans/<name>      (full GT)
//   2) ans/<name>.qgt  (QGT)
//   3) and/<name>      (full GT; legacy fallback)
//   4) and/<name>.qgt  (QGT; legacy fallback)
struct EvalPaths {
    fs::path gt_path_ans;
    fs::path qgt_path_ans;
    fs::path gt_path_and;
    fs::path qgt_path_and;
};

static EvalPaths eval_paths_for_name(const fs::path& name_file) {
    EvalPaths ep;
    const auto base = name_file.filename();
    ep.gt_path_ans  = fs::path("ans") / base;
    ep.qgt_path_ans = fs::path("ans") / (base.string() + ".qgt");
    ep.gt_path_and  = fs::path("and") / base;
    ep.qgt_path_and = fs::path("and") / (base.string() + ".qgt");
    return ep;
}

// recall@min(k, gt.k) over first eval_n points
static double recall_from_gt(const KNNGraph& g,
                             const GroundTruth& gt,
                             int eval_n) {
    const int n = g.n();
    const int k = g.k();
    const int ne = std::min(eval_n, n);
    const int ke = std::min(k, gt.k);
    if (ne <= 0 || ke <= 0) return 0.0;

    double sum = 0.0;
    for (int i = 0; i < ne; ++i) {
        const auto* nbr = g.nbr_ptr(i);
        const uint32_t* gr = gt.row(i);
        int hit = 0;
        for (int t = 0; t < ke; ++t) {
            uint32_t e = gr[t];
            for (int s = 0; s < k; ++s) {
                if (nbr[s] == e) { ++hit; break; }
            }
        }
        sum += (double)hit / (double)ke;
    }
    return sum / (double)ne;
}

struct Timer {
    std::chrono::high_resolution_clock::time_point t0;
    void tic(){ t0 = std::chrono::high_resolution_clock::now(); }
    double toc_ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double,std::milli>(t1 - t0).count();
    }
};

int main(int argc, char** argv) {
    if (argc < 3) { usage(argv[0]); return 1; }

    // Allow passing "dataset/foo.txt"; we normalize to the filename.
    const fs::path name_file = fs::path(argv[1]).filename();
    const int k = std::stoi(argv[2]);
    if (k <= 0) { std::cerr << "k must be positive\n"; return 1; }

    NNDParams p;
    p.max_iter = 10;
    p.rho = 0.5f;
    p.delta = 0.001f;
    p.seed = 12345;

    std::string init_mode = "random";

    NNDLSHInitParams ip;
    // sensible defaults (match merge side)
    ip.lsh.L = 10;
    ip.lsh.K = 4;
    ip.lsh.w = 4.0f;
    ip.lsh.bucket_cap = 64;
    ip.cand_cap = 256;
    ip.seed_k = -1;
    ip.rand_min = 2;
    ip.rank_by_distance = true;

    int eval_n = -1; // (GT: #points, QGT: #queries). -1 means "all".
    bool do_time = false;
    bool require_gt = false;

    for (int i = 3; i < argc; ++i) {
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
            else if (a == "--init") init_mode = need("--init");
            else if (a == "--lsh-L") ip.lsh.L = std::stoi(need("--lsh-L"));
            else if (a == "--lsh-K") ip.lsh.K = std::stoi(need("--lsh-K"));
            else if (a == "--lsh-w") ip.lsh.w = std::stof(need("--lsh-w"));
            else if (a == "--lsh-bucket") ip.lsh.bucket_cap = std::stoi(need("--lsh-bucket"));
            else if (a == "--lsh-cand") ip.cand_cap = std::stoi(need("--lsh-cand"));
            else if (a == "--lsh-seedk") ip.seed_k = std::stoi(need("--lsh-seedk"));
            else if (a == "--lsh-randmin") ip.rand_min = std::stoi(need("--lsh-randmin"));
            else if (a == "--lsh-norank") ip.rank_by_distance = false;
            else if (a == "--eval-n") eval_n = std::stoi(need("--eval-n"));
            else if (a == "--eval-q") eval_n = std::stoi(need("--eval-q"));
            else if (a == "--require-gt") require_gt = true;
            else if (a == "--time") do_time = true;
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

    if (init_mode != "random" && init_mode != "lsh") {
        std::cerr << "Invalid --init: " << init_mode << " (use random|lsh)\n";
        return 1;
    }

    // keep LSH deterministic w.r.t. --seed
    ip.lsh.seed = p.seed;

    // Load dataset
    const fs::path data_path = dataset_path_for_name(name_file);
    Dataset ds;
    try {
        ds = Dataset::load_text(data_path.string());
    } catch (const std::exception& e) {
        std::cerr << "Failed to load dataset: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Loaded: n=" << ds.n << " d=" << ds.d << "\n";
    if (ds.n <= k + 1) { std::cerr << "Dataset too small.\n"; return 1; }

    L2Squared dist{ds};

    Timer tt;
    double t_build = 0.0;

    tt.tic();
    KNNGraph g = (init_mode == "random")
        ? nndescent_full(ds.n, k, dist, RandomInit{}, p)
        : [&](){
            std::cout << "Build LSH index (init): L=" << ip.lsh.L << " K=" << ip.lsh.K << " w=" << ip.lsh.w
                      << " bucket_cap=" << ip.lsh.bucket_cap << " seed=" << ip.lsh.seed
                      << " cand_cap=" << ip.cand_cap << "\n";
            LSHInit init(ds, ip);
            return nndescent_full(ds.n, k, dist, init, p);
        }();
    t_build = tt.toc_ms();

    std::cout << "Done.\n";

    // Prefer GT (ans/<name>) if exists, otherwise QGT (ans/<name>.qgt).
    // Legacy fallback: ./and/
    bool has_gt = false;
    bool has_qgt = false;
    GroundTruth gt;
    QueryGT qgt;
    const EvalPaths ep = eval_paths_for_name(name_file);

    fs::path used_gt_path;
    fs::path used_qgt_path;

    if (fs::exists(ep.gt_path_ans)) {
        used_gt_path = ep.gt_path_ans;
        try {
            gt = load_gt(used_gt_path.string());
            if (gt.n != ds.n) {
                throw std::runtime_error("GT n mismatch: gt.n=" + std::to_string(gt.n) + " vs data.n=" + std::to_string(ds.n));
            }
            has_gt = true;
            std::cout << "GT loaded: " << used_gt_path.string() << " (k_gt=" << gt.k << ")\n";
        } catch (const std::exception& e) {
            std::cerr << "Failed to load GT: " << e.what() << "\n";
            return 1;
        }
    } else if (fs::exists(ep.qgt_path_ans)) {
        used_qgt_path = ep.qgt_path_ans;
        try {
            qgt = load_qgt(used_qgt_path.string());
            has_qgt = true;
            std::cout << "QGT loaded: " << used_qgt_path.string() << " (Q=" << qgt.Q << ", k_gt=" << qgt.k << ")\n";
        } catch (const std::exception& e) {
            std::cerr << "Failed to load QGT: " << e.what() << "\n";
            return 1;
        }
    } else if (fs::exists(ep.gt_path_and)) {
        used_gt_path = ep.gt_path_and;
        try {
            gt = load_gt(used_gt_path.string());
            if (gt.n != ds.n) {
                throw std::runtime_error("GT n mismatch: gt.n=" + std::to_string(gt.n) + " vs data.n=" + std::to_string(ds.n));
            }
            has_gt = true;
            std::cout << "GT loaded: " << used_gt_path.string() << " (k_gt=" << gt.k << ")\n";
        } catch (const std::exception& e) {
            std::cerr << "Failed to load GT: " << e.what() << "\n";
            return 1;
        }
    } else if (fs::exists(ep.qgt_path_and)) {
        used_qgt_path = ep.qgt_path_and;
        try {
            qgt = load_qgt(used_qgt_path.string());
            has_qgt = true;
            std::cout << "QGT loaded: " << used_qgt_path.string() << " (Q=" << qgt.Q << ", k_gt=" << qgt.k << ")\n";
        } catch (const std::exception& e) {
            std::cerr << "Failed to load QGT: " << e.what() << "\n";
            return 1;
        }
    } else {
        if (require_gt) {
            std::cerr << "Neither GT nor QGT found (required). Tried: "
                      << ep.gt_path_ans.string() << ", " << ep.qgt_path_ans.string()
                      << ", " << ep.gt_path_and.string() << ", " << ep.qgt_path_and.string() << "\n";
            return 2;
        }
        std::cerr << "GT/QGT not found (skip eval). Tried: "
                  << ep.gt_path_ans.string() << ", " << ep.qgt_path_ans.string()
                  << ", " << ep.gt_path_and.string() << ", " << ep.qgt_path_and.string() << "\n";
    }

    if (has_gt) {
        const int ne = (eval_n < 0) ? ds.n : std::min(eval_n, ds.n);
        const int ke = std::min(k, gt.k);
        double r = recall_from_gt(g, gt, ne);
        std::cout << "recall@" << ke << " over first " << ne << " points = " << r << "\n";
    } else if (has_qgt) {
        auto res = recall_from_qgt(g, qgt, eval_n, k);
        std::cout << "recall@" << res.k_eval << " over Q=" << res.used_Q << " queries = " << res.recall << "\n";
    }

    if (do_time) {
        std::cout << "time_build_ms  = " << (long long)t_build << "\n";
        std::cout << "time_total_ms  = " << (long long)t_build << "\n";
    }

    return 0;
}
