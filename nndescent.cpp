// nndescent.cpp
// Baseline NN-Descent runner (random init) with optional evaluation:
//   Prefer full GT at ans/<name.txt> if it exists, else use QGT at ans/<name.txt>.qgt if it exists.
//   (Legacy fallback: and/<name> then and/<name>.qgt)

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <chrono>
#include <filesystem>

#include "dataset.h"
#include "l2_squared.hpp"
#include "random_init.h"
#include "nnd_params.hpp"
#include "nndescent.h"
#include "gt_io.h"
#include "qgt_io.h"
#include "qgt_eval.h"
#include "exp_log.h"

namespace fs = std::filesystem;

static void usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " <name.txt> <k>\n"
        << "  reads : dataset/<name.txt>\n"
        << "  checks (preferred):\n"
        << "    1) ans/<name.txt>        (full GT)\n"
        << "    2) ans/<name.txt>.qgt    (QGT)\n"
        << "  legacy fallback:\n"
        << "    3) and/<name.txt>        (full GT)\n"
        << "    4) and/<name.txt>.qgt    (QGT)\n"
        << "Options:\n"
        << "  --rho <float>        (default 0.5)\n"
        << "  --delta <float>      (default 0.001)\n"
        << "  --iter <int>         (default 20)\n"
        << "  --seed <uint64>      (default 12345)\n"
        << "  --verbose            (iteration logs; reserved)\n"
        << "  --time               (print total time)\n"
        << "  --eval-n <int>       (default n; for QGT: #queries; for GT: #points)\n"
        << "  --eval-q <int>       (alias of --eval-n)\n"
        << "  --print-n <int>      (default 0; print first print-n rows)\n"
        << "  --require-gt         (if neither GT nor QGT exists, exit with error)\n";
}

static bool is_flag(const std::string& s, const char* name) { return s == name; }

static fs::path ans_gt_path(const fs::path& name_file) {
    return fs::path("ans") / name_file.filename();
}
static fs::path ans_qgt_path(const fs::path& name_file) {
    return fs::path("ans") / (name_file.filename().string() + ".qgt");
}
static fs::path and_gt_path(const fs::path& name_file) {
    return fs::path("and") / name_file.filename();
}
static fs::path and_qgt_path(const fs::path& name_file) {
    return fs::path("and") / (name_file.filename().string() + ".qgt");
}

// recall@k over first eval_n points
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

static void print_rows_sorted_by_dist(const KNNGraph& g, int print_n) {
    const int n = g.n();
    const int k = g.k();
    int pn = std::min(print_n, n);

    for (int i = 0; i < pn; ++i) {
        std::cout << "id :" << i << "\n";
        const auto* nbr = g.nbr_ptr(i);
        const auto* ds0 = g.dist_ptr(i);

        std::vector<int> idx(k);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b){
            return ds0[a] < ds0[b];
        });

        for (int t = 0; t < k; ++t) {
            int p = idx[t];
            std::cout << "\t" << t << "-th :" << nbr[p] << " dist=" << ds0[p] << "\n";
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(argv[0]); return 1; }

    fs::path in_name = fs::path(argv[1]).filename(); // allow "dataset/0.txt" -> "0.txt"
    int k = std::atoi(argv[2]);
    if (k <= 0) { std::cerr << "k must be positive.\n"; return 1; }

    NNDParams p;
    bool verbose = false;
    bool show_time = false;
    bool require_gt = false;
    int eval_n = -1;   // default: n (or Q for QGT)
    int print_n = 0;

    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (is_flag(a, "--rho") && i + 1 < argc) {
            p.rho = std::stof(argv[++i]);
        } else if (is_flag(a, "--delta") && i + 1 < argc) {
            p.delta = std::stof(argv[++i]);
        } else if (is_flag(a, "--iter") && i + 1 < argc) {
            p.max_iter = std::atoi(argv[++i]);
        } else if (is_flag(a, "--seed") && i + 1 < argc) {
            p.seed = (uint64_t)std::stoull(argv[++i]);
        } else if (is_flag(a, "--verbose")) {
            verbose = true; // reserved; nndescent_full may ignore this
        } else if (is_flag(a, "--time")) {
            show_time = true;
        } else if ((is_flag(a, "--eval-n") || is_flag(a, "--eval-q")) && i + 1 < argc) {
            eval_n = std::atoi(argv[++i]);
        } else if (is_flag(a, "--print-n") && i + 1 < argc) {
            print_n = std::atoi(argv[++i]);
        } else if (is_flag(a, "--require-gt")) {
            require_gt = true;
        } else {
            std::cerr << "Unknown option: " << a << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    fs::path data_path = fs::path("dataset") / in_name;

    if (!fs::exists(data_path)) {
        std::cerr << "Dataset not found: " << data_path.string() << "\n";
        return 1;
    }

    Dataset ds = Dataset::load_text(data_path.string());
    std::cout << "Loaded: n=" << ds.n << " d=" << ds.d << "\n";

    if (ds.n <= 1) { std::cerr << "Dataset must have at least 2 points.\n"; return 1; }
    if (k >= ds.n) {
        std::cerr << "Warning: k(" << k << ") >= n(" << ds.n << "), clamping to n-1.\n";
        k = ds.n - 1;
    }
    if (eval_n < 0) eval_n = ds.n;

    // Prefer full GT if exists, else QGT.
    bool has_gt = false;
    GroundTruth gt;
    fs::path gt_path;

    bool has_qgt = false;
    QueryGT qgt;
    fs::path qgt_path;

    // Search order: ans/GT -> ans/QGT -> and/GT -> and/QGT
    const fs::path gt1 = ans_gt_path(in_name);
    const fs::path qg1 = ans_qgt_path(in_name);
    const fs::path gt2 = and_gt_path(in_name);
    const fs::path qg2 = and_qgt_path(in_name);

    if (fs::exists(gt1)) {
        gt_path = gt1;
        try {
            gt = load_gt(gt_path.string());
            has_gt = true;
            if (gt.n != ds.n) {
                std::cerr << "GT n mismatch: gt.n=" << gt.n << " vs data.n=" << ds.n << "\n";
                return 1;
            }
        } catch (const std::exception& e) {
            std::cerr << "Failed to load GT: " << e.what() << "\n";
            return 1;
        }
        std::cout << "GT loaded: " << gt_path.string() << " (k_gt=" << gt.k << ")\n";
    } else if (fs::exists(qg1)) {
        qgt_path = qg1;
        try {
            qgt = load_qgt(qgt_path.string());
            has_qgt = true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load QGT: " << e.what() << "\n";
            return 1;
        }
        std::cout << "QGT loaded: " << qgt_path.string()
                  << " (Q=" << qgt.Q << ", k_gt=" << qgt.k << ")\n";
    } else if (fs::exists(gt2)) {
        gt_path = gt2;
        try {
            gt = load_gt(gt_path.string());
            has_gt = true;
            if (gt.n != ds.n) {
                std::cerr << "GT n mismatch: gt.n=" << gt.n << " vs data.n=" << ds.n << "\n";
                return 1;
            }
        } catch (const std::exception& e) {
            std::cerr << "Failed to load GT: " << e.what() << "\n";
            return 1;
        }
        std::cout << "GT loaded: " << gt_path.string() << " (k_gt=" << gt.k << ")\n";
    } else if (fs::exists(qg2)) {
        qgt_path = qg2;
        try {
            qgt = load_qgt(qgt_path.string());
            has_qgt = true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load QGT: " << e.what() << "\n";
            return 1;
        }
        std::cout << "QGT loaded: " << qgt_path.string()
                  << " (Q=" << qgt.Q << ", k_gt=" << qgt.k << ")\n";
    } else {
        if (require_gt) {
            std::cerr << "Neither GT nor QGT found (required). Tried:\n"
                      << "  " << gt1.string() << "\n"
                      << "  " << qg1.string() << "\n"
                      << "  " << gt2.string() << "\n"
                      << "  " << qg2.string() << "\n";
            return 1;
        } else {
            std::cerr << "GT/QGT not found (skip recall). Tried:\n"
                      << "  " << gt1.string() << "\n"
                      << "  " << qg1.string() << "\n"
                      << "  " << gt2.string() << "\n"
                      << "  " << qg2.string() << "\n";
        }
    }

    L2Squared dist{ds};
    RandomInit init;

    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();

    (void)verbose;
    KNNGraph g = nndescent_full(ds.n, k, dist, init, p);

    auto t1 = clk::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Done.\n";
    if (show_time) std::cout << "time_total_ms = " << total_ms << "\n";

    if (has_gt) {
        int ne = std::min(eval_n, ds.n);
        int ke = std::min(k, gt.k);
        double r = recall_from_gt(g, gt, ne);
        std::cout << "recall@" << ke << " over first " << ne << " points = " << r << "\n";
    } else if (has_qgt) {
        const int qe = (eval_n < 0) ? (int)qgt.Q : std::min<int>(eval_n, (int)qgt.Q);
        const int ke = std::min<int>(k, (int)qgt.k);
        auto rr = recall_from_qgt(g, qgt, qe, ke);
        std::cout << "recall@" << rr.k_eval << " over Q=" << rr.used_Q << " queries = " << rr.recall << "\n";
    }

    if (print_n > 0) {
        print_rows_sorted_by_dist(g, print_n);
    }

    return 0;
}
