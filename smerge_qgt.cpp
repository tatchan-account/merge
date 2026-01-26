#include <iostream>
#include <string>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <numeric>

#include "dataset.h"
#include "l2_squared.hpp"
#include "random_init.h"
#include "nndescent.h"
#include "smerge.h"
#include "gt_io.h"
#include "qgt_io.h"
#include "qgt_eval.h"

namespace fs = std::filesystem;

static void usage(const char* prog) {
    std::cerr
            << "Usage: " << prog << " <name.txt> <k> <split_ratio_or_n1>\n"
            << "  reads : dataset/<name.txt>\n"
            << "  checks: ans/<name.txt> (or and/<name.txt>) if exists\n"
            << "Examples:\n"
            << "  " << prog << " 0.txt 50 0.5\n"
            << "  " << prog << " 0.txt 50 100000\n"
            << "Options:\n"
            << "  --keep <float>      keep_ratio for S-merge (default 0.5)\n"
            << "  --rho <float>       (default 0.5)\n"
            << "  --delta <float>     (default 0.001)\n"
            << "  --iter <int>        (default 20)\n"
            << "  --seed <uint64>     (default 12345)\n"
            << "  --eval-n <int>      evaluate recall on first eval-n points (default n)\n"
            << "  --require-gt        if GT missing, exit error\n"
            << "  --print-n <int>     print first print-n rows (debug)\n"
            << "  --time              print timing summary\n";
}

static bool is_flag(const std::string& s, const char* name) { return s == name; }

// 入力が "0.txt" みたいにディレクトリ無しなら dataset/ を付ける
static fs::path dataset_path_from_arg(const std::string& arg) {
    fs::path p(arg);
    if (arg.find('/') != std::string::npos || arg.find('\\') != std::string::npos) return p;
    return fs::path("dataset") / p.filename();
}

// GT は ans/ を優先、なければ and/ を試す（ユーザーの誤記対策）
static fs::path gt_path_for_name(const fs::path& name_file) {
    fs::path ans = fs::path("ans") / name_file.filename();
    if (fs::exists(ans)) return ans;

    fs::path andd = fs::path("and") / name_file.filename();
    if (fs::exists(andd)) return andd;

    // 見つからない場合でも ans を返す（メッセージ用）
    return ans;
}



static fs::path qgt_path_for_name(const fs::path& name_file) {
    // ans/<filename>.qgt  (e.g., ans/sift1m.txt.qgt)
    return fs::path("ans") / (name_file.filename().string() + ".qgt");
}
// subset 用 distance: local index -> global index (offset を足すだけ)
struct OffsetDist {
    const L2Squared& base;
    int offset = 0;
    float operator()(int i, int j) const {
        return base(i + offset, j + offset);
    }
};

// recall@min(k, gt.k) over first eval_n points
static double recall_from_gt(const KNNGraph& g, const GroundTruth& gt, int eval_n) {
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

// デバッグ用：行を距離でソートして表示
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
    if (argc < 4) { usage(argv[0]); return 1; }

    fs::path name_file = fs::path(argv[1]).filename(); // "dataset/0.txt" でも "0.txt" にする
    const int k = std::stoi(argv[2]);
    const double split = std::stod(argv[3]);

    if (k <= 0) { std::cerr << "k must be positive.\n"; return 1; }

    // Params
    NNDParams p;
    float keep_ratio = 0.5f;
    bool require_gt = false;
    bool show_time = false;
    int eval_n = -1;
    int print_n = 0;

    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (is_flag(a, "--keep") && i + 1 < argc) {
            keep_ratio = std::stof(argv[++i]);
        } else if (is_flag(a, "--rho") && i + 1 < argc) {
            p.rho = std::stof(argv[++i]);
        } else if (is_flag(a, "--delta") && i + 1 < argc) {
            p.delta = std::stof(argv[++i]);
        } else if (is_flag(a, "--iter") && i + 1 < argc) {
            p.max_iter = std::stoi(argv[++i]);
        } else if (is_flag(a, "--seed") && i + 1 < argc) {
            p.seed = (uint64_t)std::stoull(argv[++i]);
        } else if (is_flag(a, "--eval-n") && i + 1 < argc) {
            eval_n = std::stoi(argv[++i]);
        } else if (is_flag(a, "--require-gt")) {
            require_gt = true;
        } else if (is_flag(a, "--print-n") && i + 1 < argc) {
            print_n = std::stoi(argv[++i]);
        } else if (is_flag(a, "--time")) {
            show_time = true;
        } else {
            std::cerr << "Unknown option: " << a << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    fs::path data_path = dataset_path_from_arg(name_file.string());
    if (!fs::exists(data_path)) {
        std::cerr << "Dataset not found: " << data_path.string() << "\n";
        return 1;
    }

    Dataset ds = Dataset::load_text(data_path.string());
    std::cout << "Loaded: n=" << ds.n << " d=" << ds.d << "\n";

    if (ds.n <= k + 1) {
        std::cerr << "n must be > k+1.\n";
        return 1;
    }
    if (eval_n < 0) eval_n = ds.n;
    if (eval_n < 1) eval_n = 1;

    // split -> n1
    int n1 = 0;
    if (split <= 1.0) n1 = (int)std::llround(split * (double)ds.n);
    else n1 = (int)std::llround(split);

    if (n1 < 1) n1 = 1;
    if (n1 > ds.n - 1) n1 = ds.n - 1;
    const int n2 = ds.n - n1;

    std::cout << "Split: n1=" << n1 << " n2=" << n2 << " (k=" << k << ")\n";
    if (n1 <= k || n2 <= k) {
        std::cerr << "Each subset must satisfy subset_size > k (adjust split or k).\n";
        return 1;
    }

    // GT auto load
    bool has_gt = false;
    GroundTruth gt;
    bool has_qgt = false;
    QueryGT qgt;
    fs::path gt_path = gt_path_for_name(name_file);
    fs::path qgt_path = qgt_path_for_name(name_file);

    if (fs::exists(qgt_path)) {
        try {
            qgt = load_qgt(qgt_path.string());
            has_qgt = true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load QGT: " << e.what() << "\n";
            return 1;
        }
        std::cout << "QGT loaded: " << qgt_path.string() << " (Q=" << qgt.Q << ", k_gt=" << qgt.k << ")\n";
    } else if (fs::exists(gt_path)) {
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
    } else {
        if (require_gt) {
            std::cerr << "GT not found (required): " << gt_path.string() << "\n";
            return 1;
        } else {
            std::cerr << "GT not found: " << gt_path.string() << " (skip recall)\n";
        }
    }

    // Dist + init
    L2Squared dist_global{ds};
    RandomInit init;

    using clk = std::chrono::steady_clock;
    auto t_all0 = clk::now();

    // 1) build subgraphs
    auto t0 = clk::now();
    OffsetDist dist1{dist_global, 0};
    OffsetDist dist2{dist_global, n1};

    KNNGraph g1 = nndescent_full(n1, k, dist1, init, p);
    KNNGraph g2 = nndescent_full(n2, k, dist2, init, p);
    auto t1 = clk::now();

    // 2) S-Merge
    KNNGraph g = s_merge(g1, g2, n1, dist_global, p, keep_ratio, /*verbose=*/false);
    auto t2 = clk::now();

    auto t_all1 = clk::now();

    std::cout << "Done.\n";

    if (show_time) {
        double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double merge_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double total_ms = std::chrono::duration<double, std::milli>(t_all1 - t_all0).count();
        std::cout << "time_build_subgraphs_ms = " << build_ms << "\n";
        std::cout << "time_smerge_ms          = " << merge_ms << "\n";
        std::cout << "time_total_ms           = " << total_ms << "\n";
    }

    if (has_qgt) {
        int ne = (eval_n < 0) ? (int)qgt.Q : std::min<int>(eval_n, (int)qgt.Q);
        int ke = std::min<int>(k, (int)qgt.k);
        auto rr = recall_from_qgt(g, qgt, ne, ke);
        std::cout << "recall@" << rr.k_eval << " over Q=" << rr.used_Q << " queries = " << rr.recall << "\n";
    } else if (has_gt) {
        int ne = std::min(eval_n, ds.n);
        int ke = std::min(k, gt.k);
        double r = recall_from_gt(g, gt, ne);
        std::cout << "recall@" << ke << " over first " << ne << " points = " << r << "\n";
    }

    if (print_n > 0) {
        print_rows_sorted_by_dist(g, print_n);
    }

    return 0;
}
