// gtgen.cpp
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <filesystem>

#include "dataset.h"

namespace fs = std::filesystem;

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <name.txt> <k_gt>\n"
              << "  reads : dataset/<name.txt>\n"
              << "  writes: ans/<name.txt>\n";
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(argv[0]); return 1; }

    fs::path in_name = fs::path(argv[1]).filename(); // ディレクトリが付いてても無視して filename を使う
    int k_gt = std::atoi(argv[2]);
    if (k_gt <= 0) { std::cerr << "k_gt must be positive.\n"; return 1; }

    fs::path data_path = fs::path("dataset") / in_name;
    fs::path ans_dir   = fs::path("ans");
    fs::path gt_path   = ans_dir / in_name;

    if (!fs::exists(data_path)) {
        std::cerr << "Dataset not found: " << data_path.string() << "\n";
        return 1;
    }
    if (!fs::exists(ans_dir)) fs::create_directories(ans_dir);

    Dataset ds = Dataset::load_text(data_path.string());
    std::cerr << "Loaded: n=" << ds.n << " d=" << ds.d << "\n";

    if (ds.n <= 1) { std::cerr << "Dataset must have at least 2 points.\n"; return 1; }
    if (k_gt >= ds.n) {
        std::cerr << "Warning: k_gt(" << k_gt << ") >= n(" << ds.n << "), clamping to n-1.\n";
        k_gt = ds.n - 1;
    }

    dist_func dist{ds};
    std::ofstream ofs(gt_path.string());
    if (!ofs) { std::cerr << "Failed to write: " << gt_path.string() << "\n"; return 1; }

    ofs << ds.n << " " << k_gt << "\n";

    std::vector<std::pair<float, uint32_t>> buf;
    buf.reserve((size_t)ds.n - 1);

    for (int i = 0; i < ds.n; ++i) {
        buf.clear();
        for (int j = 0; j < ds.n; ++j) {
            if (i == j) continue;
            buf.push_back({dist(i, j), (uint32_t)j});
        }
        std::nth_element(buf.begin(), buf.begin() + k_gt, buf.end(),
                         [](auto& a, auto& b){ return a.first < b.first; });
        std::sort(buf.begin(), buf.begin() + k_gt,
                  [](auto& a, auto& b){ return a.first < b.first; });

        for (int t = 0; t < k_gt; ++t) {
            ofs << buf[t].second << (t + 1 == k_gt ? '\n' : ' ');
        }
    }

    std::cerr << "GT saved: " << gt_path.string() << "\n";
    return 0;
}
