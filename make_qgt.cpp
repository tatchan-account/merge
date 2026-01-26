#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <cstdint>
#include <filesystem>

#include "dataset.h"
#include "splitmix64.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

static fs::path dataset_path_for_arg(const std::string& arg) {
    fs::path p(arg);
    if (p.has_parent_path()) return p;
    return fs::path("dataset") / p;
}

static fs::path qgt_path_for_dataset(const fs::path& name_file) {
    // ans/<filename>.qgt  (e.g., ans/sift1m.txt.qgt)
    return fs::path("ans") / (name_file.filename().string() + ".qgt");
}

struct Pair {
    float dist;
    uint32_t id;
};
struct WorseFirst {
    bool operator()(const Pair& a, const Pair& b) const { return a.dist < b.dist; } // max-heap by dist
};

static std::vector<uint32_t> reservoir_sample_u32(uint32_t n, uint32_t Q, SplitMix64& rng) {
    if (Q > n) Q = n;
    std::vector<uint32_t> res;
    res.reserve(Q);
    for (uint32_t i = 0; i < Q; ++i) res.push_back(i);

    for (uint32_t i = Q; i < n; ++i) {
        uint64_t j = rng.next_u64() % (uint64_t)(i + 1);
        if (j < Q) res[(size_t)j] = i;
    }
    // shuffle for better mix
    for (uint32_t i = Q - 1; i > 0; --i) {
        uint64_t j = rng.next_u64() % (uint64_t)(i + 1);
        std::swap(res[i], res[(size_t)j]);
    }
    return res;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <dataset_name_or_path> <Q> <k_gt> [--seed S] [--out path]\n";
        std::cerr << "  dataset: if no directory, read from dataset/<name>\n";
        std::cerr << "  output: default ans/<name>.qgt\n";
        return 1;
    }
    std::string name_arg = argv[1];
    uint32_t Q = (uint32_t)std::stoul(argv[2]);
    uint32_t kgt = (uint32_t)std::stoul(argv[3]);

    uint64_t seed = 12345;
    fs::path out_path;

    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--seed" && i + 1 < argc) { seed = std::stoull(argv[++i]); }
        else if (a == "--out" && i + 1 < argc) { out_path = fs::path(argv[++i]); }
        else {
            std::cerr << "Unknown arg: " << a << "\n";
            return 1;
        }
    }

    fs::path data_path = dataset_path_for_arg(name_arg);
    if (!fs::exists(data_path)) {
        std::cerr << "Dataset not found: " << data_path.string() << "\n";
        return 1;
    }

    Dataset ds = Dataset::load_text(data_path.string());
    std::cout << "Loaded: n=" << ds.n << " d=" << ds.d << "\n";
    if (ds.n < 2) { std::cerr << "Dataset too small.\n"; return 1; }
    if (kgt >= (uint32_t)ds.n) {
        std::cerr << "k_gt(" << kgt << ") >= n(" << ds.n << "), clamping to n-1.\n";
        kgt = (uint32_t)ds.n - 1;
    }
    if (Q > (uint32_t)ds.n) Q = (uint32_t)ds.n;

    if (out_path.empty()) out_path = qgt_path_for_dataset(data_path);
    fs::create_directories(out_path.parent_path());

    SplitMix64 rng(seed);
    std::vector<uint32_t> qids = reservoir_sample_u32((uint32_t)ds.n, Q, rng);

    dist_func dist{ds};

    // storage: Q * kgt ids
    std::vector<uint32_t> out_nbr((size_t)Q * (size_t)kgt, 0);

    std::cout << "Compute exact QGT: Q=" << Q << " k_gt=" << kgt << " ...\n";

#ifdef _OPENMP
    int maxT = omp_get_max_threads();
    std::cout << "OpenMP threads: " << maxT << "\n";
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int qi = 0; qi < (int)Q; ++qi) {
        uint32_t qid = qids[(size_t)qi];

        std::priority_queue<Pair, std::vector<Pair>, WorseFirst> heap;
        heap = decltype(heap)();

        for (uint32_t j = 0; j < (uint32_t)ds.n; ++j) {
            if (j == qid) continue;
            float d2 = dist((int)qid, (int)j);
            if ((uint32_t)heap.size() < kgt) {
                heap.push(Pair{d2, j});
            } else if (d2 < heap.top().dist) {
                heap.pop();
                heap.push(Pair{d2, j});
            }
        }

        std::vector<Pair> tmp;
        tmp.reserve(kgt);
        while (!heap.empty()) { tmp.push_back(heap.top()); heap.pop(); }
        std::sort(tmp.begin(), tmp.end(), [](const Pair& a, const Pair& b){
            if (a.dist != b.dist) return a.dist < b.dist;
            return a.id < b.id;
        });

        // write ids
        size_t base = (size_t)qi * (size_t)kgt;
        for (uint32_t t = 0; t < kgt; ++t) out_nbr[base + t] = tmp[t].id;
    }

    // write file
    std::ofstream ofs(out_path);
    if (!ofs) { std::cerr << "Cannot open output: " << out_path.string() << "\n"; return 1; }

    ofs << Q << " " << kgt << "\n";
    for (uint32_t qi = 0; qi < Q; ++qi) {
        ofs << qids[qi];
        size_t base = (size_t)qi * (size_t)kgt;
        for (uint32_t t = 0; t < kgt; ++t) ofs << " " << out_nbr[base + t];
        ofs << "\n";
    }
    ofs.close();

    std::cout << "Wrote: " << out_path.string() << "\n";
    return 0;
}
