// exp_log.hpp
#pragma once
#include <string>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>

namespace fs = std::filesystem;

inline std::string now_iso8601() {
    using clk = std::chrono::system_clock;
    auto t = clk::to_time_t(clk::now());
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

inline void ensure_logs_dir() {
    fs::path dir("logs");
    if (!fs::exists(dir)) fs::create_directories(dir);
}

inline void append_summary_csv_header_if_needed(const fs::path& path) {
    if (fs::exists(path) && fs::file_size(path) > 0) return;
    std::ofstream ofs(path, std::ios::app);
    ofs << "ts,algo,data,n,d,k,split,keep,rho,delta,max_iter,seed,eval_n,has_gt,k_gt,recall,"
           "time_total_ms,time_build_ms,time_merge_ms,iters,changes_last\n";
}

struct SummaryRow {
    std::string ts;
    std::string algo;
    std::string data; // filename only

    int n = 0;
    int d = 0;
    int k = 0;

    // optional params (set to NaN-ish string when unused)
    std::string split = ""; // e.g. "0.5" or "100000"
    std::string keep  = ""; // e.g. "0.5"

    float rho = 0.5f;
    float delta = 0.001f;
    int max_iter = 20;
    uint64_t seed = 0;

    int eval_n = 0;
    int has_gt = 0;
    int k_gt = 0;
    double recall = -1.0;

    double time_total_ms = 0.0;
    double time_build_ms = 0.0;
    double time_merge_ms = 0.0;

    int iters = 0;               // 実際に回った反復数（分かる範囲で）
    long long changes_last = -1;  // 最終反復 changes（分かる範囲で）
};

inline void append_summary_row(const SummaryRow& r) {
    ensure_logs_dir();
    fs::path path = fs::path("logs") / "summary.csv";
    append_summary_csv_header_if_needed(path);

    std::ofstream ofs(path, std::ios::app);
    ofs << r.ts << "," << r.algo << "," << r.data << ","
        << r.n << "," << r.d << "," << r.k << ","
        << r.split << "," << r.keep << ","
        << r.rho << "," << r.delta << "," << r.max_iter << "," << r.seed << ","
        << r.eval_n << "," << r.has_gt << "," << r.k_gt << "," << r.recall << ","
        << r.time_total_ms << "," << r.time_build_ms << "," << r.time_merge_ms << ","
        << r.iters << "," << r.changes_last
        << "\n";
}
