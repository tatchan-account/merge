#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>

/*
 * 正解ファイルのIOとRecall計算
 */

// 以下の入力（正確には正解ファイル）を想定
// ヘッダ"n k"があり、その後にn行"nb1 ... nbk" or "i nb1 ... nbk"が並ぶ
// ヘッダ"k"があり、その後にn行(nは数えるなりする)"nb1 ... nbk" or "i nb1 ... nbk"が並ぶ
struct FullGT {
    uint32_t n = 0;
    uint32_t k = 0;
    std::vector<uint32_t> nbr;
};

struct QueryGT {
    uint32_t Q = 0;
    uint32_t k = 0;
    std::vector<uint32_t> qid;        // size Q
    std::vector<uint32_t> nbr;        // size Q*k, row-major (distance-ordered)
};

static FullGT load_full_gt(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("load_full_gt: cannot open: " + path);

    std::string line;
    if (!std::getline(ifs, line)) throw std::runtime_error("load_full_gt: empty: " + path);

    auto parse_u64s = [](const std::string& s) {
        std::istringstream iss(s);
        std::vector<uint64_t> v;
        uint64_t x;
        while (iss >> x) v.push_back(x);
        return v;
    };

    std::vector<uint64_t> hdr = parse_u64s(line);
    if (hdr.empty()) throw std::runtime_error("load_full_gt: bad header: " + path);

    uint32_t n = 0, k = 0;
    if (hdr.size() >= 2) {
        n = (uint32_t)hdr[0];
        k = (uint32_t)hdr[1];
    } else {
        k = (uint32_t)hdr[0];
    }
    if (k == 0) throw std::runtime_error("load_full_gt: k==0: " + path);

    // read rows
    std::vector<std::vector<uint32_t>> rows;
    rows.reserve(n ? (size_t)n : 1024);

    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::vector<uint64_t> tok = parse_u64s(line);
        if (tok.empty()) continue;
        std::vector<uint32_t> r;
        r.reserve(tok.size());
        for (uint64_t t : tok) r.push_back((uint32_t)t);
        rows.push_back(std::move(r));
    }

    if (n == 0) n = (uint32_t)rows.size();
    if (rows.size() < (size_t)n) {
        throw std::runtime_error("load_full_gt: not enough rows (need n=" + std::to_string(n) +
                                 ", got " + std::to_string(rows.size()) + "): " + path);
    }

    FullGT gt;
    gt.n = n;
    gt.k = k;
    gt.nbr.resize((size_t)n * (size_t)k);

    // normalize each row--drop leading id if present; remove self & dups while preserving order; pad if short.
    for (uint32_t i = 0; i < n; ++i) {
        const auto& raw = rows[(size_t)i];
        size_t start = 0;
        if (raw.size() >= (size_t)k + 1 && raw[0] == i) start = 1; // "i nb1 ... nbk"
        std::vector<uint32_t> out;
        out.reserve(k);

        for (size_t t = start; t < raw.size() && out.size() < (size_t)k; ++t) {
            uint32_t id = raw[t];
            if (id == i) continue;
            bool dup = false;
            for (uint32_t x : out) { if (x == id) { dup = true; break; } }
            if (!dup) out.push_back(id);
        }
        if (out.empty()) out.push_back(i);
        while (out.size() < (size_t)k) out.push_back(out.back());

        std::copy(out.begin(), out.end(), gt.nbr.begin() + (size_t)i * (size_t)k);
    }

    return gt;
}

inline QueryGT load_qgt(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("load_qgt: cannot open: " + path);

    QueryGT gt;
    {
        std::string line;
        if (!std::getline(ifs, line)) throw std::runtime_error("load_qgt: empty file: " + path);
        std::istringstream iss(line);
        uint64_t Q64=0, k64=0;
        if (!(iss >> Q64 >> k64)) throw std::runtime_error("load_qgt: bad header: " + path);
        gt.Q = static_cast<uint32_t>(Q64);
        gt.k = static_cast<uint32_t>(k64);
        if (gt.Q == 0 || gt.k == 0) throw std::runtime_error("load_qgt: Q or k is zero: " + path);
    }

    gt.qid.resize(gt.Q);
    gt.nbr.resize(static_cast<size_t>(gt.Q) * gt.k);

    for (uint32_t qi = 0; qi < gt.Q; ++qi) {
        std::string line;
        if (!std::getline(ifs, line)) {
            throw std::runtime_error("load_qgt: not enough lines (expected Q lines): " + path);
        }
        std::istringstream iss(line);

        uint64_t qid64=0;
        if (!(iss >> qid64)) throw std::runtime_error("load_qgt: bad qid at line " + std::to_string(qi+2));
        const uint32_t qid = static_cast<uint32_t>(qid64);
        gt.qid[qi] = qid;

        // read neighbors, stable-unique, order-preserving
        std::vector<uint32_t> row;
        row.reserve(gt.k);

        for (uint32_t j = 0; j < gt.k; ++j) {
            uint64_t nb64=0;
            if (!(iss >> nb64)) throw std::runtime_error("load_qgt: bad neighbor count at line " + std::to_string(qi+2));
            uint32_t nb = static_cast<uint32_t>(nb64);

            if (nb == qid) continue; // safety (make_qgt excludes self)
            bool dup = false;
            for (uint32_t x : row) { if (x == nb) { dup = true; break; } }
            if (!dup) row.push_back(nb);

            if (row.size() == gt.k) break; // full (after self/dup skips)
        }

        // Pad if needed (shouldn't happen for exact QGT)
        if (row.size() < gt.k) {
            uint32_t pad = row.empty() ? qid : row.back();
            row.resize(gt.k, pad);
        } else if (row.size() > gt.k) {
            row.resize(gt.k);
        }

        std::copy(row.begin(), row.end(), gt.nbr.begin() + static_cast<size_t>(qi) * gt.k);
    }

    return gt;
}



// recall計算
static double recall_from_gt_topk(const kNNGraph& g, const FullGT& gt, int k_eval, int eval_n) {
    const int n = g.n();
    const int k_pred = g.k();
    const int en = (eval_n <= 0) ? n : std::min(eval_n, n);
    const int ke = std::min(k_eval, (int)gt.k);
    if (en <= 0 || ke <= 0) return 0.0;
    if ((int)gt.n < en) throw std::runtime_error("GT n smaller than eval_n");

    std::vector<int> idx;
    idx.reserve((size_t)k_pred);
    std::vector<uint32_t> top;
    top.reserve((size_t)ke);

    double sum = 0.0;
    for (int i = 0; i < en; ++i) {
        const uint32_t* nbr = g.nbr_ptr(i);
        const float*    ds  = g.dist_ptr(i);

        idx.clear();
        for (int t = 0; t < k_pred; ++t) {
            if (nbr[t] == kNNGraph::invalid_id()) continue;
            if (!std::isfinite(ds[t])) continue;
            idx.push_back(t);
        }
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return ds[a] < ds[b]; });

        top.clear();
        for (int t = 0; t < (int)idx.size() && (int)top.size() < ke; ++t) {
            top.push_back(nbr[idx[t]]);
        }

        const uint32_t* truth = gt.nbr.data() + (size_t)i * (size_t)gt.k;

        int hit = 0;
        for (int t = 0; t < ke; ++t) {
            const uint32_t e = truth[t];
            for (uint32_t p : top) {
                if (p == e) { ++hit; break; }
            }
        }
        sum += (double)hit / (double)ke;
    }
    return sum / (double)en;
}

static double recall_from_qgt_topk(const kNNGraph& g, const QueryGT& qgt, int k_eval, int eval_q) {
    const int k_pred = g.k();
    const int n = g.n();
    const int qe = (eval_q <= 0) ? (int)qgt.Q : std::min(eval_q, (int)qgt.Q);
    const int ke = std::min(k_eval, (int)qgt.k);
    if (ke <= 0) return 0.0;

    std::vector<int> idx;
    idx.reserve((size_t)k_pred);

    double sum = 0.0;
    for (int qi = 0; qi < qe; ++qi) {
        const uint32_t qid = qgt.qid[(size_t)qi];
        if (qid >= (uint32_t)n) throw std::runtime_error("QGT qid out of range");

        const uint32_t* nbr = g.nbr_ptr((int)qid);
        const float*    ds  = g.dist_ptr((int)qid);

        idx.clear();
        for (int t = 0; t < k_pred; ++t) {
            if (nbr[t] == kNNGraph::invalid_id()) continue;
            idx.push_back(t);
        }
        // sort by distance
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return ds[a] < ds[b]; });

        // collect top-ke ids
        std::vector<uint32_t> top;
        top.reserve((size_t)ke);
        for (int t = 0; t < (int)idx.size() && (int)top.size() < ke; ++t) {
            top.push_back(nbr[idx[t]]);
        }

        const uint32_t* truth = qgt.nbr.data() + (size_t)qi * (size_t)qgt.k;

        int hit = 0;
        for (int t = 0; t < ke; ++t) {
            const uint32_t e = truth[t];
            for (uint32_t p : top) {
                if (p == e) { ++hit; break; }
            }
        }
        sum += (double)hit / (double)ke;
    }
    return sum / (double)qe;
}