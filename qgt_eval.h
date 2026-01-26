// qgt_eval.hpp
// Recall computation against QGT (Query Ground Truth).
//
// QGT format is defined in qgt_io.hpp.
// We compute recall@k_eval averaged over first eval_q queries (eval_q<=Q).
//
// NOTE: This is meant for fast evaluation when full GT (n*k) is too expensive.

#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "knngraph.h"
#include "qgt_io.h"

struct QGTEvalResult {
    double recall = 0.0;
    int used_Q = 0;
    int k_eval = 0;
};

inline QGTEvalResult recall_from_qgt(const KNNGraph& g,
                                    const QueryGT& qgt,
                                    int eval_q,
                                    int k_eval,
                                    bool validate_ids = true) {
    const int n = g.n();
    const int k_pred = g.k();

    const int Q = (int)qgt.Q;
    const int qe = (eval_q < 0) ? Q : std::min(eval_q, Q);

    int ke = std::min(k_eval, k_pred);
    ke = std::min(ke, (int)qgt.k);

    QGTEvalResult out;
    out.used_Q = qe;
    out.k_eval = ke;

    if (qe <= 0 || ke <= 0) return out;

    double sum = 0.0;

    for (int qi = 0; qi < qe; ++qi) {
        const uint32_t qid = qgt.qid[(size_t)qi];

        if (validate_ids && (qid >= (uint32_t)n)) {
            throw std::runtime_error("QGT qid out of range: " + std::to_string(qid) +
                                     " (n=" + std::to_string(n) + ")");
        }

        const uint32_t* pred = g.nbr_ptr((int)qid);
        const uint32_t* truth = qgt.nbr.data() + (size_t)qi * (size_t)qgt.k;

        int hit = 0;
        for (int t = 0; t < ke; ++t) {
            const uint32_t e = truth[t];
            for (int s = 0; s < k_pred; ++s) {
                if (pred[s] == e) { ++hit; break; }
            }
        }
        sum += (double)hit / (double)ke;
    }

    out.recall = sum / (double)qe;
    return out;
}
