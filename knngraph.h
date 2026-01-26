#pragma once
#include <vector>
#include <cstdint>
#include <limits>
#include <algorithm>

class kNNGraph {
public:
    using id_t   = uint32_t;
    using dist_t = float;

    enum : uint8_t {
        IS_NEW  = 1u << 0,  // 次反復で "true"
        SAMPLED = 1u << 1   // 今回反復で sampled new
    };

    static constexpr id_t invalid_id() { return std::numeric_limits<id_t>::max(); }
    static constexpr dist_t inf() { return std::numeric_limits<dist_t>::infinity(); }

    kNNGraph(int k, int n)
            : k_(k), n_(n),
              nbr_((size_t)n * (size_t)k, invalid_id()),
              dist_((size_t)n * (size_t)k, inf()),
              flag_((size_t)n * (size_t)k, 0),
              worst_pos_(n, 0),
              worst_dist_(n, inf())
    {}

    int k() const { return k_; }
    int n() const { return n_; }

    size_t base(int i) const { return (size_t)i * (size_t)k_; }

    id_t*    nbr_ptr(int i)  { return nbr_.data()  + base(i); }
    dist_t*  dist_ptr(int i) { return dist_.data() + base(i); }
    uint8_t* flag_ptr(int i) { return flag_.data() + base(i); }

    const id_t*    nbr_ptr(int i)  const { return nbr_.data()  + base(i); }
    const dist_t*  dist_ptr(int i) const { return dist_.data() + base(i); }
    const uint8_t* flag_ptr(int i) const { return flag_.data() + base(i); }

    // 直接書いた後のworst計算（初期化用）
    void recompute_worst_row(int i) { recompute_worst(i); }
    void recompute_worst_all() { for (int i = 0; i < n_; ++i) recompute_worst(i); }

    // Update規約（前に合意した通り）
    bool update(int i, id_t cand, dist_t cand_dist) {
        if ((id_t)i == cand) return false;
        const size_t b = base(i);

        // duplicate check
        for (int t = 0; t < k_; ++t) {
            if (nbr_[b + t] == cand) {
                if (cand_dist < dist_[b + t]) {
                    dist_[b + t] = cand_dist;
                    flag_[b + t] = (uint8_t)((flag_[b + t] & ~SAMPLED) | IS_NEW);
                    recompute_worst(i);
                    return true;
                }
                return false;
            }
        }

        // replace worst
        if (cand_dist < worst_dist_[i]) {
            int wp = worst_pos_[i];
            nbr_[b + wp]  = cand;
            dist_[b + wp] = cand_dist;
            flag_[b + wp] = IS_NEW; // sampledは落とす
            recompute_worst(i);
            return true;
        }
        return false;
    }

private:
    int k_, n_;
    std::vector<id_t> nbr_;
    std::vector<dist_t> dist_;
    std::vector<uint8_t> flag_;
    std::vector<uint16_t> worst_pos_;
    std::vector<dist_t> worst_dist_;

    void recompute_worst(int i) {
        const size_t b = base(i);
        int wp = 0;
        dist_t wd = dist_[b + 0];
        for (int t = 1; t < k_; ++t) {
            if (dist_[b + t] > wd) { wd = dist_[b + t]; wp = t; }
        }
        worst_pos_[i]  = (uint16_t)wp;
        worst_dist_[i] = wd;
    }
};
