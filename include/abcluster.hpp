#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cmath>
#include <queue>

// ---------------------------
// Types and options
// ---------------------------

struct ClusterOptions {
    double cutoff = 0.35;        // flat cut on max intra-cluster merge distance
    double mut_value = 0.35;
    double epsilon   = 1e-3;
    int    len_penalty = 2;
    int    canonical_samples = 17; // how many samples to build canonical mutlist
};

using Mutation = std::string; // mutation identifier as string (e.g., "A23B" or similar)

struct Record {
    // One input sequence (a "sequence essence member")
    std::string junc;          // amino-acid junction
    std::string v_gene;        // V gene identifier/name
    std::string j_gene;        // J gene identifier/name
    std::vector<Mutation> mutations; // sorted ascending (we will sort)
};

struct EssenceKey {
    std::string junc;
    std::string v_gene;
    std::string j_gene;

    bool operator==(const EssenceKey& o) const noexcept {
        return v_gene == o.v_gene &&
               j_gene == o.j_gene &&
               junc   == o.junc;
    }
};

struct EssenceKeyHash {
    std::size_t operator()(const EssenceKey& k) const noexcept {
        // Simple but decent string+integers hash
        std::size_t h = std::hash<std::string>{}(k.junc);
        h ^= std::hash<std::string>{}(k.v_gene) * 0x165667b19e3779f9ULL;
        h ^= std::hash<std::string>{}(k.j_gene) * 0xdbe6d5d5fe4cce2fULL;
        return h;
    }
};

struct MutList {
    std::vector<Mutation> data; // owned, sorted unique
    int weight = 0;             // number of members with exactly this list (≥1)
};

struct Essence {
    EssenceKey key;
    int weight = 0;                                  // number of sequences in this essence
    std::vector<MutList> mutlists;                   // histogram of identical mutlists
    std::vector<Mutation> canonical_mutlist;         // computed representative (owned, sorted unique)
    // cluster label to emit back for each member of this essence:
    std::string cluster_string;
};

// ---------------------------
// Utilities
// ---------------------------

inline int hamming_distance(const std::string& a, const std::string& b) {
    const int n = (int)a.size();
    int d = 0;
    for (int i = 0; i < n; ++i) d += (a[i] != b[i]);
    return d;
}

// Memory-safe O(min(n,m)) Levenshtein
inline int levenshtein(const std::string& s1, const std::string& s2) {
    const int n = (int)s1.size();
    const int m = (int)s2.size();
    if (n == 0) return m;
    if (m == 0) return n;

    // Always allocate the shorter row
    const std::string *pa = &s1, *pb = &s2;
    int a = n, b = m;
    if (m < n) { std::swap(pa, pb); std::swap(a, b); } // a <= b
    std::vector<int> prev(a + 1), curr(a + 1);
    for (int i = 0; i <= a; ++i) prev[i] = i;

    for (int j = 1; j <= b; ++j) {
        curr[0] = j;
        const char cj = (*pb)[j - 1];
        for (int i = 1; i <= a; ++i) {
            const int cost = ((*pa)[i - 1] == cj) ? 0 : 1;
            curr[i] = std::min({ prev[i] + 1, curr[i - 1] + 1, prev[i - 1] + cost });
        }
        std::swap(prev, curr);
    }
    return prev[a];
}

inline int get_ld(const Essence& e1, const Essence& e2) {
    if (e1.key.junc.size() == e2.key.junc.size())
        return hamming_distance(e1.key.junc, e2.key.junc);
    return levenshtein(e1.key.junc, e2.key.junc);
}

inline int v_compare(const Essence& e1, const Essence& e2) {
    return 7 * (e1.key.v_gene != e2.key.v_gene);
}

inline int j_compare(const Essence& e1, const Essence& e2) {
    return 7 * (e1.key.j_gene != e2.key.j_gene);
}

inline int num_shared_muts(const std::vector<Mutation>& a,
                           const std::vector<Mutation>& b) {
    int i = 0, j = 0, n = 0;
    while (i < (int)a.size() && j < (int)b.size()) {
        if (a[i] < b[j]) ++i;
        else if (b[j] < a[i]) ++j;
        else { ++i; ++j; ++n; }
    }
    return n;
}

// Full "MutBonus" across identical-mutlist buckets with a soft ceiling
inline double mut_bonus_full(const Essence& e1, const Essence& e2,
                             double ceiling, const ClusterOptions& opt) {
    // Use double to avoid overflow
    double total_weight = double(e1.weight) * double(e2.weight);
    if (total_weight <= 0.0) return 0.0;

    double s = 0.0;
    for (const auto& m1 : e1.mutlists) {
        const double w1 = (double)m1.weight;
        for (const auto& m2 : e2.mutlists) {
            const double w2 = (double)m2.weight;
            const double shared = (double)num_shared_muts(m1.data, m2.data);
            s += std::min(ceiling, shared) * w1 * w2;
        }
    }
    return opt.mut_value * (s / total_weight);
}

inline double dissimilarity_full(const Essence& e1, const Essence& e2,
                                 int LD, const ClusterOptions& opt) {
    const int vPen = v_compare(e1, e2);
    const int jPen = j_compare(e1, e2);
    const int basic = LD + vPen + jPen;
    const double lenPen = std::abs(double(e1.key.junc.size()) - double(e2.key.junc.size())) * opt.len_penalty;
    const double editLen = (double)std::min(e1.key.junc.size(), e2.key.junc.size());
    if (editLen <= 0) return std::numeric_limits<double>::infinity();

    const double ceiling = (basic / opt.mut_value) - (opt.epsilon / opt.mut_value);
    const double bonus = mut_bonus_full(e1, e2, ceiling, opt);
    const double withBonus = (double)basic - bonus;
    return (withBonus + lenPen) / editLen;
}

// Simpler dissimilarity (canonical-mutlist only) – not used by default
inline double dissimilarity_fast(const Essence& e1, const Essence& e2,
                                 int LD, const ClusterOptions& opt) {
    const int vPen = v_compare(e1, e2);
    const int jPen = j_compare(e1, e2);
    const int basic = LD + vPen + jPen;
    const double lenPen = std::abs(double(e1.key.junc.size()) - double(e2.key.junc.size())) * opt.len_penalty;
    const double editLen = (double)std::min(e1.key.junc.size(), e2.key.junc.size());
    if (editLen <= 0) return std::numeric_limits<double>::infinity();

    const double bonus = opt.mut_value * num_shared_muts(e1.canonical_mutlist, e2.canonical_mutlist);
    const double withBonus = std::max(opt.epsilon, (double)basic - bonus);
    return (withBonus + lenPen) / editLen;
}

// ---------------------------
// Canonical mutlist builder
// ---------------------------

inline std::vector<Mutation> canonical_from_histogram(
        const std::vector<MutList>& histo, int n_members, int canonical_samples)
{
    // Select up to p samples and mark a mutation "canonical"
    // if it appears in at least floor(p/2) of them.
    const int p = std::min(n_members, canonical_samples);
    // Build counts across up to p heaviest buckets
    std::vector<std::pair<int,const MutList*>> buckets;
    buckets.reserve(histo.size());
    for (auto& m : histo) buckets.push_back({ m.weight, &m });
    std::sort(buckets.begin(), buckets.end(),
              [](auto& a, auto& b){ return a.first > b.first; });

    // Frequency accumulation
    std::vector<Mutation> merged;
    std::vector<int> counts;
    merged.reserve(256);
    counts.reserve(256);

    int taken = 0;
    for (auto& kv : buckets) {
        if (taken >= p) break;
        const auto& vec = kv.second->data;
        // merge counts (two-pointer union)
        std::vector<Mutation> out_m;
        std::vector<int>      out_c;
        out_m.reserve(merged.size() + vec.size());
        out_c.reserve(merged.size() + vec.size());
        size_t i = 0, j = 0;
        while (i < merged.size() || j < vec.size()) {
            if (j == vec.size() || (i < merged.size() && merged[i] < vec[j])) {
                out_m.push_back(merged[i]);
                out_c.push_back(counts[i]);
                ++i;
            } else if (i == merged.size() || vec[j] < merged[i]) {
                out_m.push_back(vec[j]);
                out_c.push_back(1);
                ++j;
            } else {
                out_m.push_back(merged[i]);
                out_c.push_back(counts[i] + 1);
                ++i; ++j;
            }
        }
        merged.swap(out_m);
        counts.swap(out_c);
        ++taken;
    }

    // Keep those with frequency >= p/2
    std::vector<Mutation> canonical;
    canonical.reserve(merged.size());
    const int thresh = p / 2;
    for (size_t i = 0; i < merged.size(); ++i)
        if (counts[i] >= thresh) canonical.push_back(merged[i]);
    return canonical;
}

// ---------------------------
// Average-linkage clustering
// ---------------------------

struct LinkageStep {
    int left;   // index of left child (0..n-1 for leaves; n.. for internal)
    int right;  // index of right child
    double dist;// merge distance
    int size;   // number of leaves in the merged cluster
};

// Helper: condensed index of upper-triangle without diagonal
inline std::size_t cidx(int n, int i, int j) {
    // pre: 0 <= i < j < n
    // pos = n*(n-1)/2 - (n-i)*(n-i-1)/2 + (j-i-1)
    return (std::size_t)n*(n-1)/2 - (std::size_t)(n-i)*(n-i-1)/2 + (std::size_t)(j-i-1);
}

inline std::vector<LinkageStep>
average_linkage(const std::vector<double>& D0 /* condensed */, int n) {
    if (n <= 1) return {};

    // Work on a mutable copy with current size n_curr
    std::vector<double> D = D0;
    std::vector<int> id(n);          // maps position -> current node id
    std::vector<int> sz(n, 1);       // cluster sizes
    for (int i = 0; i < n; ++i) id[i] = i;

    std::vector<LinkageStep> Z;
    Z.reserve(n - 1);

    int n_curr = n;
    while (n_curr > 1) {
        // 1) find global minimum
        double best = std::numeric_limits<double>::infinity();
        int bi = -1, bj = -1;
        for (int i = 0; i < n_curr - 1; ++i) {
            for (int j = i + 1; j < n_curr; ++j) {
                double d = D[cidx(n_curr, i, j)];
                if (d < best) { best = d; bi = i; bj = j; }
            }
        }
        if (bi < 0) throw std::runtime_error("average_linkage: no minimum found");

        // 2) record merge
        const int left_id  = id[bi];
        const int right_id = id[bj];
        const int merged_id = (int) (n + Z.size()); // new id
        const int new_sz = sz[bi] + sz[bj];
        Z.push_back(LinkageStep{left_id, right_id, best, new_sz});

        // 3) update distances for row/col bi with average (UPGMA)
        for (int k = 0; k < n_curr; ++k) if (k != bi && k != bj) {
            double dik = (k < bi) ? D[cidx(n_curr, k, bi)] : D[cidx(n_curr, bi, k)];
            double djk = (k < bj) ? D[cidx(n_curr, k, bj)] : D[cidx(n_curr, bj, k)];
            double avg = (sz[bi] * dik + sz[bj] * djk) / (double)new_sz;
            // write into position (min, max) with bi as representative
            if (k < bi) D[cidx(n_curr, k, bi)] = avg;
            else        D[cidx(n_curr, bi, k)] = avg;
        }

        // 4) compact: move last row/col (n_curr-1) into bj, then reduce size
        if (bj != n_curr - 1) {
            for (int k = 0; k < n_curr; ++k) if (k != bj) {
                double v = (k < n_curr - 1) ? D[cidx(n_curr, k, n_curr - 1)]
                                            : 0.0; // not used
                if (k < bj)      D[cidx(n_curr, k, bj)] = v;
                else if (k > bj) D[cidx(n_curr, bj, k)] = v;
            }
            id[bj] = id[n_curr - 1];
            sz[bj] = sz[n_curr - 1];
        }

        // 5) overwrite bi with new merged cluster id and size; drop last
        id[bi] = merged_id;
        sz[bi] = new_sz;
        --n_curr;

        // The condensed vector D still has the same capacity; we just interpret
        // indexes with the new n_curr on next iteration.
    }
    return Z;
}

// ---------------------------
// Flat clusters by cutoff
// ---------------------------

inline std::vector<int> flat_clusters_by_maxdist(const std::vector<LinkageStep>& Z, int n, double cutoff) {
    if (n == 0) return {};
    if (n == 1) return {1};

    // Build children arrays for internal nodes ids n..2n-2
    const int nodes = 2*n - 1;
    std::vector<int> left(nodes, -1), right(nodes, -1);
    std::vector<double> node_max(nodes, 0.0);
    for (int i = 0; i < (int)Z.size(); ++i) {
        int nid = n + i;
        left[nid] = Z[i].left;
        right[nid] = Z[i].right;
        node_max[nid] = std::max({ Z[i].dist,
                                   node_max[Z[i].left],
                                   node_max[Z[i].right] });
    }
    const int root = n + (int)Z.size() - 1;

    // DFS: whenever node_max[node] <= cutoff, assign a new cluster id to all leaves under it
    std::vector<int> leaf_label(n, 0);
    int next_label = 1;

    std::function<void(int,int)> paint = [&](int u, int label){
        if (u < n) { leaf_label[u] = label; return; }
        paint(left[u], label);
        paint(right[u], label);
    };
    std::function<void(int)> split = [&](int u){
        if (u < 0) return;
        if (node_max[u] <= cutoff) {
            paint(u, next_label++);
        } else {
            if (u < n) { // leaf (size 1) that didn't meet cutoff: its node_max = 0 ≤ cutoff in practice
                paint(u, next_label++);
            } else {
                split(left[u]);
                split(right[u]);
            }
        }
    };
    split(root);
    return leaf_label;
}

// ---------------------------
// Public API
// ---------------------------

struct ClusterResult {
    // one label per input record, in input order
    std::vector<std::string> labels;
    // optional: the linkage steps (sizes/distances) for essences (not for all sequences)
    std::vector<LinkageStep> linkage;
};

inline ClusterResult cluster_records(const std::vector<Record>& records, const ClusterOptions& opt = {}) {
    if (records.empty()) return ClusterResult{};

    // 1) Aggregate sequences into essences (by EssenceKey)
    std::unordered_map<EssenceKey, Essence, EssenceKeyHash> map;

    map.reserve(records.size());
    for (const auto& r : records) {
        EssenceKey key{r.junc, r.v_gene, r.j_gene};
        auto it = map.find(key);
        if (it == map.end()) {
            Essence e;
            e.key = key;
            it = map.emplace(key, std::move(e)).first;
        }
        Essence& e = it->second;

        // normalize mutations: sorted & unique
        std::vector<Mutation> m = r.mutations;
        std::sort(m.begin(), m.end());
        m.erase(std::unique(m.begin(), m.end()), m.end());

        // bucketing identical mutation lists
        bool placed = false;
        for (auto& ml : e.mutlists) {
            if (ml.data == m) { ++ml.weight; placed = true; break; }
        }
        if (!placed) { MutList ml; ml.data = std::move(m); ml.weight = 1; e.mutlists.push_back(std::move(ml)); }
        ++e.weight;
    }

    // 2) Finalize canonical mutlist per essence
    std::vector<Essence*> essences;
    essences.reserve(map.size());
    for (auto& kv : map) {
        Essence& e = kv.second;
        e.canonical_mutlist = canonical_from_histogram(e.mutlists, e.weight, opt.canonical_samples);
        std::sort(e.canonical_mutlist.begin(), e.canonical_mutlist.end());
        essences.push_back(&e);
    }

    const int n = (int)essences.size();
    if (n == 0) return ClusterResult{};

    // 3) Build condensed distance matrix for essences (full dissimilarity)
    std::vector<double> D((std::size_t)n*(n-1)/2, 0.0);
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const int LD = get_ld(*essences[i], *essences[j]);
            const double d = dissimilarity_full(*essences[i], *essences[j], LD, opt);
            D[cidx(n, i, j)] = d;
        }
    }

    // 4) Average-linkage
    std::vector<LinkageStep> Z = average_linkage(D, n);

    // 5) Flat clusters by cutoff
    std::vector<int> lab = flat_clusters_by_maxdist(Z, n, opt.cutoff);

    // 6) Name clusters and emit for each input record
    int maxlab = 0; for (int x : lab) if (x > maxlab) maxlab = x;
    std::vector<std::string> names(maxlab + 1);
    for (int i = 1; i <= maxlab; ++i) names[i] = std::to_string(i);

    for (int i = 0; i < n; ++i) essences[i]->cluster_string = names[lab[i]];

    std::vector<std::string> out;
    out.reserve(records.size());
    for (const auto& r : records) {
        EssenceKey key{r.junc, r.v_gene, r.j_gene};
        const Essence& e = map.at(key);
        out.push_back(e.cluster_string);
    }

    return ClusterResult{ std::move(out), std::move(Z) };
}
