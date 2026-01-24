/*
 * Clonify - Antibody Clonotype Clustering
 *
 * Portable C++ implementation for macOS/Linux.
 * Reads TSV input with abstar/airr-style columns.
 *
 * Compilation:
 *   clang++ -O3 -std=c++17 -o clonify cluster.cpp
 *   g++ -O3 -std=c++17 -o clonify cluster.cpp
 */

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <sys/time.h>

// Disable SIMD for portability
#undef SIMD
#define NO_TIMING

/*
 * Type definitions for the hierarchical clustering algorithm.
 */
typedef int_fast32_t t_index;
#define MAX_INDEX INT32_MAX

#if (LONG_MAX < MAX_INDEX)
#error The integer format "t_index" must not have a greater range than "long int".
#endif

typedef double t_float;

#define fc_isnan(X) ((X)!=(X))

// Self-destructing array pointer
template <typename type>
class auto_array_ptr {
private:
    type* ptr;
    auto_array_ptr(auto_array_ptr const&);
    auto_array_ptr& operator=(auto_array_ptr const&);
public:
    auto_array_ptr() : ptr(NULL) {}
    template <typename index>
    auto_array_ptr(index const size) : ptr(new type[size]) {}
    template <typename index, typename value>
    auto_array_ptr(index const size, value const val) : ptr(new type[size]) {
        std::fill_n(ptr, size, val);
    }
    ~auto_array_ptr() { delete[] ptr; }
    void free() { delete[] ptr; ptr = NULL; }
    template <typename index>
    void init(index const size) { ptr = new type[size]; }
    template <typename index, typename value>
    void init(index const size, value const val) {
        init(size);
        std::fill_n(ptr, size, val);
    }
    inline operator type*() const { return ptr; }
};

struct node {
    t_index node1, node2;
    t_float dist;
    inline friend bool operator<(const node a, const node b) {
        return (a.dist < b.dist);
    }
};

class cluster_result {
private:
    auto_array_ptr<node> Z;
    t_index pos;
public:
    cluster_result(const t_index size) : Z(size), pos(0) {}
    void append(const t_index node1, const t_index node2, const t_float dist) {
        Z[pos].node1 = node1;
        Z[pos].node2 = node2;
        Z[pos].dist = dist;
        ++pos;
    }
    node* operator[](const t_index idx) const { return Z + idx; }
    void sqrt() const {
        for (node* ZZ = Z; ZZ != Z + pos; ++ZZ) {
            ZZ->dist = ::sqrt(ZZ->dist);
        }
    }
};

class doubly_linked_list {
public:
    t_index start;
    auto_array_ptr<t_index> succ;
private:
    auto_array_ptr<t_index> pred;
public:
    doubly_linked_list(const t_index size)
        : start(0), succ(size + 1), pred(size + 1) {
        for (t_index i = 0; i < size; ++i) {
            pred[i + 1] = i;
            succ[i] = i + 1;
        }
    }
    ~doubly_linked_list() {}
    void remove(const t_index idx) {
        if (idx == start) {
            start = succ[idx];
        } else {
            succ[pred[idx]] = succ[idx];
            pred[succ[idx]] = pred[idx];
        }
        succ[idx] = 0;
    }
    bool is_inactive(t_index idx) const { return (succ[idx] == 0); }
};

// Indexing for condensed distance matrix
#define D_(r_,c_) ( D[(static_cast<std::ptrdiff_t>(2*N-3-(r_))*(r_)>>1)+(c_)-1] )
#define Z_(_r, _c) (Z[(_r)*4 + (_c)])

class union_find {
private:
    auto_array_ptr<t_index> parent;
    t_index nextparent;
public:
    union_find(const t_index size)
        : parent(size > 0 ? 2 * size - 1 : 0, 0), nextparent(size) {}
    t_index Find(t_index idx) const {
        if (parent[idx] != 0) {
            t_index p = idx;
            idx = parent[idx];
            if (parent[idx] != 0) {
                do {
                    idx = parent[idx];
                } while (parent[idx] != 0);
                do {
                    t_index tmp = parent[p];
                    parent[p] = idx;
                    p = tmp;
                } while (parent[p] != idx);
            }
        }
        return idx;
    }
    void Union(const t_index node1, const t_index node2) {
        parent[node1] = parent[node2] = nextparent++;
    }
};

class nan_error {};

inline static void f_average(t_float* const b, const t_float a, const t_float s, const t_float t) {
    *b = s * a + t * (*b);
    if (fc_isnan(*b)) {
        throw(nan_error());
    }
}

template <typename t_members>
static void NN_chain_core(const t_index N, t_float* const D, t_members* const members, cluster_result& Z2) {
    t_index i;
    auto_array_ptr<t_index> NN_chain(N);
    t_index NN_chain_tip = 0;
    t_index idx1, idx2;
    t_float size1, size2;
    doubly_linked_list active_nodes(N);
    t_float min;

    for (t_index j = 0; j < N - 1; ++j) {
        if (NN_chain_tip <= 3) {
            NN_chain[0] = idx1 = active_nodes.start;
            NN_chain_tip = 1;
            idx2 = active_nodes.succ[idx1];
            min = D_(idx1, idx2);
            for (i = active_nodes.succ[idx2]; i < N; i = active_nodes.succ[i]) {
                if (D_(idx1, i) < min) {
                    min = D_(idx1, i);
                    idx2 = i;
                }
            }
        } else {
            NN_chain_tip -= 3;
            idx1 = NN_chain[NN_chain_tip - 1];
            idx2 = NN_chain[NN_chain_tip];
            min = idx1 < idx2 ? D_(idx1, idx2) : D_(idx2, idx1);
        }

        do {
            NN_chain[NN_chain_tip] = idx2;
            for (i = active_nodes.start; i < idx2; i = active_nodes.succ[i]) {
                if (D_(i, idx2) < min) {
                    min = D_(i, idx2);
                    idx1 = i;
                }
            }
            for (i = active_nodes.succ[idx2]; i < N; i = active_nodes.succ[i]) {
                if (D_(idx2, i) < min) {
                    min = D_(idx2, i);
                    idx1 = i;
                }
            }
            idx2 = idx1;
            idx1 = NN_chain[NN_chain_tip++];
        } while (idx2 != NN_chain[NN_chain_tip - 2]);

        Z2.append(idx1, idx2, min);

        if (idx1 > idx2) {
            t_index tmp = idx1;
            idx1 = idx2;
            idx2 = tmp;
        }

        size1 = static_cast<t_float>(members[idx1]);
        size2 = static_cast<t_float>(members[idx2]);
        members[idx2] += members[idx1];

        active_nodes.remove(idx1);

        t_float s = size1 / (size1 + size2);
        t_float t = size2 / (size1 + size2);
        for (i = active_nodes.start; i < idx1; i = active_nodes.succ[i])
            f_average(&D_(i, idx2), D_(i, idx1), s, t);
        for (; i < idx2; i = active_nodes.succ[i])
            f_average(&D_(i, idx2), D_(idx1, i), s, t);
        for (i = active_nodes.succ[idx2]; i < N; i = active_nodes.succ[i])
            f_average(&D_(idx2, i), D_(idx1, i), s, t);
    }
}

#define size_(r_) ( ((r_<N) ? 1 : Z_(r_-N,3)) )

class linkage_output {
private:
    t_float* Z;
public:
    linkage_output(t_float* const Z_) : Z(Z_) {}
    void append(const t_index node1, const t_index node2, const t_float dist, const t_float size) {
        if (node1 < node2) {
            *(Z++) = static_cast<t_float>(node1);
            *(Z++) = static_cast<t_float>(node2);
        } else {
            *(Z++) = static_cast<t_float>(node2);
            *(Z++) = static_cast<t_float>(node1);
        }
        *(Z++) = dist;
        *(Z++) = size;
    }
};

template <const bool sorted>
static void generate_SciPy_dendrogram(t_float* const Z, cluster_result& Z2, const t_index N) {
    union_find nodes(sorted ? 0 : N);
    if (!sorted) {
        std::stable_sort(Z2[0], Z2[N - 1]);
    }
    linkage_output output(Z);
    t_index node1, node2;
    for (node const* NN = Z2[0]; NN != Z2[N - 1]; ++NN) {
        if (sorted) {
            node1 = NN->node1;
            node2 = NN->node2;
        } else {
            node1 = nodes.Find(NN->node1);
            node2 = nodes.Find(NN->node2);
            nodes.Union(node1, node2);
        }
        output.append(node1, node2, NN->dist, size_(node1) + size_(node2));
    }
}

void linkage(const size_t N, double* matrix, t_index* members, double* Z) {
    cluster_result Z2(N - 1);
    NN_chain_core<t_index>(N, matrix, members, Z2);
    generate_SciPy_dendrogram<false>(Z, Z2, N);
}

// Flat clustering from distance threshold
#define CPY_MAX(_x, _y) ((_x > _y) ? (_x) : (_y))
#define CPY_MIN(_x, _y) ((_x < _y) ? (_x) : (_y))
#define CPY_BITS_PER_CHAR (sizeof(unsigned char) * 8)
#define CPY_FLAG_ARRAY_SIZE_BYTES(num_bits) (((num_bits) + CPY_BITS_PER_CHAR - 1) / CPY_BITS_PER_CHAR)
#define CPY_GET_BIT(_xx, i) (((_xx)[(i) / CPY_BITS_PER_CHAR] >> ((CPY_BITS_PER_CHAR-1) - ((i) % CPY_BITS_PER_CHAR))) & 0x1)
#define CPY_SET_BIT(_xx, i) ((_xx)[(i) / CPY_BITS_PER_CHAR] |= ((0x1) << ((CPY_BITS_PER_CHAR-1) - ((i) % CPY_BITS_PER_CHAR))))

#define CPY_LIS 4
#define CPY_LIN_LEFT 0
#define CPY_LIN_RIGHT 1
#define CPY_LIN_DIST 2
#define CPY_LIN_CNT 3

void get_max_dist_for_each_cluster(const double* Z, double* max_dists, int n) {
    int* curNode;
    int ndid, lid, rid, k;
    unsigned char *lvisited, *rvisited;
    const double* Zrow;
    double max_dist;
    const int bff = CPY_FLAG_ARRAY_SIZE_BYTES(n);

    k = 0;
    curNode = (int*)malloc(n * sizeof(int));
    lvisited = (unsigned char*)malloc(bff);
    rvisited = (unsigned char*)malloc(bff);
    curNode[k] = (n * 2) - 2;
    memset(lvisited, 0, bff);
    memset(rvisited, 0, bff);
    while (k >= 0) {
        ndid = curNode[k];
        Zrow = Z + ((ndid - n) * CPY_LIS);
        lid = (int)Zrow[CPY_LIN_LEFT];
        rid = (int)Zrow[CPY_LIN_RIGHT];
        if (lid >= n && !CPY_GET_BIT(lvisited, ndid - n)) {
            CPY_SET_BIT(lvisited, ndid - n);
            curNode[k + 1] = lid;
            k++;
            continue;
        }
        if (rid >= n && !CPY_GET_BIT(rvisited, ndid - n)) {
            CPY_SET_BIT(rvisited, ndid - n);
            curNode[k + 1] = rid;
            k++;
            continue;
        }
        max_dist = Zrow[CPY_LIN_DIST];
        if (lid >= n) {
            max_dist = CPY_MAX(max_dist, max_dists[lid - n]);
        }
        if (rid >= n) {
            max_dist = CPY_MAX(max_dist, max_dists[rid - n]);
        }
        max_dists[ndid - n] = max_dist;
        k--;
    }
    free(curNode);
    free(lvisited);
    free(rvisited);
}

void form_flat_clusters_from_monotonic_criterion(const double* Z,
                                                  const double* mono_crit,
                                                  int* T, double cutoff, int n) {
    int* curNode;
    int ndid, lid, rid, k, ms, nc;
    unsigned char *lvisited, *rvisited;
    double max_crit;
    const double* Zrow;
    const int bff = CPY_FLAG_ARRAY_SIZE_BYTES(n);

    curNode = (int*)malloc(n * sizeof(int));
    lvisited = (unsigned char*)malloc(bff);
    rvisited = (unsigned char*)malloc(bff);

    nc = 0;
    ms = -1;
    k = 0;
    curNode[k] = (n * 2) - 2;
    memset(lvisited, 0, bff);
    memset(rvisited, 0, bff);
    ms = -1;
    while (k >= 0) {
        ndid = curNode[k];
        Zrow = Z + ((ndid - n) * CPY_LIS);
        lid = (int)Zrow[CPY_LIN_LEFT];
        rid = (int)Zrow[CPY_LIN_RIGHT];
        max_crit = mono_crit[ndid - n];
        if (ms == -1 && max_crit <= cutoff) {
            ms = k;
            nc++;
        }
        if (lid >= n && !CPY_GET_BIT(lvisited, ndid - n)) {
            CPY_SET_BIT(lvisited, ndid - n);
            curNode[k + 1] = lid;
            k++;
            continue;
        }
        if (rid >= n && !CPY_GET_BIT(rvisited, ndid - n)) {
            CPY_SET_BIT(rvisited, ndid - n);
            curNode[k + 1] = rid;
            k++;
            continue;
        }
        if (ndid >= n) {
            if (lid < n) {
                if (ms == -1) {
                    nc++;
                    T[lid] = nc;
                } else {
                    T[lid] = nc;
                }
            }
            if (rid < n) {
                if (ms == -1) {
                    nc++;
                    T[rid] = nc;
                } else {
                    T[rid] = nc;
                }
            }
            if (ms == k) {
                ms = -1;
            }
        }
        k--;
    }

    free(curNode);
    free(lvisited);
    free(rvisited);
}

void form_flat_clusters_from_dist(const double* Z, int* T, double cutoff, int n) {
    double* max_dists = (double*)malloc(sizeof(double) * n);
    get_max_dist_for_each_cluster(Z, max_dists, n);
    form_flat_clusters_from_monotonic_criterion(Z, max_dists, T, cutoff, n);
    free(max_dists);
}

using namespace std;

/*
 * Clonify clustering parameters.
 */
namespace cluster_param {
    const double cutoff = 0.35;
    const double mut_value = 0.35;
    const double epsilon = 0.001;
    const int len_penalty = 2;
}

/*
 * Input data limits.
 */
const int MAX_SEQUENCES = 25000000;
const int MAX_AVERAGE_MUTATIONS_PER_SEQUENCE = 64;
const int MAX_PARTITIONS = 8;  // V gene families 1-7 plus overflow
const int MAX_SEQUENCES_PER_PARTITION = MAX_SEQUENCES / 2;
const int MAX_ESSENCES_PER_PARTITION = MAX_SEQUENCES_PER_PARTITION / 2;
const int MAX_CLUSTERS_PER_PARTITION = 256 + MAX_SEQUENCES_PER_PARTITION / 32;
const int MAX_AA_LENGTH = 64;
const int MAX_MUTATION_LOC = 4096;

/*
 * Tweakable parameters.
 */
const int MIN_CENTER_SIZE_10K = 3;
const int MIN_CENTER_SIZE_100K = 9;
const double MIN_MEGACLUSTER_DISSIMILARITY = 0.40;
const int CANONICAL_SAMPLES = 17;

#define FOR(i,a,b) for(int i=(a); i<(b); i++)
#define REP(i,n) FOR(i,0,n)

static inline void assert_invariant(bool condition) {
    if (!condition)
        throw std::logic_error("assert_invariant failed");
}
static inline void _range_check(bool condition, const char* message) {
    if (!condition)
        throw std::out_of_range((string)"range_check(" + message + ") failed");
}
#define range_check(condition) _range_check(condition, #condition)

class Stopwatch {
    struct timeval tv_startup;
public:
    Stopwatch() { reset(); }
    void reset() {
#ifndef NO_TIMING
        gettimeofday(&tv_startup, NULL);
#endif
    }
    double time() const {
#ifndef NO_TIMING
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return (tv.tv_sec - tv_startup.tv_sec) + (tv.tv_usec - tv_startup.tv_usec) * 1e-6;
#else
        return 0;
#endif
    }
    void print(const char*) const {}
    void store(const char*) const {}
} globalStopwatch;

template<class T>
struct Slice {
    typedef T value_type;
    const T* _head;
    int _size;
    Slice<T>(const T* _head, int _size) : _head(_head), _size(_size) {}
    const T& operator[](int i) const { return _head[i]; }
    int size() const { return _size; }
    const T* begin() const { return _head; }
    const T* end() const { return _head + size(); }
};

struct StringSlice : Slice<char> {
    StringSlice(const Slice<char>& slice) : Slice<char>(slice._head, slice._size) {}
    StringSlice(const char* _head, int _size) : Slice<char>(_head, _size) {}
    bool operator<(const StringSlice& o) const {
        return lexicographical_compare(begin(), end(), o.begin(), o.end());
    }
    bool operator==(const StringSlice& o) const {
        return size() == o.size() && !memcmp(begin(), o.begin(), size());
    }
    bool operator==(const string& o) const {
        return (size_t)size() == o.size() && !memcmp(begin(), o.c_str(), size());
    }
    string to_string() const { return string(_head, _size); }
};

StringSlice make_persistent(StringSlice slice) {
    static vector<string> archive;
    archive.push_back(slice.to_string());
    const string& s = archive.back();
    return StringSlice(s.c_str(), s.size());
}

template<class K, class V>
class Interning {
    unordered_map<K, V> mapping;
    vector<K> values;
    V next;
public:
    Interning() : next(V()) {}
    V intern(const K& key) {
        auto it = mapping.find(key);
        if (it != mapping.end())
            return it->second;
        V value = next++;
        K persistent_key = make_persistent(key);
        mapping[persistent_key] = value;
        values.push_back(persistent_key);
        return value;
    }
    const K& lookup(V id) const { return values[id]; }
};

typedef uint16_t Mutation;

int min_center_size = -1;

struct MutList {
    Slice<Mutation> data;
    int _weight;

    MutList() : data(Slice<Mutation>(nullptr, 0)), _weight(0) {}
    MutList(Slice<Mutation> data, bool) : data(data), _weight(0) {}
    MutList clone() const { return *this; }
    int weight() const { return _weight; }
    bool empty() const { return data.size() == 0; }
    void finalize() {}
};

int NumSharedMuts(const MutList& m1, const MutList& m2) {
    int n = 0;
    int p1 = 0, p2 = 0;
    while (p1 < m1.data.size() && p2 < m2.data.size()) {
        if (m1.data[p1] < m2.data[p2]) {
            p1++;
        } else if (m2.data[p2] < m1.data[p1]) {
            p2++;
        } else {
            p1++;
            p2++;
            n++;
        }
    }
    return n;
}

struct MutBag {
    vector<pair<Mutation, int>> count;
    vector<Mutation> quantized;

    MutBag& operator+=(const MutList& m2) {
        int initial_size = count.size();
        int p1 = 0, p2 = 0;
        while (p1 < initial_size && p2 < m2.data.size()) {
            if (count[p1].first < m2.data[p2]) {
                p1++;
            } else if (m2.data[p2] < count[p1].first) {
                count.push_back(pair<Mutation, int>{m2.data[p2], 1});
                p2++;
            } else {
                count[p1].second++;
                p1++;
                p2++;
            }
        }
        while (p2 < m2.data.size()) {
            count.push_back(pair<Mutation, int>{m2.data[p2], 1});
            p2++;
        }
        std::inplace_merge(count.begin(), count.begin() + initial_size, count.end());
        return *this;
    }
    MutList quantize(int threshold) {
        quantized.clear();
        for (auto kv : count) {
            if (kv.second >= threshold)
                quantized.push_back(kv.first);
        }
        return MutList(Slice<Mutation>(&quantized[0], (int)quantized.size()), true);
    }
};

struct EssenceKey {
    StringSlice junc;
    uint8_t v_gene, j_gene;      // Gene-level only (e.g., IGHV3-20 -> unique ID)
    bool operator==(const EssenceKey& o) const {
        return v_gene == o.v_gene && j_gene == o.j_gene && junc == o.junc;
    }
    bool operator<(const EssenceKey& o) const {
        return v_gene < o.v_gene || (v_gene == o.v_gene && (
            j_gene < o.j_gene || (j_gene == o.j_gene && junc < o.junc)));
    }
    string to_string() const;
};

namespace std {
    template<>
    struct hash<StringSlice> {
        size_t operator()(const StringSlice& str) const {
            size_t x = 0;
            REP(i, str.size())
                x = x * 31415926535 + str[i];
            return x;
        }
    };
    template<>
    struct hash<EssenceKey> {
        size_t operator()(const EssenceKey& key) const {
            return hash<StringSlice>()(key.junc) + key.v_gene * 123546789 + key.j_gene * 987;
        }
    };
}

typedef uint64_t MutListHash;
MutListHash hash_mutations(Slice<Mutation> coll) {
    MutListHash h = 1;
    for (Mutation m : coll)
        h = h * 547129405631827 + m;
    return h;
}

struct Essence {
    EssenceKey key;
    MutList canonical_mutlist;
    unordered_map<MutListHash, MutList> mutlists;
    int _weight;
    string cluster_string;
    MutBag mutsum;

    Essence(EssenceKey key)
        : key(key), canonical_mutlist(MutList(Slice<Mutation>(nullptr, 0), false)), _weight(0) {}
    Essence(EssenceKey key, MutList&& canonical_mutlist)
        : key(key), canonical_mutlist(std::move(canonical_mutlist)), _weight(0) {}

    void push_mutlist(Slice<Mutation> mutlist) {
        MutList& m = mutlists[hash_mutations(mutlist)];
        if (m._weight)
            m._weight++;
        else {
            m.data = mutlist;
            m._weight = 1;
            m.finalize();
        }
        if (weight() > 1 && weight() <= CANONICAL_SAMPLES) {
            if (weight() == 2)
                for (auto& kv : mutlists)
                    mutsum += kv.second;
            else
                mutsum += m;
        }
        _weight++;
    }
    bool finalize_parsing() {
        int n = weight();
        if (n == 1)
            canonical_mutlist = mutlists.begin()->second.clone();
        else {
            int p = min(n, CANONICAL_SAMPLES);
            canonical_mutlist = mutsum.quantize(p / 2);
        }
        return n >= min_center_size;
    }
    int weight() const { return _weight; }
};

int LevenshteinDistance(StringSlice s1, StringSlice s2) {
    vector<vector<int>> dp(s1.size() + 1, vector<int>(s2.size() + 1));
    for (int i = 0; i <= s1.size(); i++) {
        dp[i][0] = i;
    }
    for (int i = 0; i <= s2.size(); i++) {
        dp[0][i] = i;
    }
    for (int i = 1; i <= s1.size(); i++) {
        for (int j = 1; j <= s2.size(); j++) {
            int temp = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1);
            dp[i][j] = min(temp, dp[i - 1][j - 1] + (s1[i - 1] == s2[j - 1] ? 0 : 1));
        }
    }
    return dp[s1.size()][s2.size()];
}

int hamming_distance(const char* s1, const char* s2, int n) {
    int d = 0;
    REP(i, n)
        d += s1[i] != s2[i];
    return d;
}

int hamming_distance(const Essence& s1, const Essence& s2) {
    return hamming_distance(&s1.key.junc[0], &s2.key.junc[0], s1.key.junc.size());
}

int GetLD(const Essence& s1, const Essence& s2) {
    if (s1.key.junc.size() == s2.key.junc.size()) {
        return hamming_distance(&s1.key.junc[0], &s2.key.junc[0], s1.key.junc.size());
    } else {
        return LevenshteinDistance(s1.key.junc, s2.key.junc);
    }
}

// Compare V genes (gene-level only, no family/allele distinction)
int vCompare(const Essence& s1, const Essence& s2) {
    return 8 * (s1.key.v_gene != s2.key.v_gene);
}

// Compare J genes (gene-level only)
int jCompare(const Essence& s1, const Essence& s2) {
    return 8 * (s1.key.j_gene != s2.key.j_gene);
}

double MutBonus(const Essence& s1, const Essence& s2, double ceiling) {
    int n1 = s1.weight();
    int n2 = s2.weight();
    int weight = n1 * n2;
    double s = 0;
    for (auto& kv1 : s1.mutlists) {
        double w1 = (double)kv1.second.weight();
        for (auto& kv2 : s2.mutlists) {
            int w2 = kv2.second.weight();
            s += min(ceiling, (double)NumSharedMuts(kv1.second, kv2.second)) * w1 * w2;
        }
    }
    return cluster_param::mut_value * (s / weight);
}

double merging_dissimilarity(const Essence& s1, const Essence& s2, double cutoff) {
    double lenPenalty = fabs(0. + s1.key.junc.size() - s2.key.junc.size()) * cluster_param::len_penalty;
    double editLength = (double)min(s1.key.junc.size(), s2.key.junc.size());
    if (lenPenalty >= editLength * cutoff)
        return cutoff;
    int LD = GetLD(s1, s2);
    int vPenalty = vCompare(s1, s2);
    int jPenalty = jCompare(s1, s2);
    int basic = LD + vPenalty + jPenalty;
    double mutBonus = cluster_param::mut_value * NumSharedMuts(s1.canonical_mutlist, s2.canonical_mutlist);
    double withBonus = max(cluster_param::epsilon, basic - mutBonus);
    return (withBonus + lenPenalty) / editLength;
}

double fast_dissimilarity(const Essence& s1, const Essence& s2, int LD) {
    int vPenalty = vCompare(s1, s2);
    int jPenalty = jCompare(s1, s2);
    int basic = LD + vPenalty + jPenalty;
    double mutBonus = cluster_param::mut_value * NumSharedMuts(s1.canonical_mutlist, s2.canonical_mutlist);
    double withBonus = max(cluster_param::epsilon, basic - mutBonus);
    double lenPenalty = fabs(0. + s1.key.junc.size() - s2.key.junc.size()) * cluster_param::len_penalty;
    double editLength = (double)min(s1.key.junc.size(), s2.key.junc.size());
    return (withBonus + lenPenalty) / editLength;
}

double full_dissimilarity(const Essence& s1, const Essence& s2, int LD) {
    int vPenalty = vCompare(s1, s2);
    int jPenalty = jCompare(s1, s2);
    int basic = LD + vPenalty + jPenalty;
    double mutBonus = MutBonus(s1, s2, basic * (1 / cluster_param::mut_value) - cluster_param::epsilon / cluster_param::mut_value);
    double withBonus = basic - mutBonus;
    double lenPenalty = fabs(0. + s1.key.junc.size() - s2.key.junc.size()) * cluster_param::len_penalty;
    double editLength = (double)min(s1.key.junc.size(), s2.key.junc.size());
    return (withBonus + lenPenalty) / editLength;
}

template<class T>
T* pool_new(vector<T>& pool, T&& value) {
    T* ptr = &*pool.end();
    pool.push_back(std::move(value));
    return ptr;
}

string itoa(int n) {
    ostringstream oss;
    oss << n;
    return oss.str();
}

int next_cluster = 1;

struct Megacluster {
    vector<Essence*> essences;

    void fill_base_matrix(vector<uint8_t>& base_matrix) {
        int n = essences.size();
        base_matrix.reserve(n * (n - 1) / 2);
        REP(i, n) {
            int ni = essences[i]->key.junc.size();
            FOR(j, i + 1, n) {
                int nj = essences[j]->key.junc.size();
                if (ni == nj) {
                    base_matrix.push_back(hamming_distance(&essences[i]->key.junc[0],
                                                           &essences[j]->key.junc[0], ni));
                } else {
                    base_matrix.push_back(LevenshteinDistance(essences[i]->key.junc,
                                                              essences[j]->key.junc));
                }
            }
        }
    }

    void cluster() {
        int n = essences.size();
        if (!n)
            return;

        vector<uint8_t> base_matrix;
        fill_base_matrix(base_matrix);

        vector<double> dist_matrix;
        dist_matrix.resize(n * (n - 1) / 2);
        REP(i, n) {
            FOR(j, i + 1, n) {
                int pos = n * (n - 1) / 2 - (n - i) * (n - i - 1) / 2 + j - i - 1;
                dist_matrix[pos] = full_dissimilarity(*essences[i], *essences[j], base_matrix[pos]);
            }
        }

        vector<t_index> weight(n);
        REP(i, n)
            weight[i] = essences[i]->weight();

        vector<int> flat_cluster(n);
        if (n > 1) {
            double* linkage_matrix = new double[4 * (n - 1)];
            linkage(n, &dist_matrix[0], &weight[0], linkage_matrix);
            form_flat_clusters_from_dist(linkage_matrix, &flat_cluster[0],
                                         cluster_param::cutoff, n);
            delete[] linkage_matrix;
        } else {
            flat_cluster[0] = 1;
        }

        int n_clusters = *std::max_element(flat_cluster.begin(), flat_cluster.end());
        vector<string> name(n_clusters);
        REP(i, n_clusters)
            name[i] = itoa(next_cluster + i);
        REP(i, n)
            essences[i]->cluster_string = name[flat_cluster[i] - 1];
        next_cluster += n_clusters;
    }
};

struct MegaclusterCandidate {
    Essence* ess_ptr;
    double best_dist;
    Megacluster* best_mega;

    bool finished(int ds, int s0) {
        return cluster_param::len_penalty * ds >= best_dist * s0;
    }
    void finish() {
        best_mega->essences.push_back(ess_ptr);
    }
    void try_match(const pair<Essence, Megacluster*>& center) {
        try_match(center, GetLD(*ess_ptr, center.first));
    }
    void try_match(const pair<Essence, Megacluster*>& center, uint8_t LD) {
        double dist = fast_dissimilarity(*ess_ptr, center.first, LD);
        if (best_dist > dist) {
            best_dist = dist;
            best_mega = center.second;
        }
    }
};

struct Partition {
    vector<Mutation> mutation_pool;
    vector<Essence> essence_pool;
    vector<vector<Essence*>> essence_by_length;
    unordered_map<EssenceKey, Essence*> essence_map;

    vector<Essence*> center_candidates;
    vector<Megacluster*> candidate_mega;

    vector<vector<pair<Essence, Megacluster*>>> centers;
    int n_centers = 0;
    vector<Megacluster> megaclusters;

    Partition() {
        mutation_pool.reserve(MAX_SEQUENCES_PER_PARTITION * MAX_AVERAGE_MUTATIONS_PER_SEQUENCE);
        essence_pool.reserve(MAX_ESSENCES_PER_PARTITION);
        center_candidates.reserve(MAX_ESSENCES_PER_PARTITION);
        candidate_mega.reserve(MAX_CLUSTERS_PER_PARTITION);
        megaclusters.reserve(MAX_CLUSTERS_PER_PARTITION);
        essence_by_length.resize(MAX_AA_LENGTH + 1);
        centers.resize(MAX_AA_LENGTH + 1);
    }

    Essence& essence_lookup(const EssenceKey& key) {
        auto it = essence_map.find(key);
        if (it == essence_map.end()) {
            Essence* ess = pool_new(essence_pool, Essence(key));
            essence_by_length[key.junc.size()].push_back(ess);
            return *essence_map.insert(pair<EssenceKey, Essence*>(key, ess)).first->second;
        } else {
            return *it->second;
        }
    }

    void finalize_parsing() {
        for (Essence& ess : essence_pool)
            if (ess.finalize_parsing())
                center_candidates.push_back(&ess);
    }

    Megacluster* merge(Megacluster* a, Megacluster* b) {
        if (a == b)
            return a;
        a->essences.insert(a->essences.end(), b->essences.begin(), b->essences.end());
        b->essences.clear();
        for (auto& row : centers)
            for (auto& pair : row)
                if (pair.second == b)
                    pair.second = a;
        for (auto& ptr : candidate_mega)
            if (ptr == b)
                ptr = a;
        return a;
    }

    void create_centers() {
        std::sort(center_candidates.begin(), center_candidates.end(),
                  [](Essence* i, Essence* j) { return i->weight() > j->weight(); });

        candidate_mega.reserve(center_candidates.size());
        REP(i, (int)center_candidates.size()) {
            bool good = true;
            Megacluster* mega = nullptr;
            REP(j, i) {
                double dist = merging_dissimilarity(*center_candidates[i], *center_candidates[j],
                                                    MIN_MEGACLUSTER_DISSIMILARITY);
                if (dist < MIN_MEGACLUSTER_DISSIMILARITY) {
                    if (mega != 0 && mega != candidate_mega[j])
                        mega = merge(mega, candidate_mega[j]);
                    else
                        mega = candidate_mega[j];
                }
            }
            if (!mega) {
                megaclusters.push_back(Megacluster());
                mega = &megaclusters.back();
            }
            if (good) {
                int bucket = center_candidates[i]->key.junc.size();
                centers[bucket].push_back(pair<Essence, Megacluster*>(
                    Essence(center_candidates[i]->key, center_candidates[i]->canonical_mutlist.clone()), mega));
                n_centers++;
            }
            candidate_mega.push_back(mega);
        }
    }

    void megacluster() {
        if (n_centers <= 1) {
            megaclusters.clear();
            megaclusters.push_back(Megacluster());
            Megacluster& mega = megaclusters.back();
            mega.essences.reserve(essence_pool.size());
            REP(len, MAX_AA_LENGTH + 1)
                for (Essence* ess_ptr : essence_by_length[len])
                    mega.essences.push_back(ess_ptr);
        } else {
            REP(len, MAX_AA_LENGTH + 1) {
                for (Essence* ess_ptr : essence_by_length[len]) {
                    Essence& ess = *ess_ptr;
                    int s0 = len;
                    double best_dist = numeric_limits<double>::infinity();
                    Megacluster* best_mega = nullptr;
                    REP(ds, MAX_AA_LENGTH + 1) {
                        if (cluster_param::len_penalty * ds >= best_dist * s0)
                            break;
                        if (s0 + ds <= MAX_AA_LENGTH)
                            for (const pair<Essence, Megacluster*>& center : centers[s0 + ds]) {
                                double dist = fast_dissimilarity(ess, center.first, GetLD(ess, center.first));
                                if (best_dist > dist) {
                                    best_dist = dist;
                                    best_mega = center.second;
                                }
                            }
                        if (ds > 0 && s0 - ds >= 0 && cluster_param::len_penalty * ds < best_dist * (s0 - ds))
                            for (const pair<Essence, Megacluster*>& center : centers[s0 - ds]) {
                                double dist = fast_dissimilarity(ess, center.first, GetLD(ess, center.first));
                                if (best_dist > dist) {
                                    best_dist = dist;
                                    best_mega = center.second;
                                }
                            }
                    }
                    best_mega->essences.push_back(&ess);
                }
            }
        }
    }

    void cluster() {
        for (Megacluster& mega : megaclusters)
            mega.cluster();
    }

    void process() {
        finalize_parsing();
        create_centers();
        megacluster();
        cluster();
    }
};

// Forward declaration
class Dataset;
extern Dataset dataset;

struct Dataset {
    vector<Partition> partition;
    vector<Essence*> seq_essence;
    vector<string> seq_ids;  // Store sequence IDs for output
    Interning<StringSlice, uint8_t> v_gene_map, j_gene_map;

    void initialize() {
        partition.resize(MAX_PARTITIONS);
        seq_essence.reserve(MAX_SEQUENCES);
        seq_ids.reserve(MAX_SEQUENCES);
        next_cluster = 1;
    }
    void process() {
        if (min_center_size < 0) {
            min_center_size = seq_essence.size() >= 30000 ? MIN_CENTER_SIZE_100K : MIN_CENTER_SIZE_10K;
        }
        for (Partition& p : partition)
            p.process();
    }
} dataset;

string EssenceKey::to_string() const {
    return junc.to_string() + "/" + dataset.v_gene_map.lookup(v_gene).to_string() + "/" +
           dataset.j_gene_map.lookup(j_gene).to_string();
}

/*
 * TSV Parsing utilities.
 */
vector<string> split_tsv(const string& line) {
    vector<string> fields;
    size_t start = 0;
    size_t end = line.find('\t');
    while (end != string::npos) {
        fields.push_back(line.substr(start, end - start));
        start = end + 1;
        end = line.find('\t', start);
    }
    fields.push_back(line.substr(start));
    return fields;
}

// Extract V gene family number from gene name (e.g., "IGHV3-20" -> 3)
int extract_v_family(const string& v_gene) {
    // Look for pattern like "IGHV3" or "IGKV1" or "IGLV2"
    size_t pos = v_gene.find('V');
    if (pos != string::npos && pos + 1 < v_gene.size()) {
        int family = 0;
        pos++;
        while (pos < v_gene.size() && isdigit(v_gene[pos])) {
            family = family * 10 + (v_gene[pos] - '0');
            pos++;
        }
        if (family >= 1 && family <= 7)
            return family;
    }
    return 0;  // Default partition for unknown
}

// Parse mutations from format "6:G>T|26:G>C|31:G>A"
// Returns sorted vector of Mutation values
vector<Mutation> parse_mutations(const string& mut_str) {
    vector<Mutation> mutations;
    if (mut_str.empty())
        return mutations;

    size_t start = 0;
    while (start < mut_str.size()) {
        size_t end = mut_str.find('|', start);
        if (end == string::npos)
            end = mut_str.size();

        string mut = mut_str.substr(start, end - start);

        // Parse "pos:ref>alt"
        size_t colon = mut.find(':');
        if (colon != string::npos) {
            int pos = 0;
            for (size_t i = 0; i < colon; i++) {
                if (isdigit(mut[i]))
                    pos = pos * 10 + (mut[i] - '0');
            }
            pos %= MAX_MUTATION_LOC;

            // Extract ref and alt bases
            size_t gt = mut.find('>', colon);
            if (gt != string::npos && colon + 1 < gt && gt + 1 < mut.size()) {
                char ref = mut[colon + 1];
                char alt = mut[gt + 1];

                // Encode mutation type (similar to original)
                // Uses base encoding: A=0, C=1, G=2, T=3 (approx via ASCII)
                uint8_t m1 = (uint8_t)ref / 2 % 4;
                uint8_t m2 = (uint8_t)alt / 2 % 4;
                m1 -= m2;
                if (ref == '-')
                    m1 = 0;
                int mut_code = m1 % 4 * 4 + m2;

                Mutation mutation = (pos << 4) | mut_code;
                mutations.push_back(mutation);
            }
        }

        start = end + 1;
    }

    // Sort mutations for efficient intersection
    std::sort(mutations.begin(), mutations.end());
    return mutations;
}

// Map column names to indices
struct TSVColumns {
    int sequence_id = -1;
    int v_gene = -1;
    int j_gene = -1;
    int junction_aa = -1;
    int v_mutations = -1;

    bool valid() const {
        return sequence_id >= 0 && v_gene >= 0 && j_gene >= 0 && junction_aa >= 0;
    }
};

TSVColumns find_columns(const vector<string>& header) {
    TSVColumns cols;
    for (int i = 0; i < (int)header.size(); i++) {
        if (header[i] == "sequence_id") cols.sequence_id = i;
        else if (header[i] == "v_gene") cols.v_gene = i;
        else if (header[i] == "j_gene") cols.j_gene = i;
        else if (header[i] == "junction_aa") cols.junction_aa = i;
        else if (header[i] == "v_mutations") cols.v_mutations = i;
    }
    return cols;
}

void throw_parse_error(const char* message) {
    cerr << "parse error: " << message << endl;
    std::exit(1);
}

void parse_tsv(istream& input) {
    string line;

    // Read header line
    if (!getline(input, line)) {
        throw_parse_error("empty input file");
    }
    if (!line.empty() && line.back() == '\r')
        line.pop_back();

    vector<string> header = split_tsv(line);
    TSVColumns cols = find_columns(header);

    if (!cols.valid()) {
        throw_parse_error("missing required columns (sequence_id, v_gene, j_gene, junction_aa)");
    }

    // Read data lines
    while (getline(input, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty())
            continue;

        vector<string> fields = split_tsv(line);

        if ((int)fields.size() <= max({cols.sequence_id, cols.v_gene, cols.j_gene, cols.junction_aa})) {
            continue;  // Skip malformed rows
        }

        string seq_id = fields[cols.sequence_id];
        string v_gene_str = fields[cols.v_gene];
        string j_gene_str = fields[cols.j_gene];
        string junction_aa = fields[cols.junction_aa];
        string v_mutations_str = cols.v_mutations >= 0 && cols.v_mutations < (int)fields.size()
                                 ? fields[cols.v_mutations] : "";

        // Skip sequences with empty junction
        if (junction_aa.empty())
            continue;

        // Get V gene family for partitioning
        int v_fam = extract_v_family(v_gene_str);
        Partition& partition = dataset.partition[v_fam];

        // Intern gene names (gene-level only, stripping allele info)
        // For "IGHV3-20*01", use "IGHV3-20"
        size_t star_pos = v_gene_str.find('*');
        if (star_pos != string::npos)
            v_gene_str = v_gene_str.substr(0, star_pos);
        star_pos = j_gene_str.find('*');
        if (star_pos != string::npos)
            j_gene_str = j_gene_str.substr(0, star_pos);

        uint8_t v_gene = dataset.v_gene_map.intern(StringSlice(v_gene_str.c_str(), v_gene_str.size()));
        uint8_t j_gene = dataset.j_gene_map.intern(StringSlice(j_gene_str.c_str(), j_gene_str.size()));

        // Make junction persistent
        StringSlice junc = make_persistent(StringSlice(junction_aa.c_str(), junction_aa.size()));
        if (junc.size() > MAX_AA_LENGTH)
            junc._size = MAX_AA_LENGTH;

        // Create essence key (gene-level only)
        EssenceKey key = {junc, v_gene, j_gene};
        Essence& essence = partition.essence_lookup(key);

        if (dataset.seq_essence.size() >= MAX_SEQUENCES)
            throw_parse_error("MAX_SEQUENCES exceeded");

        dataset.seq_essence.push_back(&essence);
        dataset.seq_ids.push_back(seq_id);

        // Parse and store mutations
        vector<Mutation> mutations = parse_mutations(v_mutations_str);

        // Copy mutations to partition pool
        size_t start_idx = partition.mutation_pool.size();
        for (Mutation m : mutations)
            partition.mutation_pool.push_back(m);

        Slice<Mutation> mutlist(&partition.mutation_pool[start_idx], (int)mutations.size());
        essence.push_mutlist(mutlist);
    }
}

struct Clonify {
    Clonify() {
        dataset.initialize();
    }

    void cluster(istream& input) {
        parse_tsv(input);
        dataset.process();
    }

    void write_output(ostream& output) {
        output << "sequence_id\tcluster_id\n";
        REP(i, (int)dataset.seq_essence.size()) {
            output << dataset.seq_ids[i] << "\t" << dataset.seq_essence[i]->cluster_string << "\n";
        }
    }
};

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);

    // Parse arguments
    if (argc > 1 && !strncmp(argv[1], "--min-center-size=", strlen("--min-center-size="))) {
        if (!strcmp(argv[1], "--min-center-size=disabled"))
            min_center_size = INT_MAX;
        else
            min_center_size = atoi(argv[1] + strlen("--min-center-size="));
        if (min_center_size < 1) {
            cerr << "invalid parameter for min-center-size" << endl;
            std::exit(1);
        }
        argc--;
        argv++;
    }

    if (argc != 3) {
        cerr << "Clonify - Antibody Clonotype Clustering\n\n";
        cerr << "Usage: " << argv[0] << " [options...] input.tsv output.tsv\n\n";
        cerr << "Input TSV must contain columns: sequence_id, v_gene, j_gene, junction_aa\n";
        cerr << "Optional column: v_mutations (format: pos:ref>alt|pos:ref>alt|...)\n\n";
        cerr << "Options:\n";
        cerr << "  --min-center-size=INTEGER  Set the minimum center size (default: auto)\n";
        cerr << "  --min-center-size=disabled Disable megaclustering (slow for large datasets)\n";
        std::exit(1);
    }

    string input_path = argv[1];
    string output_path = argv[2];

    Clonify clonify;

    ifstream input(input_path);
    if (!input.good()) {
        cerr << "couldn't open input file '" << input_path << "'" << endl;
        std::exit(1);
    }

    struct timeval tv_startup, tv;
    gettimeofday(&tv_startup, NULL);

    clonify.cluster(input);

    gettimeofday(&tv, NULL);
    double t = ((tv.tv_sec - tv_startup.tv_sec) + (tv.tv_usec - tv_startup.tv_usec) * 1e-6) * 1000;

    cout << "Clustering complete: " << dataset.seq_essence.size() << " sequences -> "
         << (next_cluster - 1) << " clusters in " << (int)t << " ms" << endl;

    ofstream output(output_path);
    if (!output.good()) {
        cerr << "couldn't open output file '" << output_path << "'" << endl;
        std::exit(1);
    }
    clonify.write_output(output);

    return 0;
}
