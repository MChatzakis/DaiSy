// Adapted from FADASNode.cpp in DumpyOS/src/IndexConstruction/FADASNode.cpp
// (Wang et al., "DumpyOS: A Dumpy Index for Scalable Data Series Similarity
//  Search", VLDB Journal 2024)
// Reference: SaxUtil.cpp (getMidLineFromSaxSymbolbc8, extendSax),
//            MathUtil.cpp (deviation, generateMaskSettingKbits)

#include "DumpyOS.hpp"
#include "../isax/SAX.hpp"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <map>
#include <numeric>
#include <unordered_set>
#include <vector>

namespace daisy {

// ---- bp8 : normal-distribution breakpoints for 8-bit SAX ----
// Directly from SaxUtil.cpp in the DumpyOS reference implementation.
// bp8[i] is the upper boundary of the i-th SAX symbol region.
// bp8[255] = +inf sentinel; midpoints are extrapolated for symbols 0 and 255.
static const double bp8[256] = {
    -2.660067468617458,-2.4175590162365035,-2.2662268092096522,
    -2.1538746940614573,-2.063527898316245,-1.9874278859298962,
    -1.921350774293703,-1.8627318674216515,-1.8098922384806087,
    -1.7616704103630665,-1.7172281175057411,-1.6759397227734438,
    -1.637325382768064,-1.601008664886076,-1.5666885860684134,
    -1.534120544352546,-1.5031029431292737,-1.4734675779471014,
    -1.4450725798180746,-1.4177971379962677,-1.3915374879959008,
    -1.3662038163720984,-1.341717841080254,-1.318010897303537,
    -1.2950224067058147,-1.2726986411905359,-1.2509917154625454,
    -1.229858759216589,-1.209261231709155,-1.189164350199337,
    -1.169536610207143,-1.1503493803760083,-1.131576558386188,
    -1.113194277160929,-1.0951806527613883,-1.0775155670402805,
    -1.0601804794353549,-1.0431582633184537,-1.0264330631379108,
    -1.0099901692495823,-0.993815907860883,-0.9778975439405418,
    -0.9622231952954206,-0.946781756301046,-0.9315628300071148,
    -0.9165566675331128,-0.9017541138301002,-0.8871465590188762,
    -0.8727258946270402,-0.858484474141832,-0.8444150773752572,
    -0.8305108782053992,-0.8167654153150912,-0.8031725655979178,
    -0.7897265199432658,-0.7764217611479275,-0.7632530437325706,
    -0.7502153754679404,-0.7373040004386545,-0.7245143834923653,
    -0.711842195939419,-0.69928330238322,-0.6868337485747303,
    -0.6744897501960819,-0.6622476824884141,-0.6501040706479954,
    -0.6380555809225171,-0.6260990123464211,-0.6142312890602454,
    -0.6024494531644237,-0.5907506580628189,-0.5791321622555561,
    -0.5675913235445692,-0.5561255936186916,-0.5447325129881759,
    -0.5334097062412806,-0.5221548775980015,-0.5109658067382474,
    -0.4998403448837353,-0.4887764111146696,-0.4777719889038861,
    -0.46682512285258965,-0.4559339156131388,-0.44509652498551644,
    -0.4343111611752096,-0.42357608420119974,-0.41288960144365433,
    -0.40225006532172536,-0.3916558710925915,-0.3811054547635565,
    -0.3705972911096293,-0.36012989178956945,-0.3497018035538953,
    -0.3393116065388173,-0.3289579126404911,-0.31863936396437526,
    -0.30835463134483726,-0.2981024129304869,-0.2878814328310118,
    -0.27769043982157676,-0.2675282061010972,-0.25739352610093835,
    -0.24728521534080486,-0.2372021093287877,-0.22714306250271535,
    -0.2171069472101298,-0.2070926527243603,-0.19709908429431236,
    -0.18712516222572084,-0.17716982099173986,-0.16723200837085014,
    -0.15731068461017073,-0.14740482161235488,-0.13751340214433597,
    -0.12763541906627035,-0.11776987457909531,-0.10791577948918657,
    -0.0980721524886611,-0.08823801944992447,-0.07841241273311222,
    -0.06859437050511813,-0.05878293606894307,-0.04897715720213194,
    -0.03917608550309764,-0.02937877574415705,-0.019584285230126924,
    -0.009791673161345348,
     0.0,
     0.009791673161345348, 0.019584285230126924, 0.02937877574415705,
     0.03917608550309764,  0.04897715720213194,  0.05878293606894307,
     0.06859437050511813,  0.07841241273311222,  0.08823801944992447,
     0.0980721524886611,   0.10791577948918657,  0.11776987457909531,
     0.12763541906627035,  0.13751340214433597,  0.14740482161235488,
     0.15731068461017073,  0.16723200837085014,  0.17716982099173986,
     0.18712516222572084,  0.19709908429431236,  0.2070926527243603,
     0.2171069472101298,   0.22714306250271535,  0.2372021093287877,
     0.24728521534080486,  0.25739352610093835,  0.2675282061010972,
     0.27769043982157676,  0.2878814328310118,   0.2981024129304869,
     0.30835463134483726,  0.31863936396437526,  0.3289579126404911,
     0.3393116065388173,   0.3497018035538953,   0.36012989178956945,
     0.3705972911096293,   0.3811054547635565,   0.3916558710925915,
     0.40225006532172536,  0.41288960144365433,  0.42357608420119974,
     0.4343111611752096,   0.44509652498551644,  0.4559339156131388,
     0.46682512285258965,  0.4777719889038861,   0.4887764111146696,
     0.4998403448837353,   0.5109658067382474,   0.5221548775980015,
     0.5334097062412806,   0.5447325129881759,   0.5561255936186916,
     0.5675913235445692,   0.5791321622555561,   0.5907506580628189,
     0.6024494531644237,   0.6142312890602454,   0.6260990123464211,
     0.6380555809225171,   0.6501040706479954,   0.6622476824884141,
     0.6744897501960819,   0.6868337485747303,   0.69928330238322,
     0.711842195939419,    0.7245143834923653,   0.7373040004386545,
     0.7502153754679404,   0.7632530437325706,   0.7764217611479275,
     0.7897265199432658,   0.8031725655979178,   0.8167654153150912,
     0.8305108782053992,   0.8444150773752572,   0.858484474141832,
     0.8727258946270402,   0.8871465590188762,   0.9017541138301002,
     0.9165566675331128,   0.9315628300071148,   0.946781756301046,
     0.9622231952954206,   0.9778975439405418,   0.993815907860883,
     1.0099901692495823,   1.0264330631379108,   1.0431582633184537,
     1.0601804794353549,   1.0775155670402805,   1.0951806527613883,
     1.113194277160929,    1.131576558386188,    1.1503493803760083,
     1.169536610207143,    1.189164350199337,    1.209261231709155,
     1.229858759216589,    1.2509917154625454,   1.2726986411905359,
     1.2950224067058147,   1.318010897303537,    1.341717841080254,
     1.3662038163720984,   1.3915374879959008,   1.4177971379962677,
     1.4450725798180746,   1.4734675779471014,   1.5031029431292737,
     1.534120544352546,    1.5666885860684134,   1.601008664886076,
     1.637325382768064,    1.6759397227734438,   1.7172281175057411,
     1.7616704103630665,   1.8098922384806087,   1.8627318674216515,
     1.921350774293703,    1.9874278859298962,   2.063527898316245,
     2.1538746940614573,   2.2662268092096522,   2.4175590162365035,
     2.660067468617458,
     // sentinel for symbol 255 upper bound (numeric_limits<float>::max())
     3.4028234663852886e+38
};

// ---- internal helpers (adapted from SaxUtil and MathUtil) ----

// Equivalent to SaxUtil::getMidLineFromSaxSymbolbc8.
static double get_midpoint(unsigned sym) {
    if (sym == 0)   return bp8[0] - (bp8[1] - bp8[0]);
    if (sym == 255) return bp8[254] + (bp8[254] - bp8[253]);
    return (bp8[sym - 1] + bp8[sym]) * 0.5;
}

// Equivalent to SaxUtil::extendSax(sax, bits_cardinality) — all segments.
// Produces a vertexNum (1<<w) index for the unit_size table.
static int extend_sax_all(const sax_type* sax, const int* levels, int w, int max_bits) {
    int res = 0;
    for (int s = 0; s < w; ++s) {
        int shift = max_bits - levels[s] - 1;
        int bit   = (shift >= 0) ? ((sax[s] >> shift) & 1) : 0;
        res = (res << 1) | bit;
    }
    return res;
}

// Equivalent to SaxUtil::extendSax(sax, bits_cardinality, chosenSegments).
// Produces a child ID in [0, 2^|chosen|).
static int extend_sax_chosen(const sax_type* sax, const int* levels,
                               const std::vector<int>& chosen, int max_bits) {
    int res = 0;
    for (int s : chosen) {
        int shift = max_bits - levels[s] - 1;
        int bit   = (shift >= 0) ? ((sax[s] >> shift) & 1) : 0;
        res = (res << 1) | bit;
    }
    return res;
}

// Equivalent to MathUtil::generateMaskSettingKbits(plan, k, len).
// plan must be sorted ascending.  Sets bit (w-1-plan[i]) for each i in plan.
static int generate_mask(const int* plan, int lambda, int w) {
    int res = 0, cur = 0;
    for (int i = 0; i < w; ++i) {
        res <<= 1;
        if (cur < lambda && plan[cur] == i) { res |= 1; ++cur; }
    }
    return res;
}

// Population standard deviation — equivalent to MathUtil::deviation(double*, int).
static double pop_stdev(const double* vals, int n) {
    double mean = 0.0;
    for (int i = 0; i < n; ++i) mean += vals[i];
    mean /= n;
    double var = 0.0;
    for (int i = 0; i < n; ++i) { double d = vals[i] - mean; var += d * d; }
    return std::sqrt(var / n);
}

// Equivalent to FADASNode::compute_score.
// node_n  = node->n (total series in node)
// alpha   = config_.alpha
static double compute_score_(const std::vector<int>& node_sizes,
                              const int* plan, int lambda,
                              const std::vector<double>& data_seg_stdev,
                              int node_n, int leaf_size, double alpha) {
    if (node_n < 2 * leaf_size) {
        if ((int)node_sizes.size() >= 2 &&
            (node_sizes[0] > leaf_size || node_sizes[1] > leaf_size))
            return (double)std::min(node_sizes[0], node_sizes[1]) / leaf_size;
        return data_seg_stdev[plan[0]] * 100.0;
    }
    int over = 0;
    for (int s : node_sizes) if (s > leaf_size) ++over;
    double w_frac = (double)over / (int)node_sizes.size();

    double sum_seg = 0.0;
    for (int i = 0; i < lambda; ++i) sum_seg += data_seg_stdev[plan[i]];
    sum_seg = std::exp(1.0 + std::sqrt(sum_seg / lambda));

    int nc = (int)node_sizes.size();
    std::vector<double> tmp(nc);
    for (int i = 0; i < nc; ++i) tmp[i] = (double)node_sizes[i] / leaf_size;
    double sigma_f = pop_stdev(tmp.data(), nc);
    double balance = std::exp(-(1.0 + w_frac) * sigma_f);
    return sum_seg + alpha * balance;
}

// Equivalent to FADASNode::visitPlanFromBaseTable.
// cur_lambda = parent lambda minus 1 (we're evaluating plans of size cur_lambda).
// plan       = parent plan of size cur_lambda + 1 (sorted ascending).
// base_tbl   = parent child fill counts of size 1 << (cur_lambda + 1).
static void visit_plan_from_base_table(
    std::unordered_set<int>& visited,
    int cur_lambda, const int* plan, const std::vector<int>& base_tbl,
    double* max_score, std::vector<int>& best_plan,
    int lambda_min, int mask_code,
    const std::vector<double>& data_seg_stdev,
    int node_n, int leaf_size, double alpha, int w)
{
    // base_mask has (cur_lambda+1) bits set (covers all 1<<(cur_lambda+1) entries)
    int base_mask = 1;
    for (int i = 0; i < cur_lambda; ++i) base_mask = (base_mask << 1) | 1;

    for (int i = 0; i <= cur_lambda; ++i) {
        int reset_pos       = plan[i];
        int cur_whole_mask  = mask_code - (1 << (w - 1 - reset_pos));
        if (visited.count(cur_whole_mask)) continue;
        visited.insert(cur_whole_mask);

        // Sub-plan: drop element i from parent plan
        int* new_plan = new int[cur_lambda];
        for (int j = 0, k = 0; j <= cur_lambda; ++j)
            if (j != i) new_plan[k++] = plan[j];

        // Aggregate base_tbl into cur_lambda child fill counts by masking out bit i
        int cur_base_mask = base_mask - (1 << (cur_lambda - i));
        std::map<int, int> nsmap;
        for (int j = 0; j < (int)base_tbl.size(); ++j)
            nsmap[cur_base_mask & j] += base_tbl[j];
        std::vector<int> new_tbl;
        new_tbl.reserve(1 << cur_lambda);
        for (auto& kv : nsmap) new_tbl.push_back(kv.second);

        double score = compute_score_(new_tbl, new_plan, cur_lambda,
                                       data_seg_stdev, node_n, leaf_size, alpha);
        if (score > *max_score) {
            *max_score = score;
            best_plan.assign(new_plan, new_plan + cur_lambda);
        }
        if (cur_lambda > lambda_min)
            visit_plan_from_base_table(visited, cur_lambda - 1, new_plan, new_tbl,
                                        max_score, best_plan, lambda_min,
                                        cur_whole_mask, data_seg_stdev,
                                        node_n, leaf_size, alpha, w);
        delete[] new_plan;
    }
}

// Equivalent to FADASNode::determineFanout.
static void determine_fanout(int n, int leaf_size, double f_low, double f_high, int w,
                              int* lambda_min, int* lambda_max) {
    if (n < 2 * leaf_size) { *lambda_min = 1; *lambda_max = 1; return; }
    *lambda_min = -1;
    *lambda_max = w;
    int vertex_num = 1 << w;
    double _min = (double)n / (leaf_size * f_high);
    double _max = (double)n / (leaf_size * f_low);
    if ((double)vertex_num < _min) { *lambda_min = w; *lambda_max = w; return; }
    for (int i = 1; i <= w; ++i) {
        if (*lambda_min == -1) {
            if ((double)(1 << i) >= _min) *lambda_min = i;
        } else {
            if ((double)(1 << i) == _max)      { *lambda_max = i; break; }
            else if ((double)(1 << i) > _max)  { *lambda_max = std::max(i - 1, *lambda_min); break; }
        }
    }
    if (*lambda_min == -1) { *lambda_min = w; *lambda_max = w; }
}

// Equivalent to FADASNode::determineSegments.
void DumpyOS::determineSegments_(DumpyOSNode* node) {
    int    w        = config_.paa_segments;
    int    max_bits = config_.sax_bit_cardinality;
    int    ls       = config_.leaf_size;
    double alpha    = config_.alpha;

    int lambda_min, lambda_max;
    determine_fanout(node->n, ls, config_.fill_lower, config_.fill_upper,
                     w, &lambda_min, &lambda_max);

    // Edge case: need all segments (vertexNum < _min)
    if (lambda_min == w && lambda_max == w) {
        for (int i = 0; i < w; ++i) node->chosen_segs.push_back(i);
        return;
    }

    int vertex_num = 1 << w;

    // Build unit_size table: aggregate series counts into 2^w first-level buckets
    std::vector<int> unit_size(vertex_num, 0);
    // Per-segment SAX symbol frequency (for data_seg_stdev)
    std::vector<std::unordered_map<int, int>> seg_sym_cnt(w);

    for (idx_t si : node->entries) {
        const sax_type* sax = sax_table_ + (size_t)si * w;
        for (int i = 0; i < w; ++i)
            seg_sym_cnt[i][(int)(unsigned char)sax[i]]++;
        int head = extend_sax_all(sax, node->levels.data(), w, max_bits);
        unit_size[head]++;
    }

    // Compute data_seg_mean and data_seg_stdev (variance of midpoint values)
    // Equivalent to the data_seg_mean/stdev loop in FADASNode::determineSegments.
    std::vector<double> data_seg_mean(w, 0.0), data_seg_stdev(w, 0.0);
    for (int i = 0; i < w; ++i) {
        for (auto& kv : seg_sym_cnt[i])
            data_seg_mean[i] += get_midpoint((unsigned)kv.first) * kv.second;
        data_seg_mean[i] /= node->n;
        for (auto& kv : seg_sym_cnt[i]) {
            double d = get_midpoint((unsigned)kv.first) - data_seg_mean[i];
            data_seg_stdev[i] += kv.second * d * d;
        }
        data_seg_stdev[i] /= node->n;
    }

    double            max_score = 0.0;
    std::vector<int>  best_plan;
    std::unordered_set<int> visited;

    // Enumerate all C(w, lambda_max) plans; for each, also search sub-plans
    // via visitPlanFromBaseTable (lambda_min..lambda_max-1).
    std::vector<int> idx(lambda_max);
    std::iota(idx.begin(), idx.end(), 0);

    while (true) {
        const int* plan = idx.data();

        // Compute child fill counts using unit_size table + bit masking
        int mask_code = generate_mask(plan, lambda_max, w);
        std::map<int, int> node_size_map;
        for (int j = 0; j < vertex_num; ++j)
            node_size_map[mask_code & j] += unit_size[j];
        std::vector<int> plan_node_sizes;
        plan_node_sizes.reserve(1 << lambda_max);
        for (auto& kv : node_size_map) plan_node_sizes.push_back(kv.second);

        double score = compute_score_(plan_node_sizes, plan, lambda_max,
                                       data_seg_stdev, node->n, ls, alpha);
        if (score > max_score) {
            max_score = score;
            best_plan.assign(plan, plan + lambda_max);
        }

        if (lambda_min <= lambda_max - 1)
            visit_plan_from_base_table(visited, lambda_max - 1, plan, plan_node_sizes,
                                        &max_score, best_plan, lambda_min, mask_code,
                                        data_seg_stdev, node->n, ls, alpha, w);

        // Advance to next C(w, lambda_max) combination (Gosper-style)
        int i = lambda_max - 1;
        while (i >= 0 && idx[i] == w - lambda_max + i) --i;
        if (i < 0) break;
        ++idx[i];
        for (int j = i + 1; j < lambda_max; ++j) idx[j] = idx[j - 1] + 1;
    }

    node->chosen_segs = best_plan;
}

// ---- splitNode_ ----

void DumpyOS::splitNode_(DumpyOSNode* node) {
    int w        = config_.paa_segments;
    int max_bits = config_.sax_bit_cardinality;

    determineSegments_(node);
    if (node->chosen_segs.empty()) return;  // exhausted bit budget

    int lam    = (int)node->chosen_segs.size();
    int num_ch = 1 << lam;
    node->children.assign(num_ch, nullptr);

    // Create all children with inherited levels (+ 1 for each chosen segment)
    for (int sid = 0; sid < num_ch; ++sid) {
        DumpyOSNode* child = new DumpyOSNode();
        child->levels = node->levels;
        for (int cs : node->chosen_segs) child->levels[cs]++;
        node->children[sid] = child;
    }

    // Redistribute series — equivalent to FADASNode's child routing
    for (idx_t si : node->entries) {
        const sax_type* sax = sax_table_ + (size_t)si * w;
        int sid = extend_sax_chosen(sax, node->levels.data(), node->chosen_segs, max_bits);
        node->children[sid]->entries.push_back(si);
        node->children[sid]->n++;
    }
    node->entries.clear();
    node->entries.shrink_to_fit();

    for (DumpyOSNode* child : node->children)
        if (child && child->n > config_.leaf_size)
            splitNode_(child);
}

// ---- buildIndex ----

void DumpyOS::buildIndex(DataSource* data_source) {
    dim        = data_source->getDim();
    n_database = data_source->getTotalRecords();

    const float* raw = data_source->rawPointer();
    if (raw) {
        database       = const_cast<float*>(raw);
        owns_database_ = false;
    } else {
        database       = new float[(size_t)n_database * dim];
        owns_database_ = true;
        float* ptr = database;
        while (data_source->nextRecord(ptr)) ptr += dim;
    }

    int w           = config_.paa_segments;
    int max_bits    = config_.sax_bit_cardinality;
    int cardinality = 1 << max_bits;
    int pts_per_seg = (int)dim / w;

    // Compute and cache the SAX word for every database series
    sax_table_ = new sax_type[(size_t)n_database * w];
    std::vector<ts_type> paa(w);

    for (idx_t i = 0; i < n_database; ++i) {
        paa_from_ts(database + (size_t)i * dim, paa.data(), w, pts_per_seg);
        sax_from_paa(paa.data(), sax_table_ + (size_t)i * w, w, cardinality, max_bits);
    }

    // Root: all series, all segment levels at 0
    root_ = new DumpyOSNode();
    root_->levels.assign(w, 0);
    root_->n = (int)n_database;
    root_->entries.resize(n_database);
    std::iota(root_->entries.begin(), root_->entries.end(), (idx_t)0);

    if (root_->n > config_.leaf_size)
        splitNode_(root_);
}

// ---- searchIndex ----

static float l2sq_early(const float* a, const float* b, int d, float bound) {
    float s = 0.0f;
    for (int i = 0; i < d; ++i) {
        float v = a[i] - b[i];
        s += v * v;
        if (s >= bound) return s;
    }
    return s;
}

void DumpyOS::searchIndex(const float* query, idx_t n_query, idx_t k,
                           idx_t* I, float* D) {
    if (!validateSearchParams(k, n_query)) return;

    int w           = config_.paa_segments;
    int max_bits    = config_.sax_bit_cardinality;
    int cardinality = 1 << max_bits;
    int pts_per_seg = (int)dim / w;

    std::vector<sax_type> q_sax(w);
    std::vector<ts_type>  q_paa(w);

    for (idx_t qi = 0; qi < n_query; ++qi) {
        const float* q = query + (size_t)qi * dim;

        paa_from_ts(q, q_paa.data(), w, pts_per_seg);
        sax_from_paa(q_paa.data(), q_sax.data(), w, cardinality, max_bits);

        // Route to leaf — equivalent to FADASNode::route()
        DumpyOSNode* node = root_;
        while (!node->chosen_segs.empty()) {
            int  sid = extend_sax_chosen(q_sax.data(), node->levels.data(),
                                          node->chosen_segs, max_bits);
            if (sid < 0 || sid >= (int)node->children.size() ||
                node->children[sid] == nullptr) break;
            node = node->children[sid];
        }

        // kNN via max-heap
        using Pair = std::pair<float, idx_t>;
        std::vector<Pair> heap;
        heap.reserve((size_t)k + 1);
        float bsf = FLT_MAX;

        for (idx_t si : node->entries) {
            float dist = l2sq_early(q, database + (size_t)si * dim, (int)dim, bsf);
            if ((idx_t)heap.size() < k || dist < bsf) {
                heap.push_back({dist, si});
                std::push_heap(heap.begin(), heap.end());
                if ((idx_t)heap.size() > k) {
                    std::pop_heap(heap.begin(), heap.end());
                    heap.pop_back();
                }
                if ((idx_t)heap.size() == k) bsf = heap.front().first;
            }
        }

        std::sort_heap(heap.begin(), heap.end());
        for (idx_t j = 0; j < k; ++j) {
            if (j < (idx_t)heap.size()) {
                I[qi * k + j] = heap[j].second;
                D[qi * k + j] = heap[j].first;
            } else {
                I[qi * k + j] = 0;
                D[qi * k + j] = FLT_MAX;
            }
        }
    }
}

// ---- destructor ----

void DumpyOS::destroyTree_(DumpyOSNode* node) {
    if (!node) return;
    for (DumpyOSNode* child : node->children) destroyTree_(child);
    delete node;
}

DumpyOS::DumpyOS(DistanceType distance_type)
    : SimilaritySearchAlgorithm(distance_type) {}

DumpyOS::DumpyOS(DistanceType distance_type, const DumpyOSConfig& config)
    : SimilaritySearchAlgorithm(distance_type), config_(config) {}

DumpyOS::~DumpyOS() {
    destroyTree_(root_);
    delete[] sax_table_;
    if (owns_database_) delete[] database;
}

} // namespace daisy
