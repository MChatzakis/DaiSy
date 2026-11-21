// optimizer_collector.cpp
//
// Collects training samples for the knob optimizer.
// Knobs, outputs, caveats and how to run a full collection are documented in
// docs/optimizer-notes/collector.md.
//
// For every dataset, algorithm and knob configuration of the grid below, it builds
// the index once, records the shape of the built tree, and then answers every
// query of the workload one at a time, for every k. Each answered query becomes
// one CSV row:
//
//   host, cpu_model, cpu_sockets, cpu_cores, cpu_threads, cpu_max_mhz, l2_cache_kb,
//   l3_cache_kb, numa_nodes, memory_gb, simd, gpu_count, gpu_model, compiler,
//   algorithm, dataset_name, dataset_size, dataset_dimensionality, k, query_id,
//   <one column per knob (NaN when the algorithm does not have it)>,
//   latency_ms, build_time_ms,
//   n_roots, n_leaves, leaf_fill_mean, leaf_fill_max, largest_leaf_share, tree_depth,
//   node_lb_calcs, leaves_enqueued, leaves_visited, series_lb_calcs, distance_calcs,
//   seed_kth_dist, final_kth_dist
//
// The machine columns describe the server the run is on (read from /proc and /sys
// at start-up) and are the same on every row of a run; they let samples from
// several servers be pooled. The `threads` knob (used both to build the index and
// to answer queries) is derived from the machine too: P, 3P/4, P/2, P/4 and P/8,
// where P is the largest power of two not above the hardware threads.
// latency_ms is the wall-clock time of one single-query searchIndex call.
// build_time_ms is the wall-clock time of the configuration's buildIndex call;
// it and the tree-shape columns repeat on every row of the configuration.
// The search measures come from getLastSearchStats() (lib/utils/SearchStats.hpp)
// and are NaN for an algorithm that does not collect them.
//
// Running a long collection:
//   - A configuration's rows are appended in one write after it finishes, so a
//     failed configuration leaves no rows behind.
//   - Budget: once the queries of one (configuration, k) have used --budget-s
//     seconds, the remaining queries of that k are skipped (logged as "stopped").
//   - A dataset, or Odyssey's second copy of it, is skipped when it does not fit
//     in the memory currently available.
//   - --dry-run 1 prints the plan (sizes, memory, configurations) and exits.
//
// Datasets: DATA_ROOT/<NAME>/base.<n>.fvecs and query.<n>.fvecs, prepared on each server by
// scripts/datasets/prepare_datasets.py (--data-root, or $DAISY_DATA_ROOT).
//
// Algorithms: MESSI, FreSh, SOFA (needs BUILD_SOFA), Odyssey single node (needs
// BUILD_ODYSSEY), SING (needs BUILD_SING and a GPU). Do not launch it under mpirun:
// Odyssey runs as a single-node MPI singleton.
//
// Usage:
//   optimizer_collector [--datasets SIFT1M,GIST1M|all] [--algos messi,fresh,sofa,odyssey,sing]
//                       [--k 1,10,100] [--queries 100] [--max-base N] [--budget-s 600]
//                       [--out optimizer_samples.csv] [--work-dir optimizer_work]
//                       [--data-root DIR] [--dry-run 1]

#include "algos/Messi.hpp"
#include "algos/Fresh.hpp"
#include "algos/Sofa.hpp" // empty unless SOFA_FFTW_ENABLED
#include "algos/Sing.hpp" // usable only when SING_CUDA_ENABLED (BUILD_SING, CUDA)
#if defined(ODYSSEY_MPI) && ODYSSEY_MPI
#include "algos/hodyssey/Odyssey.hpp"
#include <mpi.h>
#endif

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <csignal>
#include <sched.h>  // sched_getaffinity
#include <unistd.h> // sysconf, fsync

namespace fs = std::filesystem;
using daisy::idx_t;
using Clock = std::chrono::steady_clock;

// ============================================================================
// The machine the run is on
// ============================================================================

struct Machine
{
    std::string host = "unknown", cpu_model = "unknown", simd = "unknown", compiler = "unknown";
    int sockets = 0, cores = 0, threads = 0, numa_nodes = 0;
    double max_mhz = NAN, l2_kb = NAN, l3_kb = NAN, memory_gb = NAN;
    int gpus = 0;                  // NVIDIA GPUs
    std::string gpu_model = "none"; // model of the first one
};

// CSV-safe text: no commas or line breaks, single spaces, "unknown" when empty.
std::string csv_text(std::string s)
{
    for (char &ch : s)
        if (ch == ',' || ch == '\n' || ch == '\r' || ch == '\t')
            ch = ' ';
    std::string out;
    for (char ch : s)
        if (!(ch == ' ' && (out.empty() || out.back() == ' ')))
            out += ch;
    while (!out.empty() && out.back() == ' ')
        out.pop_back();
    return out.empty() ? "unknown" : out;
}

std::string read_line(const fs::path &path)
{
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    return line;
}

// "1024K" / "30M" (sysfs cache sizes) -> KB; NaN if unreadable.
double size_kb(const std::string &s)
{
    if (s.empty())
        return NAN;
    double v = std::atof(s.c_str());
    if (s.back() == 'M')
        v *= 1024;
    else if (s.back() == 'G')
        v *= 1024 * 1024;
    return v;
}

Machine detect_machine()
{
    Machine m;
    char host[256] = {};
    if (gethostname(host, sizeof(host) - 1) == 0)
        m.host = csv_text(host);

    // /proc/cpuinfo: one block per logical CPU.
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line, physical_id = "0", flags;
    std::set<std::string> sockets, cores;
    while (std::getline(cpuinfo, line))
    {
        const size_t colon = line.find(':');
        if (colon == std::string::npos)
            continue;
        std::string key = line.substr(0, colon), value = colon + 1 < line.size() ? line.substr(colon + 2) : "";
        key.erase(key.find_last_not_of(" \t") + 1);
        if (key == "processor")
            m.threads++;
        else if (key == "model name" && m.cpu_model == "unknown")
            m.cpu_model = csv_text(value);
        else if (key == "physical id")
        {
            physical_id = value;
            sockets.insert(value);
        }
        else if (key == "core id")
            cores.insert(physical_id + ":" + value);
        else if (key == "flags" && flags.empty())
            flags = " " + value + " ";
    }
    if (m.threads == 0)
        m.threads = (int)std::max(1u, std::thread::hardware_concurrency());
    m.sockets = sockets.empty() ? 1 : (int)sockets.size();
    m.cores = cores.empty() ? m.threads : (int)cores.size();
    if (flags.find(" avx512f ") != std::string::npos)
        m.simd = "avx512";
    else if (flags.find(" avx2 ") != std::string::npos)
        m.simd = "avx2";
    else if (flags.find(" avx ") != std::string::npos)
        m.simd = "avx";
    else if (flags.find(" sse4_2 ") != std::string::npos)
        m.simd = "sse4.2";

    const std::string khz = read_line("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
    if (!khz.empty())
        m.max_mhz = std::atof(khz.c_str()) / 1000.0;

    // Caches: L2 of one core; L3 summed over its distinct instances (sockets, CCXs).
    std::error_code ec;
    std::set<std::string> l3_instances;
    double l3_total = 0;
    for (const auto &cpu : fs::directory_iterator("/sys/devices/system/cpu", ec))
    {
        const std::string name = cpu.path().filename().string();
        if (name.size() < 4 || name.compare(0, 3, "cpu") != 0 || !std::isdigit((unsigned char)name[3]))
            continue;
        for (const auto &idx : fs::directory_iterator(cpu.path() / "cache", ec))
        {
            const std::string level = read_line(idx.path() / "level");
            const double kb = size_kb(read_line(idx.path() / "size"));
            if (level == "2" && name == "cpu0")
                m.l2_kb = kb;
            else if (level == "3" && l3_instances.insert(read_line(idx.path() / "shared_cpu_list")).second &&
                     !std::isnan(kb))
                l3_total += kb;
        }
    }
    if (!l3_instances.empty())
        m.l3_kb = l3_total;

    for (const auto &node : fs::directory_iterator("/sys/devices/system/node", ec))
    {
        const std::string name = node.path().filename().string();
        if (name.size() > 4 && name.compare(0, 4, "node") == 0 && std::isdigit((unsigned char)name[4]))
            m.numa_nodes++;
    }
    if (m.numa_nodes == 0)
        m.numa_nodes = 1;

    std::ifstream meminfo("/proc/meminfo");
    std::string key, unit;
    double kb = 0;
    while (meminfo >> key >> kb >> unit)
        if (key == "MemTotal:")
        {
            m.memory_gb = kb / 1024.0 / 1024.0;
            break;
        }

    // GPUs: the CUDA runtime when SING is built, otherwise the NVIDIA driver's
    // /proc entries (one directory per GPU).
#if SING_CUDA_ENABLED
    int n_gpus = 0;
    if (cudaGetDeviceCount(&n_gpus) == cudaSuccess && n_gpus > 0)
    {
        m.gpus = n_gpus;
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
            m.gpu_model = csv_text(prop.name);
    }
#endif
    if (m.gpus == 0)
        for (const auto &gpu : fs::directory_iterator("/proc/driver/nvidia/gpus", ec))
        {
            if (m.gpus++ > 0)
                continue;
            std::ifstream info(gpu.path() / "information");
            std::string info_line;
            while (std::getline(info, info_line))
                if (info_line.compare(0, 6, "Model:") == 0)
                    m.gpu_model = csv_text(info_line.substr(6));
        }

#if defined(__clang__)
    m.compiler = csv_text(std::string("clang ") + __clang_version__);
#elif defined(__GNUC__)
    m.compiler = csv_text(std::string("gcc ") + __VERSION__);
#endif
    return m;
}

const std::vector<std::string> MACHINE_COLUMNS = {
    "host",
    "cpu_model",
    "cpu_sockets",
    "cpu_cores",
    "cpu_threads",
    "cpu_max_mhz",
    "l2_cache_kb",
    "l3_cache_kb",
    "numa_nodes",
    "memory_gb",
    "simd",
    "gpu_count",
    "gpu_model",
    "compiler",
};

std::string fmt_number(double v, int decimals)
{
    if (std::isnan(v))
        return "NaN";
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
    return buf;
}

// The machine columns, in MACHINE_COLUMNS order, each followed by a comma.
std::string fmt_machine(const Machine &m)
{
    return m.host + "," + m.cpu_model + "," + std::to_string(m.sockets) + "," + std::to_string(m.cores) + "," +
           std::to_string(m.threads) + "," + fmt_number(m.max_mhz, 0) + "," + fmt_number(m.l2_kb, 0) + "," +
           fmt_number(m.l3_kb, 0) + "," + std::to_string(m.numa_nodes) + "," + fmt_number(m.memory_gb, 1) + "," +
           m.simd + "," + std::to_string(m.gpus) + "," + m.gpu_model + "," + m.compiler + ",";
}

// CPUs this process may run on (its affinity mask, e.g. a Slurm allocation), 0 if unknown.
int usable_cpus()
{
    cpu_set_t set;
    CPU_ZERO(&set);
    return sched_getaffinity(0, sizeof(set), &set) == 0 ? CPU_COUNT(&set) : 0;
}

// Thread values to try: P, 3P/4, P/2, P/4 and P/8, where P is the largest power of two
// that does not exceed the threads available: the hardware threads, at most the CPUs this
// process may use, and at most --max-threads if given (erato: 48 -> P = 32 -> 32, 24, 16,
// 8, 4; stella: 128 -> 128, 96, 64, 32, 16). Some of stella's nodes have SMT on (256
// hardware threads), so its Slurm jobs pass --max-threads 128 to get the same grid on
// every node. Values below 1 and duplicates (tiny machines) are dropped.
std::vector<double> thread_values(const Machine &m, size_t max_threads)
{
    int available = m.threads;
    if (int usable = usable_cpus(); usable > 0)
        available = std::min(available, usable);
    if (max_threads > 0)
        available = std::min(available, (int)max_threads);
    int p = 1;
    while (p * 2 <= available)
        p *= 2;
    std::vector<double> v;
    for (int t : {p, 3 * p / 4, p / 2, p / 4, p / 8})
        if (t >= 1 && std::find(v.begin(), v.end(), (double)t) == v.end())
            v.push_back(t);
    return v;
}

// ============================================================================
// Knob grid. Edit here to change what is explored.
// ============================================================================
//
// One column per knob, in this order. A row gets NaN for knobs its algorithm
// does not have. `threads` is used both to build the index and to answer queries.
// paa_segments is not a knob for MESSI/FreSh/Odyssey: their SIMD lower bound
// assumes exactly 16 segments.
const std::vector<std::string> KNOB_COLUMNS = {
    "threads",               // all: index-build and query threads
    "leaf_size",             // all: max series per leaf
    "sax_cardinality",       // MESSI, FreSh, Odyssey: bits per iSAX segment (1..8)
    "equidepth_breakpoints", // MESSI, FreSh: 1 = data-adaptive breakpoints (setNormalized(false))
    "word_length",           // SOFA: number of SFA coefficients
    "alphabet_size",         // SOFA: symbols per coefficient (power of 2)
    "histogram_type",        // SOFA: 1 = equi-depth bins, 2 = equi-width bins
    "is_norm",               // SOFA: 1 = skip the DC coefficient
    "sample_size",           // SOFA: series sampled to learn the bins
    "pq_th_div_factor",      // Odyssey: divides the leaves-per-work-queue estimate
};

using Grid = std::vector<std::pair<std::string, std::vector<double>>>; // knob -> values to try
using Config = std::map<std::string, double>;                          // knob -> chosen value

// `threads` comes from thread_values(): on the lab machine (48 hardware threads)
// it is {32, 24, 16, 8, 4}, on stella's CPU nodes (128) {128, 96, 64, 32, 16}.
// Limits: sax_cardinality is at most 8 bits (sax_type is one byte); SOFA's
// word_length is its number of root bits (2^word_length roots), so at most 16, and
// its alphabet_size a power of two up to 256 (log2 = bits per symbol).
Grid grid_for(const std::string &algo, const std::vector<double> &threads)
{
    if (algo == "messi" || algo == "fresh")
        return {
            {"threads", threads},
            {"leaf_size", {500, 1000, 2000, 5000, 10000}},
            {"sax_cardinality", {4, 6, 8}},
            {"equidepth_breakpoints", {0, 1}},
        };
    if (algo == "sofa")
        return {
            {"threads", threads},
            {"leaf_size", {5000, 10000, 20000}},
            {"word_length", {8, 12, 16}},
            {"alphabet_size", {8, 64, 256}},
            {"histogram_type", {1, 2}},
            {"is_norm", {0}},
            {"sample_size", {100000}},
        };
    if (algo == "odyssey")
        return {
            {"threads", threads},
            {"leaf_size", {500, 1000, 2000, 5000}}, // initial_lbl_size follows leaf_size
            {"sax_cardinality", {4, 6, 8}},
            {"pq_th_div_factor", {2, 4, 16, 64}},
        };
    if (algo == "sing")
        return {
            {"threads", threads}, // SING exposes no other knob
        };
    return {};
}

// Cartesian product of the grid values; the first knob varies slowest.
std::vector<Config> expand(const Grid &grid)
{
    std::vector<Config> configs = {Config{}};
    for (const auto &knob : grid)
    {
        std::vector<Config> next;
        for (const Config &c : configs)
            for (double v : knob.second)
            {
                Config extended = c;
                extended[knob.first] = v;
                next.push_back(extended);
            }
        configs = next;
    }
    return configs;
}

// ============================================================================
// Datasets
// ============================================================================

// Every dataset is DATA_ROOT/<name>/<base_file> and <query_file>, as
// scripts/datasets/prepare_datasets.py writes them (.fvecs: each row an int32 dimension,
// then the float32 values). The queries are never part of the base.
struct DatasetSpec
{
    std::string name;
    std::string base_file;
    std::string query_file;
};

// Run order: by base size (smallest first), then the data series.
const std::vector<DatasetSpec> DATASETS = {
    {"SIFT1M", "base.1M.fvecs", "query.10K.fvecs"},
    {"GLOVE1M", "base.1183514.fvecs", "query.10K.fvecs"},
    {"GIST1M", "base.1M.fvecs", "query.1K.fvecs"},
    {"HOTPOTQA5M", "base.5233329.fvecs", "query.7405.fvecs"},
    {"FEVER5M", "base.5416568.fvecs", "query.6666.fvecs"},
    {"MIRACLJA6M", "base.6953614.fvecs", "query.860.fvecs"},
    {"MSMARCO8M", "base.8841823.fvecs", "query.6980.fvecs"},
    {"DEEP10M", "base.10M.fvecs", "query.10K.fvecs"},
    {"DEEP50M", "base.50M.fvecs", "query.10K.fvecs"},
    {"DEEP100M", "base.100M.fvecs", "query.10K.fvecs"},
    {"MIRACLRU9M", "base.9543918.fvecs", "query.1252.fvecs"},
    {"UCASELAW7M", "base.7M.fvecs", "query.1K.fvecs"},
    {"T2I10M", "base.10M.fvecs", "query.10K.fvecs"},
    {"T2I50M", "base.50M.fvecs", "query.10K.fvecs"},
    {"T2I100M", "base.100M.fvecs", "query.10K.fvecs"},
    {"SEISMIC100M", "base.100M.fvecs", "query.10K.fvecs"},
    {"ASTRO100M", "base.100M.fvecs", "query.100.fvecs"},
    {"SALD100M", "base.100M.fvecs", "query.100.fvecs"},
    {"RANDOM100M", "base.100M.fvecs", "query.1K.fvecs"},
};

// Sizes and files of a dataset, known without reading it.
struct Shape
{
    size_t n = 0, dim = 0, nq = 0;
    std::string base_path, query_path;
};

Shape dataset_shape(const DatasetSpec &spec, const std::string &data_root, size_t n_queries, size_t max_base)
{
    Shape s;
    s.base_path = data_root + "/" + spec.name + "/" + spec.base_file;
    s.query_path = data_root + "/" + spec.name + "/" + spec.query_file;
    int32_t d = 0;
    FILE *f = std::fopen(s.base_path.c_str(), "rb");
    if (!f || std::fread(&d, 4, 1, f) != 1)
    {
        if (f)
            std::fclose(f);
        throw std::runtime_error("cannot read " + s.base_path);
    }
    std::fclose(f);
    if (!fs::exists(s.query_path))
        throw std::runtime_error("cannot read " + s.query_path);
    s.dim = (size_t)d;
    s.n = fs::file_size(s.base_path) / (4 + 4 * s.dim);
    s.nq = std::min(n_queries, (size_t)(fs::file_size(s.query_path) / (4 + 4 * s.dim)));
    if (max_base > 0)
        s.n = std::min(s.n, max_base);
    return s;
}

struct Dataset
{
    std::string name;
    size_t n = 0, dim = 0, nq = 0;
    std::unique_ptr<float[]> base;    // n x dim
    std::unique_ptr<float[]> queries; // nq x dim
    std::string raw_path;             // raw float32 copy of the base, written for Odyssey
};

// Reads the first `count` rows of an fvecs file (each row: int32 dim, dim floats).
void read_fvecs(const std::string &path, size_t count, size_t dim, float *out)
{
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f)
        throw std::runtime_error("cannot open " + path);
    for (size_t i = 0; i < count; i++)
    {
        int32_t d = 0;
        if (std::fread(&d, 4, 1, f) != 1 || (size_t)d != dim || std::fread(out + i * dim, 4, dim, f) != dim)
        {
            std::fclose(f);
            throw std::runtime_error("bad or short fvecs row " + std::to_string(i) + " in " + path);
        }
    }
    std::fclose(f);
}

Dataset load_dataset(const DatasetSpec &spec, const Shape &s)
{
    Dataset ds;
    ds.name = spec.name;
    ds.n = s.n;
    ds.dim = s.dim;
    ds.nq = s.nq;
    ds.queries.reset(new float[ds.nq * ds.dim]);
    ds.base.reset(new float[ds.n * ds.dim]);
    read_fvecs(s.query_path, ds.nq, ds.dim, ds.queries.get());
    read_fvecs(s.base_path, ds.n, ds.dim, ds.base.get());
    return ds;
}

// Odyssey builds from a raw float32 file: write the base once (a file of the right name and
// size in work_dir is reused).
void ensure_raw_file(Dataset &ds, const std::string &work_dir)
{
    ds.raw_path = work_dir + "/" + ds.name + "_n" + std::to_string(ds.n) + "_d" + std::to_string(ds.dim) + ".f32";
    const size_t bytes = ds.n * ds.dim * sizeof(float);
    if (fs::exists(ds.raw_path) && fs::file_size(ds.raw_path) == bytes)
        return;
    std::string tmp = ds.raw_path + ".tmp";
    FILE *f = std::fopen(tmp.c_str(), "wb");
    if (!f || std::fwrite(ds.base.get(), 1, bytes, f) != bytes || std::fclose(f) != 0)
        throw std::runtime_error("cannot write " + tmp);
    fs::rename(tmp, ds.raw_path);
}

// Memory the kernel could give us now (MemAvailable), falling back to physical RAM.
double mem_available_bytes()
{
    std::ifstream meminfo("/proc/meminfo");
    std::string key, unit;
    double kb = 0;
    while (meminfo >> key >> kb >> unit)
        if (key == "MemAvailable:")
            return kb * 1024.0;
    return (double)sysconf(_SC_PHYS_PAGES) * (double)sysconf(_SC_PAGE_SIZE);
}

constexpr double MEMORY_HEADROOM = 0.9; // use at most 90% of what is available
// Rough index size per series (SAX word, position, leaf-buffer pointers), used
// only for the fit check.
constexpr double INDEX_BYTES_PER_SERIES = 64;

double gb(double bytes) { return bytes / 1e9; }

// ============================================================================
// Algorithms
// ============================================================================

// Why an algorithm cannot run here, or "" if it can. SOFA, Odyssey and SING exist
// only when their optional dependencies were found; SING also needs a GPU.
std::string unavailable_reason(const std::string &algo, const Machine &m)
{
    if (algo == "messi" || algo == "fresh")
        return "";
    if (algo == "sofa")
#ifdef SOFA_FFTW_ENABLED
        return "";
#else
        return "not built (needs BUILD_SOFA and FFTW3)";
#endif
    if (algo == "odyssey")
#if defined(ODYSSEY_MPI) && ODYSSEY_MPI
        return "";
#else
        return "not built (needs BUILD_ODYSSEY and MPI)";
#endif
    if (algo == "sing")
#if SING_CUDA_ENABLED
        return m.gpus > 0 ? "" : "no GPU found";
#else
        return "not built (needs BUILD_SING and CUDA)";
#endif
    (void)m;
    return "unknown algorithm";
}

// Free GPU memory in bytes (0 without a CUDA build or a GPU).
double gpu_free_bytes()
{
#if SING_CUDA_ENABLED
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess)
        return (double)free_b;
#endif
    return 0;
}

// SING keeps each series' SAX word (16 bytes) and a 4-byte bitmap entry on the GPU.
constexpr double SING_GPU_BYTES_PER_SERIES = 20;

std::unique_ptr<daisy::SimilaritySearchAlgorithm> make_algorithm(const std::string &algo, const Config &c)
{
    auto knob = [&](const char *name)
    { return (int)c.at(name); };

    if (algo == "messi" || algo == "fresh")
    {
        std::unique_ptr<daisy::SimilaritySearchAlgorithm> a;
        auto configure = [&](auto *x)
        {
            x->setSearchWorkers(knob("threads"));
            x->setIndexWorkers(knob("threads"));
            x->setLeafSize(knob("leaf_size"));
            x->setInitialLblSize(knob("leaf_size")); // leaf buffers must hold a full leaf
            x->setSaxCardinality(knob("sax_cardinality"));
            if (knob("equidepth_breakpoints"))
                x->setNormalized(false);
        };
        if (algo == "messi")
        {
            auto *m = new daisy::Messi(daisy::DistanceType::L2_SQUARED);
            configure(m);
            a.reset(m);
        }
        else
        {
            auto *f = new daisy::Fresh(daisy::DistanceType::L2_SQUARED);
            configure(f);
            a.reset(f);
        }
        return a;
    }
#ifdef SOFA_FFTW_ENABLED
    if (algo == "sofa")
    {
        auto s = std::make_unique<daisy::Sofa>(daisy::DistanceType::L2_SQUARED);
        s->setSearchWorkers(knob("threads"));
        s->setIndexWorkers(knob("threads"));
        s->setLeafSize(knob("leaf_size"));
        s->setInitialLblSize(knob("leaf_size"));
        s->setWordLength(knob("word_length"));
        s->setAlphabetSize(knob("alphabet_size"));
        s->setHistogramType(knob("histogram_type"));
        s->setIsNorm(knob("is_norm") != 0);
        s->setSampleSize(knob("sample_size"));
        return s;
    }
#endif
#if defined(ODYSSEY_MPI) && ODYSSEY_MPI
    if (algo == "odyssey")
    {
        daisy::OdysseyConfig cfg;
        cfg.query_threads = knob("threads");
        cfg.search_workers = knob("threads");
        cfg.index_threads = knob("threads");
        cfg.leaf_size = knob("leaf_size");
        cfg.initial_lbl_size = knob("leaf_size"); // leaf buffers must hold a full leaf
        cfg.sax_cardinality = knob("sax_cardinality");
        cfg.pq_th_div_factor = knob("pq_th_div_factor");
        return std::make_unique<daisy::Odyssey>(cfg, daisy::DistanceType::L2_SQUARED, 0, nullptr);
    }
#endif
#if SING_CUDA_ENABLED
    if (algo == "sing")
    {
        auto g = std::make_unique<daisy::Sing>(daisy::DistanceType::L2_SQUARED);
        g->setNumThreads(knob("threads")); // search threads
        g->setIndexWorkers(knob("threads"));
        return g;
    }
#endif
    throw std::runtime_error(algo + " is not available in this build");
}

// ============================================================================
// Tree shape of a built index
// ============================================================================

struct TreeShape
{
    size_t roots = 0, leaves = 0, series = 0, max_fill = 0, depth = 0;
};

void walk_tree(const daisy::isax_node *node, size_t depth, TreeShape &t)
{
    if (node == nullptr)
        return;
    if (node->is_leaf)
    {
        size_t fill = 0;
        if (node->buffer != nullptr)
            fill = (size_t)node->buffer->partial_buffer_size + (size_t)node->buffer->full_buffer_size +
                   (size_t)node->buffer->tmp_full_buffer_size + (size_t)node->buffer->tmp_partial_buffer_size;
        t.leaves++;
        t.series += fill;
        t.max_fill = std::max(t.max_fill, fill);
        t.depth = std::max(t.depth, depth);
        return;
    }
    walk_tree(node->left_child, depth + 1, t);
    walk_tree(node->right_child, depth + 1, t);
}

// Walks the iSAX/SFA tree: MESSI, FreSh and SOFA chain their roots through
// first_node; Odyssey lists them through its first buffer layer.
TreeShape tree_shape(const std::string &algo, daisy::SimilaritySearchAlgorithm &a)
{
    TreeShape t;
    daisy::isax_index *index = a.getIndex();
    if (index == nullptr)
        return t;
#if defined(ODYSSEY_MPI) && ODYSSEY_MPI
    if (algo == "odyssey")
    {
        daisy::NodeList roots = daisy::initialize_node_list(index, 0);
        for (int i = 0; i < roots.node_amount; i++)
        {
            t.roots++;
            walk_tree(roots.nlist[i], 0, t);
        }
        std::free(roots.nlist);
        return t;
    }
#endif
    (void)algo;
    for (const daisy::isax_node *r = index->first_node; r != nullptr; r = r->next)
    {
        t.roots++;
        walk_tree(r, 0, t);
    }
    return t;
}

// The tree-shape columns, each preceded by a comma.
std::string fmt_tree(const TreeShape &t)
{
    char buf[160];
    const double mean = t.leaves ? (double)t.series / (double)t.leaves : 0.0;
    const double share = t.series ? (double)t.max_fill / (double)t.series : 0.0;
    std::snprintf(buf, sizeof(buf), ",%zu,%zu,%.4f,%zu,%.6f,%zu", t.roots, t.leaves, mean, t.max_fill, share, t.depth);
    return buf;
}

// ============================================================================
// CSV layout
// ============================================================================

const std::vector<std::string> TREE_COLUMNS = {
    "n_roots",
    "n_leaves",
    "leaf_fill_mean",
    "leaf_fill_max",
    "largest_leaf_share",
    "tree_depth",
};

const std::vector<std::string> MEASURE_COLUMNS = {
    "node_lb_calcs",
    "leaves_enqueued",
    "leaves_visited",
    "series_lb_calcs",
    "distance_calcs",
    "seed_kth_dist",
    "final_kth_dist",
};

std::string csv_header()
{
    std::string h;
    for (const auto &name : MACHINE_COLUMNS)
        h += name + ",";
    h += "algorithm,dataset_name,dataset_size,dataset_dimensionality,k,query_id";
    for (const auto &name : KNOB_COLUMNS)
        h += "," + name;
    h += ",latency_ms,build_time_ms";
    for (const auto &name : TREE_COLUMNS)
        h += "," + name;
    for (const auto &name : MEASURE_COLUMNS)
        h += "," + name;
    return h;
}

std::string fmt_dist(float d)
{
    if (!(d < FLT_MAX))
        return "inf";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.9g", d);
    return buf;
}

// The measure columns of one answered query, each preceded by a comma.
std::string fmt_measures(const std::vector<daisy::SearchStats> &stats, float final_kth_dist)
{
    std::string out;
    if (stats.size() == 1)
    {
        const daisy::SearchStats &s = stats[0];
        char buf[128];
        std::snprintf(buf, sizeof(buf), ",%llu,%llu,%llu,%llu,%llu,", (unsigned long long)s.node_lb_calcs,
                      (unsigned long long)s.leaves_enqueued, (unsigned long long)s.leaves_visited,
                      (unsigned long long)s.series_lb_calcs, (unsigned long long)s.distance_calcs);
        out = buf + fmt_dist(s.seed_kth_dist);
    }
    else
        out = ",NaN,NaN,NaN,NaN,NaN,NaN"; // algorithm does not collect search statistics
    return out + "," + fmt_dist(final_kth_dist);
}

std::string fmt_knob(const Config &c, const std::string &name)
{
    auto it = c.find(name);
    if (it == c.end())
        return "NaN";
    std::ostringstream os;
    os << std::setprecision(12) << it->second;
    return os.str();
}

// The knob columns of a configuration, each preceded by a comma.
std::string fmt_knobs(const Config &c)
{
    std::string knobs;
    for (const auto &name : KNOB_COLUMNS)
        knobs += "," + fmt_knob(c, name);
    return knobs;
}

// "threads=32 leaf_size=500 ..." for the progress log (only the knobs the configuration has).
std::string describe(const Config &c)
{
    std::string out;
    for (const auto &name : KNOB_COLUMNS)
        if (c.count(name))
            out += " " + name + "=" + fmt_knob(c, name);
    return out;
}

std::vector<std::string> split(const std::string &s)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty())
            out.push_back(item);
    return out;
}

// Appends a block of rows in one write and syncs it to disk.
// Set by SIGTERM/SIGINT (Slurm sends SIGTERM at the time limit, SIGKILL some seconds
// later). The handler only sets the flag, so a write in progress always completes, and
// the run stops before the next configuration. Every configuration in the CSV is then
// complete, which --resume relies on.
volatile std::sig_atomic_t stop_requested = 0;
extern "C" void request_stop(int) { stop_requested = 1; }

void install_stop_handler()
{
    struct sigaction sa = {};
    sa.sa_handler = request_stop;
    sa.sa_flags = SA_RESTART; // interrupted writes continue
    sigemptyset(&sa.sa_mask);
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT, &sa, nullptr);
}

// The key of a configuration in the CSV: algorithm, dataset_name, dataset_size and the
// knob columns, formatted as the rows have them.
std::string config_key(const std::string &algo, const std::string &dataset, size_t n, const std::string &knob_cols)
{
    return algo + "," + dataset + "," + std::to_string(n) + knob_cols;
}

// The keys of the configurations that already have rows in an existing CSV. A
// configuration's rows are appended in one write after it finishes, so one with rows
// there is complete. Lines of another field count (a cut last line) are ignored.
std::set<std::string> done_configurations(const std::string &csv, const std::string &header)
{
    std::set<std::string> done;
    std::ifstream in(csv);
    std::string line;
    if (!std::getline(in, line) || line != header)
        return done;
    auto fields_of = [](const std::string &l)
    {
        std::vector<std::string> f;
        size_t start = 0;
        for (size_t i = 0; i <= l.size(); i++)
            if (i == l.size() || l[i] == ',')
            {
                f.push_back(l.substr(start, i - start));
                start = i + 1;
            }
        return f;
    };
    const std::vector<std::string> names = fields_of(header);
    auto column = [&](const std::string &name)
    { return (size_t)(std::find(names.begin(), names.end(), name) - names.begin()); };
    const size_t algo_col = column("algorithm"), name_col = column("dataset_name"), size_col = column("dataset_size");
    std::vector<size_t> knob_cols;
    for (const auto &knob : KNOB_COLUMNS)
        knob_cols.push_back(column(knob));
    size_t cut = 0;
    while (std::getline(in, line))
    {
        const std::vector<std::string> f = fields_of(line);
        if (f.size() != names.size())
        {
            cut++;
            continue;
        }
        std::string knobs;
        for (size_t c : knob_cols)
            knobs += "," + f[c];
        done.insert(f[algo_col] + "," + f[name_col] + "," + f[size_col] + knobs);
    }
    if (cut)
        std::printf("resume: %zu lines of %s have another field count and are ignored\n", cut, csv.c_str());
    return done;
}

void append_rows(const std::string &csv, const std::string &rows)
{
    FILE *f = std::fopen(csv.c_str(), "a");
    if (!f)
        throw std::runtime_error("cannot open " + csv);
    const bool ok = std::fwrite(rows.data(), 1, rows.size(), f) == rows.size() && std::fflush(f) == 0 &&
                    fsync(fileno(f)) == 0;
    if (std::fclose(f) != 0 || !ok)
        throw std::runtime_error("cannot write " + csv);
}

// ============================================================================
// One configuration: build once, then time every (k, query) pair
// ============================================================================

struct ConfigResult
{
    double build_ms = 0;
    size_t rows = 0;
    std::string notes; // queries skipped by the budget, per k
};

// Builds the index, records its tree shape, answers every query one at a time for
// every k and appends all rows at the end.
ConfigResult run_configuration(const std::string &machine_cols, const std::string &algo, const Config &c,
                               const Dataset &ds, const std::vector<int> &ks, double budget_ms,
                               const std::string &out_csv)
{
    ConfigResult res;
    auto index = make_algorithm(algo, c);
    auto b0 = Clock::now();
    if (algo == "odyssey")
        index->buildIndex(ds.raw_path, ds.dim, ds.n); // builds from the raw file
    else
        index->buildIndex(ds.base.get(), ds.n, ds.dim);
    res.build_ms = std::chrono::duration<double, std::milli>(Clock::now() - b0).count();

    char build_col[32];
    std::snprintf(build_col, sizeof(build_col), ",%.3f", res.build_ms);
    const std::string config_cols = fmt_knobs(c);
    const std::string build_cols = build_col + fmt_tree(tree_shape(algo, *index));

    std::string rows;
    char prefix[256], latency[32];
    for (int k : ks)
    {
        std::vector<idx_t> I(k);
        std::vector<float> D(k);

        double spent_ms = 0;
        size_t answered = 0;
        for (size_t q = 0; q < ds.nq; q++)
        {
            if (budget_ms > 0 && q > 0 && spent_ms >= budget_ms)
                break;
            auto t0 = Clock::now();
            index->searchIndex(ds.queries.get() + q * ds.dim, 1, k, I.data(), D.data());
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            spent_ms += ms;
            answered++;
            std::snprintf(prefix, sizeof(prefix), "%s,%s,%zu,%zu,%d,%zu", algo.c_str(), ds.name.c_str(), ds.n,
                          ds.dim, k, q);
            std::snprintf(latency, sizeof(latency), ",%.6f", ms);
            rows += machine_cols + prefix + config_cols + latency + build_cols +
                    fmt_measures(index->getLastSearchStats(), D[k - 1]) + "\n";
        }
        res.rows += answered;
        if (answered < ds.nq)
            res.notes += " k=" + std::to_string(k) + " stopped after " + std::to_string(answered) + "/" +
                         std::to_string(ds.nq) + " queries (budget)";
    }
    append_rows(out_csv, rows);
    return res;
}

// ============================================================================
// Main
// ============================================================================

struct Options
{
    std::string datasets = "all", algos = "messi,fresh,sofa,odyssey,sing", ks = "1,10,100";
    std::string out = "optimizer_samples.csv", work_dir = "optimizer_work";
    // Where prepare_datasets.py put the datasets: $DAISY_DATA_ROOT, else the lab server's folder.
    std::string data_root = std::getenv("DAISY_DATA_ROOT") ? std::getenv("DAISY_DATA_ROOT")
                                                           : "/mnt/hddhelp/mchatzakis/datasets/processed";
    size_t queries = 100, max_base = 0;
    size_t max_threads = 0; // cap on the largest thread value (0: the CPUs available)
    double budget_s = 600;
    bool dry_run = false;
    bool resume = false; // skip the configurations that already have rows in --out
};

int parse_options(int argc, char **argv, Options &o)
{
    if (argc % 2 == 0)
    {
        std::fprintf(stderr, "every option needs a value (see docs/optimizer-notes/collector.md)\n");
        return 2;
    }
    try
    {
        for (int i = 1; i + 1 < argc; i += 2)
        {
            std::string key = argv[i], val = argv[i + 1];
            if (key == "--datasets")
                o.datasets = val;
            else if (key == "--algos")
                o.algos = val;
            else if (key == "--k")
                o.ks = val;
            else if (key == "--queries")
                o.queries = std::stoul(val);
            else if (key == "--max-base")
                o.max_base = std::stoul(val);
            else if (key == "--max-threads")
                o.max_threads = std::stoul(val);
            else if (key == "--budget-s")
                o.budget_s = std::stod(val);
            else if (key == "--out")
                o.out = val;
            else if (key == "--work-dir")
                o.work_dir = val;
            else if (key == "--data-root")
                o.data_root = val;
            else if (key == "--dry-run")
                o.dry_run = val != "0";
            else if (key == "--resume")
                o.resume = val != "0";
            else
            {
                std::fprintf(stderr, "unknown option %s (see docs/optimizer-notes/collector.md)\n", key.c_str());
                return 2;
            }
        }
    }
    catch (const std::exception &)
    {
        std::fprintf(stderr, "bad option value (see docs/optimizer-notes/collector.md)\n");
        return 2;
    }
    if (o.queries == 0)
    {
        std::fprintf(stderr, "--queries must be at least 1\n");
        return 2;
    }
    return 0;
}

// Creates the CSV with its header, or checks that an existing one has the same header.
int prepare_csv(const std::string &out)
{
    const std::string header = csv_header();
    if (!fs::exists(out) || fs::file_size(out) == 0)
    {
        FILE *f = std::fopen(out.c_str(), "w");
        if (!f)
        {
            std::fprintf(stderr, "cannot create %s\n", out.c_str());
            return 2;
        }
        std::fprintf(f, "%s\n", header.c_str());
        std::fclose(f);
        return 0;
    }
    std::ifstream existing(out);
    std::string first_line;
    std::getline(existing, first_line);
    existing.close();
    if (first_line != header)
    {
        std::fprintf(stderr,
                     "%s has a different CSV header (written by another version of the collector?).\n"
                     "Refusing to append; pass a new --out file.\n",
                     out.c_str());
        return 2;
    }
    return 0;
}

int main(int argc, char **argv)
{
    Options opt;
    if (int rc = parse_options(argc, argv, opt))
        return rc;
    std::vector<int> ks;
    for (const auto &k : split(opt.ks))
        ks.push_back(std::stoi(k));
    if (ks.empty())
    {
        std::fprintf(stderr, "--k needs at least one value\n");
        return 2;
    }

    const Machine machine = detect_machine();
    const std::string machine_cols = fmt_machine(machine);
    const std::vector<double> threads = thread_values(machine, opt.max_threads);
    std::printf("machine: %s\n", machine_cols.substr(0, machine_cols.size() - 1).c_str());
    std::string thread_list;
    for (double t : threads)
        thread_list += (thread_list.empty() ? "" : ",") + std::to_string((int)t);
    std::printf("threads grid: %s\n", thread_list.c_str());

    if (opt.dry_run)
        std::printf("dry run: nothing is built or written\n");
    else
    {
        if (int rc = prepare_csv(opt.out))
            return rc;
        fs::create_directories(opt.work_dir);
        install_stop_handler();
    }
    std::set<std::string> done;
    if (opt.resume && fs::exists(opt.out))
    {
        done = done_configurations(opt.out, csv_header());
        std::printf("resume: %zu configurations already in %s are skipped\n", done.size(), opt.out.c_str());
    }

    size_t total_configs = 0, total_rows = 0;
    for (const DatasetSpec &spec : DATASETS)
    {
        if (stop_requested)
            break;
        if (opt.datasets != "all" && ("," + opt.datasets + ",").find("," + spec.name + ",") == std::string::npos)
            continue;
        std::printf("== %s\n", spec.name.c_str());
        std::fflush(stdout);

        Shape shape;
        try
        {
            shape = dataset_shape(spec, opt.data_root, opt.queries, opt.max_base);
        }
        catch (const std::exception &e)
        {
            std::printf("   skipped: %s\n", e.what());
            continue;
        }
        const double data_bytes = (double)(shape.n + shape.nq) * shape.dim * sizeof(float);
        const double need_bytes = data_bytes + INDEX_BYTES_PER_SERIES * shape.n;
        std::printf("   n=%zu dim=%zu queries=%zu, base %.1f GB, available memory %.1f GB\n", shape.n, shape.dim,
                    shape.nq, gb(data_bytes), gb(mem_available_bytes()));

        // The grid configurations of every available algorithm.
        std::vector<std::pair<std::string, std::vector<Config>>> todo;
        for (const std::string &algo : split(opt.algos))
        {
            const std::string reason = unavailable_reason(algo, machine);
            if (!reason.empty())
            {
                std::printf("   %s: %s, skipped\n", algo.c_str(), reason.c_str());
                continue;
            }
            std::vector<Config> configs, all = expand(grid_for(algo, threads));
            for (const Config &c : all)
                if (!done.count(config_key(algo, spec.name, shape.n, fmt_knobs(c))))
                    configs.push_back(c);
            if (configs.size() == all.size())
                std::printf("   %s: %zu configurations\n", algo.c_str(), configs.size());
            else
                std::printf("   %s: %zu configurations, %zu of them already done\n", algo.c_str(), all.size(),
                            all.size() - configs.size());
            if (!configs.empty())
                todo.emplace_back(algo, configs);
        }
        if (todo.empty())
        {
            std::printf("   nothing to run\n");
            continue;
        }
        if (need_bytes > MEMORY_HEADROOM * mem_available_bytes())
        {
            std::printf("   skipped: needs about %.1f GB, %.1f GB available (use --max-base)\n", gb(need_bytes),
                        gb(mem_available_bytes()));
            continue;
        }
        if (opt.dry_run)
        {
            for (const auto &t : todo)
            {
                total_configs += t.second.size();
                total_rows += t.second.size() * ks.size() * shape.nq;
            }
            continue;
        }

        Dataset ds;
        try
        {
            std::printf("   loading\n");
            std::fflush(stdout);
            ds = load_dataset(spec, shape);
        }
        catch (const std::exception &e)
        {
            std::printf("   skipped: %s\n", e.what());
            continue;
        }

        for (const auto &t : todo)
        {
            const std::string &algo = t.first;
            const std::vector<Config> &configs = t.second;
            if (algo == "odyssey")
            {
                // Odyssey reads its own copy of the base into RAM, next to ours.
                const double copy_bytes = (double)ds.n * ds.dim * sizeof(float) + INDEX_BYTES_PER_SERIES * ds.n;
                if (copy_bytes > MEMORY_HEADROOM * mem_available_bytes())
                {
                    std::printf("   odyssey: skipped, its copy of the base needs about %.1f GB, %.1f GB available "
                                "(use --max-base)\n",
                                gb(copy_bytes), gb(mem_available_bytes()));
                    continue;
                }
                try
                {
                    ensure_raw_file(ds, opt.work_dir);
                }
                catch (const std::exception &e)
                {
                    std::printf("   odyssey: skipped: %s\n", e.what());
                    continue;
                }
            }
            if (algo == "sing")
            {
                // SING does not check its GPU allocations, so make sure they fit.
                const double gpu_bytes = SING_GPU_BYTES_PER_SERIES * ds.n;
                if (gpu_bytes > MEMORY_HEADROOM * gpu_free_bytes())
                {
                    std::printf("   sing: skipped, needs about %.1f GB of GPU memory, %.1f GB free (use --max-base)\n",
                                gb(gpu_bytes), gb(gpu_free_bytes()));
                    continue;
                }
            }
            for (size_t i = 0; i < configs.size() && !stop_requested; i++)
            {
                auto t0 = Clock::now();
                std::string status = "ok";
                ConfigResult res;
                try
                {
                    res = run_configuration(machine_cols, algo, configs[i], ds, ks, opt.budget_s * 1000.0, opt.out);
                }
                catch (const std::exception &e)
                {
                    status = std::string("error (nothing written): ") + e.what();
                }
                const double s = std::chrono::duration<double>(Clock::now() - t0).count();
                std::printf("   %s %zu/%zu [%s ]: %s (build %.1f s, total %.1f s, %zu rows)%s\n", algo.c_str(), i + 1,
                            configs.size(), describe(configs[i]).c_str(), status.c_str(), res.build_ms / 1000.0, s,
                            res.rows, res.notes.c_str());
                std::fflush(stdout);
            }
        }
    }
    if (opt.dry_run)
        std::printf("dry run: %zu configurations (%zu rows at most) to run\n", total_configs,
                    total_rows);
    else if (stop_requested)
        std::printf("stopped by a signal; the configurations written to %s are complete (rerun with --resume 1)\n",
                    opt.out.c_str());
    else
        std::printf("samples written to %s\n", opt.out.c_str());
#if defined(ODYSSEY_MPI) && ODYSSEY_MPI
    int initialized = 0, finalized = 0; // Odyssey initialized MPI when it was first constructed
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (initialized && !finalized)
        MPI_Finalize();
#endif
    return 0;
}
