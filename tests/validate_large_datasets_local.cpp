// validate_large_datasets_local.cpp
//
// Exactness validation of DaiSy's in-memory exact similarity-search algorithms on
// the large datasets stored on the lab machine. It is part of the test suite
// (CTest name validate_large_datasets_local, label large_local) and is also the
// starting point of the knob optimizer: before tuning any parameter we need to
// know that every algorithm returns exact answers on the datasets we care about.
//
// Under CTest:  ctest -L large_local          (only this test; it can take hours)
//               ctest -LE large_local         (the rest of the suite)
// Extra arguments come from the CMake cache variable DAISY_LARGE_DATASETS_ARGS.
//
// What it does, per dataset:
//   1. Loads the base (optionally only the first --max-base records) and up to
//      --queries queries into RAM.
//   2. Builds a reference answer for k_max = max(--k):
//        - the dataset's L2 ground-truth file, when it exists, covers the full
//          base, and passes a brute-force spot check (--gt-verify queries); or
//        - an exact brute-force scan (float32 candidate scan, double-precision
//          re-rank), cached under <work-dir>/reference_cache.
//   3. Runs every selected algorithm in its own forked child process (crashes,
//      exit() calls, leaks and hangs inside the library cannot take down the
//      validator), builds the index once and searches all queries for each k.
//   4. Recomputes the exact squared L2 distance of every returned id and checks:
//        - exactness: every returned id is within the reference k-th distance
//          (rtol/atol), i.e. ties are accepted but missing neighbors are not;
//        - id recall against the reference ids (informative, tie-sensitive);
//        - invalid ids, unwritten output slots, duplicate ids (padding);
//        - reported distances match the true squared L2 distances;
//        - reported distances are sorted ascending;
//        - the library did not modify the caller's base/query buffers.
//   5. Prints a per-run report, appends rows to a CSV file and prints a summary.
//
// Exit status: 0 when every executed run is PASS (SKIPs allowed), 1 otherwise,
// 77 (reported by CTest as skipped) when nothing could run because none of the
// selected datasets/algorithms is available on this machine.
//
// Datasets (default roots can be changed with --vector-root / --series-root):
//   vectors (fvecs):  SIFT1M GIST1M T2I10M HOTPOTQA5M UCASELAW7M MSMARCOMXBAI8M DEEP100M
//   data series (raw float32, len 256, z-normalized): Seismic Astro
// Data-series datasets ship without queries: the last --queries records of the
// file are held out as queries and the base is taken from the start of the file.
//
// Algorithms (the in-memory methods): MESSI, FreSh, SOFA (needs BUILD_SOFA + FFTW3),
// Odyssey (needs BUILD_ODYSSEY + MPI; runs single node as an MPI singleton).
// Do not launch this binary under mpirun.
//
// Run with --help for all options.

#include "algos/Messi.hpp"
#include "algos/Fresh.hpp"
#include "algos/Sofa.hpp" // compiles to nothing unless SOFA_FFTW_ENABLED

#if defined(ODYSSEY_MPI) && ODYSSEY_MPI
#define VALIDATOR_HAS_ODYSSEY 1
#include "algos/hodyssey/Odyssey.hpp"
#include <mpi.h>
#else
#define VALIDATOR_HAS_ODYSSEY 0
#endif

#ifdef SOFA_FFTW_ENABLED
#define VALIDATOR_HAS_SOFA 1
#else
#define VALIDATOR_HAS_SOFA 0
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace fs = std::filesystem;
using daisy::idx_t;

namespace
{

// ============================================================================
// Small utilities
// ============================================================================

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0)
{
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

std::string timestamp()
{
    std::time_t now = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return buf;
}

std::string human_bytes(double bytes)
{
    const char *units[] = {"B", "KB", "MB", "GB", "TB"};
    int u = 0;
    while (bytes >= 1024.0 && u < 4)
    {
        bytes /= 1024.0;
        u++;
    }
    std::ostringstream os;
    os << std::fixed << std::setprecision(u == 0 ? 0 : 1) << bytes << " " << units[u];
    return os.str();
}

std::string fmt(double v, int prec)
{
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << v;
    return os.str();
}

std::string fmt_sci(double v)
{
    std::ostringstream os;
    os << std::scientific << std::setprecision(2) << v;
    return os.str();
}

std::string to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

std::vector<std::string> split(const std::string &s, char sep)
{
    std::vector<std::string> out;
    std::string cur;
    std::istringstream is(s);
    while (std::getline(is, cur, sep))
    {
        if (!cur.empty())
            out.push_back(cur);
    }
    return out;
}

[[noreturn]] void die(const std::string &msg)
{
    std::fprintf(stderr, "[validator] error: %s\n", msg.c_str());
    std::exit(2);
}

void log_line(const std::string &msg)
{
    std::printf("[%s] %s\n", timestamp().c_str(), msg.c_str());
    std::fflush(stdout);
}

// Strict integer/float parsing for CLI values and knob overrides.
long long parse_int(const std::string &what, const std::string &v)
{
    try
    {
        size_t pos = 0;
        long long x = std::stoll(v, &pos);
        if (pos != v.size())
            throw std::invalid_argument("trailing characters");
        return x;
    }
    catch (const std::exception &)
    {
        throw std::invalid_argument("'" + what + "' expects an integer, got '" + v + "'");
    }
}

double parse_double(const std::string &what, const std::string &v)
{
    try
    {
        size_t pos = 0;
        double x = std::stod(v, &pos);
        if (pos != v.size())
            throw std::invalid_argument("trailing characters");
        return x;
    }
    catch (const std::exception &)
    {
        throw std::invalid_argument("'" + what + "' expects a number, got '" + v + "'");
    }
}

bool parse_bool(const std::string &what, const std::string &v)
{
    std::string l = to_lower(v);
    if (l == "1" || l == "true" || l == "yes" || l == "on")
        return true;
    if (l == "0" || l == "false" || l == "no" || l == "off")
        return false;
    throw std::invalid_argument("'" + what + "' expects a boolean, got '" + v + "'");
}

int parse_knob_int(const std::string &key, const std::string &v)
{
    long long x = parse_int(key, v);
    if (x < INT_MIN || x > INT_MAX)
        throw std::invalid_argument("'" + key + "' is out of int range");
    return (int)x;
}

size_t mem_available_bytes()
{
    std::ifstream in("/proc/meminfo");
    std::string key;
    size_t value = 0;
    std::string unit;
    while (in >> key >> value >> unit)
    {
        if (key == "MemAvailable:")
            return value * 1024;
    }
    return 0;
}

uint64_t fnv1a(const void *data, size_t bytes, uint64_t h = 1469598103934665603ULL)
{
    const unsigned char *p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < bytes; i++)
    {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

std::string hex64(uint64_t h)
{
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016" PRIx64, h);
    return buf;
}

// ============================================================================
// Aligned row-major float matrix
// ============================================================================

class FloatMatrix
{
public:
    FloatMatrix() = default;
    FloatMatrix(size_t rows, size_t cols) { allocate(rows, cols); }
    ~FloatMatrix() { std::free(data_); }

    FloatMatrix(const FloatMatrix &) = delete;
    FloatMatrix &operator=(const FloatMatrix &) = delete;
    FloatMatrix(FloatMatrix &&o) noexcept : rows_(o.rows_), cols_(o.cols_), data_(o.data_)
    {
        o.data_ = nullptr;
        o.rows_ = o.cols_ = 0;
    }
    FloatMatrix &operator=(FloatMatrix &&o) noexcept
    {
        if (this != &o)
        {
            std::free(data_);
            rows_ = o.rows_;
            cols_ = o.cols_;
            data_ = o.data_;
            o.data_ = nullptr;
            o.rows_ = o.cols_ = 0;
        }
        return *this;
    }

    void allocate(size_t rows, size_t cols)
    {
        std::free(data_);
        data_ = nullptr;
        rows_ = rows;
        cols_ = cols;
        size_t bytes = rows * cols * sizeof(float);
        size_t padded = std::max<size_t>(64, (bytes + 63) / 64 * 64);
        data_ = static_cast<float *>(std::aligned_alloc(64, padded));
        if (data_ == nullptr)
            throw std::runtime_error("failed to allocate " + human_bytes((double)padded));
    }

    void release()
    {
        std::free(data_);
        data_ = nullptr;
        rows_ = cols_ = 0;
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t bytes() const { return rows_ * cols_ * sizeof(float); }
    float *data() { return data_; }
    const float *data() const { return data_; }
    float *row(size_t i) { return data_ + i * cols_; }
    const float *row(size_t i) const { return data_ + i * cols_; }

private:
    size_t rows_ = 0;
    size_t cols_ = 0;
    float *data_ = nullptr;
};

// ============================================================================
// File reading
// ============================================================================

class File
{
public:
    File(const std::string &path, const char *mode) : path_(path)
    {
        f_ = std::fopen(path.c_str(), mode);
        if (f_ == nullptr)
            throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
    }
    ~File()
    {
        if (f_)
            std::fclose(f_);
    }
    File(const File &) = delete;
    File &operator=(const File &) = delete;

    void seek(size_t offset)
    {
        if (fseeko(f_, (off_t)offset, SEEK_SET) != 0)
            throw std::runtime_error("seek failed in " + path_);
    }
    void read_exact(void *dst, size_t bytes)
    {
        size_t got = std::fread(dst, 1, bytes, f_);
        if (got != bytes)
            throw std::runtime_error("short read in " + path_ + " (wanted " + std::to_string(bytes) +
                                     " bytes, got " + std::to_string(got) + ")");
    }
    void write_exact(const void *src, size_t bytes)
    {
        if (std::fwrite(src, 1, bytes, f_) != bytes)
            throw std::runtime_error("short write in " + path_ + ": " + std::strerror(errno));
    }
    void close()
    {
        if (f_ && std::fclose(f_) != 0)
        {
            f_ = nullptr;
            throw std::runtime_error("close failed for " + path_ + ": " + std::strerror(errno));
        }
        f_ = nullptr;
    }

private:
    std::string path_;
    FILE *f_ = nullptr;
};

// Prints throughput at most every 10 s, only for transfers larger than 1 GB.
class Progress
{
public:
    Progress(std::string what, size_t total_bytes)
        : what_(std::move(what)), total_(total_bytes), t0_(Clock::now()), last_(t0_) {}

    void update(size_t done_bytes)
    {
        if (total_ < (1ULL << 30))
            return;
        auto now = Clock::now();
        if (std::chrono::duration<double>(now - last_).count() < 10.0 && done_bytes < total_)
            return;
        last_ = now;
        double secs = std::max(1e-9, seconds_since(t0_));
        log_line("  " + what_ + ": " + fmt(100.0 * (double)done_bytes / (double)total_, 1) + "% (" +
                 human_bytes((double)done_bytes) + " / " + human_bytes((double)total_) + ", " +
                 human_bytes((double)done_bytes / secs) + "/s)");
    }

private:
    std::string what_;
    size_t total_;
    Clock::time_point t0_, last_;
};

size_t file_size_or_zero(const std::string &path)
{
    std::error_code ec;
    auto sz = fs::file_size(path, ec);
    return ec ? 0 : (size_t)sz;
}

// fvecs: every row is int32 dim followed by dim float32 values.
size_t fvecs_dim(const std::string &path)
{
    File f(path, "rb");
    int32_t d = 0;
    f.read_exact(&d, sizeof(d));
    if (d <= 0 || d > (1 << 20))
        throw std::runtime_error("unreasonable fvecs dimension " + std::to_string(d) + " in " + path);
    return (size_t)d;
}

size_t fvecs_rows(const std::string &path, size_t dim)
{
    size_t sz = file_size_or_zero(path);
    size_t row = sizeof(int32_t) + dim * sizeof(float);
    if (sz % row != 0)
        throw std::runtime_error(path + ": size " + std::to_string(sz) + " is not a multiple of the row size " +
                                 std::to_string(row));
    return sz / row;
}

void read_fvecs_rows(const std::string &path, size_t dim, size_t first, size_t count, float *out,
                     const std::string &what)
{
    File f(path, "rb");
    const size_t row_bytes = sizeof(int32_t) + dim * sizeof(float);
    f.seek(first * row_bytes);
    const size_t chunk_rows = std::max<size_t>(1, (64ULL << 20) / row_bytes);
    std::vector<char> buf(std::min(count, chunk_rows) * row_bytes);
    Progress progress(what, count * row_bytes);
    size_t done = 0;
    while (done < count)
    {
        size_t n = std::min(chunk_rows, count - done);
        f.read_exact(buf.data(), n * row_bytes);
        for (size_t r = 0; r < n; r++)
        {
            const char *src = buf.data() + r * row_bytes;
            int32_t hdr;
            std::memcpy(&hdr, src, sizeof(hdr));
            if (hdr != (int32_t)dim)
                throw std::runtime_error(path + ": row " + std::to_string(first + done + r) + " has dimension " +
                                         std::to_string(hdr) + ", expected " + std::to_string(dim));
            std::memcpy(out + (done + r) * dim, src + sizeof(int32_t), dim * sizeof(float));
        }
        done += n;
        progress.update(done * row_bytes);
    }
}

void read_raw_rows(const std::string &path, size_t dim, size_t first, size_t count, float *out,
                   const std::string &what)
{
    File f(path, "rb");
    const size_t row_bytes = dim * sizeof(float);
    f.seek(first * row_bytes);
    const size_t chunk_rows = std::max<size_t>(1, (64ULL << 20) / row_bytes);
    Progress progress(what, count * row_bytes);
    size_t done = 0;
    while (done < count)
    {
        size_t n = std::min(chunk_rows, count - done);
        f.read_exact(out + done * dim, n * row_bytes);
        done += n;
        progress.update(done * row_bytes);
    }
}

// ivecs ground truth: every row is int32 k followed by k int32 ids.
// Returns the first `keep` ids of the first `rows` rows.
std::vector<int32_t> read_ivecs_prefix(const std::string &path, size_t rows, size_t keep, size_t &row_len_out)
{
    File f(path, "rb");
    int32_t k = 0;
    f.read_exact(&k, sizeof(k));
    if (k <= 0)
        throw std::runtime_error(path + ": invalid row length " + std::to_string(k));
    row_len_out = (size_t)k;
    if ((size_t)k < keep)
        throw std::runtime_error(path + ": rows have " + std::to_string(k) + " ids, need " + std::to_string(keep));
    const size_t row_bytes = sizeof(int32_t) * (1 + (size_t)k);
    size_t available = file_size_or_zero(path) / row_bytes;
    if (available < rows)
        throw std::runtime_error(path + ": has " + std::to_string(available) + " rows, need " + std::to_string(rows));
    f.seek(0);
    std::vector<int32_t> row(1 + (size_t)k);
    std::vector<int32_t> out(rows * keep);
    for (size_t r = 0; r < rows; r++)
    {
        f.read_exact(row.data(), row_bytes);
        if (row[0] != k)
            throw std::runtime_error(path + ": row " + std::to_string(r) + " has length " + std::to_string(row[0]));
        std::copy(row.begin() + 1, row.begin() + 1 + (long)keep, out.begin() + (long)(r * keep));
    }
    return out;
}

// ============================================================================
// Distances
// ============================================================================

// Fast float32 squared L2 used only for candidate generation in the brute-force scan.
inline float l2sq_f32(const float *a, const float *b, size_t d)
{
#if defined(__AVX2__) && defined(__FMA__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= d; i += 16)
    {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
    }
    for (; i + 8 <= d; i += 8)
    {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    }
    __m256 acc = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float sum = _mm_cvtss_f32(s);
    for (; i < d; i++)
    {
        float t = a[i] - b[i];
        sum += t * t;
    }
    return sum;
#else
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    size_t i = 0;
    for (; i + 4 <= d; i += 4)
    {
        float t0 = a[i] - b[i], t1 = a[i + 1] - b[i + 1], t2 = a[i + 2] - b[i + 2], t3 = a[i + 3] - b[i + 3];
        s0 += t0 * t0;
        s1 += t1 * t1;
        s2 += t2 * t2;
        s3 += t3 * t3;
    }
    float sum = (s0 + s1) + (s2 + s3);
    for (; i < d; i++)
    {
        float t = a[i] - b[i];
        sum += t * t;
    }
    return sum;
#endif
}

// Exact squared L2 in double precision: the "truth" every answer is checked against.
inline double l2sq_exact(const float *a, const float *b, size_t d)
{
    double s = 0.0;
    for (size_t i = 0; i < d; i++)
    {
        double t = (double)a[i] - (double)b[i];
        s += t * t;
    }
    return s;
}

// ============================================================================
// Exact brute-force k-NN (reference)
// ============================================================================
//
// Parallel scan with std::thread (deliberately not OpenMP: the validator forks
// children afterwards and libgomp is not fork-safe). Each worker keeps the best
// k + margin float32 candidates per query; the merged candidates are re-ranked
// with double-precision distances, so float rounding cannot change the answer.

struct Candidate
{
    float d;
    idx_t id;
};

void brute_force_knn(const float *base, size_t n, size_t dim, const float *queries, size_t nq, size_t k,
                     unsigned threads, std::vector<idx_t> &out_ids, std::vector<double> &out_dists,
                     const std::string &label)
{
    if (k == 0 || k > n)
        throw std::invalid_argument("brute force: invalid k");
    threads = std::max(1u, threads);
    const size_t keep = std::min(n, k + 32);
    const size_t block_rows = 8192;
    const size_t n_blocks = (n + block_rows - 1) / block_rows;
    const size_t row_tile = std::max<size_t>(1, 16384 / dim);
    const size_t query_tile = std::max<size_t>(1, 32768 / dim);

    std::atomic<size_t> next_block{0};
    std::atomic<size_t> rows_done{0};
    std::atomic<unsigned> finished{0};
    std::vector<std::vector<Candidate>> heaps(threads);
    std::vector<std::vector<uint32_t>> heap_sizes(threads);

    auto cmp = [](const Candidate &a, const Candidate &b) { return a.d < b.d; };

    auto worker = [&](unsigned t) {
        std::vector<Candidate> &heap = heaps[t];
        std::vector<uint32_t> &hsize = heap_sizes[t];
        heap.assign(nq * keep, Candidate{0.0f, 0});
        hsize.assign(nq, 0);
        std::vector<float> thr(nq, std::numeric_limits<float>::infinity());
        for (;;)
        {
            size_t b = next_block.fetch_add(1);
            if (b >= n_blocks)
                break;
            size_t r0 = b * block_rows;
            size_t r1 = std::min(n, r0 + block_rows);
            for (size_t rr = r0; rr < r1; rr += row_tile)
            {
                size_t re = std::min(r1, rr + row_tile);
                for (size_t q0 = 0; q0 < nq; q0 += query_tile)
                {
                    size_t qe = std::min(nq, q0 + query_tile);
                    for (size_t r = rr; r < re; r++)
                    {
                        const float *x = base + r * dim;
                        for (size_t q = q0; q < qe; q++)
                        {
                            float d = l2sq_f32(queries + q * dim, x, dim);
                            if (d < thr[q])
                            {
                                Candidate *h = heap.data() + q * keep;
                                uint32_t &hs = hsize[q];
                                if (hs < keep)
                                {
                                    h[hs++] = Candidate{d, (idx_t)r};
                                    std::push_heap(h, h + hs, cmp);
                                    if (hs == keep)
                                        thr[q] = h[0].d;
                                }
                                else
                                {
                                    std::pop_heap(h, h + keep, cmp);
                                    h[keep - 1] = Candidate{d, (idx_t)r};
                                    std::push_heap(h, h + keep, cmp);
                                    thr[q] = h[0].d;
                                }
                            }
                        }
                    }
                }
            }
            rows_done.fetch_add(r1 - r0);
        }
        finished.fetch_add(1);
    };

    auto t0 = Clock::now();
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; t++)
        pool.emplace_back(worker, t);
    const bool chatty = (double)n * (double)nq * (double)dim > 2e11;
    auto last = Clock::now();
    while (finished.load() < threads)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (chatty && std::chrono::duration<double>(Clock::now() - last).count() >= 15.0)
        {
            last = Clock::now();
            double frac = (double)rows_done.load() / (double)n;
            double el = seconds_since(t0);
            log_line("  " + label + ": " + fmt(100.0 * frac, 1) + "% after " + fmt(el, 0) + " s" +
                     (frac > 0.01 ? " (eta " + fmt(el / frac - el, 0) + " s)" : ""));
        }
    }
    for (auto &th : pool)
        th.join();

    // Merge and re-rank in double precision.
    out_ids.assign(nq * k, 0);
    out_dists.assign(nq * k, 0.0);
    std::atomic<size_t> next_q{0};
    auto merger = [&]() {
        std::vector<std::pair<double, idx_t>> cand;
        for (;;)
        {
            size_t q = next_q.fetch_add(1);
            if (q >= nq)
                break;
            cand.clear();
            for (unsigned t = 0; t < threads; t++)
            {
                const Candidate *h = heaps[t].data() + q * keep;
                for (uint32_t i = 0; i < heap_sizes[t][q]; i++)
                    cand.emplace_back(l2sq_exact(queries + q * dim, base + h[i].id * dim, dim), h[i].id);
            }
            std::sort(cand.begin(), cand.end());
            if (cand.size() < k) // impossible: every row is scanned and keep >= k
            {
                std::fprintf(stderr, "[validator] internal error: brute force produced too few candidates\n");
                std::abort();
            }
            for (size_t j = 0; j < k; j++)
            {
                out_dists[q * k + j] = cand[j].first;
                out_ids[q * k + j] = cand[j].second;
            }
        }
    };
    std::vector<std::thread> mpool;
    for (unsigned t = 0; t < threads; t++)
        mpool.emplace_back(merger);
    for (auto &th : mpool)
        th.join();
}

// ============================================================================
// Options
// ============================================================================

using KnobMap = std::map<std::string, std::string>;

struct Options
{
    std::vector<std::string> datasets;
    std::vector<std::string> algos;
    std::vector<size_t> ks{1, 10, 100};
    size_t n_queries = 1000;
    size_t max_base = 0;
    unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    std::string vector_root = "/mnt/hddhelp/mchatzakis/datasets/processed";
    std::string series_root = "/mnt/hddhelp/mchatzakis/datasets/similarity-search-datasets";
    std::string work_dir = "validate_large_datasets_work";
    std::string csv_path;
    std::string reference = "auto"; // auto | bruteforce
    size_t gt_verify = 10;
    double rtol = 1e-4;
    double atol = 1e-5;
    double dist_rtol = 1e-3;
    double dist_atol = 1e-3;
    double timeout_s = 0.0;
    bool fork_children = true;
    bool verbose_children = false;
    bool equidepth = false;
    bool list_only = false;
    bool force = false;
    size_t details = 5;
    std::map<std::string, KnobMap> knobs; // algo key -> knob -> value
};

// ============================================================================
// Dataset catalog
// ============================================================================

enum class Format
{
    FVECS,
    RAW_F32
};

struct DatasetSpec
{
    std::string name;
    Format format;
    bool series; // true: lives under --series-root, queries are held out from the file tail
    std::string dir;
    std::string base_file;
    std::string query_file;
    std::string gt_file; // L2 ground truth (ivecs), empty if none
    size_t raw_dim = 0;
};

// Ordered from cheapest to most expensive, which is the default run order.
const std::vector<DatasetSpec> &catalog()
{
    static const std::vector<DatasetSpec> c = {
        {"SIFT1M", Format::FVECS, false, "SIFT1M", "base.1M.fvecs", "query.10K.fvecs",
         "l2.query.groundtruth.10K.k5000.ivecs", 0},
        {"GIST1M", Format::FVECS, false, "GIST1M", "base.1M.fvecs", "query.1K.fvecs",
         "l2.query.groundtruth.1K.k5000.ivecs", 0},
        {"T2I10M", Format::FVECS, false, "T2I10M", "base.10M.fvecs", "query.10K.fvecs", "", 0},
        {"HOTPOTQA5M", Format::FVECS, false, "HOTPOTQA5M", "base.5233329.fvecs", "query.7405.fvecs",
         "l2.query.groundtruth.7405.k5000.ivecs", 0},
        {"UCASELAW7M", Format::FVECS, false, "UCASELAW7M", "base.7M.fvecs", "query.1K.fvecs",
         "l2.query.groundtruth.1K.k5000.ivecs", 0},
        {"MSMARCOMXBAI8M", Format::FVECS, false, "MSMARCOMXBAI8M", "base.8841823.fvecs", "query.6980.fvecs",
         "l2.query.groundtruth.6980.k5000.ivecs", 0},
        {"DEEP100M", Format::FVECS, false, "DEEP100M", "base.100M.fvecs", "query.10K.fvecs",
         "l2.query.groundtruth.10K.k5000.ivecs", 0},
        {"Seismic", Format::RAW_F32, true, "", "data_size100M_seismic_len256_znorm.bin", "", "", 256},
        {"Astro", Format::RAW_F32, true, "", "data_size270M_astronomy_len256_znorm.bin", "", "", 256},
    };
    return c;
}

const DatasetSpec *find_dataset(const std::string &name)
{
    for (const auto &d : catalog())
        if (to_lower(d.name) == to_lower(name))
            return &d;
    return nullptr;
}

std::string dataset_path(const Options &opt, const DatasetSpec &spec, const std::string &file)
{
    if (file.empty())
        return "";
    fs::path root = spec.series ? fs::path(opt.series_root) : fs::path(opt.vector_root) / spec.dir;
    return (root / file).string();
}

struct DatasetInfo
{
    std::string base_path, query_path, gt_path;
    size_t dim = 0;
    size_t n_total = 0;      // records available in the base file
    size_t n_queries_avail = 0;
    size_t gt_rows = 0, gt_len = 0;
    bool gt_present = false;
    std::vector<std::string> warnings;
};

DatasetInfo probe_dataset(const Options &opt, const DatasetSpec &spec)
{
    DatasetInfo info;
    info.base_path = dataset_path(opt, spec, spec.base_file);
    info.query_path = dataset_path(opt, spec, spec.query_file);
    info.gt_path = dataset_path(opt, spec, spec.gt_file);
    if (!fs::exists(info.base_path))
        throw std::runtime_error("base file not found: " + info.base_path);

    if (spec.format == Format::FVECS)
    {
        info.dim = fvecs_dim(info.base_path);
        info.n_total = fvecs_rows(info.base_path, info.dim);
        if (!fs::exists(info.query_path))
            throw std::runtime_error("query file not found: " + info.query_path);
        size_t qd = fvecs_dim(info.query_path);
        if (qd != info.dim)
            throw std::runtime_error("query dimension " + std::to_string(qd) + " != base dimension " +
                                     std::to_string(info.dim));
        info.n_queries_avail = fvecs_rows(info.query_path, qd);
    }
    else
    {
        info.dim = spec.raw_dim;
        size_t sz = file_size_or_zero(info.base_path);
        size_t row = info.dim * sizeof(float);
        info.n_total = sz / row;
        if (sz % row != 0)
            info.warnings.push_back("file size is not a multiple of the record size (file still being written?); "
                                    "using the first " + std::to_string(info.n_total) + " complete records");
        // queries are the last n_queries records; the base is everything before them
        info.n_queries_avail = info.n_total;
    }

    if (!info.gt_path.empty() && fs::exists(info.gt_path))
    {
        File f(info.gt_path, "rb");
        int32_t k = 0;
        f.read_exact(&k, sizeof(k));
        info.gt_len = (size_t)std::max(0, k);
        info.gt_rows = k > 0 ? file_size_or_zero(info.gt_path) / (sizeof(int32_t) * (1 + (size_t)k)) : 0;
        info.gt_present = info.gt_len > 0;
    }
    return info;
}

// ============================================================================
// Loaded dataset and reference answers
// ============================================================================

struct LoadedDataset
{
    const DatasetSpec *spec = nullptr;
    DatasetInfo info;
    size_t n_base = 0;
    size_t dim = 0;
    size_t nq = 0;
    size_t query_first_row = 0; // for series datasets: first held-out record
    FloatMatrix base;
    FloatMatrix queries;
    std::string raw_base_path; // raw float32 file whose first n_base records equal `base` (Odyssey)
};

struct Reference
{
    size_t k = 0;
    std::vector<idx_t> ids;    // nq * k, sorted by exact distance
    std::vector<double> dists; // nq * k, exact squared L2, ascending
    std::string source;
};

void load_dataset(const Options &opt, const DatasetSpec &spec, LoadedDataset &ds)
{
    ds.spec = &spec;
    ds.info = probe_dataset(opt, spec);
    for (const auto &w : ds.info.warnings)
        log_line("  warning: " + w);
    ds.dim = ds.info.dim;

    if (spec.format == Format::FVECS)
    {
        ds.nq = std::min(opt.n_queries, ds.info.n_queries_avail);
        ds.n_base = ds.info.n_total;
        if (opt.max_base > 0 && opt.max_base < ds.n_base)
            ds.n_base = opt.max_base;
        ds.queries.allocate(ds.nq, ds.dim);
        read_fvecs_rows(ds.info.query_path, ds.dim, 0, ds.nq, ds.queries.data(), spec.name + " queries");
        log_line("  loading " + std::to_string(ds.n_base) + " x " + std::to_string(ds.dim) + " base vectors (" +
                 human_bytes((double)ds.n_base * ds.dim * 4) + ") from " + ds.info.base_path);
        ds.base.allocate(ds.n_base, ds.dim);
        read_fvecs_rows(ds.info.base_path, ds.dim, 0, ds.n_base, ds.base.data(), spec.name + " base");
    }
    else
    {
        if (ds.info.n_total < 2)
            throw std::runtime_error("data-series file has too few records");
        ds.nq = std::min(opt.n_queries, ds.info.n_total / 2);
        ds.query_first_row = ds.info.n_total - ds.nq;
        ds.n_base = ds.query_first_row;
        if (opt.max_base > 0 && opt.max_base < ds.n_base)
            ds.n_base = opt.max_base;
        ds.queries.allocate(ds.nq, ds.dim);
        read_raw_rows(ds.info.base_path, ds.dim, ds.query_first_row, ds.nq, ds.queries.data(), spec.name + " queries");
        log_line("  queries: records [" + std::to_string(ds.query_first_row) + ", " + std::to_string(ds.info.n_total) +
                 ") of the file (held out from the base)");
        log_line("  loading " + std::to_string(ds.n_base) + " x " + std::to_string(ds.dim) + " base series (" +
                 human_bytes((double)ds.n_base * ds.dim * 4) + ") from " + ds.info.base_path);
        ds.base.allocate(ds.n_base, ds.dim);
        read_raw_rows(ds.info.base_path, ds.dim, 0, ds.n_base, ds.base.data(), spec.name + " base");
        ds.raw_base_path = ds.info.base_path; // first n_base records == base
    }
}

uint64_t dataset_fingerprint(const LoadedDataset &ds, size_t k)
{
    uint64_t h = fnv1a(&ds.n_base, sizeof(ds.n_base));
    h = fnv1a(&ds.dim, sizeof(ds.dim), h);
    h = fnv1a(&k, sizeof(k), h);
    const size_t rb = ds.dim * sizeof(float);
    h = fnv1a(ds.base.row(0), rb, h);
    h = fnv1a(ds.base.row(ds.n_base / 2), rb, h);
    h = fnv1a(ds.base.row(ds.n_base - 1), rb, h);
    h = fnv1a(ds.queries.data(), ds.queries.bytes(), h);
    return h;
}

void sort_reference_rows(const LoadedDataset &ds, Reference &ref)
{
    std::vector<std::pair<double, idx_t>> row(ref.k);
    for (size_t q = 0; q < ds.nq; q++)
    {
        for (size_t j = 0; j < ref.k; j++)
            row[j] = {ref.dists[q * ref.k + j], ref.ids[q * ref.k + j]};
        std::sort(row.begin(), row.end());
        for (size_t j = 0; j < ref.k; j++)
        {
            ref.dists[q * ref.k + j] = row[j].first;
            ref.ids[q * ref.k + j] = row[j].second;
        }
    }
}

bool load_reference_cache(const std::string &path, const LoadedDataset &ds, Reference &ref)
{
    if (!fs::exists(path))
        return false;
    try
    {
        File f(path, "rb");
        char magic[8];
        uint64_t nq = 0, k = 0;
        f.read_exact(magic, sizeof(magic));
        f.read_exact(&nq, sizeof(nq));
        f.read_exact(&k, sizeof(k));
        if (std::memcmp(magic, "DSYREF1", 8) != 0 || nq != ds.nq || k != ref.k)
            return false;
        ref.ids.resize(nq * k);
        f.read_exact(ref.ids.data(), ref.ids.size() * sizeof(idx_t));
    }
    catch (const std::exception &)
    {
        return false;
    }
    ref.dists.resize(ref.ids.size());
    for (size_t q = 0; q < ds.nq; q++)
        for (size_t j = 0; j < ref.k; j++)
        {
            idx_t id = ref.ids[q * ref.k + j];
            if (id >= ds.n_base)
                return false;
            ref.dists[q * ref.k + j] = l2sq_exact(ds.queries.row(q), ds.base.row(id), ds.dim);
        }
    sort_reference_rows(ds, ref);
    return true;
}

void save_reference_cache(const std::string &path, const Reference &ref, size_t nq)
{
    fs::create_directories(fs::path(path).parent_path());
    std::string tmp = path + ".tmp";
    {
        File f(tmp, "wb");
        uint64_t q = nq, k = ref.k;
        f.write_exact("DSYREF1", 8);
        f.write_exact(&q, sizeof(q));
        f.write_exact(&k, sizeof(k));
        f.write_exact(ref.ids.data(), ref.ids.size() * sizeof(idx_t));
        f.close();
    }
    fs::rename(tmp, path);
}

// Brute-force a few evenly spaced queries and compare their k-th distance with the
// file ground truth. A stale or mismatched GT file must never be blamed on DaiSy.
bool verify_ground_truth(const Options &opt, const LoadedDataset &ds, const Reference &ref)
{
    size_t m = std::min(opt.gt_verify, ds.nq);
    if (m == 0)
        return true;
    std::vector<size_t> picks;
    for (size_t i = 0; i < m; i++)
        picks.push_back(i * ds.nq / m);
    FloatMatrix sub(m, ds.dim);
    for (size_t i = 0; i < m; i++)
        std::memcpy(sub.row(i), ds.queries.row(picks[i]), ds.dim * sizeof(float));
    std::vector<idx_t> ids;
    std::vector<double> dists;
    brute_force_knn(ds.base.data(), ds.n_base, ds.dim, sub.data(), m, ref.k, opt.threads, ids, dists,
                    "GT spot check");
    bool ok = true;
    for (size_t i = 0; i < m; i++)
    {
        size_t q = picks[i];
        double bf = dists[i * ref.k + ref.k - 1];
        double gt = ref.dists[q * ref.k + ref.k - 1];
        if (std::fabs(bf - gt) > opt.rtol * std::max(bf, gt) + opt.atol)
        {
            ok = false;
            log_line("  GT spot check FAILED for query " + std::to_string(q) + ": brute-force k-th distance " +
                     fmt_sci(bf) + " vs ground-truth file " + fmt_sci(gt));
        }
    }
    if (ok)
        log_line("  GT spot check passed (" + std::to_string(m) + " queries brute-forced)");
    return ok;
}

Reference build_reference(const Options &opt, LoadedDataset &ds, size_t k_max)
{
    Reference ref;
    ref.k = k_max;

    const bool gt_usable = opt.reference == "auto" && ds.info.gt_present && ds.n_base == ds.info.n_total &&
                           ds.info.gt_rows >= ds.nq && ds.info.gt_len >= k_max;
    if (opt.reference == "auto" && ds.info.gt_present && !gt_usable)
    {
        if (ds.n_base != ds.info.n_total)
            log_line("  ground-truth file ignored: it covers the full base, but only " + std::to_string(ds.n_base) +
                     " records are loaded");
        else
            log_line("  ground-truth file ignored: too few rows or ids per row");
    }

    if (gt_usable)
    {
        size_t row_len = 0;
        std::vector<int32_t> gt = read_ivecs_prefix(ds.info.gt_path, ds.nq, k_max, row_len);
        bool ids_ok = true;
        ref.ids.resize(ds.nq * k_max);
        ref.dists.resize(ds.nq * k_max);
        for (size_t i = 0; i < gt.size(); i++)
        {
            if (gt[i] < 0 || (size_t)gt[i] >= ds.n_base)
            {
                ids_ok = false;
                break;
            }
            ref.ids[i] = (idx_t)gt[i];
        }
        if (ids_ok)
        {
            for (size_t q = 0; q < ds.nq; q++)
                for (size_t j = 0; j < k_max; j++)
                    ref.dists[q * k_max + j] =
                        l2sq_exact(ds.queries.row(q), ds.base.row(ref.ids[q * k_max + j]), ds.dim);
            sort_reference_rows(ds, ref);
            if (verify_ground_truth(opt, ds, ref))
            {
                ref.source = "gt-file";
                log_line("  reference: " + ds.info.gt_path);
                return ref;
            }
            log_line("  ground-truth file is inconsistent with the loaded base/queries; falling back to brute force");
        }
        else
            log_line("  ground-truth file contains ids outside the base; falling back to brute force");
    }

    std::string cache = (fs::path(opt.work_dir) / "reference_cache" /
                         (ds.spec->name + "_n" + std::to_string(ds.n_base) + "_q" + std::to_string(ds.nq) + "_k" +
                          std::to_string(k_max) + "_" + hex64(dataset_fingerprint(ds, k_max)) + ".ref"))
                            .string();
    if (load_reference_cache(cache, ds, ref))
    {
        ref.source = "bruteforce-cached";
        log_line("  reference: cached brute force " + cache);
        return ref;
    }
    log_line("  computing exact brute-force reference (" + std::to_string(ds.nq) + " queries, k=" +
             std::to_string(k_max) + ", " + std::to_string(opt.threads) + " threads)");
    auto t0 = Clock::now();
    brute_force_knn(ds.base.data(), ds.n_base, ds.dim, ds.queries.data(), ds.nq, k_max, opt.threads, ref.ids,
                    ref.dists, "brute force");
    log_line("  brute force done in " + fmt(seconds_since(t0), 1) + " s");
    ref.source = "bruteforce";
    try
    {
        save_reference_cache(cache, ref, ds.nq);
    }
    catch (const std::exception &e)
    {
        log_line("  warning: could not cache the reference: " + std::string(e.what()));
    }
    return ref;
}

// Odyssey only builds from a raw float32 file. For fvecs datasets we materialize
// the loaded base once under <work-dir>/bin_cache and reuse it on later runs.
std::string ensure_raw_base_file(const Options &opt, LoadedDataset &ds)
{
    if (!ds.raw_base_path.empty())
        return ds.raw_base_path;
    fs::path dir = fs::path(opt.work_dir) / "bin_cache";
    fs::create_directories(dir);
    std::string path = (dir / (ds.spec->name + "_n" + std::to_string(ds.n_base) + "_d" + std::to_string(ds.dim) +
                               ".f32"))
                           .string();
    const size_t row = ds.dim * sizeof(float);
    if (file_size_or_zero(path) == ds.base.bytes())
    {
        std::vector<float> first(ds.dim), last(ds.dim);
        read_raw_rows(path, ds.dim, 0, 1, first.data(), "");
        read_raw_rows(path, ds.dim, ds.n_base - 1, 1, last.data(), "");
        if (std::memcmp(first.data(), ds.base.row(0), row) == 0 &&
            std::memcmp(last.data(), ds.base.row(ds.n_base - 1), row) == 0)
        {
            ds.raw_base_path = path;
            log_line("  raw float32 copy for Odyssey: reusing " + path);
            return path;
        }
    }
    log_line("  writing raw float32 copy of the base for Odyssey: " + path + " (" +
             human_bytes((double)ds.base.bytes()) + ")");
    std::string tmp = path + ".tmp";
    {
        File f(tmp, "wb");
        Progress progress("write " + ds.spec->name + " raw copy", ds.base.bytes());
        const size_t chunk = 64ULL << 20;
        const char *src = reinterpret_cast<const char *>(ds.base.data());
        for (size_t off = 0; off < ds.base.bytes(); off += chunk)
        {
            size_t n = std::min(chunk, ds.base.bytes() - off);
            f.write_exact(src + off, n);
            progress.update(off + n);
        }
        f.close();
    }
    fs::rename(tmp, path);
    ds.raw_base_path = path;
    return path;
}

// ============================================================================
// Algorithm drivers
// ============================================================================

struct RunContext
{
    const LoadedDataset *ds = nullptr;
    unsigned threads = 1;
    bool equidepth = false;
    KnobMap knobs;
};

template <class T>
using KnobTable = std::map<std::string, std::function<void(T &, const std::string &)>>;

template <class T>
void apply_knobs(T &target, const KnobTable<T> &table, const KnobMap &knobs, const std::string &algo)
{
    for (const auto &kv : knobs)
    {
        auto it = table.find(kv.first);
        if (it == table.end())
            throw std::invalid_argument("unknown knob " + algo + "." + kv.first);
        it->second(target, kv.second);
    }
}

template <class T>
std::vector<std::string> knob_names(const KnobTable<T> &table)
{
    std::vector<std::string> out;
    for (const auto &kv : table)
        out.push_back(kv.first);
    return out;
}

#define VALIDATOR_INT_KNOB(T, key, expr)                                  \
    {                                                                     \
        key, [](T &a, const std::string &v) { int x = parse_knob_int(key, v); expr; } \
    }

// MESSI and FreSh expose the same iSAX knob setters.
template <class A>
const KnobTable<A> &isax_knob_table()
{
    static const KnobTable<A> table = {
        VALIDATOR_INT_KNOB(A, "leaf_size", a.setLeafSize(x)),
        VALIDATOR_INT_KNOB(A, "min_leaf_size", a.setMinLeafSize(x)),
        VALIDATOR_INT_KNOB(A, "paa_segments", a.setPaaSegments(x)),
        VALIDATOR_INT_KNOB(A, "sax_cardinality", a.setSaxCardinality(x)),
        VALIDATOR_INT_KNOB(A, "initial_lbl_size", a.setInitialLblSize(x)),
        VALIDATOR_INT_KNOB(A, "flush_limit", a.setFlushLimit(x)),
        VALIDATOR_INT_KNOB(A, "initial_fbl_size", a.setInitialFblSize(x)),
        VALIDATOR_INT_KNOB(A, "total_loaded_leaves", a.setTotalLoadedLeaves(x)),
        VALIDATOR_INT_KNOB(A, "tight_bound", a.setTightBound(x)),
        VALIDATOR_INT_KNOB(A, "search_workers", a.setSearchWorkers(x)),
        VALIDATOR_INT_KNOB(A, "index_workers", a.setIndexWorkers(x)),
        VALIDATOR_INT_KNOB(A, "read_block_length", a.setReadBlockLength(x)),
    };
    return table;
}

#if VALIDATOR_HAS_SOFA
const KnobTable<daisy::Sofa> &sofa_knob_table()
{
    using S = daisy::Sofa;
    static const KnobTable<S> table = {
        VALIDATOR_INT_KNOB(S, "leaf_size", a.setLeafSize(x)),
        VALIDATOR_INT_KNOB(S, "min_leaf_size", a.setMinLeafSize(x)),
        VALIDATOR_INT_KNOB(S, "word_length", a.setWordLength(x)),
        VALIDATOR_INT_KNOB(S, "alphabet_size", a.setAlphabetSize(x)),
        VALIDATOR_INT_KNOB(S, "sample_size", a.setSampleSize(x)),
        VALIDATOR_INT_KNOB(S, "histogram_type", a.setHistogramType(x)),
        VALIDATOR_INT_KNOB(S, "coeff_number", a.setCoeffNumber(x)),
        VALIDATOR_INT_KNOB(S, "initial_lbl_size", a.setInitialLblSize(x)),
        VALIDATOR_INT_KNOB(S, "flush_limit", a.setFlushLimit(x)),
        VALIDATOR_INT_KNOB(S, "initial_fbl_size", a.setInitialFblSize(x)),
        VALIDATOR_INT_KNOB(S, "total_loaded_leaves", a.setTotalLoadedLeaves(x)),
        VALIDATOR_INT_KNOB(S, "tight_bound", a.setTightBound(x)),
        VALIDATOR_INT_KNOB(S, "search_workers", a.setSearchWorkers(x)),
        VALIDATOR_INT_KNOB(S, "index_workers", a.setIndexWorkers(x)),
        VALIDATOR_INT_KNOB(S, "read_block_length", a.setReadBlockLength(x)),
        {"is_norm", [](S &a, const std::string &v) { a.setIsNorm(parse_bool("is_norm", v)); }},
    };
    return table;
}
#endif

#if VALIDATOR_HAS_ODYSSEY
const KnobTable<daisy::OdysseyConfig> &odyssey_knob_table()
{
    using C = daisy::OdysseyConfig;
    static const KnobTable<C> table = {
        VALIDATOR_INT_KNOB(C, "search_workers", a.search_workers = x),
        VALIDATOR_INT_KNOB(C, "index_threads", a.index_threads = x),
        VALIDATOR_INT_KNOB(C, "query_threads", a.query_threads = x),
        VALIDATOR_INT_KNOB(C, "leaf_size", a.leaf_size = x),
        VALIDATOR_INT_KNOB(C, "paa_segments", a.paa_segments = x),
        VALIDATOR_INT_KNOB(C, "pq_th_div_factor", a.pq_th_div_factor = x),
    };
    return table;
}
#endif

#undef VALIDATOR_INT_KNOB

using AlgoPtr = std::unique_ptr<daisy::SimilaritySearchAlgorithm>;

struct AlgoDriver
{
    std::string key;   // CLI name
    std::string label; // display name
    bool available = true;
    std::string unavailable_reason;
    bool builds_from_file = false;  // needs a raw float32 base file
    bool copies_base_in_ram = false; // keeps a private copy of the base
    bool honors_breakpoint_mode = true;
    std::vector<std::string> knobs;
    std::function<AlgoPtr(const RunContext &)> make;
};

void build_in_memory(daisy::SimilaritySearchAlgorithm &algo, const RunContext &ctx)
{
    const LoadedDataset &ds = *ctx.ds;
    algo.buildIndex(const_cast<float *>(ds.base.data()), (idx_t)ds.n_base, (idx_t)ds.dim);
}

const std::vector<AlgoDriver> &drivers()
{
    static const std::vector<AlgoDriver> d = [] {
        std::vector<AlgoDriver> v;

        AlgoDriver messi;
        messi.key = "messi";
        messi.label = "MESSI";
        messi.knobs = knob_names(isax_knob_table<daisy::Messi>());
        messi.make = [](const RunContext &ctx) -> AlgoPtr {
            auto a = std::make_unique<daisy::Messi>(daisy::DistanceType::L2_SQUARED);
            a->setNumThreads((int)ctx.threads);
            a->setIndexWorkers((int)ctx.threads);
            if (ctx.equidepth)
                a->setNormalized(false);
            apply_knobs(*a, isax_knob_table<daisy::Messi>(), ctx.knobs, "messi");
            return a;
        };
        v.push_back(messi);

        AlgoDriver sofa;
        sofa.key = "sofa";
        sofa.label = "SOFA";
        sofa.honors_breakpoint_mode = false;
#if VALIDATOR_HAS_SOFA
        sofa.knobs = knob_names(sofa_knob_table());
        sofa.make = [](const RunContext &ctx) -> AlgoPtr {
            auto a = std::make_unique<daisy::Sofa>(daisy::DistanceType::L2_SQUARED);
            a->setNumThreads((int)ctx.threads);
            a->setIndexWorkers((int)ctx.threads);
            if (ctx.equidepth)
                a->setNormalized(false);
            apply_knobs(*a, sofa_knob_table(), ctx.knobs, "sofa");
            return a;
        };
#else
        sofa.available = false;
        sofa.unavailable_reason = "not compiled (configure with -DBUILD_SOFA=ON and FFTW3 single precision)";
#endif
        v.push_back(sofa);

        AlgoDriver fresh;
        fresh.key = "fresh";
        fresh.label = "FreSh";
        fresh.knobs = knob_names(isax_knob_table<daisy::Fresh>());
        fresh.make = [](const RunContext &ctx) -> AlgoPtr {
            auto a = std::make_unique<daisy::Fresh>(daisy::DistanceType::L2_SQUARED);
            a->setNumThreads((int)ctx.threads);
            a->setIndexWorkers((int)ctx.threads);
            if (ctx.equidepth)
                a->setNormalized(false);
            apply_knobs(*a, isax_knob_table<daisy::Fresh>(), ctx.knobs, "fresh");
            return a;
        };
        v.push_back(fresh);

        AlgoDriver odyssey;
        odyssey.key = "odyssey";
        odyssey.label = "Odyssey";
        odyssey.builds_from_file = true;
        odyssey.copies_base_in_ram = true;
        odyssey.honors_breakpoint_mode = false;
#if VALIDATOR_HAS_ODYSSEY
        odyssey.knobs = knob_names(odyssey_knob_table());
        odyssey.make = [](const RunContext &ctx) -> AlgoPtr {
            daisy::OdysseyConfig cfg;
            cfg.search_workers = (int)ctx.threads;
            cfg.index_threads = (int)ctx.threads;
            cfg.query_threads = (int)ctx.threads;
            apply_knobs(cfg, odyssey_knob_table(), ctx.knobs, "odyssey");
            auto a = std::make_unique<daisy::Odyssey>(cfg, daisy::DistanceType::L2_SQUARED, 0, nullptr);
            if (ctx.equidepth)
                a->setNormalized(false);
            return a;
        };
#else
        odyssey.available = false;
        odyssey.unavailable_reason = "not compiled (configure with -DBUILD_ODYSSEY=ON and MPI)";
#endif
        v.push_back(odyssey);

        return v;
    }();
    return d;
}

const AlgoDriver *find_driver(const std::string &key)
{
    for (const auto &d : drivers())
        if (d.key == to_lower(key))
            return &d;
    return nullptr;
}

// ============================================================================
// Isolated execution of one algorithm on one dataset
// ============================================================================

constexpr int kMaxK = 16;

enum Phase : int
{
    PH_START = 0,
    PH_CONFIGURE = 1,
    PH_BUILD = 2,
    PH_SEARCH = 3,
    PH_DESTROY = 4,
    PH_DONE = 5
};

const char *phase_name(int p)
{
    switch (p)
    {
    case PH_START: return "startup";
    case PH_CONFIGURE: return "configure";
    case PH_BUILD: return "buildIndex";
    case PH_SEARCH: return "searchIndex";
    case PH_DESTROY: return "destructor";
    case PH_DONE: return "done";
    default: return "?";
    }
}

enum SearchState : int
{
    SS_NOT_RUN = 0,
    SS_OK = 1,
    SS_EXCEPTION = 2
};

// Lives in a MAP_SHARED page so the parent can read it after the child dies.
struct ChildReport
{
    volatile int phase;
    volatile int k_index;
    volatile int configure_failed;
    volatile int build_failed;
    volatile int base_mutated;
    volatile int queries_mutated;
    double build_seconds;
    double search_seconds[kMaxK];
    volatile int search_state[kMaxK];
    char message[1024];
    char k_message[kMaxK][512];
};

struct ResultLayout
{
    size_t nq = 0;
    std::vector<size_t> ks;
    std::vector<size_t> i_off, d_off;
    size_t total = 0;

    ResultLayout(size_t nq_, const std::vector<size_t> &ks_) : nq(nq_), ks(ks_)
    {
        auto up = [](size_t x) { return (x + 63) / 64 * 64; };
        size_t off = up(sizeof(ChildReport));
        for (size_t k : ks)
        {
            i_off.push_back(off);
            off = up(off + nq * k * sizeof(idx_t));
            d_off.push_back(off);
            off = up(off + nq * k * sizeof(float));
        }
        total = off;
    }
};

constexpr idx_t kUnwritten = std::numeric_limits<idx_t>::max();

void copy_message(char *dst, size_t cap, const std::string &msg)
{
    std::snprintf(dst, cap, "%s", msg.c_str());
}

// Samples ~64K rows; enough to notice an algorithm normalizing or reordering the caller's data.
uint64_t sampled_signature(const FloatMatrix &m)
{
    if (m.rows() == 0)
        return 0;
    size_t step = std::max<size_t>(1, m.rows() / 65536);
    uint64_t h = 1469598103934665603ULL;
    for (size_t r = 0; r < m.rows(); r += step)
        h = fnv1a(m.row(r), m.cols() * sizeof(float), h);
    return fnv1a(m.row(m.rows() - 1), m.cols() * sizeof(float), h);
}

void execute_algorithm(const AlgoDriver &drv, const RunContext &ctx, const ResultLayout &lay, char *region)
{
    ChildReport *rep = reinterpret_cast<ChildReport *>(region);
    const LoadedDataset &ds = *ctx.ds;
    const uint64_t base_sig = sampled_signature(ds.base);
    const uint64_t query_sig = fnv1a(ds.queries.data(), ds.queries.bytes());

    AlgoPtr algo;
    rep->phase = PH_CONFIGURE;
    try
    {
        algo = drv.make(ctx);
    }
    catch (const std::invalid_argument &e) // bad knob value: a user error, not an unsupported mode
    {
        rep->configure_failed = 2;
        copy_message(rep->message, sizeof(rep->message), e.what());
        return;
    }
    catch (const std::exception &e) // an algorithm rejecting the requested mode
    {
        rep->configure_failed = 1;
        copy_message(rep->message, sizeof(rep->message), e.what());
        return;
    }

    rep->phase = PH_BUILD;
    try
    {
        auto t0 = Clock::now();
        if (drv.builds_from_file)
            algo->buildIndex(ds.raw_base_path, (idx_t)ds.dim, (idx_t)ds.n_base);
        else
            build_in_memory(*algo, ctx);
        rep->build_seconds = seconds_since(t0);
    }
    catch (const std::exception &e)
    {
        rep->build_failed = 1;
        copy_message(rep->message, sizeof(rep->message), e.what());
        return;
    }
    catch (...)
    {
        rep->build_failed = 1;
        copy_message(rep->message, sizeof(rep->message), "non-standard exception");
        return;
    }

    for (size_t i = 0; i < lay.ks.size(); i++)
    {
        rep->phase = PH_SEARCH;
        rep->k_index = (int)i;
        const size_t k = lay.ks[i];
        idx_t *I = reinterpret_cast<idx_t *>(region + lay.i_off[i]);
        float *D = reinterpret_cast<float *>(region + lay.d_off[i]);
        std::fill(I, I + ds.nq * k, kUnwritten);
        std::fill(D, D + ds.nq * k, std::numeric_limits<float>::quiet_NaN());
        try
        {
            auto t0 = Clock::now();
            algo->searchIndex(ds.queries.data(), (idx_t)ds.nq, (idx_t)k, I, D);
            rep->search_seconds[i] = seconds_since(t0);
            rep->search_state[i] = SS_OK;
        }
        catch (const std::exception &e)
        {
            rep->search_state[i] = SS_EXCEPTION;
            copy_message(rep->k_message[i], sizeof(rep->k_message[i]), e.what());
        }
        catch (...)
        {
            rep->search_state[i] = SS_EXCEPTION;
            copy_message(rep->k_message[i], sizeof(rep->k_message[i]), "non-standard exception");
        }
    }

    rep->phase = PH_DESTROY;
    algo.reset();

    rep->base_mutated = sampled_signature(ds.base) != base_sig;
    rep->queries_mutated = fnv1a(ds.queries.data(), ds.queries.bytes()) != query_sig;
    rep->phase = PH_DONE;
}

void finalize_mpi_if_needed()
{
#if VALIDATOR_HAS_ODYSSEY
    int initialized = 0, finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (initialized && !finalized)
        MPI_Finalize();
#endif
}

struct ChildOutcome
{
    enum Kind
    {
        COMPLETED,  // child reached PH_DONE and exited 0
        CRASHED,    // killed by a signal
        EXITED,     // exited before PH_DONE (exit() inside the library, or non-zero status)
        TIMED_OUT,
        FORK_FAILED
    } kind = COMPLETED;
    int code = 0; // exit status or signal
    double wall_seconds = 0.0;
};

std::vector<std::string> tail_lines(const std::string &path, size_t n)
{
    std::ifstream in(path);
    std::deque<std::string> q;
    std::string line;
    while (std::getline(in, line))
    {
        q.push_back(line);
        if (q.size() > n)
            q.pop_front();
    }
    return {q.begin(), q.end()};
}

ChildOutcome run_isolated(const Options &opt, const AlgoDriver &drv, const RunContext &ctx, const ResultLayout &lay,
                          char *region, const std::string &log_path)
{
    ChildOutcome out;
    auto t0 = Clock::now();
    ChildReport *rep = reinterpret_cast<ChildReport *>(region);

    if (!opt.fork_children)
    {
        execute_algorithm(drv, ctx, lay, region);
        out.kind = ChildOutcome::COMPLETED;
        out.wall_seconds = seconds_since(t0);
        return out;
    }

    std::fflush(stdout);
    std::fflush(stderr);
    pid_t pid = fork();
    if (pid < 0)
    {
        out.kind = ChildOutcome::FORK_FAILED;
        out.code = errno;
        return out;
    }
    if (pid == 0)
    {
        // Make this child, not the parent holding the whole base, the OOM killer's
        // first victim (raising one's own score needs no privilege).
        if (FILE *oom = std::fopen("/proc/self/oom_score_adj", "w"))
        {
            std::fputs("1000", oom);
            std::fclose(oom);
        }
        if (!opt.verbose_children)
        {
            int fd = ::open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd >= 0)
            {
                ::dup2(fd, STDOUT_FILENO);
                ::dup2(fd, STDERR_FILENO);
                ::close(fd);
            }
        }
        std::setvbuf(stdout, nullptr, _IOLBF, 0);
        std::setvbuf(stderr, nullptr, _IONBF, 0);
        execute_algorithm(drv, ctx, lay, region);
        finalize_mpi_if_needed();
        std::fflush(stdout);
        std::fflush(stderr);
        _exit(0);
    }

    int status = 0;
    auto last_beat = Clock::now();
    for (;;)
    {
        pid_t r = waitpid(pid, &status, WNOHANG);
        if (r == pid)
            break;
        if (r < 0 && errno != EINTR)
        {
            out.kind = ChildOutcome::EXITED;
            out.code = -1;
            out.wall_seconds = seconds_since(t0);
            return out;
        }
        double el = seconds_since(t0);
        if (opt.timeout_s > 0 && el > opt.timeout_s)
        {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            out.kind = ChildOutcome::TIMED_OUT;
            out.wall_seconds = el;
            return out;
        }
        if (std::chrono::duration<double>(Clock::now() - last_beat).count() >= 60.0)
        {
            last_beat = Clock::now();
            std::string where = phase_name(rep->phase);
            if (rep->phase == PH_SEARCH)
                where += " k=" + std::to_string(lay.ks[(size_t)std::max(0, (int)rep->k_index)]);
            log_line("    ... " + drv.label + " still running (" + where + ", " + fmt(el, 0) + " s)");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    out.wall_seconds = seconds_since(t0);
    if (WIFSIGNALED(status))
    {
        out.kind = ChildOutcome::CRASHED;
        out.code = WTERMSIG(status);
    }
    else if (WIFEXITED(status) && WEXITSTATUS(status) == 0 && rep->phase == PH_DONE)
        out.kind = ChildOutcome::COMPLETED;
    else if (WIFEXITED(status) && WEXITSTATUS(status) == 0 &&
             (rep->configure_failed || rep->build_failed))
        out.kind = ChildOutcome::COMPLETED; // reported failure cleanly
    else
    {
        out.kind = ChildOutcome::EXITED;
        out.code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }
    return out;
}

// ============================================================================
// Validation
// ============================================================================

struct KResult
{
    size_t k = 0;
    std::string status; // PASS WARN FAIL ERROR CRASH TIMEOUT SKIP
    size_t nq = 0;
    size_t exact_q = 0;
    size_t dup_q = 0;
    size_t invalid_q = 0;   // ids >= n_base (excluding unwritten)
    size_t unwritten_q = 0; // output slots never written by the algorithm
    size_t unsorted_q = 0;
    size_t baddist_q = 0;
    size_t sqrtdist_q = 0;  // reported distances look like L2 instead of squared L2
    size_t refsuspect_q = 0;
    double recall_mean = 0.0;
    double recall_min = 1.0;
    double kth_rel_err_max = 0.0;
    double build_s = 0.0;
    double search_s = 0.0;
    std::string note;
    std::vector<std::string> details;
};

KResult validate_k(const Options &opt, const LoadedDataset &ds, const Reference &ref, size_t k, const idx_t *I,
                   const float *D)
{
    KResult r;
    r.k = k;
    r.nq = ds.nq;
    double recall_sum = 0.0;
    std::vector<double> true_d(k);
    std::vector<double> sorted_d(k);

    for (size_t q = 0; q < ds.nq; q++)
    {
        const idx_t *ids = I + q * k;
        const float *dd = D + q * k;
        const idx_t *ref_ids = ref.ids.data() + q * ref.k;
        const double ref_kth = ref.dists[q * ref.k + (k - 1)];
        const double thr = ref_kth * (1.0 + opt.rtol) + opt.atol;

        bool invalid = false, unwritten = false, dup = false, wrong = false, unsorted = false, baddist = false,
             sqrtlike = true;
        std::unordered_set<idx_t> seen;
        for (size_t j = 0; j < k; j++)
        {
            idx_t id = ids[j];
            if (id == kUnwritten)
            {
                unwritten = true;
                true_d[j] = std::numeric_limits<double>::infinity();
                continue;
            }
            if (id >= ds.n_base)
            {
                invalid = true;
                true_d[j] = std::numeric_limits<double>::infinity();
                continue;
            }
            if (!seen.insert(id).second)
                dup = true;
            true_d[j] = l2sq_exact(ds.queries.row(q), ds.base.row(id), ds.dim);
            if (true_d[j] > thr)
                wrong = true;
            double rep_d = (double)dd[j];
            double tol = opt.dist_rtol * true_d[j] + opt.dist_atol;
            if (!(std::fabs(rep_d - true_d[j]) <= tol)) // also catches NaN
            {
                baddist = true;
                if (!(std::fabs(rep_d * rep_d - true_d[j]) <= tol))
                    sqrtlike = false;
            }
            if (j > 0 && dd[j] < dd[j - 1] - (opt.dist_rtol * std::fabs(dd[j - 1]) + opt.dist_atol))
                unsorted = true;
        }
        if (!baddist)
            sqrtlike = false;

        size_t hits = 0;
        for (size_t j = 0; j < k; j++)
            if (seen.count(ref_ids[j]))
                hits++;
        double recall = (double)hits / (double)k;
        recall_sum += recall;
        r.recall_min = std::min(r.recall_min, recall);

        std::copy(true_d.begin(), true_d.end(), sorted_d.begin());
        std::sort(sorted_d.begin(), sorted_d.end());
        double kth = sorted_d[k - 1];
        double rel = std::isfinite(kth) ? std::fabs(kth - ref_kth) / std::max(ref_kth, opt.atol)
                                        : std::numeric_limits<double>::infinity();
        r.kth_rel_err_max = std::max(r.kth_rel_err_max, rel);
        bool refsuspect = !invalid && !unwritten && !dup && kth < ref_kth * (1.0 - opt.rtol) - opt.atol;

        bool exact = !invalid && !unwritten && !dup && !wrong;
        r.exact_q += exact;
        r.invalid_q += invalid;
        r.unwritten_q += unwritten;
        r.dup_q += dup;
        r.unsorted_q += unsorted;
        r.baddist_q += baddist;
        r.sqrtdist_q += sqrtlike;
        r.refsuspect_q += refsuspect;

        if ((!exact || refsuspect) && r.details.size() < opt.details)
        {
            std::ostringstream os;
            os << "q=" << q << " k=" << k << ": reference k-th dist " << fmt_sci(ref_kth) << ", returned k-th dist "
               << (std::isfinite(kth) ? fmt_sci(kth) : std::string("inf")) << ", id recall " << fmt(recall, 3);
            if (refsuspect)
                os << " [returned set is closer than the reference: reference suspect]";
            std::vector<std::string> bits;
            for (size_t j = 0; j < k && bits.size() < 6; j++)
            {
                idx_t id = ids[j];
                if (id == kUnwritten)
                    bits.push_back("slot " + std::to_string(j) + " unwritten");
                else if (id >= ds.n_base)
                    bits.push_back("slot " + std::to_string(j) + " invalid id " + std::to_string(id));
                else if (true_d[j] > thr)
                    bits.push_back("wrong id " + std::to_string(id) + " (dist " + fmt_sci(true_d[j]) + ")");
            }
            if (dup)
            {
                std::map<idx_t, int> cnt;
                for (size_t j = 0; j < k; j++)
                    cnt[ids[j]]++;
                for (const auto &c : cnt)
                    if (c.second > 1 && bits.size() < 8)
                        bits.push_back("duplicate id " + std::to_string(c.first) + " x" + std::to_string(c.second));
            }
            // Reference neighbors that are strictly closer than the worst returned one were missed.
            size_t missing_shown = 0;
            for (size_t j = 0; j < k && missing_shown < 3; j++)
            {
                const double rd = ref.dists[q * ref.k + j];
                if (!seen.count(ref_ids[j]) && (!std::isfinite(kth) || rd * (1.0 + opt.rtol) + opt.atol < kth))
                {
                    bits.push_back("missed id " + std::to_string(ref_ids[j]) + " (dist " + fmt_sci(rd) + ")");
                    missing_shown++;
                }
            }
            for (const auto &b : bits)
                os << "; " << b;
            r.details.push_back(os.str());
        }
    }
    r.recall_mean = ds.nq ? recall_sum / (double)ds.nq : 0.0;

    if (r.exact_q < r.nq || r.dup_q || r.invalid_q || r.unwritten_q)
        r.status = "FAIL";
    else if (r.unsorted_q || r.baddist_q)
        r.status = "WARN";
    else
        r.status = "PASS";

    std::vector<std::string> notes;
    if (r.unwritten_q)
        notes.push_back(std::to_string(r.unwritten_q) + " queries with unwritten output slots");
    if (r.invalid_q)
        notes.push_back(std::to_string(r.invalid_q) + " queries with out-of-range ids");
    if (r.dup_q)
        notes.push_back(std::to_string(r.dup_q) + " queries with duplicate ids");
    if (r.baddist_q)
        notes.push_back(std::to_string(r.baddist_q) + " queries whose reported distances differ from true squared L2" +
                        (r.sqrtdist_q ? " (" + std::to_string(r.sqrtdist_q) + " look like non-squared L2)" : ""));
    if (r.unsorted_q)
        notes.push_back(std::to_string(r.unsorted_q) + " queries with unsorted distances");
    if (r.refsuspect_q)
        notes.push_back(std::to_string(r.refsuspect_q) + " queries closer than the reference (reference suspect)");
    for (size_t i = 0; i < notes.size(); i++)
        r.note += (i ? "; " : "") + notes[i];
    return r;
}

// ============================================================================
// Reporting
// ============================================================================

struct RunRow
{
    std::string dataset;
    size_t n_base = 0, dim = 0, nq = 0;
    std::string reference;
    std::string algo;
    KResult res;
};

std::string csv_escape(const std::string &s)
{
    if (s.find_first_of(",\"\n") == std::string::npos)
        return s;
    std::string out = "\"";
    for (char c : s)
        out += (c == '"') ? std::string("\"\"") : std::string(1, c);
    return out + "\"";
}

void append_csv(const std::string &path, const RunRow &row)
{
    bool fresh = !fs::exists(path) || file_size_or_zero(path) == 0;
    std::ofstream out(path, std::ios::app);
    if (!out)
    {
        std::fprintf(stderr, "[validator] warning: cannot write %s\n", path.c_str());
        return;
    }
    if (fresh)
        out << "timestamp,dataset,n_base,dim,n_queries,reference,algorithm,k,status,exact_queries,exact_pct,"
               "recall_mean,recall_min,kth_rel_err_max,dup_queries,invalid_queries,unwritten_queries,"
               "unsorted_queries,bad_distance_queries,reference_suspect_queries,build_s,search_s,qps,note\n";
    const KResult &r = row.res;
    double exact_pct = r.nq ? 100.0 * (double)r.exact_q / (double)r.nq : 0.0;
    double qps = r.search_s > 0 ? (double)r.nq / r.search_s : 0.0;
    out << timestamp() << "," << row.dataset << "," << row.n_base << "," << row.dim << "," << row.nq << ","
        << row.reference << "," << row.algo << "," << r.k << "," << r.status << "," << r.exact_q << ","
        << fmt(exact_pct, 2) << "," << fmt(r.recall_mean, 6) << "," << fmt(r.recall_min, 6) << ","
        << fmt_sci(r.kth_rel_err_max) << "," << r.dup_q << "," << r.invalid_q << "," << r.unwritten_q << ","
        << r.unsorted_q << "," << r.baddist_q << "," << r.refsuspect_q << "," << fmt(r.build_s, 3) << ","
        << fmt(r.search_s, 3) << "," << fmt(qps, 2) << "," << csv_escape(r.note) << "\n";
}

void print_k_line(const std::string &label, const KResult &r)
{
    std::ostringstream os;
    os << "    " << std::left << std::setw(9) << label << " k=" << std::setw(4) << r.k << std::setw(8) << r.status;
    if (r.nq > 0 && (r.status == "PASS" || r.status == "WARN" || r.status == "FAIL"))
    {
        double qps = r.search_s > 0 ? (double)r.nq / r.search_s : 0.0;
        os << "exact " << r.exact_q << "/" << r.nq << "  recall " << fmt(r.recall_mean, 4) << " (min "
           << fmt(r.recall_min, 3) << ")  kth-rel-err " << fmt_sci(r.kth_rel_err_max) << "  build "
           << fmt(r.build_s, 1) << "s  search " << fmt(r.search_s, 2) << "s (" << fmt(qps, 1) << " qps)";
    }
    if (!r.note.empty())
        os << "\n" << std::string(14, ' ') << "note: " << r.note;
    std::printf("%s\n", os.str().c_str());
    for (const auto &d : r.details)
        std::printf("              %s\n", d.c_str());
    std::fflush(stdout);
}

// ============================================================================
// Main driver
// ============================================================================

void print_usage(const char *argv0)
{
    std::printf(
        "Usage: %s [options]\n"
        "\n"
        "Exactness validation of DaiSy in-memory algorithms on large local datasets,\n"
        "against ground truth / brute force. Exit: 0 all PASS, 1 any problem, 77 nothing available.\n"
        "\n"
        "Selection:\n"
        "  --datasets LIST       comma list or 'all' (default all): SIFT1M,GIST1M,T2I10M,HOTPOTQA5M,\n"
        "                        UCASELAW7M,MSMARCOMXBAI8M,DEEP100M,Seismic,Astro\n"
        "  --algos LIST          comma list or 'all' (default all): messi,sofa,fresh,odyssey\n"
        "  --k LIST              k values (default 1,10,100)\n"
        "  --queries N           max queries per dataset (default 1000)\n"
        "  --max-base N          use only the first N base records (0 = all; default 0)\n"
        "  --threads N           threads for algorithms and brute force (default: all cores)\n"
        "\n"
        "Paths:\n"
        "  --vector-root DIR     fvecs datasets root (default /mnt/hddhelp/mchatzakis/datasets/processed)\n"
        "  --series-root DIR     data-series root (default /mnt/hddhelp/mchatzakis/datasets/similarity-search-datasets)\n"
        "  --work-dir DIR        caches and logs (default ./validate_large_datasets_work)\n"
        "  --csv FILE            append per-run rows here (default <work-dir>/results.csv)\n"
        "\n"
        "Reference and tolerances:\n"
        "  --reference MODE      auto (GT file when valid, else brute force) | bruteforce (default auto)\n"
        "  --gt-verify N         queries brute-forced to validate a GT file (default 10, 0 = trust GT)\n"
        "  --rtol X --atol X     exactness tolerance on the k-th distance (default 1e-4, 1e-5)\n"
        "  --dist-rtol X --dist-atol X  tolerance for reported vs true distances (default 1e-3, 1e-3)\n"
        "\n"
        "Algorithm configuration:\n"
        "  --bp gaussian|equidepth  breakpoint declaration via setNormalized (default gaussian = library default)\n"
        "  --set ALGO.KNOB=VALUE    override a knob, repeatable (e.g. --set messi.leaf_size=5000)\n"
        "  --list-knobs             print the accepted knobs per algorithm and exit\n"
        "\n"
        "Execution:\n"
        "  --timeout SEC         kill an algorithm run after SEC seconds (default 0 = no limit)\n"
        "  --no-fork             run algorithms in-process (for gdb; a crash then stops the validator)\n"
        "  --verbose             let algorithm output through instead of writing <work-dir>/logs\n"
        "  --force               run Odyssey even if the memory pre-check fails\n"
        "  --details N           failing queries printed per run (default 5)\n"
        "  --list                print the resolved datasets and exit\n"
        "  -h, --help\n",
        argv0);
}

Options parse_args(int argc, char **argv)
{
    Options opt;
    std::string datasets = "all", algos = "all";
    bool list_knobs = false;
    auto need = [&](int &i) -> std::string {
        if (i + 1 >= argc)
            die(std::string("missing value for ") + argv[i]);
        return argv[++i];
    };
    try
    {
        for (int i = 1; i < argc; i++)
        {
            std::string a = argv[i];
            if (a == "-h" || a == "--help")
            {
                print_usage(argv[0]);
                std::exit(0);
            }
            else if (a == "--datasets")
                datasets = need(i);
            else if (a == "--algos")
                algos = need(i);
            else if (a == "--k")
            {
                opt.ks.clear();
                for (const auto &s : split(need(i), ','))
                    opt.ks.push_back((size_t)parse_int("--k", s));
            }
            else if (a == "--queries")
                opt.n_queries = (size_t)parse_int(a, need(i));
            else if (a == "--max-base")
                opt.max_base = (size_t)parse_int(a, need(i));
            else if (a == "--threads")
                opt.threads = (unsigned)parse_int(a, need(i));
            else if (a == "--vector-root")
                opt.vector_root = need(i);
            else if (a == "--series-root")
                opt.series_root = need(i);
            else if (a == "--work-dir")
                opt.work_dir = need(i);
            else if (a == "--csv")
                opt.csv_path = need(i);
            else if (a == "--reference")
                opt.reference = to_lower(need(i));
            else if (a == "--gt-verify")
                opt.gt_verify = (size_t)parse_int(a, need(i));
            else if (a == "--rtol")
                opt.rtol = parse_double(a, need(i));
            else if (a == "--atol")
                opt.atol = parse_double(a, need(i));
            else if (a == "--dist-rtol")
                opt.dist_rtol = parse_double(a, need(i));
            else if (a == "--dist-atol")
                opt.dist_atol = parse_double(a, need(i));
            else if (a == "--bp")
            {
                std::string v = to_lower(need(i));
                if (v != "gaussian" && v != "equidepth")
                    die("--bp expects gaussian or equidepth");
                opt.equidepth = (v == "equidepth");
            }
            else if (a == "--set")
            {
                std::string v = need(i);
                size_t dot = v.find('.'), eq = v.find('=');
                if (dot == std::string::npos || eq == std::string::npos || eq < dot)
                    die("--set expects ALGO.KNOB=VALUE, got '" + v + "'");
                opt.knobs[to_lower(v.substr(0, dot))][v.substr(dot + 1, eq - dot - 1)] = v.substr(eq + 1);
            }
            else if (a == "--list-knobs")
                list_knobs = true;
            else if (a == "--timeout")
                opt.timeout_s = parse_double(a, need(i));
            else if (a == "--no-fork")
                opt.fork_children = false;
            else if (a == "--verbose")
                opt.verbose_children = true;
            else if (a == "--force")
                opt.force = true;
            else if (a == "--details")
                opt.details = (size_t)parse_int(a, need(i));
            else if (a == "--list")
                opt.list_only = true;
            else
                die("unknown option '" + a + "' (see --help)");
        }
    }
    catch (const std::invalid_argument &e)
    {
        die(e.what());
    }

    if (list_knobs)
    {
        for (const auto &d : drivers())
        {
            std::printf("%-9s", d.key.c_str());
            if (!d.available)
                std::printf(" (%s)", d.unavailable_reason.c_str());
            for (const auto &k : d.knobs)
                std::printf(" %s", k.c_str());
            std::printf("\n");
        }
        std::exit(0);
    }

    if (to_lower(datasets) == "all")
        for (const auto &d : catalog())
            opt.datasets.push_back(d.name);
    else
        for (const auto &s : split(datasets, ','))
        {
            const DatasetSpec *d = find_dataset(s);
            if (!d)
                die("unknown dataset '" + s + "' (see --help)");
            opt.datasets.push_back(d->name);
        }

    if (to_lower(algos) == "all")
        for (const auto &d : drivers())
            opt.algos.push_back(d.key);
    else
        for (const auto &s : split(algos, ','))
        {
            const AlgoDriver *d = find_driver(s);
            if (!d)
                die("unknown algorithm '" + s + "' (see --help)");
            opt.algos.push_back(d->key);
        }

    for (const auto &entry : opt.knobs)
    {
        const AlgoDriver *d = find_driver(entry.first);
        if (!d)
            die("--set refers to unknown algorithm '" + entry.first + "'");
        for (const auto &kv : entry.second)
            if (d->available && std::find(d->knobs.begin(), d->knobs.end(), kv.first) == d->knobs.end())
                die("unknown knob '" + entry.first + "." + kv.first + "' (see --list-knobs)");
    }

    if (opt.ks.empty() || opt.ks.size() > (size_t)kMaxK)
        die("--k needs between 1 and " + std::to_string(kMaxK) + " values");
    for (size_t k : opt.ks)
        if (k == 0)
            die("--k values must be positive");
    if (opt.n_queries == 0)
        die("--queries must be positive");
    if (opt.threads == 0)
        die("--threads must be positive");
    if (opt.reference != "auto" && opt.reference != "bruteforce")
        die("--reference expects auto or bruteforce");
    if (opt.csv_path.empty())
        opt.csv_path = (fs::path(opt.work_dir) / "results.csv").string();
    return opt;
}

void list_datasets(const Options &opt)
{
    std::printf("%-15s %-6s %12s %6s %8s  %-28s %s\n", "dataset", "format", "base_rows", "dim", "queries",
                "ground truth", "base file");
    for (const auto &name : opt.datasets)
    {
        const DatasetSpec &spec = *find_dataset(name);
        try
        {
            DatasetInfo info = probe_dataset(opt, spec);
            std::string gt = spec.series ? "none (brute force)"
                                         : info.gt_present ? std::to_string(info.gt_rows) + " rows x k=" +
                                                                 std::to_string(info.gt_len)
                                                           : "none (brute force)";
            size_t q = spec.series ? std::min(opt.n_queries, info.n_total / 2)
                                   : std::min(opt.n_queries, info.n_queries_avail);
            size_t base_rows = spec.series ? info.n_total - q : info.n_total;
            std::printf("%-15s %-6s %12zu %6zu %8zu  %-28s %s\n", spec.name.c_str(),
                        spec.format == Format::FVECS ? "fvecs" : "f32", base_rows, info.dim, q, gt.c_str(),
                        info.base_path.c_str());
            for (const auto &w : info.warnings)
                std::printf("%15s warning: %s\n", "", w.c_str());
        }
        catch (const std::exception &e)
        {
            std::printf("%-15s unavailable: %s\n", spec.name.c_str(), e.what());
        }
    }
    std::printf("\nalgorithms:");
    for (const auto &d : drivers())
        std::printf(" %s%s", d.label.c_str(), d.available ? "" : "(not built)");
    std::printf("\n");
}

KResult skip_result(size_t k, const std::string &status, const std::string &note)
{
    KResult r;
    r.k = k;
    r.status = status;
    r.note = note;
    return r;
}

} // namespace

int main(int argc, char **argv)
{
    Options opt = parse_args(argc, argv);

    if (opt.list_only)
    {
        list_datasets(opt);
        return 0;
    }

    if (opt.fork_children && (std::getenv("OMPI_COMM_WORLD_SIZE") || std::getenv("PMI_RANK") ||
                              std::getenv("PMIX_RANK")))
        die("do not launch the validator under mpirun: Odyssey runs as an MPI singleton inside a forked child. "
            "Run the binary directly (or use --no-fork).");

    fs::create_directories(opt.work_dir);
    fs::create_directories(fs::path(opt.work_dir) / "logs");
    if (!fs::path(opt.csv_path).parent_path().empty())
        fs::create_directories(fs::path(opt.csv_path).parent_path());

    const size_t k_max = *std::max_element(opt.ks.begin(), opt.ks.end());

    log_line("DaiSy large-dataset exactness validation");
    {
        std::string ks, ds, al;
        for (size_t k : opt.ks)
            ks += (ks.empty() ? "" : ",") + std::to_string(k);
        for (const auto &d : opt.datasets)
            ds += (ds.empty() ? "" : ",") + d;
        for (const auto &a : opt.algos)
            al += (al.empty() ? "" : ",") + a;
        log_line("  datasets: " + ds);
        log_line("  algorithms: " + al);
        log_line("  k: " + ks + "  queries: " + std::to_string(opt.n_queries) + "  threads: " +
                 std::to_string(opt.threads) + "  breakpoints: " + (opt.equidepth ? "equidepth" : "gaussian"));
        log_line("  work dir: " + fs::absolute(opt.work_dir).string() + "  csv: " + opt.csv_path);
        for (const auto &entry : opt.knobs)
            for (const auto &kv : entry.second)
                log_line("  knob override: " + entry.first + "." + kv.first + "=" + kv.second);
    }

    std::vector<RunRow> rows;
    bool any_problem = false;

    for (const auto &ds_name : opt.datasets)
    {
        const DatasetSpec &spec = *find_dataset(ds_name);
        std::printf("\n");
        log_line("==== " + spec.name + " ====");

        LoadedDataset ds;
        Reference ref;
        std::string ds_error;
        try
        {
            auto t0 = Clock::now();
            load_dataset(opt, spec, ds);
            log_line("  loaded in " + fmt(seconds_since(t0), 1) + " s: n_base=" + std::to_string(ds.n_base) +
                     " dim=" + std::to_string(ds.dim) + " queries=" + std::to_string(ds.nq));
            if (k_max > ds.n_base)
                throw std::runtime_error("k_max exceeds the base size");
            ref = build_reference(opt, ds, k_max);
        }
        catch (const std::exception &e)
        {
            ds_error = e.what();
        }

        for (const auto &algo_key : opt.algos)
        {
            const AlgoDriver &drv = *find_driver(algo_key);
            auto emit = [&](const KResult &r) {
                RunRow row{spec.name, ds.n_base, ds.dim, ds.nq, ref.source, drv.label, r};
                print_k_line(drv.label, r);
                append_csv(opt.csv_path, row);
                rows.push_back(row);
                if (r.status != "PASS" && r.status != "SKIP")
                    any_problem = true;
            };
            auto emit_all = [&](const std::string &status, const std::string &note) {
                for (size_t k : opt.ks)
                    emit(skip_result(k, status, note));
            };

            if (!ds_error.empty())
            {
                log_line("  dataset unavailable: " + ds_error);
                emit_all("SKIP", "dataset unavailable: " + ds_error);
                continue;
            }
            if (!drv.available)
            {
                emit_all("SKIP", drv.unavailable_reason);
                continue;
            }

            // Resource pre-checks for algorithms that duplicate the base.
            if (drv.copies_base_in_ram && !opt.force)
            {
                size_t need = ds.base.bytes() + ds.base.bytes() / 5;
                size_t avail = mem_available_bytes();
                if (avail && need > avail)
                {
                    emit_all("SKIP", "needs ~" + human_bytes((double)need) + " RAM for its private base copy, only " +
                                         human_bytes((double)avail) + " available (use --max-base or --force)");
                    continue;
                }
            }
            if (drv.builds_from_file)
            {
                try
                {
                    ensure_raw_base_file(opt, ds);
                }
                catch (const std::exception &e)
                {
                    emit_all("ERROR", std::string("cannot prepare raw base file: ") + e.what());
                    continue;
                }
            }

            RunContext ctx;
            ctx.ds = &ds;
            ctx.threads = opt.threads;
            ctx.equidepth = opt.equidepth;
            if (opt.knobs.count(drv.key))
                ctx.knobs = opt.knobs.at(drv.key);
            const std::string log_path =
                (fs::absolute(opt.work_dir) / "logs" / (spec.name + "__" + drv.key + ".log")).string();

            ResultLayout lay(ds.nq, opt.ks);
            void *mem = mmap(nullptr, lay.total, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
            if (mem == MAP_FAILED)
            {
                emit_all("ERROR", "mmap of the result area failed");
                continue;
            }
            char *region = static_cast<char *>(mem);
            std::memset(region, 0, sizeof(ChildReport));

            log_line("  running " + drv.label + (opt.fork_children && !opt.verbose_children ? " (log: " + log_path + ")" : ""));
            if (opt.equidepth && !drv.honors_breakpoint_mode)
                log_line("    note: " + drv.label + " does not consume the breakpoint mode; --bp equidepth has no effect");
            ChildOutcome outc = run_isolated(opt, drv, ctx, lay, region, log_path);
            const ChildReport *rep = reinterpret_cast<const ChildReport *>(region);

            std::string event;
            std::string fatal_status;
            switch (outc.kind)
            {
            case ChildOutcome::COMPLETED:
                break;
            case ChildOutcome::CRASHED:
                fatal_status = "CRASH";
                event = std::string("killed by signal ") + std::to_string(outc.code) + " (" + strsignal(outc.code) +
                        ") during " + phase_name(rep->phase) +
                        (outc.code == SIGKILL ? " (SIGKILL without a timeout: most likely the kernel OOM killer)" : "");
                break;
            case ChildOutcome::EXITED:
                fatal_status = "CRASH";
                event = "process exited with status " + std::to_string(outc.code) + " during " +
                        phase_name(rep->phase) + " (exit() called inside the library?)";
                break;
            case ChildOutcome::TIMED_OUT:
                fatal_status = "TIMEOUT";
                event = "timed out after " + fmt(outc.wall_seconds, 0) + " s during " + phase_name(rep->phase);
                break;
            case ChildOutcome::FORK_FAILED:
                fatal_status = "ERROR";
                event = std::string("fork failed: ") + std::strerror(outc.code);
                break;
            }
            if (rep->phase == PH_SEARCH && !event.empty())
                event += " k=" + std::to_string(opt.ks[(size_t)std::max(0, (int)rep->k_index)]);

            if (rep->configure_failed == 1)
                emit_all("SKIP", std::string("configuration not supported: ") + rep->message);
            else if (rep->configure_failed == 2)
                emit_all("ERROR", std::string("invalid configuration: ") + rep->message);
            else if (rep->build_failed)
                emit_all("ERROR", std::string("buildIndex threw: ") + rep->message);
            else
            {
                for (size_t i = 0; i < opt.ks.size(); i++)
                {
                    const size_t k = opt.ks[i];
                    KResult r;
                    if (rep->search_state[i] == SS_OK)
                    {
                        r = validate_k(opt, ds, ref, k, reinterpret_cast<const idx_t *>(region + lay.i_off[i]),
                                       reinterpret_cast<const float *>(region + lay.d_off[i]));
                        if (!event.empty())
                        {
                            r.note += std::string(r.note.empty() ? "" : "; ") + "later " + event;
                            if (r.status == "PASS")
                                r.status = "WARN";
                        }
                    }
                    else if (rep->search_state[i] == SS_EXCEPTION)
                        r = skip_result(k, "ERROR", std::string("searchIndex threw: ") + rep->k_message[i]);
                    else
                        r = skip_result(k, fatal_status.empty() ? "ERROR" : fatal_status,
                                        event.empty() ? "search not executed" : event);
                    r.build_s = rep->build_seconds;
                    r.search_s = rep->search_seconds[i];
                    if (outc.kind == ChildOutcome::COMPLETED && (rep->base_mutated || rep->queries_mutated))
                    {
                        r.note += std::string(r.note.empty() ? "" : "; ") + "algorithm modified the caller's " +
                                  (rep->base_mutated ? "base" : "query") + " buffer";
                        if (r.status == "PASS")
                            r.status = "WARN";
                    }
                    emit(r);
                }
            }

            if (!event.empty() && opt.fork_children && !opt.verbose_children)
            {
                std::printf("    last lines of %s:\n", log_path.c_str());
                for (const auto &l : tail_lines(log_path, 15))
                    std::printf("      | %s\n", l.c_str());
                std::fflush(stdout);
            }

            munmap(mem, lay.total);
        }
    }

    // ---- Summary ----
    std::printf("\n");
    log_line("==== Summary ====");
    std::printf("%-15s %-9s", "dataset", "algorithm");
    for (size_t k : opt.ks)
        std::printf("  %-16s", ("k=" + std::to_string(k)).c_str());
    std::printf("\n");
    for (size_t i = 0; i < rows.size();)
    {
        const std::string &dsn = rows[i].dataset;
        const std::string &al = rows[i].algo;
        std::printf("%-15s %-9s", dsn.c_str(), al.c_str());
        while (i < rows.size() && rows[i].dataset == dsn && rows[i].algo == al)
        {
            const KResult &r = rows[i].res;
            std::string cell = r.status;
            if (r.nq > 0 && (r.status == "PASS" || r.status == "WARN" || r.status == "FAIL"))
                cell += " " + fmt(100.0 * (double)r.exact_q / (double)r.nq, 1) + "%";
            std::printf("  %-16s", cell.c_str());
            i++;
        }
        std::printf("\n");
    }
    std::printf("\nexact%% = queries whose returned ids are all within the reference k-th distance.\n");
    std::printf("Details per run: %s ; algorithm logs: %s\n", opt.csv_path.c_str(),
                (fs::absolute(opt.work_dir) / "logs").string().c_str());
    std::fflush(stdout);

    if (!opt.fork_children)
        finalize_mpi_if_needed();
    if (any_problem)
        return 1;
    const bool nothing_ran = std::all_of(rows.begin(), rows.end(),
                                         [](const RunRow &r) { return r.res.status == "SKIP"; });
    if (nothing_ran)
    {
        std::printf("Nothing could run on this machine (all datasets/algorithms unavailable): reporting SKIP.\n");
        return 77; // CTest SKIP_RETURN_CODE
    }
    return 0;
}
