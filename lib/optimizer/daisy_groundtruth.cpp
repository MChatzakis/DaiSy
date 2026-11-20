// daisy_groundtruth.cpp
//
// Exact k-nearest-neighbour ground truth of a query file against a base file, computed with
// DaiSy's MESSI. Used by scripts/datasets/prepare_datasets.py.
//
// Inputs and outputs are .fvecs/.ivecs files (each row: int32 dimension, then the values).
// Writes <out>.ivecs (neighbour ids, 0-based base rows) and <out>.fvecs, one row of k per query:
//   --metric l2  squared L2 distances, ascending (the default)
//   --metric ip  inner products, descending. MESSI searches squared L2, so every base vector
//                x gets the extra coordinate sqrt(M^2 - |x|^2) (M: the largest base norm) and
//                every query a 0: the squared distance |q|^2 + M^2 - 2 q.x is then smallest for
//                the largest q.x (an exact reduction).
// The written values are recomputed exactly (double precision) for the returned ids.
//
// Every answer is checked: each id's lifted squared distance must be the one MESSI reported,
// ids must be distinct, and the first --check queries of the run are also answered by brute
// force and must get the same values. Any mismatch exits with code 1.
//
// Queries are answered in batches of --batch; finished batches go to <out>.ivecs.partial and
// <out>.fvecs.partial, so a stopped run continues after them when started again.
//
// Usage:
//   daisy_groundtruth --base base.fvecs --queries query.fvecs --k 5000 --out PREFIX
//                     [--metric l2|ip] [--nq N] [--threads T] [--check 10] [--batch 1000]

#include "algos/Messi.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <omp.h>

namespace fs = std::filesystem;
using daisy::idx_t;
using Clock = std::chrono::steady_clock;

// Reads the first `limit` rows (0 = all) of an .fvecs file into rows of `width` floats
// (width >= the file's dimension; extra columns are 0).
std::vector<float> read_fvecs(const std::string &path, size_t limit, size_t width_extra, size_t &rows, size_t &dim)
{
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f)
        throw std::runtime_error("cannot open " + path);
    int32_t d = 0;
    if (std::fread(&d, 4, 1, f) != 1 || d <= 0)
        throw std::runtime_error("bad header in " + path);
    dim = (size_t)d;
    rows = fs::file_size(path) / (4 + 4 * dim);
    if (limit > 0)
        rows = std::min(rows, limit);
    const size_t width = dim + width_extra;
    std::vector<float> out(rows * width, 0.0f);
    std::fseek(f, 0, SEEK_SET);
    for (size_t i = 0; i < rows; i++)
    {
        if (std::fread(&d, 4, 1, f) != 1 || (size_t)d != dim || std::fread(&out[i * width], 4, dim, f) != dim)
            throw std::runtime_error("bad or short row " + std::to_string(i) + " in " + path);
    }
    std::fclose(f);
    return out;
}

size_t vecs_rows(const std::string &path, size_t dim)
{
    return fs::exists(path) ? fs::file_size(path) / (4 + 4 * dim) : 0;
}

template <typename T>
void append_vecs(const std::string &path, const T *values, size_t rows, size_t dim)
{
    FILE *f = std::fopen(path.c_str(), "ab");
    if (!f)
        throw std::runtime_error("cannot write " + path);
    const int32_t d = (int32_t)dim;
    for (size_t i = 0; i < rows; i++)
        if (std::fwrite(&d, 4, 1, f) != 1 || std::fwrite(&values[i * dim], sizeof(T), dim, f) != dim)
            throw std::runtime_error("cannot write " + path);
    if (std::fclose(f) != 0)
        throw std::runtime_error("cannot write " + path);
}

double squared_l2(const float *a, const float *b, size_t dim)
{
    double s = 0;
    for (size_t j = 0; j < dim; j++)
    {
        const double diff = (double)a[j] - (double)b[j];
        s += diff * diff;
    }
    return s;
}

double dot(const float *a, const float *b, size_t dim)
{
    double s = 0;
    for (size_t j = 0; j < dim; j++)
        s += (double)a[j] * (double)b[j];
    return s;
}

bool close(double a, double b, double scale) { return std::fabs(a - b) <= 1e-3 * std::max({1.0, std::fabs(b), scale}); }

int main(int argc, char **argv)
{
    std::string base_path, query_path, out, metric = "l2";
    size_t k = 100, nq_limit = 0, check = 10, batch = 1000;
    int threads = (int)std::max(1u, std::thread::hardware_concurrency());
    for (int i = 1; i + 1 < argc; i += 2)
    {
        const std::string key = argv[i], val = argv[i + 1];
        if (key == "--base") base_path = val;
        else if (key == "--queries") query_path = val;
        else if (key == "--out") out = val;
        else if (key == "--metric") metric = val;
        else if (key == "--k") k = std::stoul(val);
        else if (key == "--nq") nq_limit = std::stoul(val);
        else if (key == "--threads") threads = std::stoi(val);
        else if (key == "--check") check = std::stoul(val);
        else if (key == "--batch") batch = std::max<size_t>(1, std::stoul(val));
        else
        {
            std::fprintf(stderr, "unknown option %s\n", key.c_str());
            return 2;
        }
    }
    if (base_path.empty() || query_path.empty() || out.empty() || argc % 2 == 0 || (metric != "l2" && metric != "ip"))
    {
        std::fprintf(stderr, "usage: daisy_groundtruth --base B.fvecs --queries Q.fvecs --k K --out PREFIX "
                             "[--metric l2|ip] [--nq N] [--threads T] [--check C] [--batch B]\n");
        return 2;
    }
    const bool ip = metric == "ip";

    try
    {
        auto t0 = Clock::now();
        size_t n = 0, d = 0, nq = 0, dq = 0;
        std::vector<float> base = read_fvecs(base_path, 0, ip ? 1 : 0, n, d);
        std::vector<float> queries = read_fvecs(query_path, nq_limit, ip ? 1 : 0, nq, dq);
        if (d != dq)
            throw std::runtime_error("base and queries have different dimensions");
        if (k > n)
            throw std::runtime_error("k is larger than the base");
        const size_t w = ip ? d + 1 : d; // indexed width
        double max_sq = 0;
        if (ip)
        {
            std::vector<double> norms(n);
#pragma omp parallel for num_threads(threads) reduction(max : max_sq)
            for (size_t i = 0; i < n; i++)
            {
                norms[i] = dot(&base[i * w], &base[i * w], d);
                max_sq = std::max(max_sq, norms[i]);
            }
#pragma omp parallel for num_threads(threads)
            for (size_t i = 0; i < n; i++)
                base[i * w + d] = (float)std::sqrt(std::max(0.0, max_sq - norms[i]));
        }
        std::printf("read %zu x %zu base and %zu queries in %.1f s (metric %s)\n", n, d, nq,
                    std::chrono::duration<double>(Clock::now() - t0).count(), metric.c_str());

        const std::string ids_part = out + ".ivecs.partial", values_part = out + ".fvecs.partial";
        size_t done = std::min(vecs_rows(ids_part, k), vecs_rows(values_part, k));
        if (done > 0)
        {
            fs::resize_file(ids_part, done * (4 + 4 * k)); // drop a half-written batch, if any
            fs::resize_file(values_part, done * (4 + 4 * k));
            std::printf("resuming after %zu queries already in %s\n", done, ids_part.c_str());
        }
        else
        {
            fs::remove(ids_part);
            fs::remove(values_part);
        }

        // MESSI with data-adaptive breakpoints: exact for any data, and much faster than the
        // Gaussian ones on vectors that are not z-normalized.
        auto t1 = Clock::now();
        daisy::Messi messi(daisy::DistanceType::L2_SQUARED);
        messi.setIndexWorkers(threads);
        messi.setSearchWorkers(threads);
        messi.setNormalized(false);
        messi.buildIndex(base.data(), n, w);
        std::printf("built MESSI with %d threads in %.1f s\n", threads,
                    std::chrono::duration<double>(Clock::now() - t1).count());

        const size_t first = done;
        std::vector<idx_t> I;
        std::vector<float> D;
        std::vector<int32_t> ids;
        std::vector<float> values;
        auto t2 = Clock::now();
        for (size_t start = first; start < nq; start += batch)
        {
            const size_t m = std::min(batch, nq - start);
            I.assign(m * k, 0);
            D.assign(m * k, 0.0f);
            messi.searchIndex(&queries[start * w], m, k, I.data(), D.data());

            // Exact values for the returned ids, sorted; and the checks.
            ids.assign(m * k, 0);
            values.assign(m * k, 0.0f);
            bool ok = true;
#pragma omp parallel for num_threads(threads) reduction(&& : ok)
            for (size_t q = 0; q < m; q++)
            {
                const float *query = &queries[(start + q) * w];
                std::vector<std::pair<double, idx_t>> row(k);
                for (size_t j = 0; j < k; j++)
                {
                    const idx_t id = I[q * k + j];
                    if (id >= n)
                    {
                        ok = false;
                        break;
                    }
                    ok = ok && close(squared_l2(query, &base[id * w], w), D[q * k + j], ip ? max_sq : 0.0);
                    row[j] = {ip ? -dot(query, &base[id * w], d) : squared_l2(query, &base[id * w], d), id};
                }
                std::sort(row.begin(), row.end());
                for (size_t j = 0; j < k; j++)
                {
                    ok = ok && (j == 0 || row[j].second != row[j - 1].second || row[j].first != row[j - 1].first);
                    ids[q * k + j] = (int32_t)row[j].second;
                    values[q * k + j] = (float)(ip ? -row[j].first : row[j].first);
                }
                std::vector<idx_t> seen(I.begin() + q * k, I.begin() + (q + 1) * k);
                std::sort(seen.begin(), seen.end());
                ok = ok && std::adjacent_find(seen.begin(), seen.end()) == seen.end();
            }
            if (!ok)
                throw std::runtime_error("MESSI returned an id out of range, a wrong distance, or a repeated id");

            // Brute force on the first `check` queries of this run.
            for (size_t q = 0; start == first && q < std::min(check, m); q++)
            {
                const float *query = &queries[(start + q) * w];
                std::vector<double> all(n);
#pragma omp parallel for num_threads(threads)
                for (size_t i = 0; i < n; i++)
                    all[i] = ip ? -dot(query, &base[i * w], d) : squared_l2(query, &base[i * w], d);
                std::nth_element(all.begin(), all.begin() + (k - 1), all.end());
                std::sort(all.begin(), all.begin() + k);
                for (size_t j = 0; j < k; j++)
                {
                    const double expected = ip ? -all[j] : all[j];
                    if (!close(values[q * k + j], expected, 0.0))
                        throw std::runtime_error("MESSI differs from brute force on query " + std::to_string(start + q) +
                                                 ", rank " + std::to_string(j));
                }
            }

            append_vecs(ids_part, ids.data(), m, k);
            append_vecs(values_part, values.data(), m, k);
            const double s = std::chrono::duration<double>(Clock::now() - t2).count();
            const size_t finished = start + m - first;
            std::printf("queries %zu-%zu of %zu: %.2f s per query, about %.0f s left\n", start, start + m, nq,
                        s / finished, s / finished * (nq - start - m));
            std::fflush(stdout);
        }
        fs::rename(ids_part, out + ".ivecs");
        fs::rename(values_part, out + ".fvecs");
        std::printf("checked every answer, and the first %zu of this run against brute force; wrote %s.ivecs and "
                    "%s.fvecs (%.1f s in total)\n",
                    std::min(check, nq - first), out.c_str(), out.c_str(),
                    std::chrono::duration<double>(Clock::now() - t0).count());
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
