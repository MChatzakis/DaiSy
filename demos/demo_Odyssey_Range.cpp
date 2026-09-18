// Odyssey range search: report every series within squared-L2 distance r of a query.
// Odyssey is MPI-distributed: rank 0 stages the data on disk, every rank indexes and
// searches it, and rank 0 reports the merged result.

#include "../commons/dataloaders.hpp"
#include "../lib/daisy.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#if ODYSSEY_MPI
#include <mpi.h>
#endif

#if defined(__unix__) || defined(__APPLE__)
#include <limits.h>
#include <stdlib.h>
#endif

static void reportRange(const char *name, float r,
                        const std::vector<std::vector<daisy::idx_t>> &I,
                        const std::vector<std::vector<float>> &D)
{
    printf("=== %s range search (squared L2 <= %.1f) ===\n", name, r);
    for (size_t qi = 0; qi < I.size(); qi++)
    {
        printf("Query %zu: %zu hits", qi, I[qi].size());
        if (!I[qi].empty())
        {
            const auto minmax = std::minmax_element(D[qi].begin(), D[qi].end());
            printf("  [closest=%.4f, farthest=%.4f]", *minmax.first, *minmax.second);
        }
        printf("\n");
    }
}

// Range search returns an unordered set of hits, so compare sets rather than positions.
// "missing" are hits brute force found and the index did not; "extra" are the other way
// round, which for an exact index means false positives.
static bool matchesBruteforce(const char *name,
                              const std::vector<std::vector<daisy::idx_t>> &I,
                              const std::vector<std::vector<daisy::idx_t>> &gt_I)
{
    bool all_match = true;
    printf("\n=== Verification against brute force ===\n");
    for (size_t qi = 0; qi < I.size(); qi++)
    {
        const std::set<daisy::idx_t> got(I[qi].begin(), I[qi].end());
        const std::set<daisy::idx_t> expected(gt_I[qi].begin(), gt_I[qi].end());

        size_t missing = 0;
        for (daisy::idx_t idx : expected)
            if (got.count(idx) == 0) missing++;
        size_t extra = 0;
        for (daisy::idx_t idx : got)
            if (expected.count(idx) == 0) extra++;

        all_match = all_match && (missing == 0 && extra == 0);
        printf("Query %zu: %s=%zu, bruteforce=%zu, missing=%zu, extra=%zu\n",
               qi, name, got.size(), expected.size(), missing, extra);
    }
    return all_match;
}

int main(int argc, char *argv[])
{
    daisy::idx_t n_database = 200000;
    unsigned long long dim = 96;
    unsigned long long n_query = 10;

    // Independent z-normalized series sit around 2*dim apart, so a radius below that
    // keeps the result sets small without leaving every query empty.
    const float r = 0.60f * 2.0f * dim;

    std::string temp_db_file = "odyssey_range_db.bin";

    daisy::OdysseyConfig odyssey_config;
    odyssey_config.search_workers = 2;
    odyssey_config.index_threads = 4;
    odyssey_config.query_threads = 2;
    odyssey_config.leaf_size = 1000;
    odyssey_config.paa_segments = 16;
    odyssey_config.replication_groups = 0;

    daisy::Odyssey odyssey(odyssey_config, daisy::DistanceType::L2_SQUARED, argc, argv);
    int rank = odyssey.getMyRank();

    float *database = nullptr;
    if (rank == 0)
    {
        remove(temp_db_file.c_str());
        database = loadRandomData(n_database, dim, 100, true);
        printf("Loaded %llu database points with dimension %llu\n", n_database, dim);

        FILE *fp = fopen(temp_db_file.c_str(), "wb");
        if (fp == nullptr)
        {
            fprintf(stderr, "Error: Could not create temporary database file\n");
            delete[] database;
            return 1;
        }
        size_t to_write = static_cast<size_t>(n_database) * static_cast<size_t>(dim);
        size_t written = fwrite(database, sizeof(float), to_write, fp);
        fclose(fp);
        if (written != to_write)
        {
            fprintf(stderr, "Error: wrote only %zu floats (expected %zu)\n", written, to_write);
            delete[] database;
            return 1;
        }
    }

#if ODYSSEY_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    static const int PATH_MAX_MPI = 1024;
    char path_buf[PATH_MAX_MPI];
    std::memset(path_buf, 0, PATH_MAX_MPI);
    if (rank == 0)
    {
#if (defined(__unix__) || defined(__APPLE__)) && defined(PATH_MAX)
        char resolved[PATH_MAX];
        if (realpath(temp_db_file.c_str(), resolved) != nullptr)
            std::strncpy(path_buf, resolved, PATH_MAX_MPI - 1);
        else
#endif
            std::strncpy(path_buf, temp_db_file.c_str(), PATH_MAX_MPI - 1);
        path_buf[PATH_MAX_MPI - 1] = '\0';
    }
#if ODYSSEY_MPI
    MPI_Bcast(path_buf, PATH_MAX_MPI, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif
    std::string path_to_use(path_buf);

    float *query = loadRandomData(n_query, dim, 50, true);
    if (rank == 0)
        printf("Loaded %llu query points with dimension %llu\n", n_query, dim);

    daisy::FileDataSource data_source(path_to_use.c_str(), dim, n_database);
    odyssey.buildIndex(&data_source);
    if (rank == 0)
        printf(">>> Finished indexing\n");

    daisy::SearchConfig config;
    config.type = daisy::QueryType::RANGE;
    config.r = r;

    std::vector<std::vector<daisy::idx_t>> I;
    std::vector<std::vector<float>> D;
    odyssey.searchIndex(query, n_query, config, I, D);

    if (rank == 0)
    {
        reportRange("Odyssey", r, I, D);

        daisy::BruteForceSearch ground_truth(daisy::DistanceType::L2_SQUARED);
        ground_truth.buildIndex(database, n_database, dim);
        std::vector<std::vector<daisy::idx_t>> gt_I;
        std::vector<std::vector<float>> gt_D;
        ground_truth.searchIndex(query, n_query, config, gt_I, gt_D);

        printf("\nAll queries match: %s\n",
               matchesBruteforce("Odyssey", I, gt_I) ? "yes" : "no");
    }

    delete[] database;
    delete[] query;
    if (rank == 0)
        remove(path_to_use.c_str());

#if ODYSSEY_MPI
    MPI_Finalize();
#endif

    return 0;
}
