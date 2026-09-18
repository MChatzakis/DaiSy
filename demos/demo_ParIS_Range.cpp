// ParIS range search: report every series within squared-L2 distance r of a query.
// ParIS indexes from a file, so the generated data is staged on disk first.

#include "../commons/dataloaders.hpp"
#include "../lib/daisy.hpp"

#include <algorithm>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

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

int main()
{
    daisy::idx_t n_database = 200000;
    unsigned long long dim = 96;
    unsigned long long n_query = 10;

    // Independent z-normalized series sit around 2*dim apart, so a radius below that
    // keeps the result sets small without leaving every query empty.
    const float r = 0.60f * 2.0f * dim;

    float *database = loadRandomData(n_database, dim, 100, true);
    float *query = loadRandomData(n_query, dim, 50, true);

    printf("Loaded %llu database points and %llu query points with dimension %llu\n",
           n_database, n_query, dim);

    std::string temp_db_file = "/tmp/paris_range_db.bin";
    FILE *fp = fopen(temp_db_file.c_str(), "wb");
    if (fp == nullptr) {
        fprintf(stderr, "Error: Could not create temporary database file\n");
        delete[] database;
        delete[] query;
        return 1;
    }
    fwrite(database, sizeof(float), n_database * dim, fp);
    fclose(fp);

    daisy::SearchConfig config;
    config.type = daisy::QueryType::RANGE;
    config.r = r;

    daisy::ParIS paris_search(daisy::DistanceType::L2_SQUARED);
    paris_search.setNumThreads(4);
    paris_search.buildIndex(temp_db_file, dim, n_database);

    std::vector<std::vector<daisy::idx_t>> I;
    std::vector<std::vector<float>> D;
    paris_search.searchIndex(query, n_query, config, I, D);
    reportRange("ParIS", r, I, D);

    daisy::BruteForceSearch ground_truth(daisy::DistanceType::L2_SQUARED);
    ground_truth.buildIndex(database, n_database, dim);
    std::vector<std::vector<daisy::idx_t>> gt_I;
    std::vector<std::vector<float>> gt_D;
    ground_truth.searchIndex(query, n_query, config, gt_I, gt_D);

    printf("\nAll queries match: %s\n",
           matchesBruteforce("ParIS", I, gt_I) ? "yes" : "no");

    delete[] database;
    delete[] query;
    remove(temp_db_file.c_str());

    return 0;
}
