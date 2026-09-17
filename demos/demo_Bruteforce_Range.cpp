// Bruteforce range search: report every series within squared-L2 distance r of a query.
// Brute force is the exact baseline that every other range demo checks itself against.

#include "../commons/dataloaders.hpp"
#include "../lib/daisy.hpp"

#include <algorithm>
#include <cstdio>
#include <set>
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

    daisy::SearchConfig config;
    config.type = daisy::QueryType::RANGE;
    config.r = r;

    daisy::BruteForceSearch bf_search(daisy::DistanceType::L2_SQUARED);
    bf_search.setNumThreads(4);
    bf_search.buildIndex(database, n_database, dim);

    std::vector<std::vector<daisy::idx_t>> I;
    std::vector<std::vector<float>> D;
    bf_search.searchIndex(query, n_query, config, I, D);
    reportRange("Bruteforce", r, I, D);

    delete[] database;
    delete[] query;

    return 0;
}
