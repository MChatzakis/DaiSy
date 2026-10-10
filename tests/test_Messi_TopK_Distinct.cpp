#include <gtest/gtest.h>

#include <algos/Bruteforce.hpp>
#include <algos/Messi.hpp>

#include <algorithm>
#include <random>
#include <set>
#include <vector>

// Regression test: MESSI's approximate phase scans one leaf to seed the
// best-so-far queue, then the parallel phase scans every leaf again. Without
// de-duplication by position the seeded leaf's series were inserted twice, so
// the k-NN list contained the same series several times and displaced real
// neighbours.
namespace
{
    constexpr daisy::idx_t kRows = 20000;
    constexpr daisy::idx_t kDim = 128;
    constexpr daisy::idx_t kQueries = 50;
    constexpr daisy::idx_t kK = 10;

    // 50 Gaussian clusters with isotropic noise: dense neighbourhoods, so the
    // leaf scanned by the approximate phase holds members of the final top-k.
    std::vector<float> clusteredData(daisy::idx_t rows, unsigned seed)
    {
        std::mt19937 centre_rng(42);
        std::normal_distribution<float> normal;
        std::vector<float> centres(50 * kDim);
        for (auto &v : centres)
            v = 6.0F * normal(centre_rng);
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> pick(0, 49);
        std::vector<float> data(static_cast<size_t>(rows) * kDim);
        for (daisy::idx_t r = 0; r < rows; ++r)
        {
            const float *centre = centres.data() + pick(rng) * kDim;
            for (daisy::idx_t d = 0; d < kDim; ++d)
                data[static_cast<size_t>(r) * kDim + d] = centre[d] + normal(rng);
        }
        return data;
    }
}

TEST(MessiTopKDistinctTest, ReturnsDistinctIdsAndMatchesBruteForce)
{
    auto database = clusteredData(kRows, 1);
    auto queries = clusteredData(kQueries, 2);

    daisy::BruteForceSearch brute(daisy::DistanceType::L2_SQUARED);
    brute.buildIndex(database.data(), kRows, kDim);
    std::vector<daisy::idx_t> expected_ids(kQueries * kK);
    std::vector<float> expected_distances(kQueries * kK);
    brute.searchIndex(queries.data(), kQueries, kK, expected_ids.data(),
                      expected_distances.data());

    daisy::MessiConfig config;
    config.search_workers = 4;
    config.index_workers = 2;
    daisy::Messi messi(daisy::DistanceType::L2_SQUARED, config);
    messi.setNormalized(false);
    messi.buildIndex(database.data(), kRows, kDim);
    std::vector<daisy::idx_t> ids(kQueries * kK);
    std::vector<float> distances(kQueries * kK);
    messi.searchIndex(queries.data(), kQueries, kK, ids.data(), distances.data());

    size_t found = 0;
    for (daisy::idx_t q = 0; q < kQueries; ++q)
    {
        std::set<daisy::idx_t> returned(ids.begin() + q * kK, ids.begin() + (q + 1) * kK);
        EXPECT_EQ(returned.size(), static_cast<size_t>(kK))
            << "query " << q << " returned duplicate ids";
        EXPECT_EQ(ids[q * kK], expected_ids[q * kK]) << "nearest neighbour differs, query " << q;
        for (daisy::idx_t j = 0; j < kK; ++j)
            found += returned.count(expected_ids[q * kK + j]);
    }
    const double recall = static_cast<double>(found) / (kQueries * kK);
    EXPECT_GE(recall, 0.99) << "MESSI top-" << kK << " recall vs brute force";
}
