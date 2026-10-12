#include <gtest/gtest.h>

#include <algos/Bruteforce.hpp>
#include <algos/Messi.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <tuple>
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

    for (daisy::idx_t q = 0; q < kQueries; ++q)
    {
        std::set<daisy::idx_t> returned(ids.begin() + q * kK, ids.begin() + (q + 1) * kK);
        EXPECT_EQ(returned.size(), static_cast<size_t>(kK))
            << "query " << q << " returned duplicate ids";
        EXPECT_EQ(ids[q * kK], expected_ids[q * kK]) << "nearest neighbour differs, query " << q;
        for (daisy::idx_t j = 0; j < kK; ++j)
        {
            EXPECT_EQ(returned.count(expected_ids[q * kK + j]), 1u)
                << "missing exact neighbor, query " << q << ", rank " << j;
            ASSERT_LT(ids[q * kK + j], kRows);
            double actual = 0.0;
            for (daisy::idx_t d = 0; d < kDim; ++d)
            {
                double delta = static_cast<double>(database[ids[q * kK + j] * kDim + d]) - queries[q * kDim + d];
                actual += delta * delta;
            }
            EXPECT_NEAR(distances[q * kK + j], actual, 1e-5 + 1e-5 * std::fabs(actual))
                << "returned ID/distance pair differs, query " << q << ", rank " << j;
            EXPECT_NEAR(distances[q * kK + j], expected_distances[q * kK + j],
                        1e-5 + 1e-5 * std::fabs(expected_distances[q * kK + j]))
                << "exact top-k distance differs, query " << q << ", rank " << j;
        }
    }
}

class MessiExactnessTest : public ::testing::TestWithParam<std::tuple<bool, int>> {};

TEST_P(MessiExactnessTest, RemainsExactAcrossStreamingStagesAndRepeatedQueries)
{
    constexpr daisy::idx_t dim = 64, rows = 4096, nquery = 12;
    const auto [normalized, workers] = GetParam();
    std::mt19937 rng(731);
    std::normal_distribution<float> normal;
    std::vector<float> database(rows * dim), query(nquery * dim);
    auto fill = [&](std::vector<float> &values, daisy::idx_t n) {
        for (daisy::idx_t r = 0; r < n; ++r)
        {
            double mean = 0.0, variance = 0.0;
            for (daisy::idx_t d = 0; d < dim; ++d)
            {
                float value = normal(rng);
                values[r * dim + d] = normalized ? value : 75.0f + 8.0f * value;
                mean += values[r * dim + d];
            }
            if (normalized)
            {
                mean /= dim;
                for (daisy::idx_t d = 0; d < dim; ++d)
                    variance += std::pow(values[r * dim + d] - mean, 2);
                const double scale = std::sqrt(variance / dim);
                for (daisy::idx_t d = 0; d < dim; ++d)
                    values[r * dim + d] = (values[r * dim + d] - mean) / scale;
            }
        }
    };
    fill(database, rows);
    fill(query, nquery);
    // Include exact matches in the initial tree and both appended batches.
    for (daisy::idx_t q = 0; q < 3; ++q)
        std::copy_n(database.data() + (q == 0 ? 17 : q == 1 ? 3200 : 4000) * dim,
                    dim, query.data() + q * dim);
    daisy::MessiConfig config;
    config.index_workers = 2;
    config.search_workers = workers;
    daisy::Messi messi(daisy::DistanceType::L2_SQUARED, config);
    messi.setNormalized(normalized);
    messi.buildIndex(database.data(), 3072, dim);
    daisy::idx_t previous = 3072;
    for (daisy::idx_t n : {3072ULL, 3584ULL, 4096ULL})
    {
        if (n > previous)
            messi.insertBatch(database.data() + previous * dim, n - previous);
        previous = n;
        daisy::BruteForceSearch brute(daisy::DistanceType::L2_SQUARED);
        brute.buildIndex(database.data(), n, dim);
        for (daisy::idx_t k : {1ULL, 10ULL, 100ULL, 1000ULL})
        {
            std::vector<daisy::idx_t> expected_ids(nquery * k), ids(nquery * k);
            std::vector<float> expected_distances(nquery * k), distances(nquery * k);
            brute.searchIndex(query.data(), nquery, k, expected_ids.data(), expected_distances.data());
            for (int repeat = 0; repeat < 2; ++repeat)
            {
                SCOPED_TRACE(::testing::Message() << "normalized=" << normalized << ", workers=" << workers
                    << ", n=" << n << ", k=" << k << ", repeat=" << repeat);
                messi.searchIndex(query.data(), nquery, k, ids.data(), distances.data());
                for (daisy::idx_t q = 0; q < nquery; ++q)
                {
                    std::set<daisy::idx_t> returned(ids.begin() + q * k, ids.begin() + (q + 1) * k);
                    EXPECT_EQ(returned.size(), k) << "duplicate IDs, query " << q;
                    for (daisy::idx_t j = 0; j < k; ++j)
                    {
                        EXPECT_EQ(returned.count(expected_ids[q * k + j]), 1u)
                            << "missing exact neighbor, query " << q << ", rank " << j;
                        ASSERT_LT(ids[q * k + j], n);
                        double actual = 0.0;
                        for (daisy::idx_t d = 0; d < dim; ++d)
                        {
                            double delta = static_cast<double>(database[ids[q * k + j] * dim + d]) - query[q * dim + d];
                            actual += delta * delta;
                        }
                        EXPECT_NEAR(distances[q * k + j], actual, 1e-5 + 1e-5 * actual)
                            << "returned ID/distance mismatch, query " << q << ", rank " << j;
                        EXPECT_NEAR(distances[q * k + j], expected_distances[q * k + j],
                                    1e-5 + 1e-5 * expected_distances[q * k + j])
                            << "exact distance mismatch, query " << q << ", rank " << j;
                    }
                }
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(BreakpointsAndWorkers, MessiExactnessTest,
                        ::testing::Combine(::testing::Bool(), ::testing::Values(1, 4)));

TEST(MessiTopKDistinctTest, ExactTiesAllowAnyDistinctEquallyDistantNeighbors)
{
    constexpr daisy::idx_t dim = 64, rows = 144, nquery = 4;
    std::vector<float> database(rows * dim), query(nquery * dim);
    for (daisy::idx_t r = 0; r < rows; ++r)
        std::fill_n(database.data() + r * dim, dim, r < 40 || r >= 128 ? 0.0f : r < 80 ? 1.0f : 2.0f);
    for (daisy::idx_t q = 0; q < nquery; ++q)
        std::fill_n(query.data() + q * dim, dim, static_cast<float>(q));
    daisy::MessiConfig config;
    config.index_workers = 2;
    config.search_workers = 4;
    daisy::Messi messi(daisy::DistanceType::L2_SQUARED, config);
    messi.setNormalized(false);
    messi.buildIndex(database.data(), 96, dim);
    for (daisy::idx_t n : {96ULL, rows})
    {
        if (n == rows)
            messi.insertBatch(database.data() + 96 * dim, rows - 96);
        daisy::BruteForceSearch brute(daisy::DistanceType::L2_SQUARED);
        brute.buildIndex(database.data(), n, dim);
        for (daisy::idx_t k : {1ULL, 10ULL, 80ULL})
        {
            std::vector<daisy::idx_t> expected_ids(nquery * k), ids(nquery * k);
            std::vector<float> expected_distances(nquery * k), distances(nquery * k);
            brute.searchIndex(query.data(), nquery, k, expected_ids.data(), expected_distances.data());
            messi.searchIndex(query.data(), nquery, k, ids.data(), distances.data());
            for (daisy::idx_t q = 0; q < nquery; ++q)
            {
                std::set<daisy::idx_t> returned(ids.begin() + q * k, ids.begin() + (q + 1) * k);
                EXPECT_EQ(returned.size(), k);
                for (daisy::idx_t j = 0; j < k; ++j)
                {
                    ASSERT_LT(ids[q * k + j], n);
                    float actual = 0.0f;
                    for (daisy::idx_t d = 0; d < dim; ++d)
                    {
                        float delta = database[ids[q * k + j] * dim + d] - query[q * dim + d];
                        actual += delta * delta;
                    }
                    EXPECT_FLOAT_EQ(distances[q * k + j], actual);
                    EXPECT_FLOAT_EQ(distances[q * k + j], expected_distances[q * k + j]);
                }
            }
        }
    }
}
