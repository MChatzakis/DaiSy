#include <gtest/gtest.h>

#include "../lib/algos/Bruteforce.hpp"
#include "../lib/algos/Faiss.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace
{
    constexpr int DIM = 32;

    std::vector<float> makeSeries(int n, int first_phase = 0)
    {
        std::vector<float> data(static_cast<size_t>(n) * DIM);
        for (int i = 0; i < n; ++i)
        {
            float *series = data.data() + static_cast<size_t>(i) * DIM;
            const float phase = static_cast<float>(first_phase + i) * 0.37f;
            double mean = 0.0;
            for (int j = 0; j < DIM; ++j)
            {
                series[j] = std::sin(0.21f * j + phase) +
                            0.35f * std::cos(0.47f * j - phase);
                mean += series[j];
            }
            mean /= DIM;

            double variance = 0.0;
            for (int j = 0; j < DIM; ++j)
                variance += (series[j] - mean) * (series[j] - mean);
            const double stddev = std::sqrt(variance / DIM);
            for (int j = 0; j < DIM; ++j)
                series[j] = static_cast<float>((series[j] - mean) / stddev);
        }
        return data;
    }
}

TEST(FaissStreamingTest, RejectsDtw)
{
    EXPECT_THROW(daisy::FaissFlat(daisy::DistanceType::DTW), std::invalid_argument);
}

TEST(FaissStreamingTest, RequiresBuildAndRejectsNullInput)
{
    daisy::FaissFlat search(daisy::DistanceType::L2_SQUARED);
    EXPECT_THROW(search.insert(nullptr), std::invalid_argument);
    EXPECT_THROW(search.insertBatch(nullptr, 1), std::runtime_error);

    auto initial = makeSeries(2);
    search.buildIndex(initial.data(), 2, DIM);
    EXPECT_THROW(search.insert(nullptr), std::invalid_argument);
    EXPECT_THROW(search.insertBatch(nullptr, 1), std::invalid_argument);
}

TEST(FaissStreamingTest, SingleAndBatchInsertsAreImmediatelySearchable)
{
    auto all = makeSeries(8);
    daisy::FaissFlat search(daisy::DistanceType::L2_SQUARED);
    search.buildIndex(all.data(), 3, DIM);

    daisy::SimilaritySearchAlgorithm *streaming = &search;
    streaming->insert(all.data() + 3 * DIM);
    streaming->insertBatch(all.data() + 4 * DIM, 4);

    ASSERT_EQ(search.getNDatabase(), 8u);
    for (int j = 0; j < DIM; ++j)
        EXPECT_FLOAT_EQ(search.getDatabase()[7 * DIM + j], all[7 * DIM + j]);

    daisy::idx_t index = 0;
    float distance = -1.0f;
    search.searchIndex(all.data() + 7 * DIM, 1, 1, &index, &distance);
    EXPECT_EQ(index, 7u);
    EXPECT_NEAR(distance, 0.0f, 1e-5f);
}

// A flat FAISS index stays exact across inserts: no breakpoints, centroids or
// codebooks are frozen at build time, so an incrementally grown index must agree
// with a bruteforce scan over the same data, index for index.
TEST(FaissStreamingTest, StreamedIndexMatchesBruteforceExactly)
{
    constexpr int TOTAL = 64;
    constexpr int INITIAL = 10;
    constexpr int N_QUERY = 5;
    constexpr int K = 7;

    auto data = makeSeries(TOTAL, 3);
    auto queries = makeSeries(N_QUERY, 900);

    daisy::FaissFlat streamed(daisy::DistanceType::L2_SQUARED);
    streamed.buildIndex(data.data(), INITIAL, DIM);
    streamed.insertBatch(data.data() + INITIAL * DIM, 20);
    for (int i = INITIAL + 20; i < TOTAL; ++i)
        streamed.insert(data.data() + static_cast<size_t>(i) * DIM);
    ASSERT_EQ(streamed.getNDatabase(), static_cast<daisy::idx_t>(TOTAL));

    daisy::BruteForceSearch ground_truth(daisy::DistanceType::L2_SQUARED);
    ground_truth.setNumThreads(1);
    ground_truth.buildIndex(data.data(), TOTAL, DIM);

    std::vector<daisy::idx_t> expected_I(static_cast<size_t>(N_QUERY) * K);
    std::vector<float> expected_D(static_cast<size_t>(N_QUERY) * K);
    std::vector<daisy::idx_t> actual_I(static_cast<size_t>(N_QUERY) * K);
    std::vector<float> actual_D(static_cast<size_t>(N_QUERY) * K);

    ground_truth.searchIndex(queries.data(), N_QUERY, K, expected_I.data(), expected_D.data());
    streamed.searchIndex(queries.data(), N_QUERY, K, actual_I.data(), actual_D.data());

    for (size_t i = 0; i < actual_I.size(); ++i)
    {
        EXPECT_EQ(actual_I[i], expected_I[i]) << "at result " << i;
        EXPECT_NEAR(actual_D[i], expected_D[i], 1e-3f) << "at result " << i;
    }
}

// Range search must use DaiSy's inclusive boundary (dist <= r) and be sorted by
// distance, even though FAISS keeps only dist < radius and returns hits unsorted.
TEST(FaissStreamingTest, RangeSearchMatchesBruteforceIncludingTheBoundary)
{
    constexpr int TOTAL = 40;
    auto data = makeSeries(TOTAL, 77);

    daisy::FaissFlat streamed(daisy::DistanceType::L2_SQUARED);
    streamed.buildIndex(data.data(), 8, DIM);
    streamed.insertBatch(data.data() + 8 * DIM, TOTAL - 8);
    ASSERT_EQ(streamed.getNDatabase(), static_cast<daisy::idx_t>(TOTAL));

    daisy::BruteForceSearch ground_truth(daisy::DistanceType::L2_SQUARED);
    ground_truth.setNumThreads(1);
    ground_truth.buildIndex(data.data(), TOTAL, DIM);

    // A query drawn from the database has an exact-zero hit, so r = 0 pins down
    // the inclusive boundary: FAISS alone (dist < radius) would return nothing.
    for (float r : {0.0f, 5.0f, 30.0f})
    {
        daisy::SearchConfig config;
        config.type = daisy::QueryType::RANGE;
        config.r = r;

        std::vector<std::vector<daisy::idx_t>> expected_I, actual_I;
        std::vector<std::vector<float>> expected_D, actual_D;

        ground_truth.searchIndex(data.data() + 12 * DIM, 1, config, expected_I, expected_D);
        streamed.searchIndex(data.data() + 12 * DIM, 1, config, actual_I, actual_D);

        ASSERT_EQ(actual_I.size(), 1u) << "r=" << r;
        ASSERT_EQ(actual_I[0].size(), expected_I[0].size()) << "r=" << r;
        for (size_t j = 0; j < actual_I[0].size(); ++j)
        {
            EXPECT_EQ(actual_I[0][j], expected_I[0][j]) << "r=" << r << " hit " << j;
            EXPECT_NEAR(actual_D[0][j], expected_D[0][j], 1e-3f) << "r=" << r << " hit " << j;
        }
        EXPECT_TRUE(std::is_sorted(actual_D[0].begin(), actual_D[0].end())) << "r=" << r;
        for (float d : actual_D[0])
            EXPECT_LE(d, r) << "hit outside the radius at r=" << r;
    }
}

// FAISS storage moves when it grows, so inserting a slice of the live database
// would otherwise read from a buffer that add() is reallocating underneath it.
TEST(FaissStreamingTest, CanInsertFromItsOwnDatabaseAcrossReallocation)
{
    auto initial = makeSeries(4, 150);
    daisy::FaissFlat search(daisy::DistanceType::L2_SQUARED);
    search.buildIndex(initial.data(), 4, DIM);

    search.insert(search.getDatabase() + 2 * DIM);
    ASSERT_EQ(search.getNDatabase(), 5u);

    daisy::idx_t index = 0;
    float distance = -1.0f;
    search.searchIndex(search.getDatabase() + 4 * DIM, 1, 1, &index, &distance);
    EXPECT_EQ(index, 2u);
    EXPECT_NEAR(distance, 0.0f, 1e-5f);
}
