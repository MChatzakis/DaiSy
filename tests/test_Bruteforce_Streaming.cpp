#include <gtest/gtest.h>

#include "../lib/algos/Bruteforce.hpp"

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

TEST(BruteforceStreamingTest, RequiresBuildAndRejectsNullInput)
{
    daisy::BruteForceSearch search(daisy::DistanceType::L2_SQUARED);
    EXPECT_THROW(search.insert(nullptr), std::runtime_error);

    auto initial = makeSeries(2);
    search.buildIndex(initial.data(), 2, DIM);
    EXPECT_THROW(search.insert(nullptr), std::invalid_argument);
}

TEST(BruteforceStreamingTest, SingleAndBatchInsertsAreImmediatelySearchable)
{
    auto all = makeSeries(8);
    daisy::BruteForceSearch search(daisy::DistanceType::L2_SQUARED);
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
    EXPECT_FLOAT_EQ(distance, 0.0f);

    daisy::SearchConfig range;
    range.type = daisy::QueryType::RANGE;
    range.r = 0.0f;
    std::vector<std::vector<daisy::idx_t>> indices;
    std::vector<std::vector<float>> distances;
    search.searchIndex(all.data() + 3 * DIM, 1, range, indices, distances);
    ASSERT_EQ(indices.size(), 1u);
    ASSERT_EQ(indices[0].size(), 1u);
    EXPECT_EQ(indices[0][0], 3u);
    EXPECT_FLOAT_EQ(distances[0][0], 0.0f);
}

TEST(BruteforceStreamingTest, DtwSearchIncludesInsertedSeries)
{
    auto all = makeSeries(6, 20);
    daisy::BruteForceSearch search(daisy::DistanceType::DTW);
    search.buildIndex(all.data(), 3, DIM);
    search.insert(all.data() + 3 * DIM);
    search.insertBatch(all.data() + 4 * DIM, 2);

    daisy::idx_t index = 0;
    float distance = -1.0f;
    search.searchIndex(all.data() + 5 * DIM, 1, 1, &index, &distance);
    EXPECT_EQ(index, 5u);
    EXPECT_NEAR(distance, 0.0f, 1e-6f);
}

TEST(BruteforceStreamingTest, CanInsertFromItsOwnDatabaseAcrossReallocation)
{
    auto initial = makeSeries(4, 150);
    daisy::BruteForceSearch search(daisy::DistanceType::L2_SQUARED);
    search.buildIndex(initial.data(), 4, DIM);

    search.insert(search.getDatabase() + 2 * DIM);
    ASSERT_EQ(search.getNDatabase(), 5u);

    daisy::idx_t index = 0;
    float distance = -1.0f;
    search.searchIndex(search.getDatabase() + 4 * DIM, 1, 1, &index, &distance);
    EXPECT_EQ(index, 2u);
    EXPECT_FLOAT_EQ(distance, 0.0f);
}
