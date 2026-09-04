#include <gtest/gtest.h>

#include "../lib/algos/Bruteforce.hpp"
#include "../lib/algos/LbBruteforce.hpp"

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
            const float phase = static_cast<float>(first_phase + i) * 0.31f;
            double mean = 0.0;
            for (int j = 0; j < DIM; ++j)
            {
                series[j] = std::sin(0.19f * j + phase) +
                            0.4f * std::cos(0.43f * j - 0.5f * phase);
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

    void expectMatchesBruteforce(daisy::LbBruteforce &search,
                                 const std::vector<float> &data,
                                 int n_database,
                                 const std::vector<float> &queries,
                                 int n_query,
                                 int k)
    {
        daisy::BruteForceSearch ground_truth(daisy::DistanceType::L2_SQUARED);
        ground_truth.buildIndex(const_cast<float *>(data.data()), n_database, DIM);

        std::vector<daisy::idx_t> expected_indices(static_cast<size_t>(n_query) * k);
        std::vector<float> expected_distances(static_cast<size_t>(n_query) * k);
        std::vector<daisy::idx_t> actual_indices(static_cast<size_t>(n_query) * k);
        std::vector<float> actual_distances(static_cast<size_t>(n_query) * k);

        ground_truth.searchIndex(queries.data(), n_query, k,
                                 expected_indices.data(), expected_distances.data());
        search.searchIndex(queries.data(), n_query, k,
                           actual_indices.data(), actual_distances.data());

        for (size_t i = 0; i < actual_indices.size(); ++i)
        {
            EXPECT_EQ(actual_indices[i], expected_indices[i]);
            EXPECT_NEAR(actual_distances[i], expected_distances[i], 1e-4f);
        }
    }
}

TEST(LbBruteforceStreamingTest, RequiresBuildAndRejectsNullInput)
{
    daisy::LbBruteforce search(daisy::DistanceType::L2_SQUARED);
    EXPECT_THROW(search.insert(nullptr), std::runtime_error);

    auto initial = makeSeries(2);
    search.buildIndex(initial.data(), 2, DIM);
    EXPECT_THROW(search.insert(nullptr), std::invalid_argument);
}

TEST(LbBruteforceStreamingTest, SingleAndBatchInsertsUpdateDataAndSaxSummaries)
{
    auto all = makeSeries(15);
    auto queries = makeSeries(4, 40);

    daisy::LbBruteforce search(daisy::DistanceType::L2_SQUARED);
    search.setNumThreads(1);
    search.buildIndex(all.data(), 8, DIM);
    expectMatchesBruteforce(search, all, 8, queries, 4, 4);

    daisy::SimilaritySearchAlgorithm *streaming = &search;
    streaming->insert(all.data() + 8 * DIM);
    expectMatchesBruteforce(search, all, 9, queries, 4, 4);

    streaming->insertBatch(all.data() + 9 * DIM, 6);
    ASSERT_EQ(search.getNDatabase(), 15u);
    expectMatchesBruteforce(search, all, 15, queries, 4, 4);

    daisy::idx_t index = 0;
    float distance = -1.0f;
    search.searchIndex(all.data() + 14 * DIM, 1, 1, &index, &distance);
    EXPECT_EQ(index, 14u);
    EXPECT_FLOAT_EQ(distance, 0.0f);

    daisy::SearchConfig range;
    range.type = daisy::QueryType::RANGE;
    range.r = 0.0f;
    std::vector<std::vector<daisy::idx_t>> indices;
    std::vector<std::vector<float>> distances;
    search.searchIndex(all.data() + 8 * DIM, 1, range, indices, distances);
    ASSERT_EQ(indices.size(), 1u);
    ASSERT_EQ(indices[0].size(), 1u);
    EXPECT_EQ(indices[0][0], 8u);
    EXPECT_FLOAT_EQ(distances[0][0], 0.0f);
}

TEST(LbBruteforceStreamingTest, DtwSearchIncludesInsertedSaxRecords)
{
    auto all = makeSeries(7, 60);
    daisy::LbBruteforce search(daisy::DistanceType::DTW);
    search.setNumThreads(1);
    search.buildIndex(all.data(), 4, DIM);
    search.insert(all.data() + 4 * DIM);
    search.insertBatch(all.data() + 5 * DIM, 2);

    daisy::idx_t index = 0;
    float distance = -1.0f;
    search.searchIndex(all.data() + 6 * DIM, 1, 1, &index, &distance);
    EXPECT_EQ(index, 6u);
    EXPECT_NEAR(distance, 0.0f, 1e-6f);
}

TEST(LbBruteforceStreamingTest, EquidepthInsertsReuseInitialBreakpoints)
{
    auto all = makeSeries(14, 90);
    auto queries = makeSeries(3, 120);

    // Scaling and offset make the input intentionally non-z-normalized.
    for (float &value : all)
        value = 25.0f + 7.0f * value;
    for (float &value : queries)
        value = 25.0f + 7.0f * value;

    daisy::LbBruteforce search(daisy::DistanceType::L2_SQUARED);
    search.setNormalized(false);
    search.setNumThreads(1);
    search.buildIndex(all.data(), 8, DIM);
    search.insertBatch(all.data() + 8 * DIM, 6);

    expectMatchesBruteforce(search, all, 14, queries, 3, 4);
}

TEST(LbBruteforceStreamingTest, CanInsertFromItsOwnDatabaseAcrossReallocation)
{
    auto initial = makeSeries(4, 180);
    daisy::LbBruteforce search(daisy::DistanceType::L2_SQUARED);
    search.setNumThreads(1);
    search.buildIndex(initial.data(), 4, DIM);

    search.insert(search.getDatabase() + DIM);
    ASSERT_EQ(search.getNDatabase(), 5u);

    daisy::idx_t index = 0;
    float distance = -1.0f;
    search.searchIndex(search.getDatabase() + 4 * DIM, 1, 1, &index, &distance);
    EXPECT_EQ(index, 1u);
    EXPECT_FLOAT_EQ(distance, 0.0f);
}
