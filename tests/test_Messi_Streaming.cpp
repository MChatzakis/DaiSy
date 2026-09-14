#include <gtest/gtest.h>

#include "../lib/algos/Bruteforce.hpp"
#include "../lib/algos/Messi.hpp"

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
            const float phase = static_cast<float>(first_phase + i) * 0.29f;
            double mean = 0.0;
            for (int j = 0; j < DIM; ++j)
            {
                series[j] = std::sin(0.17f * j + phase) +
                            0.45f * std::cos(0.41f * j - 0.3f * phase);
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

    daisy::MessiConfig streamingConfig()
    {
        daisy::MessiConfig config;
        config.search_workers = 1;
        config.index_workers = 1;
        config.leaf_size = 4;
        config.paa_segments = 16;
        return config;
    }

    void expectL2MatchesBruteforce(daisy::Messi &search,
                                   const std::vector<float> &data,
                                   int n_database,
                                   const std::vector<float> &queries,
                                   int n_query,
                                   int k)
    {
        daisy::BruteForceSearch ground_truth(daisy::DistanceType::L2_SQUARED);
        ground_truth.setNumThreads(1);
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

TEST(MessiStreamingTest, RequiresBuildAndRejectsNullInput)
{
    daisy::Messi search(daisy::DistanceType::L2_SQUARED, streamingConfig());
    EXPECT_THROW(search.insert(nullptr), std::runtime_error);

    auto initial = makeSeries(2);
    search.buildIndex(initial.data(), 2, DIM);
    EXPECT_THROW(search.insert(nullptr), std::invalid_argument);
    EXPECT_NO_THROW(search.insertBatch(nullptr, 0));
}

TEST(MessiStreamingTest, SingleAndBatchInsertsMatchBruteforce)
{
    auto all = makeSeries(18);
    auto queries = makeSeries(5, 50);
    daisy::Messi search(daisy::DistanceType::L2_SQUARED, streamingConfig());
    search.buildIndex(all.data(), 7, DIM);
    expectL2MatchesBruteforce(search, all, 7, queries, 5, 4);

    daisy::SimilaritySearchAlgorithm *streaming = &search;
    streaming->insert(all.data() + 7 * DIM);
    expectL2MatchesBruteforce(search, all, 8, queries, 5, 4);

    streaming->insertBatch(all.data() + 8 * DIM, 10);
    ASSERT_EQ(search.getNDatabase(), 18u);
    expectL2MatchesBruteforce(search, all, 18, queries, 5, 4);

    daisy::idx_t index = 0;
    float distance = -1.0f;
    search.searchIndex(all.data() + 17 * DIM, 1, 1, &index, &distance);
    EXPECT_EQ(index, 17u);
    EXPECT_FLOAT_EQ(distance, 0.0f);

    daisy::SearchConfig range;
    range.type = daisy::QueryType::RANGE;
    range.r = 1e-5f;
    std::vector<std::vector<daisy::idx_t>> indices;
    std::vector<std::vector<float>> distances;
    search.searchIndex(all.data() + 7 * DIM, 1, range, indices, distances);
    ASSERT_EQ(indices.size(), 1u);
    ASSERT_EQ(indices[0].size(), 1u);
    EXPECT_EQ(indices[0][0], 7u);
    EXPECT_FLOAT_EQ(distances[0][0], 0.0f);
}

TEST(MessiStreamingTest, CreatesMissingRootAndSplitsExistingLeaf)
{
    std::vector<float> initial(static_cast<size_t>(2) * DIM);
    for (int row = 0; row < 2; ++row)
    {
        for (int j = 0; j < DIM; ++j)
        {
            const int segment = j / 8;
            initial[static_cast<size_t>(row) * DIM + j] =
                static_cast<float>(segment * 2 - 3);
        }
    }
    std::vector<float> opposite(DIM);
    for (int j = 0; j < DIM; ++j)
        opposite[j] = -initial[j];

    daisy::MessiConfig config;
    config.search_workers = 1;
    config.index_workers = 1;
    config.leaf_size = 2;
    config.paa_segments = 4;
    daisy::Messi search(daisy::DistanceType::L2_SQUARED, config);
    search.buildIndex(initial.data(), 2, DIM);

    ASSERT_NE(search.getIndex(), nullptr);
    ASSERT_NE(search.getIndex()->first_node, nullptr);
    daisy::isax_node *initial_root = search.getIndex()->first_node;
    const unsigned long roots_before = search.getIndex()->root_nodes;

    // Same SAX root at capacity: the incremental insert must split the leaf.
    search.insert(initial.data());
    EXPECT_FALSE(initial_root->is_leaf);

    // Negating every segment flips the root SAX mask, forcing root creation.
    search.insert(opposite.data());
    EXPECT_GT(search.getIndex()->root_nodes, roots_before);

    daisy::idx_t index = 0;
    float distance = -1.0f;
    search.searchIndex(opposite.data(), 1, 1, &index, &distance);
    EXPECT_EQ(index, 3u);
    EXPECT_FLOAT_EQ(distance, 0.0f);
}

TEST(MessiStreamingTest, DtwSearchIncludesInsertedSeries)
{
    auto all = makeSeries(8, 80);
    daisy::Messi search(daisy::DistanceType::DTW, streamingConfig());
    search.buildIndex(all.data(), 4, DIM);
    search.insert(all.data() + 4 * DIM);
    search.insertBatch(all.data() + 5 * DIM, 3);

    daisy::idx_t index = 0;
    float distance = -1.0f;
    search.searchIndex(all.data() + 7 * DIM, 1, 1, &index, &distance);
    EXPECT_EQ(index, 7u);
    EXPECT_NEAR(distance, 0.0f, 1e-6f);
}

TEST(MessiStreamingTest, EquidepthInsertsReuseInitialBreakpoints)
{
    auto all = makeSeries(16, 110);
    auto queries = makeSeries(4, 140);
    for (float &value : all)
        value = 30.0f + 6.0f * value;
    for (float &value : queries)
        value = 30.0f + 6.0f * value;

    daisy::Messi search(daisy::DistanceType::L2_SQUARED, streamingConfig());
    search.setNormalized(false);
    search.buildIndex(all.data(), 8, DIM);
    search.insertBatch(all.data() + 8 * DIM, 8);

    expectL2MatchesBruteforce(search, all, 16, queries, 4, 4);
}

TEST(MessiStreamingTest, CanInsertFromItsOwnDatabaseAcrossReallocation)
{
    auto initial = makeSeries(4, 180);
    daisy::Messi search(daisy::DistanceType::L2_SQUARED, streamingConfig());
    search.buildIndex(initial.data(), 4, DIM);

    search.insert(search.getDatabase() + DIM);
    ASSERT_EQ(search.getNDatabase(), 5u);
    const float *owned_database = search.getDatabase();
    search.insertBatch(owned_database, 4);
    ASSERT_EQ(search.getNDatabase(), 9u);

    for (int i = 0; i < 4 * DIM; ++i)
        EXPECT_FLOAT_EQ(search.getDatabase()[5 * DIM + i], initial[i]);
}
