#include "test_utils.hpp"
#include "../commons/test_bm_utils.hpp"
#include "../commons/paramSetup.hpp"
#include "../commons/dataloaders.hpp"

std::string prefix = "bruteForce";

TEST_P(BruteforceDTWParameterizedTest, AllConfigurations)
{
    const SSTestConfig &config = GetParam();

    std::string gt_I_path = config.gt_I_prefix + std::to_string(config.k_value) + ".txt";
    std::string gt_D_path = config.gt_D_prefix + std::to_string(config.k_value) + ".txt";

    double dtw_rtol = 0.0;
    double dtw_atol = 2.0;

    runSSTWithDistance(
        diNoLib::DistanceType::DTW,
        prefix,
        gt_I_path,
        gt_D_path,
        config.dataset_path,
        config.query_path,
        config.thread_count,
        dtw_rtol,
        dtw_atol);
}

INSTANTIATE_TEST_SUITE_P(
    BruteforceDTWTests,
    BruteforceDTWParameterizedTest,
    ::testing::ValuesIn(test_configs_dtw),
    [](const ::testing::TestParamInfo<SSTestConfig> &info)
    {
        return info.param.name + "_k" + std::to_string(info.param.k_value) +
               "_thread" + std::to_string(info.param.thread_count) +
               "_idx" + std::to_string(info.index); 
    });

// Additional manual tests for DTW-specific functionality
TEST(BruteforceDTWManualTests, BasicDTWFunctionality)
{
    printf("Testing BruteForce DTW integration...\n");

    // Test 1: Basic functionality test
    {
        printf("Test 1: Basic DTW functionality...\n");
        diNoLib::idx_t n_database = 10;
        unsigned long long dim = 5;
        unsigned long long n_query = 1;
        diNoLib::idx_t k = 3;

        float *database = loadRandomData(n_database, dim, true, 42);
        float *query = loadRandomData(n_query, dim, true, 24);

        diNoLib::BruteForceSearch bf_search_dtw(diNoLib::DistanceType::DTW);
        bf_search_dtw.buildIndex(database, n_database, dim);

        diNoLib::idx_t *I = new diNoLib::idx_t[n_query * k];
        float *D = new float[n_query * k];
        bf_search_dtw.searchIndex(query, n_query, k, I, D);

        bool has_valid_results = true;
        for (diNoLib::idx_t i = 0; i < k; i++)
        {
            if (I[i] >= n_database || D[i] < 0)
            {
                has_valid_results = false;
                break;
            }
        }

        EXPECT_TRUE(has_valid_results) << "DTW Results should be valid";

        delete[] database;
        delete[] query;
        delete[] I;
        delete[] D;
    }
}

TEST(BruteforceDTWManualTests, DTWVsL2SquaredDifference)
{
    // Test 2: Compare DTW vs L2_SQUARED results are different
    {
        printf("Test 2: DTW vs L2_SQUARED produce different results...\n");
        diNoLib::idx_t n_database = 20;
        unsigned long long dim = 8;
        unsigned long long n_query = 1;
        diNoLib::idx_t k = 5;

        float *database = loadRandomData(n_database, dim, true, 100);
        float *query = loadRandomData(n_query, dim, true, 50);

        // DTW search
        diNoLib::BruteForceSearch bf_search_dtw(diNoLib::DistanceType::DTW);
        bf_search_dtw.buildIndex(database, n_database, dim);
        diNoLib::idx_t *I_dtw = new diNoLib::idx_t[n_query * k];
        float *D_dtw = new float[n_query * k];
        bf_search_dtw.searchIndex(query, n_query, k, I_dtw, D_dtw);

        // L2_SQUARED search
        diNoLib::BruteForceSearch bf_search_l2(diNoLib::DistanceType::L2_SQUARED);
        bf_search_l2.buildIndex(database, n_database, dim);
        diNoLib::idx_t *I_l2 = new diNoLib::idx_t[n_query * k];
        float *D_l2 = new float[n_query * k];
        bf_search_l2.searchIndex(query, n_query, k, I_l2, D_l2);

        // Check if results are different (they should be)
        bool results_different = false;
        for (diNoLib::idx_t i = 0; i < k; i++)
        {
            if (I_dtw[i] != I_l2[i] || fabs(D_dtw[i] - D_l2[i]) > 0.001)
            {
                results_different = true;
                break;
            }
        }

        EXPECT_TRUE(results_different) << "DTW and L2_SQUARED should produce different results";

        delete[] database;
        delete[] query;
        delete[] I_dtw;
        delete[] D_dtw;
        delete[] I_l2;
        delete[] D_l2;
    }
}

TEST(BruteforceDTWManualTests, DTWMultiThreading)
{
    // Test 3: Thread safety test
    {
        printf("Test 3: DTW multi-threading...\n");
        diNoLib::idx_t n_database = 50;
        unsigned long long dim = 12;
        unsigned long long n_query = 5;
        diNoLib::idx_t k = 3;

        float *database = loadRandomData(n_database, dim, true, 200);
        float *query = loadRandomData(n_query, dim, true, 150);

        diNoLib::BruteForceSearch bf_search_dtw(diNoLib::DistanceType::DTW);
        bf_search_dtw.setNumThreads(4);
        bf_search_dtw.buildIndex(database, n_database, dim);

        diNoLib::idx_t *I = new diNoLib::idx_t[n_query * k];
        float *D = new float[n_query * k];
        bf_search_dtw.searchIndex(query, n_query, k, I, D);

        bool all_valid = true;
        for (diNoLib::idx_t i = 0; i < n_query * k; i++)
        {
            if (I[i] >= n_database || D[i] < 0)
            {
                all_valid = false;
                break;
            }
        }

        EXPECT_TRUE(all_valid) << "Multi-threaded DTW results should be valid";

        delete[] database;
        delete[] query;
        delete[] I;
        delete[] D;
    }

    printf("\nAll DTW integration tests completed!\n");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
