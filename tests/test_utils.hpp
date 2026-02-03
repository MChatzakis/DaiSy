#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <gtest/gtest.h>

#include "../commons/test_bm_utils.hpp"
#include "../lib/algos/SimilaritySearchAlgorithm.hpp"
#include "../lib/algos/Bruteforce.hpp"
#include "../lib/algos/LbBruteforce.hpp"
#include "../lib/algos/Messi.hpp"
#if ODYSSEY_MPI
#include "../lib/algos/hodyssey/Odyssey.hpp"
#endif
#include "../lib/algos/ParIS.hpp"
#include "../lib/algos/Sing.hpp"

/**
 * @brief Parameterized test fixture for similarity search
 */
class SimilaritySearchTest : public ::testing::Test
{
protected:
    /**
     * @brief Run a brute-force search test using the provided dataset and compare results to ground truth.
     *
     * @param search The similarity search algorithm instance
     * @param prefix_name name of the search type
     * @param gt_I Path to ground-truth index file
     * @param gt_D Path to ground-truth distance file
     * @param dataset_path Path to the database binary file
     * @param query_path Path to the query binary file
     * @param num_thread Number of threads to use during search
     */
    void runSST(diNoLib::SimilaritySearchAlgorithm *search,
                const std::string &prefix_name,
                const std::string &gt_I,
                const std::string &gt_D,
                const std::string &dataset_path,
                const std::string &query_path,
                int num_thread = 1,
                double rtol = 1e-2,
                double atol = 1e-8);

    /**
     * @brief Run a similarity search test with a specific distance type
     *
     * @param distance_type The distance metric to use (L2_SQUARED, DTW, etc.)
     * @param prefix_name name of the search type
     * @param gt_I Path to ground-truth index file
     * @param gt_D Path to ground-truth distance file
     * @param dataset_path Path to the database binary file
     * @param query_path Path to the query binary file
     * @param num_thread Number of threads to use during search
     */
    void runSSTWithDistance(diNoLib::DistanceType distance_type,
                            const std::string &prefix_name,
                            const std::string &gt_I,
                            const std::string &gt_D,
                            const std::string &dataset_path,
                            const std::string &query_path,
                            int num_thread = 1,
                            double rtol = 1e-2,
                            double atol = 1e-8);
};

/**
 * @brief BruteforceDTWParameterizedTest
 */
class BruteforceDTWParameterizedTest : public SimilaritySearchTest,
                                       public ::testing::WithParamInterface<SSTestConfig>
{
protected:
    using SimilaritySearchTest::runSST;
    using SimilaritySearchTest::runSSTWithDistance;

    // Required Google Test setup/teardown methods
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

/**
 * @brief BruteforceParameterizedTest
 */
class BruteforceParameterizedTest : public SimilaritySearchTest,
                                    public ::testing::WithParamInterface<SSTestConfig>
{
protected:
    using SimilaritySearchTest::runSST;

    // Required Google Test setup/teardown methods
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

/**
 * @brief LbBruteforceParameterizedTest
 */
class LbBruteforceParameterizedTest : public SimilaritySearchTest,
                                      public ::testing::WithParamInterface<SSTestConfig>
{
protected:
    using SimilaritySearchTest::runSST;

    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

/**
 * @brief LbBruteforceDTWParameterizedTest
 */
class LbBruteforceDTWParameterizedTest : public SimilaritySearchTest,
                                         public ::testing::WithParamInterface<SSTestConfig>
{
protected:
    using SimilaritySearchTest::runSST;

    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

/**
 * @brief MessiParameterizedTest
 */
class MessiParameterizedTest : public SimilaritySearchTest,
                               public ::testing::WithParamInterface<SSTestConfig>
{
protected:
    using SimilaritySearchTest::runSST;

    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

/**
 * @brief OdysseyParameterizedTest
 */
class OdysseyParameterizedTest : public SimilaritySearchTest,
                                 public ::testing::WithParamInterface<SSTestConfig>
{
protected:
    using SimilaritySearchTest::runSST;

    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

/**
 * @brief ParISParameterizedTest
 */
class ParISParameterizedTest : public SimilaritySearchTest,
                               public ::testing::WithParamInterface<SSTestConfig>
{
protected:
    using SimilaritySearchTest::runSST;

    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

/**
 * @brief SingParameterizedTest
 */
class SingParameterizedTest : public SimilaritySearchTest,
                              public ::testing::WithParamInterface<SSTestConfig>
{
protected:
    using SimilaritySearchTest::runSST;

    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

#endif // TEST_UTILS_HPP
