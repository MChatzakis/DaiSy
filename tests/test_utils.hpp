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
#include "../lib/algos/hsofa/Sofa.hpp"
#include "../lib/algos/hhercules/Hercules.hpp"

class SimilaritySearchTest : public ::testing::Test
{
protected:
    
    void runSST(daisy::SimilaritySearchAlgorithm *search,
                const std::string &prefix_name,
                const std::string &gt_I,
                const std::string &gt_D,
                const std::string &dataset_path,
                const std::string &query_path,
                int num_thread = 1,
                double rtol = 1e-2,
                double atol = 1e-8);

    void runSSTWithDistance(daisy::DistanceType distance_type,
                            const std::string &prefix_name,
                            const std::string &gt_I,
                            const std::string &gt_D,
                            const std::string &dataset_path,
                            const std::string &query_path,
                            int num_thread = 1,
                            double rtol = 1e-2,
                            double atol = 1e-8);
};

class BruteforceDTWParameterizedTest : public SimilaritySearchTest,
                                       public ::testing::WithParamInterface<SSTestConfig>
{
protected:
    using SimilaritySearchTest::runSST;
    using SimilaritySearchTest::runSSTWithDistance;

    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

class BruteforceParameterizedTest : public SimilaritySearchTest,
                                    public ::testing::WithParamInterface<SSTestConfig>
{
protected:
    using SimilaritySearchTest::runSST;

    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

class LbBruteforceParameterizedTest : public SimilaritySearchTest,
                                      public ::testing::WithParamInterface<SSTestConfig>
{
protected:
    using SimilaritySearchTest::runSST;

    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

class LbBruteforceDTWParameterizedTest : public SimilaritySearchTest,
                                         public ::testing::WithParamInterface<SSTestConfig>
{
protected:
    using SimilaritySearchTest::runSST;

    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

class MessiParameterizedTest : public SimilaritySearchTest,
                               public ::testing::WithParamInterface<SSTestConfig>
{
protected:
    using SimilaritySearchTest::runSST;

    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

class OdysseyParameterizedTest : public SimilaritySearchTest,
                                 public ::testing::WithParamInterface<SSTestConfig>
{
protected:
    using SimilaritySearchTest::runSST;

    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

class ParISParameterizedTest : public SimilaritySearchTest,
                               public ::testing::WithParamInterface<SSTestConfig>
{
protected:
    using SimilaritySearchTest::runSST;

    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

class SingParameterizedTest : public SimilaritySearchTest,
                              public ::testing::WithParamInterface<SSTestConfig>
{
protected:
    using SimilaritySearchTest::runSST;

    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

class SofaParameterizedTest : public SimilaritySearchTest,
                              public ::testing::WithParamInterface<SSTestConfig>
{
protected:
    using SimilaritySearchTest::runSST;

    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

class HerculesParameterizedTest : public SimilaritySearchTest,
                                  public ::testing::WithParamInterface<SSTestConfig>
{
protected:
    using SimilaritySearchTest::runSST;

    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

#endif
