#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <string>
#include <cstddef> 
#include "../lib/algos/SimilaritySearchAlgorithm.hpp" 
#include <gtest/gtest.h>

#include "../lib/algos/BruteforceSearch.hpp" 
#include "../lib/algos/LbBruteforce.hpp"
#include "../lib/algos/Messi.hpp" 
#include "../lib/algos/Odyssey.hpp" 
#include "../lib/algos/Paris.hpp" 
#include "../lib/algos/Sing.hpp" 

/**
* @brief Integer equality.
*
* @param a First index
* @param b Second index
* @return True if equal, false otherwise
*/ 
bool isclose(int a, int b, double rtol = 1e-5, double atol = 1e-8);

/**
 * @brief Floating point ; A equivalent function of numpy.isclose -- absolute(a - b) <= (atol + rtol * absolute(b))
 *
 * @param a First value
 * @param b Second value
 * @param rtol Relative tolerance
 * @param atol Absolute tolerance
 * @return True if values are close, false otherwise
 */
bool isclose(double a, double b, double rtol = 1e-5, double atol = 1e-8);

/**
 * @brief Read a text file containing floating-point numbers and return them as a dynamically allocated array.
 *
 * @param filepath Path to the file
 * @param outSize Output variable storing the number of floats read
 * @return Pointer to the array of floats, or nullptr on error
 */
float* readFile(const std::string& filepath, size_t& outSize);

/**
 * @brief Extract filename from full path.
 *
 * @param path Full file path
 * @return Filename portion of the path
 */
std::string pathToFilename(std::string path);

/**
 * @brief Parse dataset filename to extract configuration parameters.
 *
 * @param filename Name of the dataset file
 * @param dim Output dimension of the data vectors
 * @param n_database Output number of vectors in the database
 * @param n_query Output number of query vectors
 * @param k Output number of nearest neighbors
 * @return True if all required parameters were successfully parsed, false otherwise
 */
bool parseFilenameForConfig(const std::string& filename,
                            const std::string& prefix,
                            diNoLib::idx_t &dim,
                            diNoLib::idx_t &n_database,
                            diNoLib::idx_t &n_query,
                            diNoLib::idx_t &k); 

                            /**
 * @brief Compare brute-force search results with ground truth and report mismatches or close results.
 *
 * @param pathI Path to ground-truth index file
 * @param pathD Path to ground-truth distance file
 * @param I Computed index results
 * @param D Computed distance results
 * @param n_query Number of queries
 * @param k Number of top results
 */
void compareWithGroundTruth(const std::string& pathI,
                                const std::string& pathD,
                                const diNoLib::idx_t* I,
                                float* D,
                                diNoLib::idx_t n_query,
                                diNoLib::idx_t k);

/**
 * @brief Parameterized test fixture for similarity search
 */
class SimilaritySearchTest : public ::testing::Test
{
protected:
    /**
     * @brief Run a brute-force search test using the provided dataset and compare results to ground truth.
     *
     * @param gt_I Path to ground-truth index file
     * @param gt_D Path to ground-truth distance file
     * @param dataset_path Path to the database binary file
     * @param query_path Path to the query binary file
     * @param num_thread Number of threads to use during search
     */
    void runSST(diNoLib::SimilaritySearchAlgorithm* search,  
                const std::string& prefix_name,
                const std::string& gt_I, 
                const std::string& gt_D, 
                const std::string& dataset_path, 
                const std::string& query_path,
                int num_thread = 1
                );
};     

/**
 * @brief Configuration structure for similarity search tests
 */
struct SSTestConfig {
    std::string name;
    std::string dataset_path;
    std::string query_path;
    std::string gt_I_prefix;
    std::string gt_D_prefix;
    int thread_count;
    int k_value;
};

/**
 * @brief BruteforceParameterizedTest
 */
class BruteforceParameterizedTest : public SimilaritySearchTest,
                                   public ::testing::WithParamInterface<SSTestConfig> {
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
                                   public ::testing::WithParamInterface<SSTestConfig> {
protected:
    using SimilaritySearchTest::runSST;
    
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

/**
 * @brief MessiParameterizedTest
 */
class MessiParameterizedTest : public SimilaritySearchTest,
                                   public ::testing::WithParamInterface<SSTestConfig> {
protected:
    using SimilaritySearchTest::runSST;
    
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

/**
 * @brief OdysseyParameterizedTest
 */
class OdysseyParameterizedTest : public SimilaritySearchTest,
                                   public ::testing::WithParamInterface<SSTestConfig> {
protected:
    using SimilaritySearchTest::runSST;
    
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

/**
 * @brief ParisParameterizedTest
 */
class ParisParameterizedTest : public SimilaritySearchTest,
                                   public ::testing::WithParamInterface<SSTestConfig> {
protected:
    using SimilaritySearchTest::runSST;
    
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

/**
 * @brief SingParameterizedTest
 */
class SingParameterizedTest : public SimilaritySearchTest,
                                   public ::testing::WithParamInterface<SSTestConfig> {
protected:
    using SimilaritySearchTest::runSST;
    
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
};

#endif // TEST_UTILS_HPP
