#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <regex>
#include <cassert>
#include <cmath>

#include <gtest/gtest.h>
                    
#include "../commons/dataloaders.hpp"
#include "../lib/algos/BruteforceSearch.hpp"

/**
* @brief Integer equality.
*
* @param a First index
* @param b Second index
* @return True if equal, false otherwise
*/  
bool isclose(diNoLib::idx_t a, diNoLib::idx_t b) 
{  
    return a == b;
}

/**
 * @brief Floating point ; A equivalent function of numpy.isclose -- absolute(a - b) <= (atol + rtol * absolute(b))
 *
 * @param a First value
 * @param b Second value
 * @param rtol Relative tolerance
 * @param atol Absolute tolerance
 * @return True if values are close, false otherwise
 */
bool isclose(double a, double b, double rtol=1e-5, double atol=1e-8) 
{
    return std::fabs(a - b) <= (atol + rtol * std::fabs(b));
}

/**
 * @brief Extract filename from full path.
 *
 * @param path Full file path
 * @return Filename portion of the path
 */
std::string pathToFilename(std::string path)
{
    return path.substr(path.find_last_of("/\\") + 1);
}

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
                            diNoLib::idx_t &dim,
                            diNoLib::idx_t &n_database,
                            diNoLib::idx_t &n_query,
                            diNoLib::idx_t &k)
{
    std::smatch match;
    std::regex prefix_rx("^bruteFSS");
    std::regex len_rx("len(\\d+)");
    std::regex size_rx("size(\\d+)");
    std::regex q_rx("q(\\d+)");
    std::regex k_rx("k(\\d+)");

    bool success = true;

    if (std::regex_search(filename, match, len_rx)) 
    {
        dim = std::stoi(match[1]);
    } 
    else 
    {
        success = false;
    }

    if (std::regex_search(filename, match, size_rx)) 
    {
        n_database = std::stoi(match[1]);
    } else 
    {
        success = false;
    }

    if (std::regex_search(filename, prefix_rx)) 
    {
        if (std::regex_search(filename, match, q_rx)) 
        {
            n_query = std::stoi(match[1]);
        } 
        else 
        {
            success = false;
        }

        if (std::regex_search(filename, match, k_rx)) 
        {
            k = std::stoi(match[1]);
        } 
        else 
        {
            success = false;
        }
    } 
    else 
    {
        n_query = 0;
        k = 0;
    }

    return success;
}

/**
 * @brief Read a text file containing floating-point numbers and return them as a dynamically allocated array.
 *
 * @param filepath Path to the file
 * @param outSize Output variable storing the number of floats read
 * @return Pointer to the array of floats, or nullptr on error
 */
float* readFile(const std::string& filepath, size_t& outSize)
{
    std::ifstream file(filepath);
    if (!file)
    {
        std::cerr << "Error opening file: " << filepath << std::endl;
        outSize = 0;
        return nullptr;
    }

    std::vector<float> temp;
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        float number;
        while (iss >> number)
        {
            temp.push_back(static_cast<float>(number));
        }
    }

    outSize = temp.size();
    float* D = new float[outSize];
    for (size_t i = 0; i < outSize; ++i)
    {
        D[i] = temp[i];
    }

    return D;
}

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
void compareWithGroundTruthTEST(const std::string& pathI, 
                            const std::string& pathD, 
                            const diNoLib::idx_t *I, 
                            float *D,
                            diNoLib::idx_t n_query, 
                            diNoLib::idx_t k) 
{
    size_t sizeI;
    size_t sizeD;
    float* arrayI_gt = readFile(pathI, sizeI);
    float* arrayD_gt = readFile(pathD, sizeD);

    ASSERT_EQ(sizeI, n_query * k) << "Mismatch in Index file size.";
    ASSERT_EQ(sizeD, n_query * k) << "Mismatch in Distance file size.";

    for (size_t i = 0; i < n_query; ++i) 
    {
        for (size_t j = 0; j < k; ++j) 
        {
            auto idx = i * k + j;
            bool I_equal = isclose(I[idx], static_cast<diNoLib::idx_t>(arrayI_gt[idx]));
            bool D_close = isclose(D[idx], arrayD_gt[idx], 1e-2, 1e-8);

        if (!I_equal && !D_close) 
        {
            // Error case 1
            ADD_FAILURE() << "ERROR 1: Indices mismatch AND distance mismatch at (" << i << "," << j << "): "
                          << "expected label " << arrayI_gt[idx] << ", got " << I[idx] << "; "
                          << "expected distance " << arrayD_gt[idx] << ", got " << D[idx];
        }
        else if (I_equal && !D_close) 
        {
            // Error case 2
            ADD_FAILURE() << "ERROR 2: Indices match BUT distance mismatch at (" << i << "," << j << "): "
                          << "label " << I[idx] << "; "
                          << "expected distance " << arrayD_gt[idx] << ", got " << D[idx];
        }
        else if (!I_equal && D_close) 
        {
            // Warning case
            std::cerr << "WARNING: Indices mismatch but distances are close at (" << i << "," << j << "): "
                      << "expected label " << arrayI_gt[idx] << ", got " << I[idx] << "; "
                      << "distance close to " << D[idx] << std::endl;
        }
        else 
        {
            SUCCEED(); // Everything matches or labels and distances both close: test passes
        }
        }
    }

    delete[] arrayI_gt;
    delete[] arrayD_gt;
}

/**
 * Test fixture class for brute-force search tests using GoogleTest.
 */
class BruteForceSSTest : public ::testing::Test 
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
    void runBruteForceTest(const std::string& gt_I, 
                           const std::string& gt_D, 
                           const std::string& dataset_path, 
                           const std::string& query_path,
                           int num_thread=1) 
    {
        std::string filename_gt = pathToFilename(gt_I);
        std::string dataset_name = pathToFilename(dataset_path);

        diNoLib::idx_t dim_gt, n_database_gt, n_query, k;
        ASSERT_TRUE(parseFilenameForConfig(filename_gt, dim_gt, n_database_gt, n_query, k));

        diNoLib::idx_t dim, n_database, _, __;
        ASSERT_TRUE(parseFilenameForConfig(dataset_name, dim, n_database, _, __));

        ASSERT_EQ(dim_gt, dim);
        ASSERT_EQ(n_database_gt, n_database);

        float* database = loadBinData(dataset_path.c_str(), n_database, dim);
        float* query = loadBinData(query_path.c_str(), n_query, dim);

        diNoLib::BruteForceSearch bf_search(diNoLib::DistanceType::L2_SQUARED);
        bf_search.buildIndex(database, n_database, dim);
        bf_search.setNumThreads(num_thread);

        diNoLib::idx_t* I = new diNoLib::idx_t[n_query * k];
        float* D = new float[n_query * k];
        bf_search.searchIndex(query, n_query, k, I, D);

        compareWithGroundTruthTEST(gt_I, gt_D, I, D, n_query, k);

        delete[] database;
        delete[] query;
        delete[] I;
        delete[] D;
    }
};

/*******************************************************/
/************************ ASTRO ************************/
/*******************************************************/
const char *astro_data = "../data/astronomy.data.len256.size50000.znorm.bin";
const char *astro_query = "../data/astronomy.query.len256.size100.znorm.bin";

/************************ THREAD 1 ************************/
TEST_F(BruteForceSSTest, AstronomyData_q100_k1_thread1) 
{
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_astronomy_len256_size50000_q100_k1.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_astronomy_len256_size50000_q100_k1.txt",
        astro_data,
        astro_query
    );
}

TEST_F(BruteForceSSTest, AstronomyData_q100_k10_thread1) 
{
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_astronomy_len256_size50000_q100_k10.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_astronomy_len256_size50000_q100_k10.txt",
        astro_data,
        astro_query
    );
}

TEST_F(BruteForceSSTest, AstronomyData_q100_k100_thread1) 
{
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_astronomy_len256_size50000_q100_k100.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_astronomy_len256_size50000_q100_k100.txt",
        astro_data,
        astro_query
    );
}

/************************ THREAD 4 ************************/
TEST_F(BruteForceSSTest, AstronomyData_q100_k1_thread4) 
{
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_astronomy_len256_size50000_q100_k1.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_astronomy_len256_size50000_q100_k1.txt",
        astro_data,
        astro_query,
        4
    );
}

TEST_F(BruteForceSSTest, AstronomyData_q100_k10_thread4) 
{
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_astronomy_len256_size50000_q100_k10.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_astronomy_len256_size50000_q100_k10.txt",
        astro_data,
        astro_query,
        4
    );
}

TEST_F(BruteForceSSTest, AstronomyData_q100_k100_thread4) 
{
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_astronomy_len256_size50000_q100_k100.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_astronomy_len256_size50000_q100_k100.txt",
        astro_data,
        astro_query,
        4
    );
}

/************************ THREAD 8  ************************/
TEST_F(BruteForceSSTest, AstronomyData_q100_k1_thread8) 
{
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_astronomy_len256_size50000_q100_k1.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_astronomy_len256_size50000_q100_k1.txt",
        astro_data,
        astro_query,
        8
    );
}

TEST_F(BruteForceSSTest, AstronomyData_q100_k10_thread8) 
{
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_astronomy_len256_size50000_q100_k10.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_astronomy_len256_size50000_q100_k10.txt",
        astro_data,
        astro_query,
        8
    );
}

TEST_F(BruteForceSSTest, AstronomyData_q100_k100_thread8) 
{
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_astronomy_len256_size50000_q100_k100.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_astronomy_len256_size50000_q100_k100.txt",
        astro_data,
        astro_query,
        8
    );
}

/********************************************************/
/************************ RANDOM ************************/
/********************************************************/
const char *random_data = "../data/random.data.randwalk.len96.size200000.znorm.bin";
const char *random_query = "../data/random.query.randwalk.len96.size1000.bin";

/************************ THREAD 1 ************************/
TEST_F(BruteForceSSTest, RandomWalkData_q1000_k1_thread1) 
{
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_random_len96_size200000_q1000_k1.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_random_len96_size200000_q1000_k1.txt",
        random_data,
        random_query
    );
}

TEST_F(BruteForceSSTest, RandomWalkData_q1000_k10_thread1) 
{
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_random_len96_size200000_q1000_k10.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_random_len96_size200000_q1000_k10.txt",
        random_data,
        random_query
    );
}

TEST_F(BruteForceSSTest, RandomWalkData_q1000_k100_thread1) 
{
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_random_len96_size200000_q1000_k100.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_random_len96_size200000_q1000_k100.txt",
        random_data,
        random_query
    );
}

/************************ THREAD 4 ************************/
TEST_F(BruteForceSSTest, RandomWalkData_q1000_k1_thread4) 
{
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_random_len96_size200000_q1000_k1.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_random_len96_size200000_q1000_k1.txt",
        random_data,
        random_query,
        4
    );
}

TEST_F(BruteForceSSTest, RandomWalkData_q1000_k10_thread4) 
{
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_random_len96_size200000_q1000_k10.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_random_len96_size200000_q1000_k10.txt",
        random_data,
        random_query,
        4
    );
}

TEST_F(BruteForceSSTest, RandomWalkData_q1000_k100_thread4) 
{
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_random_len96_size200000_q1000_k100.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_random_len96_size200000_q1000_k100.txt",
        random_data,
        random_query,
        4
    );
}

/************************ THREAD 8 ************************/
TEST_F(BruteForceSSTest, RandomWalkData_q1000_k1_thread8) 
{
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_random_len96_size200000_q1000_k1.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_random_len96_size200000_q1000_k1.txt",
        random_data,
        random_query,
        8
    );
}

TEST_F(BruteForceSSTest, RandomWalkData_q1000_k10_thread8) 
{
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_random_len96_size200000_q1000_k10.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_random_len96_size200000_q1000_k10.txt",
        random_data,
        random_query,
        8
    );
}

TEST_F(BruteForceSSTest, RandomWalkData_q1000_k100_thread8) 
{
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_random_len96_size200000_q1000_k100.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_random_len96_size200000_q1000_k100.txt",
        random_data,
        random_query,
        8
    );
}

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}