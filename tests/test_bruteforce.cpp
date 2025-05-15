#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <regex>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <unordered_map>

// header of the GoogleTest
#include <gtest/gtest.h>

#include "../commons/dataloaders.hpp"
#include "../lib/algos/BruteforceSearch.hpp"

// A equivalent function of numpy.isclose ; absolute(a - b) <= (atol + rtol * absolute(b))
bool isclose(diNoLib::idx_t a, diNoLib::idx_t b, double rtol = 1e-5, double atol = 1e-8) {
    return std::fabs(a - b) <= atol || std::fabs(a - b) <= rtol * std::fabs(b);
}

std::string pathToFilename(std::string path)
{
    return path.substr(path.find_last_of("/\\") + 1);
}

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

    if (std::regex_search(filename, match, len_rx)) {
        dim = std::stoi(match[1]);
    } else {
        success = false;
    }

    if (std::regex_search(filename, match, size_rx)) {
        n_database = std::stoi(match[1]);
    } else {
        success = false;
    }

    if (std::regex_search(filename, prefix_rx)) {
        if (std::regex_search(filename, match, q_rx)) {
            n_query = std::stoi(match[1]);
        } else {
            success = false;
        }

        if (std::regex_search(filename, match, k_rx)) {
            k = std::stoi(match[1]);
        } else {
            success = false;
        }
    } else {
        n_query = 0;
        k = 0;
    }

    return success;
}

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

bool compareWithGroundTruth(const std::string& pathI, 
                            const std::string& pathD, 
                            const diNoLib::idx_t *I, 
                            float *D,
                            diNoLib::idx_t n_query, 
                            diNoLib::idx_t k) 
{
    size_t sizeI;
    size_t sizeD;
    float* arrayI = readFile(pathI, sizeI);
    float* arrayD = readFile(pathD, sizeD);
    
    if (sizeI != n_query * k || sizeD != n_query * k) 
    {
        std::cerr << "Size mismatch " << std::endl;
        return false;
    }

    for (size_t i = 0; i < n_query; ++i) 
    {
        for (size_t j = 0; j < k; ++j) 
        {
            diNoLib::idx_t expectedI = I[i * k + j];
            diNoLib::idx_t actualI = arrayI[i * k + j];
            diNoLib::idx_t expectedD = D[i * k + j];
            diNoLib::idx_t actualD = arrayD[i * k + j];                       

            if (!isclose(expectedI, actualI))            
            {
                std::cerr << "I: Mismatch at i=" << i << ", j=" << j
                          << ": expected " << expectedI << ", got " << actualI << std::endl;
                return false;
            }
           
            if (!isclose(actualD, expectedD)) 
            {
                std::cerr << "D: Mismatch at i=" << i << ", j=" << j
                          << ": expected " << expectedD << ", got " << actualD << std::endl;
                return false;
            }
        }
    }

    return true;
}

void printResults(diNoLib::idx_t n_query, diNoLib::idx_t k, const diNoLib::idx_t *I, float *D, const char* label = "") 
{
    for (diNoLib::idx_t i = 0; i < n_query; ++i) 
    {
        if (label) printf("%s", label);
        printf("Query %llu: ", i);
        for (diNoLib::idx_t j = 0; j < k; ++j) 
        {
            printf("%llu ", I[i * k + j]);
        }
        printf("\n");
    }

    for (diNoLib::idx_t i = 0; i < n_query; ++i) 
    {
        if (label) printf("%s", label);
        printf("Query %llu: ", i);
        for (diNoLib::idx_t j = 0; j < k; ++j) 
        {
            printf("%.8f ", D[i * k + j]);
        }
        printf("\n");
    }
}

bool testBruteForceSS(const char *path_I_gt,
                    const char *path_D_gt, 
                    const char *dataset_path, 
                    const char *query_path)
{
    std::string filename_gt = pathToFilename(path_I_gt);
    std::string dataset_name = pathToFilename(dataset_path); 
    
    // 0. Configuration
    diNoLib::idx_t dim_gt, n_database_gt, n_query, k;

    if (!parseFilenameForConfig(filename_gt, dim_gt, n_database_gt, n_query, k)) 
    {
        std::cerr << "Failed to parse ground truth file." << std::endl;
        return false;
    }    

    diNoLib::idx_t dim, n_database, noQuery, noK;
    if (!parseFilenameForConfig(dataset_name, dim, n_database, noQuery, noK)) 
    {
        std::cerr << "Failed to parse test file." << std::endl;
        return false;
    }

    assert ((dim_gt=dim) && (n_database_gt == n_database));

    // 1. Load data and queries
    float *database = loadBinData(dataset_path, n_database, dim);
    float *query = loadBinData(query_path, n_query, dim);

    // 2. Create a brute-force search object
    diNoLib::BruteForceSearch bf_search(diNoLib::DistanceType::L2_SQUARED);  

    // 3. Build the index
    bf_search.buildIndex(database, n_database, dim);
    diNoLib::idx_t *I = new diNoLib::idx_t[n_query * k];
    float *D = new float[n_query * k];
    
    // 4. Search the index
    bf_search.searchIndex(query, n_query, k, I, D);
    
    // 5. Compare with ground truth
    bool ok = compareWithGroundTruth(path_I_gt, path_D_gt, I, D, n_query, k);
    if (ok)
    {
        std::cout << "Test PASSED! :)" << std::endl;
    }
    else
    {
        std::cerr << "Test FAILED" << std::endl;
    }
    delete[] database;
    delete[] query;
    delete[] I;
    delete[] D;

    return true;
}

struct DatasetPairing {
    std::string data_path;
    std::string query_path;
    std::vector<std::string> gt_I_paths;
    std::vector<std::string> gt_D_paths;
};


void compareWithGroundTruthTEST(const std::string& pathI, 
                            const std::string& pathD, 
                            const diNoLib::idx_t *I, 
                            float *D,
                            diNoLib::idx_t n_query, 
                            diNoLib::idx_t k) 
{
    size_t sizeI;
    size_t sizeD;
    float* arrayI = readFile(pathI, sizeI);
    float* arrayD = readFile(pathD, sizeD);

    ASSERT_EQ(sizeI, n_query * k) << "Mismatch in Index file size.";
    ASSERT_EQ(sizeD, n_query * k) << "Mismatch in Distance file size.";

    for (size_t i = 0; i < n_query; ++i) {
        for (size_t j = 0; j < k; ++j) {
            auto idx = i * k + j;

            EXPECT_TRUE(isclose(I[idx], static_cast<diNoLib::idx_t>(arrayI[idx])))
                << "Index mismatch at (" << i << "," << j << "): expected " << arrayI[idx] << ", got " << I[idx];

            EXPECT_TRUE(isclose(D[idx], arrayD[idx]))
                << "Distance mismatch at (" << i << "," << j << "): expected " << arrayD[idx] << ", got " << D[idx];
        }
    }

    delete[] arrayI;
    delete[] arrayD;
}

class BruteForceSSTest : public ::testing::Test {
protected:
    void runBruteForceTest(const std::string& gt_I, 
                           const std::string& gt_D, 
                           const std::string& dataset_path, 
                           const std::string& query_path) 
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

TEST_F(BruteForceSSTest, RandomWalkData_k1) {
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_random_len96_size200000_q4_k1.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_random_len96_size200000_q4_k1.txt",
        "../data/random.data.randwalk.len96.size200000.znorm.bin",
        "../data/random.query.randwalk.len96.size1000.bin"
    );
}

TEST_F(BruteForceSSTest, RandomWalkData_k10) {
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_random_len96_size200000_q4_k10.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_random_len96_size200000_q4_k10.txt",
        "../data/random.data.randwalk.len96.size200000.znorm.bin",
        "../data/random.query.randwalk.len96.size1000.bin"
    );
}

TEST_F(BruteForceSSTest, RandomWalkData_k100) {
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_random_len96_size200000_q4_k100.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_random_len96_size200000_q4_k100.txt",
        "../data/random.data.randwalk.len96.size200000.znorm.bin",
        "../data/random.query.randwalk.len96.size1000.bin"
    );
}

TEST_F(BruteForceSSTest, AstronomyData_k1) {
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_astronomy_len256_size50000_q4_k1.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_astronomy_len256_size50000_q4_k1.txt",
        "../data/astronomy.data.len256.size50000.znorm.bin",
        "../data/astronomy.query.len256.size50000.znorm.bin"
    );
}

TEST_F(BruteForceSSTest, AstronomyData_k10) {
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_astronomy_len256_size50000_q4_k10.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_astronomy_len256_size50000_q4_k10.txt",
        "../data/astronomy.data.len256.size50000.znorm.bin",
        "../data/astronomy.query.len256.size50000.znorm.bin"
    );
}

TEST_F(BruteForceSSTest, AstronomyData_k100) {
    runBruteForceTest(
        "../tests/gt/Indices/bruteFSS_gt_I_astronomy_len256_size50000_q4_k100.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_astronomy_len256_size50000_q4_k100.txt",
        "../data/astronomy.data.len256.size50000.znorm.bin",
        "../data/astronomy.query.len256.size50000.znorm.bin"
    );
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}