#include "test_utils.hpp"

#include "../commons/dataloaders.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <regex>
#include <cmath>
 
bool isclose(diNoLib::idx_t a, diNoLib::idx_t b) 
{  
    return a == b;
}

bool isclose(double a, double b, double rtol, double atol) 
{
    return std::fabs(a - b) <= (atol + rtol * std::fabs(b));
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

void compareWithGroundTruth(const std::string& pathI, 
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
            SUCCEED(); 
            }
        }
    }

    delete[] arrayI_gt;
    delete[] arrayD_gt;
}

void SimilaritySearchTest::runSST(diNoLib::SimilaritySearchAlgorithm* search,
                                const std::string& gt_I, 
                                const std::string& gt_D, 
                                const std::string& dataset_path, 
                                const std::string& query_path,
                                int num_thread) 
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

    search->buildIndex(database, n_database, dim);
    search->setNumThreads(num_thread);

    diNoLib::idx_t* I = new diNoLib::idx_t[n_query * k];
    float* D = new float[n_query * k];
    search->searchIndex(query, n_query, k, I, D);

    compareWithGroundTruth(gt_I, gt_D, I, D, n_query, k);

    delete[] database;
    delete[] query;
    delete[] I;
    delete[] D;
}