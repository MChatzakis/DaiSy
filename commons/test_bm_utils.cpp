#include "test_bm_utils.hpp"
#include "../commons/dataloaders.hpp"

#include <iostream>
#include <fstream>
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

float *readFile(const std::string &filepath, size_t &outSize)
{
    std::ifstream file(filepath);
    if (!file)
    {
        std::cerr << "(readFile) Error opening file: " << filepath << std::endl;
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
    float *D = new float[outSize];
    for (size_t i = 0; i < outSize; ++i)
    {
        D[i] = temp[i];
    }

    return D;
}

bool parseFilenameForConfig(const std::string &filename,
                            const std::string &prefix,
                            diNoLib::idx_t &dim,
                            diNoLib::idx_t &n_database,
                            diNoLib::idx_t &n_query,
                            diNoLib::idx_t &k)
{
    std::smatch match;
    std::regex len_rx("len(\\d+)");
    std::regex size_rx("size(\\d+)");
    std::regex q_rx("q(\\d+)");
    std::regex k_rx("k(\\d+)");
    std::regex prefix_rx("^" + prefix);

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
    }
    else
    {
        success = false;
    }

    if (std::regex_search(filename, prefix_rx))
    {
        std::smatch match_q, match_k;
        if (std::regex_search(filename, match_q, q_rx))
        {
            n_query = std::stoi(match_q[1]);
        }
        else
        {
            success = false;
        }

        if (std::regex_search(filename, match_k, k_rx))
        {
            k = std::stoi(match_k[1]);
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

void assert_eq(size_t a, size_t b, const std::string &msg)
{
    if (a != b)
    {
        std::cerr << "Assertion failed: " << msg << " (got " << a << ", expected " << b << ")" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void add_failure(const std::string &msg)
{
    std::cerr << "FAILURE: " << msg << std::endl;
}

void compareWithGroundTruth(const std::string &pathI,
                            const std::string &pathD,
                            const diNoLib::idx_t *I,
                            float *D,
                            diNoLib::idx_t n_query,
                            diNoLib::idx_t k)
{
    size_t sizeI;
    size_t sizeD;
    float *arrayI_gt = readFile(pathI, sizeI);
    float *arrayD_gt = readFile(pathD, sizeD);

    assert_eq(sizeI, n_query * k, "Mismatch in Index file size.");
    assert_eq(sizeD, n_query * k, "Mismatch in Distance file size.");

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
                add_failure("ERROR 1: Indices mismatch AND distance mismatch at (" + std::to_string(i) + "," + std::to_string(j) + "): " + "expected label " + std::to_string(arrayI_gt[idx]) + ", got " + std::to_string(I[idx]) + "; " + "expected distance " + std::to_string(arrayD_gt[idx]) + ", got " + std::to_string(D[idx]));
            }
            else if (I_equal && !D_close)
            {
                // Error case 2
                add_failure("ERROR 2: Indices mismatch AND distance mismatch at (" + std::to_string(i) + "," + std::to_string(j) + "): " + "expected label " + std::to_string(arrayI_gt[idx]) + ", got " + std::to_string(I[idx]) + "; " + "expected distance " + std::to_string(arrayD_gt[idx]) + ", got " + std::to_string(D[idx]));
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
                // SUCCEED: do nothing
            }
        }
    }

    delete[] arrayI_gt;
    delete[] arrayD_gt;
}