#include "test_bm_utils.hpp"
#include "../commons/dataloaders.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <vector>

#if __has_include(<gtest/gtest.h>)
#include <gtest/gtest.h>
#define REPORT_FAILURE(msg) ADD_FAILURE() << msg
#else
#define REPORT_FAILURE(msg) std::cerr << "FAILURE: " << msg << std::endl
#endif

bool isclose(daisy::idx_t a, daisy::idx_t b)
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
                            daisy::idx_t &dim,
                            daisy::idx_t &n_database,
                            daisy::idx_t &n_query,
                            daisy::idx_t &k)
{
    std::smatch match;
    std::regex len_rx("len(\\d+)");
    std::regex size_rx("size(\\d+)([MK])?");
    std::regex ctrl_rx("ctrl(\\d+)");
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
        n_database = static_cast<daisy::idx_t>(std::stoll(match[1]));
        if (match[2].matched && match[2].str() == "M")
            n_database *= 1000000;
        else if (match[2].matched && match[2].str() == "K")
            n_database *= 1000;
    }
    else if (std::regex_search(filename, match, ctrl_rx))
    {
        n_database = static_cast<daisy::idx_t>(std::stoll(match[1]));
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
    
    REPORT_FAILURE(msg);
}

void compareWithGroundTruth(const std::string &pathI,
                            const std::string &pathD,
                            const daisy::idx_t *I,
                            float *D,
                            daisy::idx_t n_query,
                            daisy::idx_t k,
                            double rtol,
                            double atol)
{
    size_t sizeI;
    size_t sizeD;
    float *arrayI_gt = readFile(pathI, sizeI);
    float *arrayD_gt = readFile(pathD, sizeD);

    assert_eq(sizeI, n_query * k, "Mismatch in Index file size.");
    assert_eq(sizeD, n_query * k, "Mismatch in Distance file size.");

    for (size_t i = 0; i < n_query; ++i)
    {
        
        std::vector<std::pair<float, daisy::idx_t>> our_pairs;
        our_pairs.reserve(k);
        for (size_t j = 0; j < k; ++j)
        {
            auto idx = i * k + j;
            our_pairs.emplace_back(D[idx], I[idx]);
        }
        std::sort(our_pairs.begin(), our_pairs.end(), [](const auto &a, const auto &b) {
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        });

        std::vector<std::pair<float, daisy::idx_t>> gt_pairs;
        gt_pairs.reserve(k);
        for (size_t j = 0; j < k; ++j)
        {
            auto idx = i * k + j;
            daisy::idx_t gt_idx = static_cast<daisy::idx_t>(std::round(arrayI_gt[idx]));
            gt_pairs.emplace_back(arrayD_gt[idx], gt_idx);
        }
        std::sort(gt_pairs.begin(), gt_pairs.end(), [](const auto &a, const auto &b) {
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        });

        size_t our_unique = 0;
        for (size_t j = 0; j < k; ++j)
        {
            if (j == 0 || our_pairs[j].second != our_pairs[j - 1].second ||
                std::fabs(our_pairs[j].first - our_pairs[j - 1].first) > atol + rtol * std::fabs(our_pairs[j - 1].first))
                our_unique++;
        }
        size_t compare_count = (our_unique < k) ? our_unique : k;

        for (size_t j = 0; j < compare_count; ++j)
        {
            float our_d = our_pairs[j].first;
            daisy::idx_t our_i = our_pairs[j].second;
            float gt_d = gt_pairs[j].first;
            daisy::idx_t gt_idx = gt_pairs[j].second;

            bool I_equal = isclose(our_i, gt_idx);
            bool D_close = isclose(our_d, gt_d, rtol, atol);

            if (!I_equal && !D_close)
            {
                add_failure("ERROR 1: Indices mismatch AND distance mismatch at (query " + std::to_string(i) + ", rank " + std::to_string(j) + "): "
                            "expected (label " + std::to_string(gt_idx) + ", dist " + std::to_string(gt_d) + "), "
                            "got (label " + std::to_string(our_i) + ", dist " + std::to_string(our_d) + ")");
            }
            else if (I_equal && !D_close)
            {
                add_failure("ERROR 2: Indices match BUT distance mismatch at (query " + std::to_string(i) + ", rank " + std::to_string(j) + "): "
                            "label " + std::to_string(our_i) + "; expected distance " + std::to_string(gt_d) + ", got " + std::to_string(our_d));
            }
            else if (!I_equal && D_close)
            {
                std::cerr << "WARNING: Indices mismatch but distances close at (query " << i << ", rank " << j << "): "
                          << "expected label " << gt_idx << ", got " << our_i << "; distance ~" << our_d << std::endl;
            }
        }

    }

    delete[] arrayI_gt;
    delete[] arrayD_gt;
}
