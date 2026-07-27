// Static COCONUT: exact kNN on a z-normalized dataset must match brute-force ground truth
// (a wrong sortable-SAX / MINDIST path would drop true neighbours).

#include "test_utils.hpp"  // gtest + algorithm headers + BruteForceSearch
#include "../lib/algos/Coconut.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using daisy::idx_t;

namespace
{
    // Z-normalized random-walk series (the setting SAX/Gaussian breakpoints target).
    std::vector<float> genZNorm(int n, int dim, unsigned seed)
    {
        std::mt19937 rng(seed);
        std::normal_distribution<float> step(0.0f, 1.0f);
        std::vector<float> data((size_t)n * dim);
        for (int i = 0; i < n; i++)
        {
            float *s = data.data() + (size_t)i * dim;
            float acc = 0.0f;
            for (int j = 0; j < dim; j++) { acc += step(rng); s[j] = acc; }
            double mean = 0, sd = 0;
            for (int j = 0; j < dim; j++) mean += s[j];
            mean /= dim;
            for (int j = 0; j < dim; j++) sd += (s[j] - mean) * (s[j] - mean);
            sd = std::sqrt(sd / dim);
            if (sd < 1e-6) sd = 1.0;
            for (int j = 0; j < dim; j++) s[j] = (float)((s[j] - mean) / sd);
        }
        return data;
    }

    bool distClose(double a, double b) { return std::fabs(a - b) <= 1e-8 + 1e-2 * std::fabs(b); }
}

TEST(CoconutTest, StaticKnnMatchesBruteForce)
{
    const int N = 2000, DIM = 64, NQ = 30, K = 10;
    auto data = genZNorm(N, DIM, 7);
    auto query = genZNorm(NQ, DIM, 77);

    daisy::BruteForceSearch bf(daisy::DistanceType::L2_SQUARED);
    bf.buildIndex(data.data(), N, DIM);
    std::vector<idx_t> gtI((size_t)NQ * K);
    std::vector<float> gtD((size_t)NQ * K);
    bf.searchIndex(query.data(), NQ, K, gtI.data(), gtD.data());

    daisy::Coconut cc(daisy::DistanceType::L2_SQUARED);
    cc.buildIndex(data.data(), N, DIM);
    cc.setNumThreads(1);
    std::vector<idx_t> I((size_t)NQ * K);
    std::vector<float> D((size_t)NQ * K);
    cc.searchIndex(query.data(), NQ, K, I.data(), D.data());

    for (int q = 0; q < NQ; q++)
    {
        std::vector<float> gd(gtD.begin() + (size_t)q * K, gtD.begin() + (size_t)(q + 1) * K);
        std::vector<float> od(D.begin() + (size_t)q * K, D.begin() + (size_t)(q + 1) * K);
        std::sort(gd.begin(), gd.end());
        std::sort(od.begin(), od.end());
        for (int j = 0; j < K; j++)
            EXPECT_TRUE(distClose(od[j], gd[j]))
                << "query " << q << " nn " << j << ": coconut " << od[j] << " vs GT " << gd[j];
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
