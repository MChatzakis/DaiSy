// Verifies the data-adaptive (equi-depth) breakpoint path on NON-z-normalized data.
//
// The iSAX-based algorithms use SAX breakpoints for MINDIST lower bounds. With the
// hardcoded Gaussian table those bounds are only meaningful for z-normalized data.
// setNormalized(false) switches an index to per-index equi-depth breakpoints computed
// from the data. This test builds each equi-depth-capable algorithm on raw
// (non-normalized) data and checks that kNN results match the brute-force ground truth.

#include "test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

using daisy::idx_t;

namespace
{
    std::vector<float> genNonNormalized(int n, int dim, unsigned seed)
    {
        std::mt19937 rng(seed);
        std::normal_distribution<float> step(0.0f, 1.0f);
        std::uniform_real_distribution<float> mean_dist(50.0f, 150.0f);
        std::uniform_real_distribution<float> scale_dist(2.0f, 20.0f);

        std::vector<float> data((size_t)n * dim);
        for (int i = 0; i < n; i++)
        {
            float mean = mean_dist(rng);
            float scale = scale_dist(rng);
            float acc = 0.0f;
            for (int j = 0; j < dim; j++)
            {
                acc += step(rng);
                data[(size_t)i * dim + j] = mean + scale * acc;
            }
        }
        return data;
    }

    constexpr double RTOL = 1e-2;
    constexpr double ATOL = 1e-8;

    bool distClose(double a, double b)
    {
        return std::fabs(a - b) <= ATOL + RTOL * std::fabs(b);
    }
}  // namespace

class EquidepthTest : public ::testing::Test
{
protected:
    static constexpr int N = 3000;
    static constexpr int DIM = 64;
    static constexpr int NQ = 40;
    static constexpr int K = 10;

    std::vector<float> data, query;
    std::vector<std::vector<float>> gtDist;

    void SetUp() override
    {
        data = genNonNormalized(N, DIM, 1234);
        query = genNonNormalized(NQ, DIM, 9999);

        // Exact ground truth (distances).
        daisy::BruteForceSearch bf(daisy::DistanceType::L2_SQUARED);
        bf.buildIndex(data.data(), N, DIM);
        std::vector<idx_t> gtI((size_t)NQ * K);
        std::vector<float> gtD((size_t)NQ * K);
        bf.searchIndex(query.data(), NQ, K, gtI.data(), gtD.data());
        gtDist.assign(NQ, {});
        for (int q = 0; q < NQ; q++)
        {
            gtDist[q].assign(gtD.begin() + (size_t)q * K, gtD.begin() + (size_t)(q + 1) * K);
            std::sort(gtDist[q].begin(), gtDist[q].end());
        }
    }

    void checkExactOnNonNormalized(daisy::SimilaritySearchAlgorithm *algo)
    {
        algo->setNormalized(false);
        EXPECT_FALSE(algo->getNormalized()) << "setNormalized(false) did not take effect";

        algo->buildIndex(data.data(), N, DIM);
        algo->setNumThreads(1);

        std::vector<idx_t> I((size_t)NQ * K);
        std::vector<float> D((size_t)NQ * K);
        algo->searchIndex(query.data(), NQ, K, I.data(), D.data());

        for (int q = 0; q < NQ; q++)
        {
            std::vector<std::pair<float, idx_t>> ours(K);
            for (int j = 0; j < K; j++)
                ours[j] = {D[(size_t)q * K + j], I[(size_t)q * K + j]};
            std::sort(ours.begin(), ours.end());

            int uniq = 0;
            for (int j = 0; j < K; j++)
                if (j == 0 || ours[j].second != ours[j - 1].second ||
                    !distClose(ours[j].first, ours[j - 1].first))
                    uniq++;

            int compareCount = std::min(uniq, K);
            for (int j = 0; j < compareCount; j++)
                EXPECT_TRUE(distClose(ours[j].first, gtDist[q][j]))
                    << "query " << q << " neighbour " << j
                    << ": got " << ours[j].first << ", exact GT " << gtDist[q][j]
                    << " (equi-depth kNN not exact on non-normalized data)";
        }
    }
};

TEST_F(EquidepthTest, LbBruteforce)
{
    daisy::LbBruteforce algo(daisy::DistanceType::L2_SQUARED);
    checkExactOnNonNormalized(&algo);
}

TEST_F(EquidepthTest, Messi)
{
    daisy::Messi algo(daisy::DistanceType::L2_SQUARED);
    checkExactOnNonNormalized(&algo);
}

TEST_F(EquidepthTest, Fresh)
{
    daisy::Fresh algo(daisy::DistanceType::L2_SQUARED);
    checkExactOnNonNormalized(&algo);
}

// Hercules only supports z-normalized data: setNormalized(false) must be rejected.
TEST_F(EquidepthTest, HerculesRejectsNonNormalized)
{
    daisy::Hercules algo(daisy::DistanceType::L2_SQUARED);
    EXPECT_THROW(algo.setNormalized(false), std::runtime_error);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
