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
#include <atomic>
#include <random>
#include <set>
#include <thread>
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

    constexpr double RTOL = 1e-5;
    constexpr double ATOL = 1e-5;

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
    std::vector<idx_t> gtIds;

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
        gtIds = gtI;
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

            std::set<idx_t> returned;
            for (int j = 0; j < K; j++)
            {
                ASSERT_LT(ours[j].second, static_cast<idx_t>(N));
                EXPECT_TRUE(returned.insert(ours[j].second).second)
                    << "duplicate ID, query " << q << ", rank " << j;
                double actual = 0.0;
                for (int d = 0; d < DIM; ++d)
                {
                    double delta = static_cast<double>(data[ours[j].second * DIM + d]) - query[q * DIM + d];
                    actual += delta * delta;
                }
                EXPECT_TRUE(distClose(ours[j].first, actual))
                    << "returned ID/distance mismatch, query " << q << ", rank " << j;
                EXPECT_TRUE(distClose(ours[j].first, gtDist[q][j]))
                    << "query " << q << " neighbour " << j
                    << ": got " << ours[j].first << ", exact GT " << gtDist[q][j]
                    << " (equi-depth kNN not exact on non-normalized data)";
            }
            for (int j = 0; j < K; ++j)
                EXPECT_EQ(returned.count(gtIds[q * K + j]), 1u)
                    << "missing exact neighbor, query " << q << ", rank " << j;
        }
    }

    void checkSearchRunsOnNonNormalized(daisy::SimilaritySearchAlgorithm *algo)
    {
        algo->setNormalized(false);
        EXPECT_FALSE(algo->getNormalized()) << "setNormalized(false) did not take effect";

        algo->buildIndex(data.data(), N, DIM);
        algo->setNumThreads(1);

        std::vector<idx_t> I((size_t)NQ * K);
        std::vector<float> D((size_t)NQ * K);
        algo->searchIndex(query.data(), NQ, K, I.data(), D.data());

        for (size_t i = 0; i < D.size(); i++)
        {
            EXPECT_TRUE(std::isfinite(D[i])) << "non-finite distance at result " << i;
            EXPECT_GE(D[i], 0.0f) << "negative distance at result " << i;
            EXPECT_LT(I[i], static_cast<idx_t>(N)) << "out-of-range index at result " << i;
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
    checkSearchRunsOnNonNormalized(&algo);
}

// Hercules only supports z-normalized data: setNormalized(false) must be rejected.
TEST_F(EquidepthTest, HerculesRejectsNonNormalized)
{
    daisy::Hercules algo(daisy::DistanceType::L2_SQUARED);
    EXPECT_THROW(algo.setNormalized(false), std::runtime_error);
}

// Two live indices with different equi-depth breakpoints.
//
// Breakpoint tables used to be installed into a process-global before every search, so an
// index's lower bounds were only correct while no other index had installed its own. The
// query paths of Messi and LbBruteforce now pass their own tables down explicitly.
//
// Each index must remain exact against brute force whether or not a second
// index with different breakpoints exists.
class TwoIndexEquidepthTest : public ::testing::Test
{
protected:
    static constexpr int N = 3000;
    static constexpr int DIM = 64;
    static constexpr int NQ = 20;
    static constexpr int K = 10;

    std::vector<float> dataA, queryA, dataB;

    void SetUp() override
    {
        dataA = genNonNormalized(N, DIM, 4242);
        queryA = genNonNormalized(NQ, DIM, 777);
        // Shift B far away so its equi-depth breakpoints cannot stand in for A's.
        dataB = genNonNormalized(N, DIM, 8888);
        for (float &v : dataB)
            v = v * 25.0f + 100000.0f;
    }

    template <typename Algo>
    void expectSecondIndexDoesNotPerturbFirst()
    {
        daisy::BruteForceSearch brute(daisy::DistanceType::L2_SQUARED);
        brute.buildIndex(dataA.data(), N, DIM);
        std::vector<idx_t> expected_I((size_t)NQ * K);
        std::vector<float> expected_D((size_t)NQ * K);
        brute.searchIndex(queryA.data(), NQ, K, expected_I.data(), expected_D.data());

        Algo a(daisy::DistanceType::L2_SQUARED);
        a.setNormalized(false);
        a.buildIndex(dataA.data(), N, DIM);
        a.setNumThreads(1);

        std::vector<idx_t> alone_I((size_t)NQ * K);
        std::vector<float> alone_D((size_t)NQ * K);
        a.searchIndex(queryA.data(), NQ, K, alone_I.data(), alone_D.data());

        // Building B installs B's tables into the legacy global.
        Algo b(daisy::DistanceType::L2_SQUARED);
        b.setNormalized(false);
        b.buildIndex(dataB.data(), N, DIM);
        b.setNumThreads(1);

        std::vector<idx_t> after_I((size_t)NQ * K);
        std::vector<float> after_D((size_t)NQ * K);
        a.searchIndex(queryA.data(), NQ, K, after_I.data(), after_D.data());

        for (size_t i = 0; i < alone_D.size(); i++)
        {
            EXPECT_EQ(after_I[i], alone_I[i])
                << "result " << i << " changed once a second index existed";
            EXPECT_FLOAT_EQ(after_D[i], alone_D[i])
                << "result " << i << " changed once a second index existed";
        }
        for (int q = 0; q < NQ; ++q)
        {
            std::set<idx_t> alone(alone_I.begin() + q * K, alone_I.begin() + (q + 1) * K);
            std::set<idx_t> after(after_I.begin() + q * K, after_I.begin() + (q + 1) * K);
            EXPECT_EQ(alone.size(), static_cast<size_t>(K));
            EXPECT_EQ(after.size(), static_cast<size_t>(K));
            for (int j = 0; j < K; ++j)
            {
                EXPECT_EQ(alone.count(expected_I[q * K + j]), 1u);
                EXPECT_EQ(after.count(expected_I[q * K + j]), 1u);
                EXPECT_TRUE(distClose(alone_D[q * K + j], expected_D[q * K + j]));
                EXPECT_TRUE(distClose(after_D[q * K + j], expected_D[q * K + j]));
            }
        }
    }
};

TEST_F(TwoIndexEquidepthTest, MessiQueriesAreUnaffectedByASecondIndex)
{
    expectSecondIndexDoesNotPerturbFirst<daisy::Messi>();
}

TEST_F(TwoIndexEquidepthTest, LbBruteforceQueriesAreUnaffectedByASecondIndex)
{
    expectSecondIndexDoesNotPerturbFirst<daisy::LbBruteforce>();
}

// Concurrent queries against two indices with different breakpoints. A process-global
// could not express this at all: the two searches would race to install their tables.
TEST_F(TwoIndexEquidepthTest, ConcurrentQueriesOnTwoIndicesStayStable)
{
    daisy::LbBruteforce a(daisy::DistanceType::L2_SQUARED);
    a.setNormalized(false);
    a.buildIndex(dataA.data(), N, DIM);
    a.setNumThreads(1);

    std::vector<idx_t> want_I((size_t)NQ * K);
    std::vector<float> want_D((size_t)NQ * K);
    a.searchIndex(queryA.data(), NQ, K, want_I.data(), want_D.data());

    daisy::LbBruteforce b(daisy::DistanceType::L2_SQUARED);
    b.setNormalized(false);
    b.buildIndex(dataB.data(), N, DIM);
    b.setNumThreads(1);

    constexpr int ROUNDS = 40;
    std::atomic<bool> stop{false};

    // Keep B querying so its searches overlap A's for the whole run.
    std::thread noise([&]
    {
        std::vector<idx_t> I((size_t)NQ * K);
        std::vector<float> D((size_t)NQ * K);
        while (!stop)
            b.searchIndex(dataB.data(), NQ, K, I.data(), D.data());
    });

    std::vector<idx_t> I((size_t)NQ * K);
    std::vector<float> D((size_t)NQ * K);
    for (int r = 0; r < ROUNDS; r++)
    {
        a.searchIndex(queryA.data(), NQ, K, I.data(), D.data());
        for (size_t i = 0; i < D.size(); i++)
        {
            ASSERT_EQ(I[i], want_I[i]) << "round " << r << ", result " << i;
            ASSERT_FLOAT_EQ(D[i], want_D[i]) << "round " << r << ", result " << i;
        }
    }
    stop = true;
    noise.join();
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
