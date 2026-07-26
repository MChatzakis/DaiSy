// Streaming COCONUT: build on an initial batch, then insert more batches; after each step
// exact kNN must match brute-force ground truth over the data seen so far.

#include "test_utils.hpp"
#include "../lib/algos/Coconut.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using daisy::idx_t;

namespace
{
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

    // exact kNN distances (sorted) of `query` over the first `n_seen` series of `data`
    void bruteForceKnn(const std::vector<float> &data, int n_seen, int dim,
                       const float *query, int NQ, int K, std::vector<std::vector<float>> &out)
    {
        daisy::BruteForceSearch bf(daisy::DistanceType::L2_SQUARED);
        bf.buildIndex(const_cast<float *>(data.data()), n_seen, dim);
        std::vector<idx_t> gtI((size_t)NQ * K);
        std::vector<float> gtD((size_t)NQ * K);
        bf.searchIndex(const_cast<float *>(query), NQ, K, gtI.data(), gtD.data());
        out.assign(NQ, {});
        for (int q = 0; q < NQ; q++)
        {
            out[q].assign(gtD.begin() + (size_t)q * K, gtD.begin() + (size_t)(q + 1) * K);
            std::sort(out[q].begin(), out[q].end());
        }
    }

    void checkAgainstGT(daisy::Coconut &cc, const std::vector<float> &data, int n_seen, int dim,
                        const std::vector<float> &query, int NQ, int K, const char *phase)
    {
        std::vector<std::vector<float>> gt;
        bruteForceKnn(data, n_seen, dim, query.data(), NQ, K, gt);

        std::vector<idx_t> I((size_t)NQ * K);
        std::vector<float> D((size_t)NQ * K);
        cc.searchIndex(query.data(), NQ, K, I.data(), D.data());

        for (int q = 0; q < NQ; q++)
        {
            std::vector<float> od(D.begin() + (size_t)q * K, D.begin() + (size_t)(q + 1) * K);
            std::sort(od.begin(), od.end());
            for (int j = 0; j < K; j++)
                EXPECT_TRUE(distClose(od[j], gt[q][j]))
                    << phase << " query " << q << " nn " << j
                    << ": coconut " << od[j] << " vs GT " << gt[q][j];
        }
    }
}

TEST(CoconutTest, StreamingKnnMatchesBruteForce)
{
    const int DIM = 64, NQ = 20, K = 10;
    const int INITIAL = 800;
    const int BATCH = 400;
    const int BATCHES = 3;                 // total = 800 + 3*400 = 2000
    const int N = INITIAL + BATCHES * BATCH;

    auto data = genZNorm(N, DIM, 11);
    auto query = genZNorm(NQ, DIM, 111);

    daisy::Coconut cc(daisy::DistanceType::L2_SQUARED);
    cc.buildIndex(data.data(), INITIAL, DIM);   // static bottom-up build on the first batch
    cc.setNumThreads(1);
    checkAgainstGT(cc, data, INITIAL, DIM, query, NQ, K, "after build");

    int seen = INITIAL;
    for (int b = 0; b < BATCHES; b++)
    {
        cc.insertBatch(data.data() + (size_t)seen * DIM, BATCH);  // stream the next batch
        seen += BATCH;
        checkAgainstGT(cc, data, seen, DIM, query, NQ, K, "after insert");
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
