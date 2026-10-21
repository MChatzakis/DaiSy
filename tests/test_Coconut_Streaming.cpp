// Streaming COCONUT: build on an initial batch, then insert more batches; after each step
// exact kNN must match brute-force ground truth over the data seen so far.

#include "test_utils.hpp"
#include "../lib/algos/Coconut.hpp"
#include "../lib/algos/CoconutBounds.hpp"

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

TEST(CoconutTest, DiskMergesPreserveExactPhysicalNeighbors)
{
    constexpr int dim = 12, total = 123, k = 10;
    std::vector<float> data(total * dim), query(3 * dim);
    for (int i = 0; i < total; ++i)
        for (int d = 0; d < dim; ++d)
            data[i * dim + d] = (float)((i * 17 + d * 13) % 101 - 50);
    for (int q = 0; q < 3; ++q)
        std::copy_n(data.data() + (q == 0 ? 0 : q == 1 ? 41 : 122) * dim, dim, query.data() + q * dim);
    daisy::Coconut index(daisy::DistanceType::L2_SQUARED);
    index.setPaaSegments(3); index.setBufferCapacity(7); index.setLeafSize(5);
    index.setNumThreads(3); index.buildIndex(data.data(), 19, dim);
    int previous = 19;
    for (int seen : {19, 37, 83, 123})
    {
        if (seen > previous) index.insertBatch(data.data() + previous * dim, seen - previous);
        EXPECT_LE(index.getPeakBufferedRecords(), 7u);
        EXPECT_LT(index.getBufferedRecords(), 7u);
        EXPECT_EQ(index.getDiskRecords() + index.getBufferedRecords(), (idx_t)seen);
        std::vector<idx_t> ids(3 * k); std::vector<float> distances(3 * k);
        index.searchIndex(query.data(), 3, k, ids.data(), distances.data());
        for (int q = 0; q < 3; ++q)
        {
            std::vector<std::pair<float, idx_t>> expected;
            for (int i = 0; i < seen; ++i)
            {
                float squared = 0;
                for (int d = 0; d < dim; ++d)
                {
                    float delta = data[i * dim + d] - query[q * dim + d]; squared += delta * delta;
                }
                expected.emplace_back(squared, i);
            }
            std::sort(expected.begin(), expected.end());
            for (int rank = 0; rank < k; ++rank)
            {
                EXPECT_EQ(distances[q * k + rank], expected[rank].first);
                EXPECT_EQ(ids[q * k + rank], expected[rank].second);
            }
        }
        previous = seen;
    }
    index.flush();
    EXPECT_EQ(index.getBufferedRecords(), 0u);
    EXPECT_EQ(index.getDiskRecords(), (idx_t)total);
    index.waitForMerges();
    EXPECT_GT(index.getMergeCount(), 0u);
    EXPECT_EQ(index.getResidentSummaryBytes(), (idx_t)total * 3);
    std::vector<idx_t> recent_ids(3 * k); std::vector<float> recent_distances(3 * k);
    index.searchIndexRecent(query.data(), 3, k, 17, recent_ids.data(), recent_distances.data());
    for (int q = 0; q < 3; ++q)
    {
        std::vector<std::pair<float, idx_t>> expected;
        for (int i = total - 17; i < total; ++i)
        {
            float distance = 0;
            for (int d = 0; d < dim; ++d)
            { float delta = data[i * dim + d] - query[q * dim + d]; distance += delta * delta; }
            expected.emplace_back(distance, i);
        }
        std::sort(expected.begin(), expected.end());
        for (int j = 0; j < k; ++j)
        { EXPECT_EQ(recent_ids[q * k + j], expected[j].second); EXPECT_EQ(recent_distances[q * k + j], expected[j].first); }
    }
}

TEST(CoconutTest, QueryBoundLookupMatchesReferenceAndPrunesSafely)
{
    std::mt19937 rng(112677);
    std::uniform_real_distribution<float> values(-4, 4);
    for (int segments : {1, 3, 16, 32, 48, 96})
        for (int bits = 1; bits <= 8; ++bits)
        {
            int alphabet = 1 << bits;
            std::vector<float> paa(segments);
            std::vector<daisy::sax_type> word(segments), cardinalities(segments, bits);
            for (auto &v : paa) v = values(rng);
            daisy::coconut_detail::QueryBounds lookup;
            lookup.reset(paa.data(), segments, bits, 6);
            for (int trial = 0; trial < 512; ++trial)
            {
                for (auto &v : word) v = rng() % alphabet;
                float expected = daisy::minidist_paa_to_isax(paa.data(), word.data(), cardinalities.data(),
                    bits, alphabet, segments, MINVAL, MAXVAL, 6, sax_breakpoints);
                EXPECT_EQ(lookup(word.data(), FLT_MAX), expected);
                for (float limit : {0.0f, expected * .5f, std::nextafter(expected, 0.0f), expected,
                                    std::nextafter(expected, FLT_MAX)})
                {
                    float actual = lookup(word.data(), limit);
                    if (expected <= limit) EXPECT_EQ(actual, expected);
                    else { EXPECT_GT(actual, limit); EXPECT_LE(actual, expected); }
                }
            }
        }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
