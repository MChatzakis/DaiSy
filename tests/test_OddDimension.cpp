#include <gtest/gtest.h>

#include <algos/Fresh.hpp>
#include <algos/Messi.hpp>
#include <algos/Sofa.hpp> // empty unless SOFA_FFTW_ENABLED

#include <algorithm>
#include <memory>
#include <random>
#include <vector>

// Regression test: exact top-k on vectors whose dimension is not a multiple of 8 (the
// width of the SIMD distance kernels) or of 16 (MESSI's PAA segments). The kernels used to
// read past the end of each vector when dim % 8 != 0 (GloVe's 100 dimensions), and the
// iSAX lower bound was scaled by dim / 16 unrounded while PAA averages floor(dim / 16)
// points per segment, which made the bound too large for dim % 16 != 0 (100, 200).
// The reference is a scalar brute force computed here, not with the library's kernels.
namespace
{
    constexpr daisy::idx_t kRows = 20000;
    constexpr daisy::idx_t kQueries = 20;
    constexpr daisy::idx_t kK = 10;

    std::vector<float> clusteredData(daisy::idx_t rows, daisy::idx_t dim, unsigned seed)
    {
        std::mt19937 centre_rng(42);
        std::normal_distribution<float> normal;
        std::vector<float> centres(50 * dim);
        for (auto &v : centres)
            v = 4.0F * normal(centre_rng);
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> pick(0, 49);
        std::vector<float> data(rows * dim);
        for (daisy::idx_t r = 0; r < rows; ++r)
        {
            const float *centre = centres.data() + pick(rng) * dim;
            for (daisy::idx_t d = 0; d < dim; ++d)
                data[r * dim + d] = centre[d] + normal(rng);
        }
        return data;
    }

    // The k smallest squared L2 distances of a query, by a plain scalar scan.
    std::vector<float> bruteForce(const std::vector<float> &base, const float *query, daisy::idx_t dim)
    {
        std::vector<float> all(kRows);
        for (daisy::idx_t r = 0; r < kRows; ++r)
        {
            double s = 0;
            for (daisy::idx_t d = 0; d < dim; ++d)
            {
                const double diff = (double)base[r * dim + d] - (double)query[d];
                s += diff * diff;
            }
            all[r] = (float)s;
        }
        std::partial_sort(all.begin(), all.begin() + kK, all.end());
        all.resize(kK);
        return all;
    }

    void expectExact(daisy::SimilaritySearchAlgorithm &algo, daisy::idx_t dim)
    {
        auto base = clusteredData(kRows, dim, 1);
        auto queries = clusteredData(kQueries, dim, 2);
        algo.buildIndex(base.data(), kRows, dim);
        std::vector<daisy::idx_t> I(kQueries * kK);
        std::vector<float> D(kQueries * kK);
        algo.searchIndex(queries.data(), kQueries, kK, I.data(), D.data());
        for (daisy::idx_t q = 0; q < kQueries; ++q)
        {
            const std::vector<float> expected = bruteForce(base, queries.data() + q * dim, dim);
            for (daisy::idx_t j = 0; j < kK; ++j)
                EXPECT_NEAR(D[q * kK + j], expected[j], 1e-3F * std::max(1.0F, expected[j]))
                    << "dim " << dim << ", query " << q << ", rank " << j;
        }
    }
}

class OddDimensionTest : public ::testing::TestWithParam<daisy::idx_t>
{
};

TEST_P(OddDimensionTest, MessiIsExact)
{
    daisy::Messi messi(daisy::DistanceType::L2_SQUARED);
    messi.setSearchWorkers(4);
    messi.setIndexWorkers(2);
    messi.setNormalized(false);
    expectExact(messi, GetParam());
}

TEST_P(OddDimensionTest, FreshIsExact)
{
    daisy::Fresh fresh(daisy::DistanceType::L2_SQUARED);
    fresh.setSearchWorkers(4);
    fresh.setIndexWorkers(2);
    fresh.setNormalized(false);
    expectExact(fresh, GetParam());
}

#ifdef SOFA_FFTW_ENABLED
TEST_P(OddDimensionTest, SofaIsExact)
{
    daisy::SofaConfig config;
    config.search_workers = 4;
    config.index_workers = 2;
    config.leaf_size = 1000;
    config.min_leaf_size = 1000;
    config.sample_size = static_cast<int>(kRows);
    daisy::Sofa sofa(daisy::DistanceType::L2_SQUARED, config);
    expectExact(sofa, GetParam());
}
#endif

// 100: GloVe (not a multiple of 8 or 16); 200: T2I (multiple of 8, not of 16); 36: neither.
INSTANTIATE_TEST_SUITE_P(Dimensions, OddDimensionTest, ::testing::Values(100, 200, 36));
