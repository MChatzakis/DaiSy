// Every top-k entry point must reject k > number of indexed series with
// std::invalid_argument (ValueError in Python) instead of returning padded or
// placeholder results. k == n must keep working.
#include "test_utils.hpp"
#if DAISY_TEST_COCONUT
#include "../lib/algos/Coconut.hpp"
#endif

#include <cstdio>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

using daisy::idx_t;

namespace
{
    constexpr idx_t N = 64;
    constexpr idx_t DIM = 64;
    constexpr idx_t NQ = 2;

    std::vector<float> randomSeries(idx_t n, unsigned seed)
    {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> v(n * DIM);
        for (float &x : v)
            x = dist(rng);
        return v;
    }

    struct Fixture
    {
        std::vector<float> data = randomSeries(N, 7);
        std::vector<float> queries = randomSeries(NQ, 11);
        std::string data_file;

        Fixture()
        {
            data_file = ::testing::TempDir() + "daisy_topk_k_gt_n.bin";
            FILE *f = std::fopen(data_file.c_str(), "wb");
            std::fwrite(data.data(), sizeof(float), data.size(), f);
            std::fclose(f);
        }
        ~Fixture() { std::remove(data_file.c_str()); }
    };

    void expectKGreaterThanNRejected(daisy::SimilaritySearchAlgorithm &search, const float *queries)
    {
        std::vector<idx_t> I(NQ * (N + 1));
        std::vector<float> D(NQ * (N + 1));
        EXPECT_THROW(search.searchIndex(queries, NQ, N + 1, I.data(), D.data()), std::invalid_argument);

        daisy::SearchConfig config;
        config.k = N + 1;
        std::vector<std::vector<idx_t>> VI;
        std::vector<std::vector<float>> VD;
        EXPECT_THROW(search.searchIndex(queries, NQ, config, VI, VD), std::invalid_argument);

        EXPECT_NO_THROW(search.searchIndex(queries, NQ, N, I.data(), D.data()));
    }
}

class TopKGreaterThanN : public ::testing::TestWithParam<daisy::DistanceType>
{
protected:
    Fixture fx;
};

TEST_P(TopKGreaterThanN, BruteForceSearch)
{
    daisy::BruteForceSearch search(GetParam());
    search.buildIndex(fx.data.data(), N, DIM);
    expectKGreaterThanNRejected(search, fx.queries.data());
}

TEST_P(TopKGreaterThanN, LbBruteforce)
{
    daisy::LbBruteforce search(GetParam());
    search.buildIndex(fx.data.data(), N, DIM);
    expectKGreaterThanNRejected(search, fx.queries.data());
}

TEST_P(TopKGreaterThanN, Messi)
{
    daisy::Messi search(GetParam());
    search.buildIndex(fx.data.data(), N, DIM);
    expectKGreaterThanNRejected(search, fx.queries.data());
}

TEST_P(TopKGreaterThanN, ParIS)
{
    daisy::ParIS search(GetParam());
    search.buildIndex(fx.data_file, DIM, N);
    expectKGreaterThanNRejected(search, fx.queries.data());
}

TEST_P(TopKGreaterThanN, Fresh)
{
    daisy::Fresh search(GetParam());
    search.buildIndex(fx.data.data(), N, DIM);
    expectKGreaterThanNRejected(search, fx.queries.data());
}

TEST_P(TopKGreaterThanN, DumpyOS)
{
    daisy::DumpyOS search(GetParam());
    search.buildIndex(fx.data.data(), N, DIM);
    expectKGreaterThanNRejected(search, fx.queries.data());
}

INSTANTIATE_TEST_SUITE_P(
    Metrics, TopKGreaterThanN,
    ::testing::Values(daisy::DistanceType::L2_SQUARED, daisy::DistanceType::DTW),
    [](const ::testing::TestParamInfo<daisy::DistanceType> &info)
    { return info.param == daisy::DistanceType::DTW ? std::string("DTW") : std::string("L2Squared"); });

// L2-only algorithms.
TEST(TopKGreaterThanNL2, Hercules)
{
    Fixture fx;
    daisy::HerculesConfig cfg;
    cfg.index_dir = ::testing::TempDir() + "daisy_topk_k_gt_n_hercules";
    daisy::Hercules search(daisy::DistanceType::L2_SQUARED, cfg);
    search.buildIndex(fx.data.data(), N, DIM);
    expectKGreaterThanNRejected(search, fx.queries.data());
}

#if DAISY_TEST_COCONUT
TEST(TopKGreaterThanNL2, Coconut)
{
    Fixture fx;
    daisy::Coconut search(daisy::DistanceType::L2_SQUARED);
    search.buildIndex(fx.data.data(), N, DIM);
    expectKGreaterThanNRejected(search, fx.queries.data());

    std::vector<idx_t> I(NQ * (N + 1));
    std::vector<float> D(NQ * (N + 1));
    EXPECT_THROW(search.searchIndexRecent(fx.queries.data(), NQ, N + 1, N, I.data(), D.data()),
                 std::invalid_argument);
}
#endif
