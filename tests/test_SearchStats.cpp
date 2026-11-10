#include <gtest/gtest.h>

#include <algos/Bruteforce.hpp>
#include <algos/Fresh.hpp>
#include <algos/Messi.hpp>
#include <algos/Sofa.hpp> // empty unless SOFA_FFTW_ENABLED
#if defined(ODYSSEY_MPI) && ODYSSEY_MPI
#include <algos/hodyssey/Odyssey.hpp>
#include <mpi.h>
#endif

#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <unistd.h> // getpid

// Search statistics (SimilaritySearchAlgorithm::getLastSearchStats): one entry
// per query of the last top-k call for MESSI, FreSh, SOFA and single-node Odyssey
// with L2 squared, and no entries where they are not collected.
//
// The counters depend on thread timing, so the checks are relations that hold on
// every run: enough work to produce the answer, no more than the data allows,
// pruning on clustered data, and a seed k-th distance never below the exact one.
//
// Run this binary directly, not under mpirun: Odyssey must run as a single node.
namespace
{
    constexpr daisy::idx_t kRows = 20000;
    constexpr daisy::idx_t kDim = 128;
    constexpr daisy::idx_t kQueries = 8;

    // 50 Gaussian clusters with unit noise: tight neighbourhoods that an iSAX or
    // SFA index with data-adaptive breakpoints can prune well.
    std::vector<float> clusteredData(daisy::idx_t rows, unsigned seed)
    {
        std::mt19937 centre_rng(42);
        std::normal_distribution<float> normal;
        std::vector<float> centres(50 * kDim);
        for (auto &v : centres)
            v = 6.0F * normal(centre_rng);
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> pick(0, 49);
        std::vector<float> data(static_cast<size_t>(rows) * kDim);
        for (daisy::idx_t r = 0; r < rows; ++r)
        {
            const float *centre = centres.data() + pick(rng) * kDim;
            for (daisy::idx_t d = 0; d < kDim; ++d)
                data[static_cast<size_t>(r) * kDim + d] = centre[d] + normal(rng);
        }
        return data;
    }

    const std::vector<float> &database()
    {
        static const std::vector<float> data = clusteredData(kRows, 1);
        return data;
    }

    const std::vector<float> &queries()
    {
        static const std::vector<float> data = clusteredData(kQueries, 2);
        return data;
    }

    // Runs top-k for k = 1 and 10 and checks the statistics of every query.
    // leaf_scans_once: every queued leaf is scanned at most once in the exact
    // phase (not FreSh, whose helper threads may scan a leaf again), so series
    // lower bounds cannot exceed the dataset size.
    void expectConsistentStats(daisy::SimilaritySearchAlgorithm &algo, bool leaf_scans_once)
    {
        for (daisy::idx_t k : {daisy::idx_t(1), daisy::idx_t(10)})
        {
            std::vector<daisy::idx_t> I(kQueries * k);
            std::vector<float> D(kQueries * k);
            algo.searchIndex(queries().data(), kQueries, k, I.data(), D.data());

            const std::vector<daisy::SearchStats> &stats = algo.getLastSearchStats();
            ASSERT_EQ(stats.size(), static_cast<size_t>(kQueries)) << "k=" << k;
            uint64_t total_distances = 0;
            for (daisy::idx_t q = 0; q < kQueries; ++q)
            {
                const daisy::SearchStats &s = stats[q];
                SCOPED_TRACE("k=" + std::to_string(k) + " query=" + std::to_string(q));
                EXPECT_GE(s.leaves_visited, 1u) << "the approximate seed scans one leaf";
                EXPECT_GE(s.distance_calcs, k) << "k neighbours need at least k real distances";
                EXPECT_GE(s.node_lb_calcs, s.leaves_enqueued) << "a queued leaf had its bound computed";
                // Seed leaf + refinement + exact phase: each scans a series at most once.
                EXPECT_LE(s.distance_calcs, 3 * kRows);
                if (leaf_scans_once)
                    EXPECT_LE(s.series_lb_calcs, kRows);
                // The exact answer can only improve on the seed.
                EXPECT_GE(s.seed_kth_dist, D[q * k + k - 1] * (1.0F - 1e-5F));
                total_distances += s.distance_calcs;
            }
            // On clustered data the index must prune: fewer real distances than a scan.
            EXPECT_LT(total_distances, kQueries * kRows) << "no pruning visible in distance_calcs, k=" << k;
        }

        // A call with fewer queries replaces the previous entries.
        std::vector<daisy::idx_t> I(3);
        std::vector<float> D(3);
        algo.searchIndex(queries().data(), 3, 1, I.data(), D.data());
        EXPECT_EQ(algo.getLastSearchStats().size(), 3u);
    }
}

TEST(SearchStatsTest, Messi)
{
    daisy::MessiConfig config;
    config.search_workers = 4;
    config.index_workers = 2;
    config.leaf_size = 500;
    daisy::Messi messi(daisy::DistanceType::L2_SQUARED, config);
    messi.setNormalized(false);
    messi.buildIndex(const_cast<float *>(database().data()), kRows, kDim);
    expectConsistentStats(messi, true);

    // Range search collects no statistics and clears the previous ones.
    daisy::SearchConfig range;
    range.type = daisy::QueryType::RANGE;
    range.r = 1.0F;
    std::vector<std::vector<daisy::idx_t>> RI;
    std::vector<std::vector<float>> RD;
    messi.searchIndex(queries().data(), 2, range, RI, RD);
    EXPECT_TRUE(messi.getLastSearchStats().empty());
}

TEST(SearchStatsTest, Fresh)
{
    daisy::FreshConfig config;
    config.search_workers = 4;
    config.index_workers = 2;
    config.leaf_size = 500;
    daisy::Fresh fresh(daisy::DistanceType::L2_SQUARED, config);
    fresh.setNormalized(false);
    fresh.buildIndex(const_cast<float *>(database().data()), kRows, kDim);
    expectConsistentStats(fresh, false);
}

#ifdef SOFA_FFTW_ENABLED
TEST(SearchStatsTest, Sofa)
{
    daisy::SofaConfig config;
    config.search_workers = 4;
    config.index_workers = 2;
    config.leaf_size = 500;
    config.min_leaf_size = 500;
    config.sample_size = static_cast<int>(kRows);
    daisy::Sofa sofa(daisy::DistanceType::L2_SQUARED, config);
    sofa.buildIndex(const_cast<float *>(database().data()), kRows, kDim);
    expectConsistentStats(sofa, true);
}
#endif

#if defined(ODYSSEY_MPI) && ODYSSEY_MPI
TEST(SearchStatsTest, OdysseySingleNode)
{
    // Odyssey builds from a raw float32 file.
    const std::string path = (std::filesystem::temp_directory_path() /
                              ("daisy_search_stats_" + std::to_string(::getpid()) + ".f32"))
                                 .string();
    FILE *f = std::fopen(path.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    ASSERT_EQ(std::fwrite(database().data(), sizeof(float), database().size(), f), database().size());
    std::fclose(f);

    daisy::OdysseyConfig config;
    config.search_workers = 4;
    config.query_threads = 4;
    config.index_threads = 2;
    config.leaf_size = 500;
    config.pq_th_div_factor = 4;
    daisy::Odyssey odyssey(config, daisy::DistanceType::L2_SQUARED, 0, nullptr);
    odyssey.buildIndex(path, kDim, kRows);
    std::filesystem::remove(path);
    expectConsistentStats(odyssey, true);
}
#endif

TEST(SearchStatsTest, EmptyWhereNotCollected)
{
    std::vector<daisy::idx_t> I(2);
    std::vector<float> D(2);

    daisy::BruteForceSearch brute(daisy::DistanceType::L2_SQUARED);
    brute.buildIndex(const_cast<float *>(database().data()), kRows, kDim);
    brute.searchIndex(queries().data(), 2, 1, I.data(), D.data());
    EXPECT_TRUE(brute.getLastSearchStats().empty());

    daisy::MessiConfig config;
    config.search_workers = 2;
    config.index_workers = 2;
    daisy::Messi messi_dtw(daisy::DistanceType::DTW, config);
    messi_dtw.buildIndex(const_cast<float *>(database().data()), 2000, kDim);
    messi_dtw.searchIndex(queries().data(), 2, 1, I.data(), D.data());
    EXPECT_TRUE(messi_dtw.getLastSearchStats().empty()) << "DTW search collects no statistics";
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
#if defined(ODYSSEY_MPI) && ODYSSEY_MPI
    int initialized = 0, finalized = 0; // Odyssey initializes MPI as a singleton
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (initialized && !finalized)
        MPI_Finalize();
#endif
    return result;
}
