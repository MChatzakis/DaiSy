#include "test_utils.hpp"
#include "../commons/test_bm_utils.hpp"
#include "../commons/paramSetup.hpp"
#if ODYSSEY_MPI
#include <mpi.h>
#endif

std::string prefix = "bruteForce";

// Odyssey needs argc/argv for MPI_Init; store them in main() and use in tests
static int g_argc = 0;
static char **g_argv = nullptr;

TEST_P(OdysseyParameterizedTest, AllConfigurations)
{
    const SSTestConfig &config = GetParam();
    diNoLib::DistanceType dist_L2Squared = diNoLib::DistanceType::L2_SQUARED;

    diNoLib::OdysseyConfig odyssey_config;
    odyssey_config.search_workers = 2;
    odyssey_config.index_threads = 2;
    odyssey_config.query_threads = 2;
    odyssey_config.leaf_size = 64;
    odyssey_config.paa_segments = 16;
    odyssey_config.replication_groups = 0;

    diNoLib::Odyssey search(odyssey_config, dist_L2Squared, g_argc, g_argv);

    std::string gt_I_path = config.gt_I_prefix + std::to_string(config.k_value) + ".txt";
    std::string gt_D_path = config.gt_D_prefix + std::to_string(config.k_value) + ".txt";

    runSST(
        &search,
        prefix,
        gt_I_path,
        gt_D_path,
        config.dataset_path,
        config.query_path,
        config.thread_count);
}

INSTANTIATE_TEST_SUITE_P(
    OdysseyTests,
    OdysseyParameterizedTest,
    ::testing::ValuesIn(test_configs),
    [](const ::testing::TestParamInfo<SSTestConfig> &info)
    {
        return info.param.name + "_k" + std::to_string(info.param.k_value) +
               "_thread" + std::to_string(info.param.thread_count) +
               "_idx" + std::to_string(info.index);
    });

int main(int argc, char **argv)
{
    g_argc = argc;
    g_argv = argv;
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
#if ODYSSEY_MPI
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized)
        MPI_Finalize();
#endif
    return result;
}
