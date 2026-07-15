#include "test_utils.hpp"
#include "../commons/test_bm_utils.hpp"
#include "../commons/paramSetup.hpp"
#if ODYSSEY_MPI
#include <mpi.h>
#endif

static int g_argc = 0;
static char **g_argv = nullptr;

TEST_P(OdysseyRangeParameterizedTest, AllConfigurations)
{
    const RangeTestConfig &config = GetParam();

    daisy::OdysseyConfig odyssey_config;
    odyssey_config.search_workers  = 2;
    odyssey_config.index_threads   = 2;
    odyssey_config.query_threads   = 2;
    odyssey_config.leaf_size       = 256;
    odyssey_config.paa_segments    = 16;
    odyssey_config.replication_groups = 0;
    odyssey_config.pq_th_div_factor   = 4;

    daisy::Odyssey search(odyssey_config, daisy::DistanceType::L2_SQUARED, g_argc, g_argv);
    runSSTRange(&search, config);
}

INSTANTIATE_TEST_SUITE_P(
    OdysseyRangeTests,
    OdysseyRangeParameterizedTest,
    ::testing::ValuesIn(range_test_configs),
    [](const ::testing::TestParamInfo<RangeTestConfig> &info) {
        return info.param.name + "_r" + std::to_string((int)info.param.r_value) +
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
