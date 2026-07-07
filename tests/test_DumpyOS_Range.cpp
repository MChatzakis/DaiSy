#include "test_utils.hpp"
#include "../commons/test_bm_utils.hpp"
#include "../commons/paramSetup.hpp"

TEST_P(DumpyOSRangeParameterizedTest, AllConfigurations)
{
    const RangeTestConfig &config = GetParam();
    for (int i = 0; i < 3; ++i) {
        daisy::DumpyOS search(daisy::DistanceType::L2_SQUARED);
        runSSTRange(&search, config);
    }
}

INSTANTIATE_TEST_SUITE_P(
    DumpyOSRangeTests,
    DumpyOSRangeParameterizedTest,
    ::testing::ValuesIn(range_test_configs),
    [](const ::testing::TestParamInfo<RangeTestConfig> &info) {
        return info.param.name + "_r" + std::to_string((int)info.param.r_value) +
               "_thread" + std::to_string(info.param.thread_count) +
               "_idx" + std::to_string(info.index);
    });

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
