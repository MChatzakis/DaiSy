#include "test_utils.hpp"
#include "../lib/algos/Coconut.hpp"
#include "../commons/test_bm_utils.hpp"
#include "../commons/paramSetup.hpp"
#include <algorithm>
#include <cmath>

TEST_P(CoconutRangeParameterizedTest, AllConfigurations)
{
    const RangeTestConfig &config = GetParam();
    // Historical Astronomy fixture contains two non-finite series. It cannot
    // establish exact L2 range correctness under the finite-input contract.
    // Detect invalid fixture data explicitly; do not silently filter/reindex it.
    daisy::FileDataSource fixture(config.dataset_path.c_str(), config.dim, config.n_database);
    std::vector<float> record(config.dim);
    while (fixture.nextRecord(record.data()))
        if (std::any_of(record.begin(), record.end(), [](float x) { return !std::isfinite(x); }))
            GTEST_SKIP() << "Non-finite range fixture: " << config.dataset_path
                         << "; finite-input rejection is tested separately";
    for (int i = 0; i < 3; ++i) {
        daisy::Coconut search(daisy::DistanceType::L2_SQUARED);
        runSSTRange(&search, config);
    }
}

INSTANTIATE_TEST_SUITE_P(
    CoconutRangeTests,
    CoconutRangeParameterizedTest,
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
