#include "test_utils.hpp"
#include "paramSetup.hpp"

std::string prefix = "bruteFSS";

TEST_P(LbBruteforceParameterizedTest, AllConfigurations) {
    const SSTestConfig& config = GetParam();
    diNoLib::DistanceType dist_L2Squared = diNoLib::DistanceType::L2_SQUARED;
    diNoLib::LbBruteforce search(dist_L2Squared);
    
    std::string gt_I_path = config.gt_I_prefix + std::to_string(config.k_value) + ".txt";
    std::string gt_D_path = config.gt_D_prefix + std::to_string(config.k_value) + ".txt";
    
    runSST(
        &search,
        prefix,
        gt_I_path,
        gt_D_path,
        config.dataset_path,
        config.query_path,
        config.thread_count
    );
}

INSTANTIATE_TEST_SUITE_P(
    LbBruteforceTests,
    LbBruteforceParameterizedTest,
    ::testing::ValuesIn(test_configs),
    [](const ::testing::TestParamInfo<SSTestConfig>& info) {
        return info.param.name + "_k" + std::to_string(info.param.k_value) + 
               "_thread" + std::to_string(info.param.thread_count);
    }
);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}