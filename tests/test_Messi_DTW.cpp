#include "test_utils.hpp"
#include "../commons/test_bm_utils.hpp"
#include "../commons/paramSetup.hpp"

std::string prefix = "bruteForce";

TEST_P(LbBruteforceDTWParameterizedTest, AllConfigurations)
{
    const SSTestConfig &config = GetParam();
    diNoLib::DistanceType dist_DTW = diNoLib::DistanceType::DTW;
    diNoLib::Messi search(dist_DTW);

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
    LbBruteforceDTWTests,
    LbBruteforceDTWParameterizedTest,
    ::testing::ValuesIn(test_configs_dtw),
    [](const ::testing::TestParamInfo<SSTestConfig> &info)
    {
        return info.param.name + "_k" + std::to_string(info.param.k_value) +
               "_thread" + std::to_string(info.param.thread_count) +
               "_idx" + std::to_string(info.index); 

    });

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
