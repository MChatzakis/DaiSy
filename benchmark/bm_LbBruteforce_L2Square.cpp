#include <benchmark/benchmark.h>
#include "bm_utils.hpp"
#include "../lib/algos/LbBruteforce.hpp" 

static void BM_LbBruteforce(benchmark::State& state) {
    int config_idx = static_cast<int>(state.range(0));
    const SSTestConfig& config = test_configs[config_idx];

    diNoLib::LbBruteforce search(diNoLib::DistanceType::L2_SQUARED);


    for (auto _ : state) {
        runSSTBenchmark(&search, config.dataset_path, config.query_path, config.thread_count, config.k_value);
    }
}

BENCHMARK(BM_LbBruteforce)->Arg(0)->MinTime(2.0)->Unit(benchmark::kMillisecond); //TO BE CONFIRMED

BENCHMARK_MAIN();