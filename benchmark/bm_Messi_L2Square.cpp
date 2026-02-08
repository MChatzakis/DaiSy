#include <benchmark/benchmark.h>
#include "bm_utils.hpp"
#include "../lib/algos/Messi.hpp" 

static void BM_Messi(benchmark::State& state) {
    int config_idx = static_cast<int>(state.range(0));
    const SSTestConfig& config = test_configs[config_idx];

    daisy::Messi search(daisy::DistanceType::L2_SQUARED);


    for (auto _ : state) {
        runSSTBenchmark(&search, config.dataset_path, config.query_path, config.thread_count, config.k_value);
    }
}

BENCHMARK(BM_Messi)->Arg(0)->MinTime(2.0)->Unit(benchmark::kMillisecond); //TO BE CONFIRMED

BENCHMARK_MAIN();