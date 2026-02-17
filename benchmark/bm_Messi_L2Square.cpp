#include <benchmark/benchmark.h>
#include <cstdio>
#include "bm_utils.hpp"
#include "../lib/algos/Messi.hpp"

struct MessiSearchOnlyFixture : public benchmark::Fixture {
    daisy::Messi* search = nullptr;
    float* query = nullptr;
    daisy::idx_t* I = nullptr;
    float* D = nullptr;
    daisy::idx_t n_query = 0;
    size_t k = 0;

    void SetUp(const benchmark::State& state) override {
        int config_idx = static_cast<int>(state.range(0));
        const SSTestConfig& config = test_configs_large[config_idx];

        search = new daisy::Messi(daisy::DistanceType::L2_SQUARED);
        runSSTBenchmarkSetup(
            search, config.dataset_path, config.query_path,
            config.thread_count, static_cast<size_t>(config.k_value),
            query, I, D, n_query);
        k = static_cast<size_t>(config.k_value);
        fprintf(stderr, "[MESSI] n_query=%zu k=%zu threads=%d\n",
                (size_t)n_query, k, config.thread_count);
    }

    void TearDown(const benchmark::State&) override {
        delete search;
        delete[] query;
        delete[] I;
        delete[] D;
        search = nullptr;
        query = nullptr;
        I = nullptr;
        D = nullptr;
    }
};

BENCHMARK_DEFINE_F(MessiSearchOnlyFixture, BM_Messi_SearchOnly)(benchmark::State& state) {
    for (auto _ : state) {
        search->searchIndex(query, n_query, k, I, D);
    }
}

BENCHMARK_REGISTER_F(MessiSearchOnlyFixture, BM_Messi_SearchOnly)
    ->Arg(1)   // Astronomy270M
    ->Arg(2)   // DEEP10m fbin (first config: 1 thread, k=1)
    ->Iterations(1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
