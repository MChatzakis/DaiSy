#include <benchmark/benchmark.h>

#include "../lib/algos/Messi.hpp"

#include <cmath>
#include <vector>

namespace
{
    constexpr int DIM = 96;
    constexpr int INITIAL = 4096;

    std::vector<float> makeSeries(int n, int phase_offset)
    {
        std::vector<float> data(static_cast<size_t>(n) * DIM);
        for (int i = 0; i < n; ++i)
        {
            float *series = data.data() + static_cast<size_t>(i) * DIM;
            const float phase = static_cast<float>(phase_offset + i) * 0.013f;
            for (int j = 0; j < DIM; ++j)
                series[j] = std::sin(0.07f * j + phase) +
                            0.5f * std::cos(0.19f * j - phase);
        }
        return data;
    }
}

static void BM_Messi_InsertBatch(benchmark::State &state)
{
    state.PauseTiming();
    const daisy::idx_t batch_size = static_cast<daisy::idx_t>(state.range(0));
    auto initial = makeSeries(INITIAL, 0);
    auto batch = makeSeries(static_cast<int>(batch_size), INITIAL);

    daisy::MessiConfig config;
    config.index_workers = 2;
    config.search_workers = 2;
    daisy::Messi search(daisy::DistanceType::L2_SQUARED, config);
    search.buildIndex(initial.data(), INITIAL, DIM);
    state.ResumeTiming();

    for (auto _ : state)
    {
        search.insertBatch(batch.data(), batch_size);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(batch_size));
}

BENCHMARK(BM_Messi_InsertBatch)
    ->Arg(1)
    ->Arg(64)
    ->Arg(1024)
    ->Iterations(20)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
