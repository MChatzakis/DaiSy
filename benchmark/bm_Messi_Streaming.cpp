#include <benchmark/benchmark.h>

#include "../lib/algos/Messi.hpp"

#include <cmath>
#include <vector>

namespace
{
    constexpr int DIM = 96;
    constexpr int INITIAL = 4096;
    constexpr int N_QUERY = 16;
    constexpr int K = 10;

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
    // Setup before the KeepRunning loop is already outside the timed region.
    // Calling PauseTiming() here instead corrupts the real-time measurement,
    // because the timer has not been started yet.
    const daisy::idx_t batch_size = static_cast<daisy::idx_t>(state.range(0));
    auto initial = makeSeries(INITIAL, 0);
    auto batch = makeSeries(static_cast<int>(batch_size), INITIAL);

    daisy::MessiConfig config;
    config.index_workers = 2;
    config.search_workers = 2;
    daisy::Messi search(daisy::DistanceType::L2_SQUARED, config);
    search.buildIndex(initial.data(), INITIAL, DIM);

    // One untimed insert absorbs the one-off costs of the first update: MESSI
    // converts the borrowed build buffer into owned, growable storage and grows
    // its SAX cache. Leaving it inside the loop would spread that fixed cost
    // over the iterations and understate steady-state insert throughput.
    search.insertBatch(batch.data(), batch_size);

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

// One insert batch followed by one top-k query round, repeated: the workload a
// streaming index actually serves. Insert-only numbers flatter a flat baseline,
// which buys its cheap append by rescanning the whole database on every query.
// The counterpart is BM_Faiss_StreamingMixed, with matching DIM/INITIAL/args.
// Inserts and queries are issued sequentially; MESSI does not support
// concurrent updates and searches.
static void BM_Messi_StreamingMixed(benchmark::State &state)
{
    const daisy::idx_t batch_size = static_cast<daisy::idx_t>(state.range(0));
    auto initial = makeSeries(INITIAL, 0);
    auto batch = makeSeries(static_cast<int>(batch_size), INITIAL);
    auto queries = makeSeries(N_QUERY, INITIAL / 2);

    daisy::MessiConfig config;
    config.index_workers = 2;
    config.search_workers = 2;
    daisy::Messi search(daisy::DistanceType::L2_SQUARED, config);
    search.buildIndex(initial.data(), INITIAL, DIM);

    std::vector<daisy::idx_t> I(static_cast<size_t>(N_QUERY) * K);
    std::vector<float> D(static_cast<size_t>(N_QUERY) * K);

    search.insertBatch(batch.data(), batch_size);
    search.searchIndex(queries.data(), N_QUERY, K, I.data(), D.data());

    for (auto _ : state)
    {
        search.insertBatch(batch.data(), batch_size);
        search.searchIndex(queries.data(), N_QUERY, K, I.data(), D.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(batch_size));
    state.counters["n_database_final"] = static_cast<double>(search.getNDatabase());
}

BENCHMARK(BM_Messi_StreamingMixed)
    ->Arg(1)
    ->Arg(64)
    ->Arg(1024)
    ->Iterations(20)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
