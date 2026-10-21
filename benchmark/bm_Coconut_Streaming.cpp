#include <benchmark/benchmark.h>
#include "../commons/dataloaders.hpp"
#include "../lib/algos/Coconut.hpp"
#include <memory>
#include <vector>

// Match the existing streaming benchmark pattern. Unlike an enqueue-only
// measurement, wall time includes every flush and background compaction.
static void BM_Coconut_Streaming(benchmark::State &state)
{
    constexpr daisy::idx_t dim = 96, initial = 4096, queries = 4;
    const auto batch = static_cast<daisy::idx_t>(state.range(0));
    std::unique_ptr<float[]> data(loadRandomData(initial + batch, dim, 100, true));
    std::unique_ptr<float[]> query(loadRandomData(queries, dim, 50, true));
    daisy::Coconut index(daisy::DistanceType::L2_SQUARED);
    index.setBufferCapacity(512); index.setNumThreads(2);
    index.setQueryBatchSize(static_cast<daisy::idx_t>(state.range(1)));
    index.buildIndex(data.get(), initial, dim);
    std::vector<daisy::idx_t> ids(queries); std::vector<float> distances(queries);
    for (auto _ : state)
    {
        index.insertBatch(data.get() + initial * dim, batch);
        index.waitForMerges();
        index.searchIndex(query.get(), queries, 1, ids.data(), distances.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * batch);
    state.counters["n_database_final"] = static_cast<double>(index.getNDatabase());
    state.counters["resident_summary_bytes"] = static_cast<double>(index.getResidentSummaryBytes());
}
BENCHMARK(BM_Coconut_Streaming)->Args({1024, 1})->Args({1024, 32})->Args({8192, 32})->Iterations(5)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();
