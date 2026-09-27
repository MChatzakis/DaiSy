// FAISS streaming baseline.
//
// faiss::IndexFlatL2::add() is a pure append into the flat code storage: no
// training, no rebuild, and the index stays exact. That makes it the natural
// lower bound on what an incremental insert can cost, and the reference point
// for the DaiSy streaming algorithms (Bruteforce, LbBruteforce, Messi, Coconut).
//
// The series generator, DIM, INITIAL and the batch sizes mirror
// bm_Messi_Streaming.cpp so the insert numbers are directly comparable.
//
// Insert-only numbers alone are misleading: a flat index buys its cheap insert
// by paying O(n) per query. BM_Faiss_StreamingMixed therefore interleaves
// inserts and queries, which is where an indexed competitor earns its keep.

#include <benchmark/benchmark.h>

#include <faiss/IndexFlat.h>

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

static void BM_Faiss_InsertBatch(benchmark::State &state)
{
    const faiss::idx_t batch_size = static_cast<faiss::idx_t>(state.range(0));
    const auto initial = makeSeries(INITIAL, 0);
    const auto batch = makeSeries(static_cast<int>(batch_size), INITIAL);

    faiss::IndexFlatL2 index(DIM);
    index.add(INITIAL, initial.data());

    // One untimed insert absorbs the reallocation that the first add() after a
    // bulk build triggers (~2ms at INITIAL=4096, measured). Inside the loop that
    // single spike dominates: it alone accounts for ~108us of an apparent 113us
    // mean over 20 iterations, while steady-state add(1) is ~0.2us.
    index.add(batch_size, batch.data());

    for (auto _ : state)
    {
        index.add(batch_size, batch.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(batch_size));
}

BENCHMARK(BM_Faiss_InsertBatch)
    ->Arg(1)
    ->Arg(64)
    ->Arg(1024)
    ->Iterations(20)
    ->Unit(benchmark::kMicrosecond);

// One insert batch followed by one top-k query round, repeated. This is the
// workload a streaming index actually serves, and it exposes the O(n) scan that
// the flat index pays back on every query.
static void BM_Faiss_StreamingMixed(benchmark::State &state)
{
    const faiss::idx_t batch_size = static_cast<faiss::idx_t>(state.range(0));
    const auto initial = makeSeries(INITIAL, 0);
    const auto batch = makeSeries(static_cast<int>(batch_size), INITIAL);
    const auto queries = makeSeries(N_QUERY, INITIAL / 2);

    faiss::IndexFlatL2 index(DIM);
    index.add(INITIAL, initial.data());

    std::vector<faiss::idx_t> I(static_cast<size_t>(N_QUERY) * K);
    std::vector<float> D(static_cast<size_t>(N_QUERY) * K);

    index.add(batch_size, batch.data());
    index.search(N_QUERY, queries.data(), K, D.data(), I.data());

    for (auto _ : state)
    {
        index.add(batch_size, batch.data());
        index.search(N_QUERY, queries.data(), K, D.data(), I.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(batch_size));
    state.counters["n_database_final"] = static_cast<double>(index.ntotal);
}

BENCHMARK(BM_Faiss_StreamingMixed)
    ->Arg(1)
    ->Arg(64)
    ->Arg(1024)
    ->Iterations(20)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
