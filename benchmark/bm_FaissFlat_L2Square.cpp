#include <benchmark/benchmark.h>
#include <faiss/IndexFlat.h>
#include <omp.h>

#include "../commons/dataloaders.hpp"
#include "../commons/paramSetup.hpp"
#include "../commons/test_bm_utils.hpp"

static void BM_FaissFlat(benchmark::State& state) {
    int config_idx = static_cast<int>(state.range(0));
    const SSTestConfig& config = test_configs[config_idx];

    std::string dataset_filename = pathToFilename(config.dataset_path);
    std::string query_filename = pathToFilename(config.query_path);

    daisy::idx_t dim, nb, _nq, __;
    if (!parseFilenameForConfig(dataset_filename, "bruteForce", dim, nb, __, __)) {
        state.SkipWithError("Failed to parse dataset config from filename");
        return;
    }

    daisy::idx_t dim_q, nq, ___, ____;
    if (!parseFilenameForConfig(query_filename, "bruteForce", dim_q, nq, ___, ____)) {
        state.SkipWithError("Failed to parse query config from filename");
        return;
    }

    if (dim != dim_q) {
        state.SkipWithError("Dimension mismatch between dataset and queries");
        return;
    }

    size_t k = static_cast<size_t>(config.k_value);
    omp_set_num_threads(config.thread_count);

    state.PauseTiming();
    float* xb = loadBinData(config.dataset_path.c_str(), nb, dim, false);
    float* xq = loadBinData(config.query_path.c_str(), nq, dim, false);

    faiss::IndexFlatL2 index(static_cast<faiss::idx_t>(dim));
    index.add(static_cast<faiss::idx_t>(nb), xb);

    faiss::idx_t* I = new faiss::idx_t[nq * k];
    float* D = new float[nq * k];
    state.ResumeTiming();

    for (auto _ : state) {
        index.search(static_cast<faiss::idx_t>(nq), xq, static_cast<faiss::idx_t>(k), D, I);
    }

    state.PauseTiming();
    delete[] I;
    delete[] D;
    delete[] xb;
    delete[] xq;
    state.ResumeTiming();
}

BENCHMARK(BM_FaissFlat)->Arg(0)->MinTime(2.0)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
