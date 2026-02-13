#include <benchmark/benchmark.h>
#include <cstdio>
#include <faiss/IndexFlat.h>
#include <omp.h>

#include "../commons/dataloaders.hpp"
#include "../commons/paramSetup.hpp"
#include "../commons/test_bm_utils.hpp"

static void runFaissFlatBenchmark(
    const std::string& dataset_path,
    const std::string& query_path,
    int num_thread,
    size_t k)
{
    std::string dataset_filename = pathToFilename(dataset_path);
    std::string query_filename = pathToFilename(query_path);

    daisy::idx_t dim, n_database, _, __;
    if (!parseFilenameForConfig(dataset_filename, "bruteForce", dim, n_database, _, __)) {
        return;
    }

    daisy::idx_t dim_q, n_query, ___, ____;
    if (!parseFilenameForConfig(query_filename, "bruteForce", dim_q, n_query, ___, ____)) {
        return;
    }

    if (dim != dim_q) return;

    float* xb = loadBinData(dataset_path.c_str(), n_database, dim, false);
    float* xq = loadBinData(query_path.c_str(), n_query, dim, false);

    faiss::IndexFlatL2 index(static_cast<faiss::idx_t>(dim));
    index.add(static_cast<faiss::idx_t>(n_database), xb);
    fprintf(stderr, ">>> Finished indexing\n");

    omp_set_num_threads(num_thread);

    faiss::idx_t* I = new faiss::idx_t[n_query * k];
    float* D = new float[n_query * k];

    index.search(static_cast<faiss::idx_t>(n_query), xq, static_cast<faiss::idx_t>(k), D, I);
    fprintf(stderr, ">>> Finished querying.\n");

    delete[] xb;
    delete[] xq;
    delete[] I;
    delete[] D;
}

static void BM_FaissFlat(benchmark::State& state) {
    int config_idx = static_cast<int>(state.range(0));
    const SSTestConfig& config = test_configs[config_idx];

    for (auto _ : state) {
        runFaissFlatBenchmark(
            config.dataset_path,
            config.query_path,
            config.thread_count,
            static_cast<size_t>(config.k_value));
    }
}

BENCHMARK(BM_FaissFlat)->Arg(0)->MinTime(2.0)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
