#include <benchmark/benchmark.h>
#include <cstdio>
#include <faiss/IndexFlat.h>
#include <omp.h>

#include "../commons/dataloaders.hpp"
#include "../commons/paramSetup.hpp"
#include "../commons/test_bm_utils.hpp"

struct FaissFlatSearchOnlyFixture : public benchmark::Fixture {
    float* xb = nullptr;
    float* xq = nullptr;
    faiss::IndexFlatL2* index = nullptr;
    faiss::idx_t* I = nullptr;
    float* D = nullptr;
    faiss::idx_t nq = 0;
    faiss::idx_t k = 0;

    void SetUp(const benchmark::State& state) override {
        int config_idx = static_cast<int>(state.range(0));
        const SSTestConfig& config = test_configs_large[config_idx];

        std::string dataset_filename = pathToFilename(config.dataset_path);
        std::string query_filename = pathToFilename(config.query_path);

        daisy::idx_t dim, nb, __, ___;
        if (!parseFilenameForConfig(dataset_filename, "bruteForce", dim, nb, __, ___)) return;

        daisy::idx_t dim_q, nq_val, ____, _____;
        if (!parseFilenameForConfig(query_filename, "bruteForce", dim_q, nq_val, ____, _____)) return;
        if (dim != dim_q) return;

        k = static_cast<faiss::idx_t>(config.k_value);
        nq = static_cast<faiss::idx_t>(nq_val);
        omp_set_num_threads(config.thread_count);

        xb = loadBinData(config.dataset_path.c_str(), nb, dim, false);
        xq = loadBinData(config.query_path.c_str(), nq_val, dim, false);

        index = new faiss::IndexFlatL2(static_cast<faiss::idx_t>(dim));
        index->add(static_cast<faiss::idx_t>(nb), xb);
        fprintf(stderr, ">>> Finished indexing\n");

        I = new faiss::idx_t[nq * k];
        D = new float[nq * k];
    }

    void TearDown(const benchmark::State&) override {
        delete[] I;
        delete[] D;
        delete[] xb;
        delete[] xq;
        delete index;
    }
};

BENCHMARK_DEFINE_F(FaissFlatSearchOnlyFixture, BM_FaissFlat_SearchOnly)(benchmark::State& state) {
    for (auto _ : state) {
        index->search(nq, xq, k, D, I);
    }
}

BENCHMARK_REGISTER_F(FaissFlatSearchOnlyFixture, BM_FaissFlat_SearchOnly)
    ->Arg(0)
    ->MinTime(2.0)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
