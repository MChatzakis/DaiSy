#include <benchmark/benchmark.h>
#include <cstdio>
#include "bm_utils.hpp"
#include "../commons/dataloaders.hpp"
#include "../commons/test_bm_utils.hpp"
#include <faiss/IndexFlat.h>

#ifdef _OPENMP
#include <omp.h>
#endif

struct FaissFlatSearchOnlyFixture : public benchmark::Fixture {
    faiss::IndexFlatL2* index = nullptr;
    float* query = nullptr;
    faiss::idx_t* I = nullptr;
    float* D = nullptr;
    faiss::idx_t n_query = 0;
    size_t k = 0;

    void SetUp(const benchmark::State& state) override {
        int config_idx = static_cast<int>(state.range(0));
        const SSTestConfig& config = test_configs_large[config_idx];

        std::string dataset_filename = pathToFilename(config.dataset_path);
        std::string query_filename = pathToFilename(config.query_path);

        daisy::idx_t dim, n_database, _, __;
        if (!parseFilenameForConfig(dataset_filename, "bruteForce", dim, n_database, _, __)) {
            std::cerr << "Failed to parse dataset config from filename: " << dataset_filename << std::endl;
            return;
        }

        daisy::idx_t dim_q, n_q, ___, ____;
        if (!parseFilenameForConfig(query_filename, "bruteForce", dim_q, n_q, ___, ____)) {
            std::cerr << "Failed to parse query config from filename: " << query_filename << std::endl;
            return;
        }

        if (dim != static_cast<daisy::idx_t>(dim_q)) {
            std::cerr << "Dimension mismatch between dataset and queries" << std::endl;
            return;
        }

        float* database = loadBinData(config.dataset_path.c_str(), n_database, dim, false);
        if (!database) {
            std::cerr << "Failed to load dataset" << std::endl;
            return;
        }

        query = loadBinData(config.query_path.c_str(), n_q, dim_q, false);
        if (!query) {
            std::cerr << "Failed to load queries" << std::endl;
            delete[] database;
            return;
        }

#ifdef _OPENMP
        omp_set_num_threads(config.thread_count);
#endif

        index = new faiss::IndexFlatL2(static_cast<int>(dim));
        index->add(static_cast<faiss::idx_t>(n_database), database);
        delete[] database;

        k = static_cast<size_t>(config.k_value);
        n_query = static_cast<faiss::idx_t>(n_q);
        I = new faiss::idx_t[n_query * k];
        D = new float[n_query * k];
    }

    void TearDown(const benchmark::State&) override {
        delete[] query;
        delete[] I;
        delete[] D;
        delete index;
    }
};

BENCHMARK_DEFINE_F(FaissFlatSearchOnlyFixture, BM_FaissFlat_SearchOnly)(benchmark::State& state) {
    for (auto _ : state) {
        index->search(n_query, query, static_cast<faiss::idx_t>(k), D, I);
    }
}

BENCHMARK_REGISTER_F(FaissFlatSearchOnlyFixture, BM_FaissFlat_SearchOnly)
    ->Arg(0)
    ->MinTime(2.0)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
