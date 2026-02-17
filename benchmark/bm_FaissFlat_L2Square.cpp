#include <benchmark/benchmark.h>
#include <cstdio>
#include <string>
#include "bm_utils.hpp"
#include "../commons/dataloaders.hpp"
#include "../commons/VectorDataLoader.h"
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
    std::string dataset_name;
    size_t n_database = 0;
    int thread_count = 0;

    static bool endsWith(const std::string& s, const std::string& suffix) {
        return s.size() >= suffix.size() &&
               s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    void SetUp(const benchmark::State& state) override {
        int config_idx = static_cast<int>(state.range(0));
        const SSTestConfig& config = test_configs_deep_seismic[config_idx];

        const bool use_fvecs = endsWith(config.dataset_path, ".fvecs") || endsWith(config.query_path, ".fvecs");
        size_t dim_u = 0, n_database_u = 0, n_q_u = 0;
        float* database = nullptr;

        if (use_fvecs) {
            database = fvecs_read(config.dataset_path.c_str(), &dim_u, &n_database_u, 0);
            if (!database) {
                std::cerr << "Failed to load dataset (fvecs)" << std::endl;
                return;
            }
            const size_t query_limit = (config.query_limit > 0) ? static_cast<size_t>(config.query_limit) : 0;
            query = fvecs_read(config.query_path.c_str(), &dim_u, &n_q_u, query_limit);
            if (!query) {
                std::cerr << "Failed to load queries (fvecs)" << std::endl;
                delete[] database;
                return;
            }
        } else {
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

            dim_u = static_cast<size_t>(dim);
            n_database_u = static_cast<size_t>(n_database);
            if (config.query_limit > 0 && static_cast<daisy::idx_t>(config.query_limit) < n_q)
                n_q = static_cast<daisy::idx_t>(config.query_limit);
            n_q_u = static_cast<size_t>(n_q);

            database = loadBinData(config.dataset_path.c_str(), n_database, dim, false);
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
        }

#ifdef _OPENMP
        omp_set_num_threads(config.thread_count);
#endif

        index = new faiss::IndexFlatL2(static_cast<int>(dim_u));
        index->add(static_cast<faiss::idx_t>(n_database_u), database);
        fprintf(stderr, ">>> Finished indexing\n");
        delete[] database;

        k = static_cast<size_t>(config.k_value);
        n_query = static_cast<faiss::idx_t>(n_q_u);
        I = new faiss::idx_t[n_query * k];
        D = new float[n_query * k];

        dataset_name = config.name;
        n_database = n_database_u;
        thread_count = config.thread_count;

        fprintf(stderr, "[FAISS] n_database=%zu n_query=%zu dim=%zu k=%zu threads=%d\n",
                n_database_u, n_query, dim_u, k, config.thread_count);
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
        fprintf(stderr, "[FAISS] --- Query phase ---\n");
        fprintf(stderr, "[FAISS]   dataset=%s  n_database=%zu\n", dataset_name.c_str(), n_database);
        fprintf(stderr, "[FAISS]   search_threads=%d  n_query=%zu  k=%zu\n", thread_count, (size_t)n_query, k);
        fflush(stderr);
        index->search(n_query, query, static_cast<faiss::idx_t>(k), D, I);
        fprintf(stderr, ">>> Finished querying\n");
    }
}

BENCHMARK_REGISTER_F(FaissFlatSearchOnlyFixture, BM_FaissFlat_SearchOnly)
    // DEEP100M+Seismic: q=100 all k (0-7), then k=10 all q (8-13)
    ->Args({0})->Args({1})->Args({2})->Args({3})->Args({4})->Args({5})->Args({6})->Args({7})
    ->Args({8})->Args({9})->Args({10})->Args({11})->Args({12})->Args({13})
    ->Iterations(1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
