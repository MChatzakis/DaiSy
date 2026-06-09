#include <benchmark/benchmark.h>
#include <cstdio>
#include <string>
#include "bm_utils.hpp"
#include "../commons/dataloaders.hpp"
#include "../commons/VectorDataLoader.h"
#include "../commons/test_bm_utils.hpp"
#include "../lib/algos/hsofa/Sofa.hpp"
#include "../lib/algos/DataSource.hpp"

static bool endsWith(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

struct SofaSearchOnlyFixture : public benchmark::Fixture {
    daisy::Sofa* search = nullptr;
    float* database = nullptr; 
    float* query = nullptr;
    daisy::idx_t* I = nullptr;
    float* D = nullptr;
    daisy::idx_t n_query = 0;
    size_t k = 0;
    std::string dataset_name;
    size_t n_database = 0;
    int thread_count = 0;

    void SetUp(const benchmark::State& state) override {
        int config_idx = static_cast<int>(state.range(0));
        const SSTestConfig& config = test_configs[config_idx];

        const bool use_fvecs = endsWith(config.dataset_path, ".fvecs") || endsWith(config.query_path, ".fvecs");
        size_t dim_u = 0, n_database_u = 0, n_q_u = 0;
        database = nullptr;

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

        search = new daisy::Sofa(daisy::DistanceType::L2_SQUARED);
        search->setNumThreads(config.thread_count);
        search->setIndexWorkers(config.thread_count);

        fprintf(stderr, "[SOFA] Before buildIndex (n_database=%zu dim=%zu).\n", n_database_u, dim_u);
        fflush(stderr);

        daisy::InMemoryDataSource data_source(database, static_cast<daisy::idx_t>(n_database_u), static_cast<daisy::idx_t>(dim_u));
        fprintf(stderr, "created data source\n");
        fflush(stderr);
        search->buildIndex(&data_source);

        fprintf(stderr, "[SOFA] Indexing finished (n_database=%zu dim=%zu).\n", n_database_u, dim_u);
        fflush(stderr);

        k = static_cast<size_t>(config.k_value);
        n_query = static_cast<daisy::idx_t>(n_q_u);
        I = new daisy::idx_t[n_query * k];
        D = new float[n_query * k];

        dataset_name = config.name;
        n_database = n_database_u;
        thread_count = config.thread_count;

        fprintf(stderr, "[SOFA] n_database=%zu n_query=%zu dim=%zu k=%zu threads=%d\n",
                n_database_u, (size_t)n_query, dim_u, k, config.thread_count);
        fflush(stderr);
    }

    void TearDown(const benchmark::State&) override {
        fprintf(stderr, "[SOFA] TearDown start.\n");
        fflush(stderr);
        delete search;
        delete[] database;
        delete[] query;
        delete[] I;
        delete[] D;
        search = nullptr;
        database = nullptr;
        query = nullptr;
        I = nullptr;
        D = nullptr;
        fprintf(stderr, "[SOFA] TearDown done.\n");
        fflush(stderr);
    }
};

BENCHMARK_DEFINE_F(SofaSearchOnlyFixture, BM_Sofa_SearchOnly)(benchmark::State& state) {
    for (auto _ : state) {
        fprintf(stderr, "[SOFA] --- Query phase ---\n");
        fprintf(stderr, "[SOFA]   dataset=%s  n_database=%zu\n", dataset_name.c_str(), n_database);
        fprintf(stderr, "[SOFA]   search_threads=%d  n_query=%zu  k=%zu\n", thread_count, (size_t)n_query, k);
        fflush(stderr);
        search->searchIndex(query, n_query, static_cast<daisy::idx_t>(k), I, D);
        fprintf(stderr, "[SOFA] Querying finished (n_query=%zu k=%zu).\n", (size_t)n_query, k);
        fflush(stderr);
    }
}

// test_configs: Astronomy + RandomWalk, threads={1,4,8}, k={1,10,100} => 18 configs (indices 0-17)
BENCHMARK_REGISTER_F(SofaSearchOnlyFixture, BM_Sofa_SearchOnly)
    ->Args({0})->Args({1})->Args({2})->Args({3})->Args({4})->Args({5})
    ->Args({6})->Args({7})->Args({8})->Args({9})->Args({10})->Args({11})
    ->Args({12})->Args({13})->Args({14})->Args({15})->Args({16})->Args({17})
    ->Iterations(1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
