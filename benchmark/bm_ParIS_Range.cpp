#include <benchmark/benchmark.h>
#include <cstdio>
#include <vector>
#include "bm_utils.hpp"
#include "../commons/dataloaders.hpp"
#include "../commons/test_bm_utils.hpp"
#include "../lib/algos/ParIS.hpp"
#include "../lib/algos/SimilaritySearchAlgorithm.hpp"

struct ParISRangeFixture : public benchmark::Fixture {
    daisy::ParIS* search   = nullptr;
    float*        query    = nullptr;
    daisy::idx_t  n_query  = 0;
    float         r        = 0.0f;
    std::string   dataset_name;
    int           thread_count = 0;

    void SetUp(const benchmark::State& state) override {
        int config_idx = static_cast<int>(state.range(0));
        const RangeTestConfig& config = range_test_configs[config_idx];

        daisy::idx_t n_q = config.query_limit > 0
                               ? (daisy::idx_t)config.query_limit
                               : config.n_query;

        query = loadBinData(config.query_path.c_str(), n_q, config.dim, false);

        search = new daisy::ParIS(daisy::DistanceType::L2_SQUARED);
        search->setNumThreads(config.thread_count);

        fprintf(stderr, "[PARIS-RANGE] Before buildIndex (n=%llu dim=%llu).\n",
                (unsigned long long)config.n_database, (unsigned long long)config.dim);
        fflush(stderr);

        search->buildIndex(config.dataset_path, config.dim, config.n_database);

        fprintf(stderr, "[PARIS-RANGE] Indexing done. n_query=%llu r=%.2f threads=%d\n",
                (unsigned long long)n_q, config.r_value, config.thread_count);
        fflush(stderr);

        n_query      = n_q;
        r            = config.r_value;
        dataset_name = config.name;
        thread_count = config.thread_count;
    }

    void TearDown(const benchmark::State&) override {
        delete search; search = nullptr;
        delete[] query; query = nullptr;
    }
};

BENCHMARK_DEFINE_F(ParISRangeFixture, BM_ParIS_RangeSearch)(benchmark::State& state) {
    for (auto _ : state) {
        fprintf(stderr, "[PARIS-RANGE] dataset=%s threads=%d r=%.2f\n",
                dataset_name.c_str(), thread_count, r);
        fflush(stderr);
        daisy::SearchConfig cfg;
        cfg.type = daisy::QueryType::RANGE;
        cfg.r    = r;
        std::vector<std::vector<daisy::idx_t>> I;
        std::vector<std::vector<float>> D;
        search->searchIndex(query, n_query, cfg, I, D);
        fprintf(stderr, "[PARIS-RANGE] Done (n_query=%llu r=%.2f).\n",
                (unsigned long long)n_query, r);
        fflush(stderr);
    }
}

BENCHMARK_REGISTER_F(ParISRangeFixture, BM_ParIS_RangeSearch)
    ->Args({0})->Args({1})->Args({2})->Args({3})->Args({4})->Args({5})->Args({6})->Args({7})
    ->Args({8})->Args({9})->Args({10})->Args({11})->Args({12})->Args({13})->Args({14})->Args({15})
    ->Args({16})->Args({17})
    ->Iterations(1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
