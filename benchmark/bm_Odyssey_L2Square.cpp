#include <benchmark/benchmark.h>
#include "bm_utils.hpp"
#include "../lib/algos/hodyssey/Odyssey.hpp"
#if ODYSSEY_MPI
#include <mpi.h>
#endif

static int g_argc = 0;
static char **g_argv = nullptr;

static void BM_Odyssey(benchmark::State& state) {
    int config_idx = static_cast<int>(state.range(0));
    const SSTestConfig& config = test_configs[config_idx];

    daisy::OdysseyConfig odyssey_config;
    odyssey_config.search_workers = 2;
    odyssey_config.index_threads = 2;
    odyssey_config.query_threads = 2;
    odyssey_config.leaf_size = 256;
    odyssey_config.paa_segments = 16;
    odyssey_config.replication_groups = 0;
    odyssey_config.pq_th_div_factor = 4;

    daisy::Odyssey search(odyssey_config, daisy::DistanceType::L2_SQUARED, g_argc, g_argv);

    for (auto _ : state) {
        runSSTBenchmark(&search, config.dataset_path, config.query_path, config.thread_count, config.k_value);
    }
}

BENCHMARK(BM_Odyssey)->Arg(0)->MinTime(2.0)->Unit(benchmark::kMillisecond);

int main(int argc, char **argv)
{
#if ODYSSEY_MPI
    MPI_Init(&argc, &argv);
#endif
    g_argc = argc;
    g_argv = argv;

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

#if ODYSSEY_MPI
    MPI_Finalize();
#endif
    return 0;
}
