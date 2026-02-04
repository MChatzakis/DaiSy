#include "../commons/dataloaders.hpp"
#include "../lib/algos/Sing.hpp"
#include "../lib/algos/DataSource.hpp"
#include <chrono>
#include <cstdio>

int main(){
    // 0. Configuration (stesso dataset della demo Messi)
    diNoLib::idx_t n_database = 200000;
    unsigned long long dim = 96;
    unsigned long long n_query = 10;
    diNoLib::idx_t k = 5;

    printf("=== Sing L2Square demo (stesso dataset di demo_Messi_L2Square) ===\n");
    printf("n_database=%llu dim=%llu n_query=%llu k=%llu\n", n_database, dim, n_query, k);

    // 1. Generate random data and queries (stessi seed di Messi: 100 database, 50 query)
    float *database = loadRandomData(n_database, dim, 100, true);
    float *query = loadRandomData(n_query, dim, 50, true);
    printf("Loaded %llu database points and %llu query points with dimension %llu\n", n_database, n_query, dim);

    // 2. Create Sing search object
    diNoLib::Sing sing_search(diNoLib::DistanceType::L2_SQUARED);

    // 3. Build the index (con stampe di debug dai worker)
    diNoLib::InMemoryDataSource data_source(database, n_database, dim);
    auto t0 = std::chrono::steady_clock::now();
    sing_search.buildIndex(&data_source);
    auto t1 = std::chrono::steady_clock::now();
    double build_ms = 1e-6 * (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    printf("buildIndex done in %.2f ms\n", build_ms);

    sing_search.printBuildIndexDebug();

    // 4. Search the index
    diNoLib::idx_t *I = new diNoLib::idx_t[n_query * k];
    float *D = new float[n_query * k];
    sing_search.searchIndex(query, n_query, k, I, D);

    // 5. Print the results
    for (diNoLib::idx_t i = 0; i < n_query; i++)
    {
        printf("Query %llu: ", i);
        for (diNoLib::idx_t j = 0; j < k; j++)
        {
            printf("%llu ", I[i * k + j]);
        }
        printf("\n");
    }

    // 6. Clean up
    delete[] database;
    delete[] query;
    delete[] I;
    delete[] D; 

    return 0;
}