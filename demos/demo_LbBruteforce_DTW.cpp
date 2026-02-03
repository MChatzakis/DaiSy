#include "../commons/dataloaders.hpp"
#include "../lib/algos/LbBruteforce.hpp"

int main()
{
    // 0. Configuration of the variables
    diNoLib::idx_t n_database = 200000;
    unsigned long long dim = 96;
    unsigned long long n_query = 10;
    diNoLib::idx_t k = 5;

    // 1. Generate random data and queries
    float *database = loadRandomData(n_database, dim, 100, true);
    float *query = loadRandomData(n_query, dim, 50, true);

    printf("Loaded %llu database points and %llu query points with dimension %llu\n", n_database, n_query, dim);

    // 2. Create a LbBruteforce search object
    diNoLib::LbBruteforce bf_search(diNoLib::DistanceType::DTW);

    // 3. Build the index (simplified API - no need for DataSource!)
    bf_search.buildIndex(database, n_database, dim);

    // 4. Search the index
    diNoLib::idx_t *I = new diNoLib::idx_t[n_query * k];
    float *D = new float[n_query * k];
    bf_search.searchIndex(query, n_query, k, I, D);

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