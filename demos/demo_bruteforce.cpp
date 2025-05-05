#include <stdio.h>

#include "../commons/dataloaders.hpp"
#include "../lib/algos/BruteforceSearch.hpp"

int main(){
    // 1. Load dummy data and queries
    diNoLib::idx_t n_database = 200000;
    diNoLib::idx_t dim = 96;
    const char *dataset_name = "/home/mchatzakis/diNoSimilaritySearch/data/data.randwalk.len96.size200000.znorm.bin";
    float *database = loadBinData(dataset_name, n_database, dim);

    diNoLib::idx_t n_query = 5;
    const char *query_name = "/home/mchatzakis/diNoSimilaritySearch/data/query.randwalk.len96.size1000.bin";
    float *query = loadBinData(query_name, n_query, dim);

    printf("Loaded %llu database points and %llu query points with dimension %llu\n", n_database, n_query, dim);

    // 2. Create a brute-force search object
    diNoLib::BruteForceSearch bf_search(diNoLib::DistanceType::L2_SQUARED);

    // 3. Build the index
    bf_search.buildIndex(database, n_database, dim);

    // 4. Search the index
    diNoLib::idx_t k = 10;
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