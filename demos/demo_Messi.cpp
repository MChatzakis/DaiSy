#include "../commons/dataloaders.hpp"
#include "../lib/algos/Messi.hpp"

int main(){
    // 0. Configuration of the variables
    diNoLib::idx_t n_database = 200000;
    unsigned long long dim = 96;
    unsigned long long n_query = 10;
    diNoLib::idx_t k = 5;

    // 1. Generate random data and queries
    float *database = loadRandomData(n_database, dim, true, 100);
    float *query = loadRandomData(n_query, dim, true, 50);

    printf("Loaded %llu database points and %llu query points with dimension %llu\n", n_database, n_query, dim);

    // diNoLib::DistanceType dist_type = diNoLib::DistanceType::L2_SQUARED;
    // 2. Create a brute-force search object
    // diNoLib::Messi bf_search(diNoLib::DistanceType::L2_SQUARED);

    // // 3. Build the index
    // bf_search.buildIndex(database, n_database, dim);

    // // 4. Search the index
    // diNoLib::idx_t *I = new diNoLib::idx_t[n_query * k];
    // float *D = new float[n_query * k];
    // bf_search.searchIndex(query, n_query, k, I, D);

    // // 5. Print the results
    // for (diNoLib::idx_t i = 0; i < n_query; i++)
    // {
    //     printf("Query %llu: ", i);
    //     for (diNoLib::idx_t j = 0; j < k; j++)
    //     {
    //         printf("%llu ", I[i * k + j]);
    //     }
    //     printf("\n");
    // }

    // // 6. Clean up
    // delete[] database;
    // delete[] query;
    // delete[] I;
    // delete[] D; 

    return 0;
}