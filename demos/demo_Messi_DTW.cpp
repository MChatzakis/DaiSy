#include "../commons/dataloaders.hpp"
#include "../lib/algos/Messi.hpp"
#include "../lib/algos/DataSource.hpp"
#include <chrono>

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

    // // 2. Create a brute-force search object
    // diNoLib::Messi messi_search(diNoLib::DistanceType::L2_SQUARED);
    // messi_search.setNumThreads(1); 

    // // 3. Build the index
    // diNoLib::InMemoryDataSource data_source(database, n_database, dim);
    // messi_search.buildIndex(&data_source);

    // // 4. Search the index
    // diNoLib::idx_t *I = new diNoLib::idx_t[n_query * k];
    // float *D = new float[n_query * k];
    // messi_search.searchIndex(query, n_query, k, I, D);

    
    // 2. Create a DTW search object
    diNoLib::Messi messi_search(diNoLib::DistanceType::DTW);
    messi_search.setNumThreads(4);

    int warp_window = std::max(1, static_cast<int>(dim * 0.1));
    messi_search.setWarpingWindow(warp_window);  // Set warping window (typically 10% of time series length)
    // 3. Build the index
    diNoLib::InMemoryDataSource data_source(database, n_database, dim);
    messi_search.buildIndex(&data_source);
    printf(">>> Finished indexing \n");

    // 4. Search the index
    printf("@ Starting search\n");
    diNoLib::idx_t *I = new diNoLib::idx_t[n_query * k];
    float *D = new float[n_query * k];
    printf("@ Starting search Variables are been set\n");
    printf("@ going searchIndex constructor\n");                
    messi_search.searchIndex(query, n_query, k, I, D);
    printf(">>> Finished search \n");

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