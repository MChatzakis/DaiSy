#include "../commons/dataloaders.hpp"
#include "../lib/algos/Messi.hpp"
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
    // messi_search.buildIndex(database, n_database, dim);

    // // 4. Search the index
    // diNoLib::idx_t *I = new diNoLib::idx_t[n_query * k];
    // float *D = new float[n_query * k];
    // messi_search.searchIndex(query, n_query, k, I, D);

    
    // 2. Create a brute-force search object
    diNoLib::Messi messi_search(diNoLib::DistanceType::L2_SQUARED);
    messi_search.setNumThreads(4);

    // 3. Build the index
    auto start_index = std::chrono::high_resolution_clock::now();
    messi_search.buildIndex(database, n_database, dim);
    auto end_index = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> index_duration = end_index - start_index;
    printf(">>> Finished indexing in %.4f seconds\n", index_duration.count());

    // 4. Search the index
    printf("Starting search\n");
    diNoLib::idx_t *I = new diNoLib::idx_t[n_query * k];
    float *D = new float[n_query * k];
    printf("Starting search Variables are been set\n");
    auto start_search = std::chrono::high_resolution_clock::now();
    messi_search.searchIndex(query, n_query, k, I, D);
    auto end_search = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> search_duration = end_search - start_search;
    printf(">>> Finished search in %.4f seconds\n", search_duration.count());

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