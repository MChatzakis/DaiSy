#include "../commons/dataloaders.hpp"
#include "../lib/algos/BruteforceSearch.hpp"

#include <chrono>
#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>

void printResults(diNoLib::idx_t n_query, diNoLib::idx_t k, const diNoLib::idx_t *I, const char* label = "") 
{
    for (diNoLib::idx_t i = 0; i < n_query; ++i) 
    {
        if (label) printf("%s", label);
        printf("Query %llu: ", i);
        for (diNoLib::idx_t j = 0; j < k; ++j) 
        {
            printf("%llu ", I[i * k + j]);
        }
        printf("\n");
    }
}

void loadDataCHECK()
{
    // 0. Configuration of the variables
    const char *dataset_name = "../data/random.data.randwalk.len96.size200000.znorm.bin"; 
    const char *query_name = "../data/random.query.randwalk.len96.size1000.bin";

    diNoLib::idx_t n_database = 200000;
    diNoLib::idx_t dim = 96;
    unsigned long long n_query = 1000;
    diNoLib::idx_t k = 10;

    // 1. Load dummy data and queries
    float *database = loadBinData(dataset_name, n_database, dim);
    float *query = loadBinData(query_name, n_query, dim);

    printf("Loaded %llu database points and %llu query points with dimension %llu\n", n_database, n_query, dim);

    // 2. Create a brute-force search object
    diNoLib::BruteForceSearch bf_search(diNoLib::DistanceType::L2_SQUARED);

    // 3. Build the index
    bf_search.buildIndex(database, n_database, dim);

    // 4. Search the index
    diNoLib::idx_t *I = new diNoLib::idx_t[n_query * k];   
    float *D = new float[n_query * k];

    bf_search.searchIndex(query, n_query, k, I, D);

    // 5. Print the results
    printResults(n_query, k, I);

    // 6. Clean up
    delete[] database;
    delete[] query;
    delete[] I;
    delete[] D;    
}

void genarateRandomDataCHECK()
{
    // 0. Configuration of the variables
    diNoLib::idx_t n_database = 200000;
    unsigned long long dim = 96;
    unsigned long long n_query = 1000;
    diNoLib::idx_t k = 10;

    // 1. Generate random data and queries
    float *database = loadRandomData(n_database, dim, true, 100);
    float *query = loadRandomData(n_query, dim, true, 50);

    printf("Loaded %llu database points and %llu query points with dimension %llu\n", n_database, n_query, dim);

    // 2. Create a brute-force search object
    diNoLib::BruteForceSearch bf_search(diNoLib::DistanceType::L2_SQUARED);

    // 3. Build the index
    bf_search.buildIndex(database, n_database, dim);

    // 4. Search the index
    diNoLib::idx_t *I = new diNoLib::idx_t[n_query * k];
    float *D = new float[n_query * k];
    bf_search.searchIndex(query, n_query, k, I, D);

    // 5. Print the results
    printResults(n_query, k, I);

    // 6. Clean up
    delete[] database;
    delete[] query;
    delete[] I;
    delete[] D;       
}

void checkThreadBruteForceSearch(const diNoLib::idx_t *I, diNoLib::idx_t *I2, float *D, float *D2, diNoLib::idx_t n_query, diNoLib::idx_t k) 
{
    bool ok = true;
    for (diNoLib::idx_t i = 0; i < n_query * k; ++i) {
        if (I[i] != I2[i] || std::abs(D[i] - D2[i]) > 1e-4f) {
            printf("Mismatch at index %llu: I = %llu, I2 = %llu --- D = %.5f, D2 = %.5f\n",
                   i, I[i], I2[i], D[i], D2[i]);
            ok = false;
        }
    }
}

void threadingCHECK()
{
    // 0. Configuration of the variables
    const char *dataset_name = "../data/random.data.randwalk.len96.size200000.znorm.bin"; 
    const char *query_name = "../data/random.query.randwalk.len96.size1000.bin";

    diNoLib::idx_t n_database = 200000;
    diNoLib::idx_t dim = 96;
    unsigned long long n_query = 1000;
    diNoLib::idx_t k = 10;

    // 1. Load dummy data and queries
    float *database = loadBinData(dataset_name, n_database, dim);
    float *query = loadBinData(query_name, n_query, dim);

    printf("Loaded %llu database points and %llu query points with dimension %llu\n", n_database, n_query, dim);

    // 2. Create a brute-force search object
    diNoLib::BruteForceSearch bf_search_singleThread(diNoLib::DistanceType::L2_SQUARED);
    diNoLib::BruteForceSearch bf_search_multiThread(diNoLib::DistanceType::L2_SQUARED);

    // 3. Build the index
    bf_search_singleThread.buildIndex(database, n_database, dim);
    bf_search_multiThread.buildIndex(database, n_database, dim);
    bf_search_multiThread.setNumThreads(4); // Setting the Thread number

    // 4. Search the index
    diNoLib::idx_t *I = new diNoLib::idx_t[n_query * k];
    diNoLib::idx_t *I2 = new diNoLib::idx_t[n_query * k];    
    
    float *D = new float[n_query * k];
    float *D2 = new float[n_query * k];   

    auto start = std::chrono::high_resolution_clock::now();
    bf_search_singleThread.searchIndex(query, n_query, k, I, D);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();

    auto start2 = std::chrono::high_resolution_clock::now();
    bf_search_multiThread.searchIndex(query, n_query, k, I2, D2);
    auto end2 = std::chrono::high_resolution_clock::now();
    double duration2 = std::chrono::duration<double>(end2 - start2).count();

    // 5. Print the results
    printResults(n_query, k, I);
    printResults(n_query, k, I2, "[Threaded] ");

    /*CHECKS*/
    printf("Search took %.4f s with %d threads\n", duration, 1);    

    printf("Search took %.4f s with %d threads\n", duration2, bf_search_multiThread.getNumThreads());    

    checkThreadBruteForceSearch(I, I2, D, D2, n_query, k);

    // 6. Clean up
    delete[] database;
    delete[] query;
    delete[] I;
    delete[] D;    
    delete[] I2;
    delete[] D2;     
}

int main(){
    loadDataCHECK();

    genarateRandomDataCHECK();
    
    threadingCHECK(); 

    return 0;
}