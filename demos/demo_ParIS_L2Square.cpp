#include "../commons/dataloaders.hpp"
#include "../lib/algos/ParIS.hpp"
#include <cstdio>
#include <cstring>
#include <string>

int main(){
    // 0. Configuration of the variables
    daisy::idx_t n_database = 200000;
    unsigned long long dim = 96;
    unsigned long long n_query = 10;
    daisy::idx_t k = 5;

    // 1. Generate random data and queries (same as LbBruteforce demo)
    float *database = loadRandomData(n_database, dim, 100, true);
    float *query = loadRandomData(n_query, dim, 50, true);

    printf("Loaded %llu database points and %llu query points with dimension %llu\n", n_database, n_query, dim);

    // 2. Write database to temporary file (ParIS requires file-based data)
    std::string temp_db_file = "/tmp/paris_test_db.bin";
    FILE *fp = fopen(temp_db_file.c_str(), "wb");
    if (fp == nullptr) {
        fprintf(stderr, "Error: Could not create temporary database file\n");
        delete[] database;
        delete[] query;
        return 1;
    }
    fwrite(database, sizeof(float), n_database * dim, fp);
    fclose(fp);

    // 3. Create ParIS search object
    daisy::ParIS paris_search(daisy::DistanceType::L2_SQUARED);
    paris_search.setNumThreads(4);

    // 4. Build the index (simplified API - just pass filename!)
    paris_search.buildIndex(temp_db_file, dim, n_database);

    // 5. Search the index
    daisy::idx_t *I = new daisy::idx_t[n_query * k];
    float *D = new float[n_query * k];
    paris_search.searchIndex(query, n_query, k, I, D);

    // 6. Print the results
    for (daisy::idx_t i = 0; i < n_query; i++)
    {
        printf("Query %llu: ", i);
        for (daisy::idx_t j = 0; j < k; j++)
        {
            printf("%llu ", I[i * k + j]);
        }
        printf("\n");
    }

    // 7. Clean up
    delete[] database;
    delete[] query;
    delete[] I;
    delete[] D;
    remove(temp_db_file.c_str());  // Remove temporary file

    return 0;
}
