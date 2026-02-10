#include "../commons/dataloaders.hpp"
#include "../lib/daisy.hpp"
#include <cstdio>
#include <cstring>
#include <string>

int main(){
    
    daisy::idx_t n_database = 200000;
    unsigned long long dim = 96;
    unsigned long long n_query = 10;
    daisy::idx_t k = 5;

    float *database = loadRandomData(n_database, dim, 100, true);
    float *query = loadRandomData(n_query, dim, 50, true);

    printf("Loaded %llu database points and %llu query points with dimension %llu\n", n_database, n_query, dim);

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

    daisy::ParIS paris_search(daisy::DistanceType::DTW);
    paris_search.setNumThreads(4);

    int warp_window = std::max(1, static_cast<int>(dim * 0.1));
    paris_search.setWarpingWindow(warp_window);

    paris_search.buildIndex(temp_db_file, dim, n_database);

    daisy::idx_t *I = new daisy::idx_t[n_query * k];
    float *D = new float[n_query * k];
    paris_search.searchIndex(query, n_query, k, I, D);

    for (daisy::idx_t i = 0; i < n_query; i++)
    {
        printf("Query %llu: ", i);
        for (daisy::idx_t j = 0; j < k; j++)
        {
            printf("%llu ", I[i * k + j]);
        }
        printf("\n");
    }

    delete[] database;
    delete[] query;
    delete[] I;
    delete[] D;
    remove(temp_db_file.c_str());  

    return 0;
}
