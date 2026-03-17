    #include "../commons/dataloaders.hpp"
    #include "../lib/daisy.hpp"
    #include <chrono>
    #include <algorithm>

    int main() {
        daisy::idx_t n_database = 200000;
        unsigned long long dim = 96;
        unsigned long long n_query = 10;
        daisy::idx_t k = 1; // current Sing DTW implementation supports only k = 1

        float *database = loadRandomData(n_database, dim, 100, true);
        float *query = loadRandomData(n_query, dim, 50, true);

        daisy::Sing sing_search(daisy::DistanceType::DTW);
        sing_search.setNumThreads(1);

        // Sing::buildIndex expects a DataSource*, use InMemoryDataSource as in other components.
        daisy::InMemoryDataSource data_source(database, n_database, dim);
        printf("indexing\n");
        sing_search.buildIndex(&data_source);
        printf("finished indexing\n");

        daisy::idx_t *I = new daisy::idx_t[n_query * k];
        float *D = new float[n_query * k];
        printf("querying\n");
        sing_search.searchIndex(query, n_query, k, I, D);
        printf("finished querying\n");

        for (daisy::idx_t i = 0; i < n_query; i++) {
            printf("Query %llu:\n", i);
            printf("  Distances: ");
            for (daisy::idx_t j = 0; j < k; j++) {
                printf("%f ", D[i * k + j]);
            }
        }

        delete[] database;
        delete[] query;
        delete[] I;
        delete[] D;

        return 0;
    }

