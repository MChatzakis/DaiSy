#include "../commons/dataloaders.hpp"
#include "../lib/daisy.hpp"

int main(){
    
    daisy::idx_t n_database = 200000;
    unsigned long long dim = 96;
    unsigned long long n_query = 10;
    daisy::idx_t k = 5;

    float *database = loadRandomData(n_database, dim, 100, true);
    float *query = loadRandomData(n_query, dim, 50, true);

    printf("Loaded %llu database points and %llu query points with dimension %llu\n", n_database, n_query, dim);

    daisy::LbBruteforce bf_search(daisy::DistanceType::L2_SQUARED);

    bf_search.buildIndex(database, n_database, dim);

    daisy::idx_t *I = new daisy::idx_t[n_query * k];
    float *D = new float[n_query * k];
    bf_search.searchIndex(query, n_query, k, I, D);

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

    return 0;
}