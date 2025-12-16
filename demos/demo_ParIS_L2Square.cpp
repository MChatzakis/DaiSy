#include "../commons/dataloaders.hpp"
#include "../lib/algos/ParIS.hpp"
#include "../lib/algos/DataSource.hpp"

int main(){
    // 0. Configuration of the variables
    diNoLib::idx_t n_database = 200000;
    unsigned long long dim = 96;
    unsigned long long n_query = 10;
    diNoLib::idx_t k = 5;
    const char *db_file = "../data/random.data.randwalk.len96.size200000.znorm.bin";
    const char *query_file = "../data/random.query.randwalk.len96.size1000.bin";

    // 1. Load queries from binary file
    float *query = loadBinData(query_file, n_query, dim);

    printf("Loaded %llu database points from %s\n", n_database, db_file);
    printf("Loaded %llu query points from %s with dimension %llu\n", n_query, query_file, dim);

    // 2. Create FileDataSource and ParIS search object
    diNoLib::FileDataSource file_source(db_file, dim, n_database);
    diNoLib::ParIS paris_search(diNoLib::DistanceType::L2_SQUARED);
    paris_search.setNumThreads(4);

    // 3. Build the index
    paris_search.buildIndex(&file_source);
    printf(">>> Finished indexing\n");

    // 4. Search the index
    diNoLib::idx_t *I = new diNoLib::idx_t[n_query * k];
    float *D = new float[n_query * k];
    paris_search.searchIndex(query, n_query, k, I, D);
    printf(">>> Finished search\n");

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
    delete[] query;
    delete[] I;
    delete[] D;

    return 0;
}
