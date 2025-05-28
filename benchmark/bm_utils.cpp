#include "bm_utils.hpp"

#include "../commons/dataloaders.hpp"

void runSSTBenchmark(
    diNoLib::SimilaritySearchAlgorithm* search,
    const std::string& dataset_path,
    const std::string& query_path,
    int num_thread,
    size_t k
) {

    std::string dataset_filename = pathToFilename(dataset_path);
    std::string query_filename = pathToFilename(query_path);

    diNoLib::idx_t dim, n_database, _;
    if (!parseFilenameForConfig(dataset_filename, "bruteFSS", dim, n_database, _, _)) {
        std::cerr << "Failed to parse dataset config from filename: " << dataset_filename << std::endl;
        return;
    }

    diNoLib::idx_t dim_q, n_query, __, ___;
    if (!parseFilenameForConfig(query_filename, "bruteFSS", dim_q, n_query, __, ___)) {
        std::cerr << "Failed to parse query config from filename: " << query_filename << std::endl;
        return;
    }

    if (dim != dim_q) {
        std::cerr << "Dimension mismatch between dataset and queries: "
                  << dim << " vs " << dim_q << std::endl;
        return;
    }

    float* database = loadBinData(dataset_path.c_str(), n_database, dim);
    float* query = loadBinData(query_path.c_str(), n_query, dim);

    search->buildIndex(database, n_database, dim);
    search->setNumThreads(num_thread);

    diNoLib::idx_t* I = new diNoLib::idx_t[n_query * k];
    float* D = new float[n_query * k];

    // Run the search without validation
    search->searchIndex(query, n_query, k, I, D);

    delete[] database;
    delete[] query;
    delete[] I;
    delete[] D;
}