#include "bm_utils.hpp"

#include "../commons/dataloaders.hpp"
#include "../lib/algos/DataSource.hpp"
#include "../lib/algos/ParIS.hpp"
#if ODYSSEY_MPI
#include "../lib/algos/hodyssey/Odyssey.hpp"
#endif

void runSSTBenchmark(
    daisy::SimilaritySearchAlgorithm *search,
    const std::string &dataset_path,
    const std::string &query_path,
    int num_thread,
    size_t k)
{

    std::string dataset_filename = pathToFilename(dataset_path);
    std::string query_filename = pathToFilename(query_path);

    daisy::idx_t dim, n_database, _;
    if (!parseFilenameForConfig(dataset_filename, "bruteForce", dim, n_database, _, _))
    {
        std::cerr << "Failed to parse dataset config from filename: " << dataset_filename << std::endl;
        return;
    }

    daisy::idx_t dim_q, n_query, __, ___;
    if (!parseFilenameForConfig(query_filename, "bruteForce", dim_q, n_query, __, ___))
    {
        std::cerr << "Failed to parse query config from filename: " << query_filename << std::endl;
        return;
    }

    if (dim != dim_q)
    {
        std::cerr << "Dimension mismatch between dataset and queries: "
                  << dim << " vs " << dim_q << std::endl;
        return;
    }

    float *query = loadBinData(query_path.c_str(), n_query, dim, false);

#if ODYSSEY_MPI
    if (daisy::Odyssey *odyssey_search = dynamic_cast<daisy::Odyssey *>(search))
    {
        daisy::FileDataSource data_source(dataset_path.c_str(), dim, n_database);
        search->buildIndex(&data_source);
    }
    else
#endif
    if (daisy::ParIS *paris_search = dynamic_cast<daisy::ParIS *>(search))
    {
        search->buildIndex(dataset_path, dim, n_database);
    }
    else
    {
        float *database = loadBinData(dataset_path.c_str(), n_database, dim, false);
        daisy::InMemoryDataSource data_source(database, n_database, dim);
        search->buildIndex(&data_source);
        delete[] database;
    }

    search->setNumThreads(num_thread);

    daisy::idx_t *I = new daisy::idx_t[n_query * k];
    float *D = new float[n_query * k];

    search->searchIndex(query, n_query, k, I, D);

    delete[] query;
    delete[] I;
    delete[] D;
}

void runSSTBenchmarkSetup(
    daisy::SimilaritySearchAlgorithm* search,
    const std::string& dataset_path,
    const std::string& query_path,
    int num_thread,
    size_t k,
    float*& query_out,
    daisy::idx_t*& I_out,
    float*& D_out,
    daisy::idx_t& n_query_out)
{
    std::string dataset_filename = pathToFilename(dataset_path);
    std::string query_filename = pathToFilename(query_path);

    daisy::idx_t dim, n_database, _;
    if (!parseFilenameForConfig(dataset_filename, "bruteForce", dim, n_database, _, _))
    {
        std::cerr << "Failed to parse dataset config from filename: " << dataset_filename << std::endl;
        return;
    }

    daisy::idx_t dim_q, n_query, __, ___;
    if (!parseFilenameForConfig(query_filename, "bruteForce", dim_q, n_query, __, ___))
    {
        std::cerr << "Failed to parse query config from filename: " << query_filename << std::endl;
        return;
    }

    if (dim != dim_q)
    {
        std::cerr << "Dimension mismatch between dataset and queries" << std::endl;
        return;
    }

    query_out = loadBinData(query_path.c_str(), n_query, dim, false);

#if ODYSSEY_MPI
    if (daisy::Odyssey* odyssey_search = dynamic_cast<daisy::Odyssey*>(search))
    {
        daisy::FileDataSource data_source(dataset_path.c_str(), dim, n_database);
        search->buildIndex(&data_source);
    }
    else
#endif
    if (daisy::ParIS* paris_search = dynamic_cast<daisy::ParIS*>(search))
    {
        search->buildIndex(dataset_path, dim, n_database);
    }
    else
    {
        float* database = loadBinData(dataset_path.c_str(), n_database, dim, false);
        daisy::InMemoryDataSource data_source(database, n_database, dim);
        search->buildIndex(&data_source);
        delete[] database;
    }

    search->setNumThreads(num_thread);

    I_out = new daisy::idx_t[n_query * k];
    D_out = new float[n_query * k];
    n_query_out = n_query;
}