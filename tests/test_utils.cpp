#include "test_utils.hpp"

#include "../commons/dataloaders.hpp"
#include "../lib/algos/DataSource.hpp"
#if ODYSSEY_MPI
#include "../lib/algos/hodyssey/Odyssey.hpp"
#endif

void SimilaritySearchTest::runSST(diNoLib::SimilaritySearchAlgorithm *search,
                                  const std::string &prefix_name,
                                  const std::string &gt_I,
                                  const std::string &gt_D,
                                  const std::string &dataset_path,
                                  const std::string &query_path,
                                  int num_thread,
                                  double rtol,
                                  double atol)
{
    std::string filename_gt = pathToFilename(gt_I);
    std::string dataset_name = pathToFilename(dataset_path);

    diNoLib::idx_t dim_gt, n_database_gt, n_query, k;
    ASSERT_TRUE(parseFilenameForConfig(filename_gt, prefix_name, dim_gt, n_database_gt, n_query, k));

    diNoLib::idx_t dim, n_database, _, __;
    ASSERT_TRUE(parseFilenameForConfig(dataset_name, prefix_name, dim, n_database, _, __));

    ASSERT_EQ(dim_gt, dim);
    ASSERT_EQ(n_database_gt, n_database);

    float *database = loadBinData(dataset_path.c_str(), n_database, dim);
    float *query = loadBinData(query_path.c_str(), n_query, dim);

    // Odyssey requires FileDataSource and MPI (argc/argv passed in test main)
#if ODYSSEY_MPI
    diNoLib::Odyssey *odyssey_search = dynamic_cast<diNoLib::Odyssey *>(search);
    if (odyssey_search != nullptr) {
        diNoLib::FileDataSource data_source(dataset_path.c_str(), dim, n_database);
        search->buildIndex(&data_source);
    } else
#endif
    // ParIS uses file-based buildIndex
    if (dynamic_cast<diNoLib::ParIS *>(search) != nullptr) {
        search->buildIndex(dataset_path, dim, n_database);
    } else {
        // Other algorithms use in-memory buildIndex
        search->buildIndex(database, n_database, dim);
    }
    
    search->setNumThreads(num_thread);

    diNoLib::idx_t *I = new diNoLib::idx_t[n_query * k];
    float *D = new float[n_query * k];
    search->searchIndex(query, n_query, k, I, D);

    /* For MPI (e.g. Odyssey) only rank 0 has merged results to compare; other ranks skip. */
    if (search->getResultCompareRank() == 0)
        compareWithGroundTruth(gt_I, gt_D, I, D, n_query, k);

    delete[] database;
    delete[] query;
    delete[] I;
    delete[] D;
}

void SimilaritySearchTest::runSSTWithDistance(diNoLib::DistanceType distance_type,
                                              const std::string &prefix_name,
                                              const std::string &gt_I,
                                              const std::string &gt_D,
                                              const std::string &dataset_path,
                                              const std::string &query_path,
                                              int num_thread,
                                              double rtol,
                                              double atol)
{
    diNoLib::BruteForceSearch search(distance_type);
    runSST(&search, prefix_name, gt_I, gt_D, dataset_path, query_path, num_thread, rtol, atol);
}