#include "test_utils.hpp"

#include "../commons/dataloaders.hpp"
#include "../lib/algos/DataSource.hpp"

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

    diNoLib::InMemoryDataSource data_source(database, n_database, dim);
    search->buildIndex(&data_source);
    search->setNumThreads(num_thread);

    diNoLib::idx_t *I = new diNoLib::idx_t[n_query * k];
    float *D = new float[n_query * k];
    search->searchIndex(query, n_query, k, I, D);

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