#include "test_utils.hpp"

#include "../commons/dataloaders.hpp"
#include "../lib/algos/DataSource.hpp"
#if ODYSSEY_MPI
#include "../lib/algos/hodyssey/Odyssey.hpp"
#endif

#include <set>

void SimilaritySearchTest::runSST(daisy::SimilaritySearchAlgorithm *search,
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

    daisy::idx_t dim_gt, n_database_gt, n_query, k;
    ASSERT_TRUE(parseFilenameForConfig(filename_gt, prefix_name, dim_gt, n_database_gt, n_query, k));

    daisy::idx_t dim, n_database, _, __;
    ASSERT_TRUE(parseFilenameForConfig(dataset_name, prefix_name, dim, n_database, _, __));

    ASSERT_EQ(dim_gt, dim);
    ASSERT_EQ(n_database_gt, n_database);

    float *database = loadBinData(dataset_path.c_str(), n_database, dim, false);
    float *query = loadBinData(query_path.c_str(), n_query, dim, false);

#if ODYSSEY_MPI
    daisy::Odyssey *odyssey_search = dynamic_cast<daisy::Odyssey *>(search);
    if (odyssey_search != nullptr) {
        daisy::FileDataSource data_source(dataset_path.c_str(), dim, n_database);
        search->buildIndex(&data_source);
    } else
#endif
    
    if (dynamic_cast<daisy::ParIS *>(search) != nullptr) {
        search->buildIndex(dataset_path, dim, n_database);
    } else {
        
        search->buildIndex(database, n_database, dim);
    }
    
    search->setNumThreads(num_thread);

    daisy::idx_t *I = new daisy::idx_t[n_query * k];
    float *D = new float[n_query * k];
    search->searchIndex(query, n_query, k, I, D);

    if (search->getResultCompareRank() == 0)
        compareWithGroundTruth(gt_I, gt_D, I, D, n_query, k, rtol, atol);

    delete[] database;
    delete[] query;
    delete[] I;
    delete[] D;
}

void SimilaritySearchTest::runSSTWithDistance(daisy::DistanceType distance_type,
                                              const std::string &prefix_name,
                                              const std::string &gt_I,
                                              const std::string &gt_D,
                                              const std::string &dataset_path,
                                              const std::string &query_path,
                                              int num_thread,
                                              double rtol,
                                              double atol)
{
    daisy::BruteForceSearch search(distance_type);
    runSST(&search, prefix_name, gt_I, gt_D, dataset_path, query_path, num_thread, rtol, atol);
}

void SimilaritySearchTest::runSSTRange(daisy::SimilaritySearchAlgorithm *search,
                                       const RangeTestConfig &config,
                                       double tol)
{
    daisy::idx_t n_query = config.query_limit > 0
                               ? (daisy::idx_t)config.query_limit
                               : config.n_query;

    float *database = loadBinData(config.dataset_path.c_str(), config.n_database, config.dim, false);
    float *query    = loadBinData(config.query_path.c_str(),   n_query,            config.dim, false);

    daisy::SearchConfig cfg;
    cfg.type = daisy::QueryType::RANGE;
    cfg.r    = config.r_value;

    daisy::BruteForceSearch gt(daisy::DistanceType::L2_SQUARED);
    gt.buildIndex(database, config.n_database, config.dim);
    std::vector<std::vector<daisy::idx_t>> gt_I;
    std::vector<std::vector<float>> gt_D;
    gt.searchIndex(query, n_query, cfg, gt_I, gt_D);

#if ODYSSEY_MPI
    if (dynamic_cast<daisy::Odyssey *>(search) != nullptr) {
        daisy::FileDataSource data_source(config.dataset_path.c_str(), config.dim, config.n_database);
        search->buildIndex(&data_source);
    } else
#endif
    if (dynamic_cast<daisy::ParIS *>(search) != nullptr) {
        search->buildIndex(config.dataset_path, config.dim, config.n_database);
    } else {
        search->buildIndex(database, config.n_database, config.dim);
    }
    search->setNumThreads(config.thread_count);

    std::vector<std::vector<daisy::idx_t>> I;
    std::vector<std::vector<float>> D;
    search->searchIndex(query, n_query, cfg, I, D);

    for (daisy::idx_t qi = 0; qi < n_query; qi++) {
        for (float d : D[qi]) {
            EXPECT_LE(d, (double)config.r_value * (1.0 + tol))
                << "False positive at query " << qi;
        }

        std::set<daisy::idx_t> gt_set(gt_I[qi].begin(), gt_I[qi].end());
        std::set<daisy::idx_t> algo_set(I[qi].begin(), I[qi].end());
        size_t missing = 0;
        for (daisy::idx_t idx : gt_set) {
            if (!algo_set.count(idx)) missing++;
        }
        size_t allowed = std::max<size_t>(1, (size_t)(tol * gt_set.size()));
        EXPECT_LE(missing, allowed)
            << "query " << qi << ": " << missing << "/" << gt_set.size()
            << " GT results missing (allowed " << allowed << ")";
    }

    delete[] database;
    delete[] query;
}