#ifndef BRUTEFORCE_HPP
#define BRUTEFORCE_HPP

#include "SimilaritySearchAlgorithm.hpp"

#include <queue>
#include <cfloat>
#include <omp.h>

namespace daisy
{
    /**
     * @class BruteForceSearch
     * @brief Exhaustive similarity search without indexing
     *
     * Performs brute-force comparison of queries against all database vectors.
     * Supports both L2 Euclidean and DTW distance metrics with optional multi-threading.
     * Requires in-memory data; file-based input is not supported.
     *
     * @note This is a baseline algorithm useful for ground truth and small datasets.
     */
    class BruteForceSearch : public SimilaritySearchAlgorithm
    {
    private:
        void searchIndexL2Squared(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D);
        void searchIndexDTW(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D);

    public:
        BruteForceSearch(DistanceType distance_type);
        void setNumThreads(int num_threads);
        int getNumThreads() const { return this->num_threads; }

        // Bring base class buildIndex overloads into scope
        using SimilaritySearchAlgorithm::buildIndex;

        void buildIndex(DataSource *data_source) override;

        // BruteForceSearch only supports in-memory data
        void buildIndex(const std::string &filename, idx_t dim, idx_t n_database = 0) override
        {
            throw std::runtime_error("BruteForceSearch requires in-memory data. Use buildIndex(database, n_database, dim) instead.");
        }

        void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) override;

        ~BruteForceSearch();
    };

} // namespace daisy

#endif // BRUTEFORCE_HPP