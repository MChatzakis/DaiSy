#ifndef BRUTEFORCESEARCH_HPP
#define BRUTEFORCESEARCH_HPP

#include "SimilaritySearchAlgorithm.hpp"

namespace diNoLib
{
    class BruteForceSearch : public SimilaritySearchAlgorithm
    {
    public:
        BruteForceSearch(DistanceType distance_type) : SimilaritySearchAlgorithm(distance_type)
        {
        }

        void buildIndex(const float *database, const idx_t n_database, const idx_t dim) override
        {
            this->database = new float[n_database * dim];
            std::copy(database, database + n_database * dim, this->database);
            this->n_database = n_database;
            this->dim = dim;
        }

        void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) override
        {
            // @Gayatiri: TODO
            // The brute-force search algorithm checks all the points in the database and computers the distance to each point in the query.
            // The final results should be stored sorted in the I and D arrays.
        }

        ~BruteForceSearch()
        {
            delete[] database;
        }
    };
} // namespace diNoLib

#endif // BRUTEFORCESEARCH_HPP