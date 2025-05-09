#include "BruteforceSearch.hpp"

namespace diNoLib
{

    BruteForceSearch::BruteForceSearch(DistanceType distance_type)
        : SimilaritySearchAlgorithm(distance_type)
    {
    }

    void BruteForceSearch::buildIndex(const float *database, const idx_t n_database, const idx_t dim)
    {
        this->database = new float[n_database * dim];
        std::copy(database, database + n_database * dim, this->database);
        this->n_database = n_database;
        this->dim = dim;
    }

    void BruteForceSearch::searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        for (idx_t qi = 0; qi < n_query; qi++)
        {
            std::priority_queue<std::pair<float, idx_t>> pq;
            const float *q_vec = query + qi * dim;

            for (idx_t dbi = 0; dbi < n_database; ++dbi)
            {
                const float *db_vec = database + dbi * dim;
                float dist = this->distance_computer->compute_dist(const_cast<float *>(q_vec), const_cast<float *>(db_vec), dim, FLT_MAX);

                if ((idx_t)pq.size() < k)
                {
                    pq.emplace(dist, dbi); // equivalent to pq.push(make_pair(dist, dbi));
                }
                else if (dist < pq.top().first) 
                {
                    pq.pop();
                    pq.emplace(dist, dbi);
                }
            }

            for (idx_t j = k; j > 0; --j)
            {
                D[qi * k + (j - 1)] = pq.top().first;
                I[qi * k + (j - 1)] = pq.top().second;
                pq.pop();
            }
        }
    }
    BruteForceSearch::~BruteForceSearch()
    {
        delete[] database;
    }
}