#ifndef BRUTEFORCESEARCH_HPP
#define BRUTEFORCESEARCH_HPP

#include "SimilaritySearchAlgorithm.hpp"

#include <queue>
#include <cfloat>
using namespace std; // remove std 

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

            /**
             * @param query a pointor to an array of float
             * @param n_query number of query vectors
             * @param k number of nearest neighbors
             * @param I output indices of nearest neighbors
             * @param D output distances of nearest neighbors
             */

            
            //DistanceComputer dc(distance_type);

            for (idx_t qi = 0; qi < n_query; qi++)
            {
                priority_queue<pair<float, idx_t>> pq;
                const float *q_vec = query + qi * dim;

                for (idx_t dbi = 0; dbi < n_database; ++dbi)
                {
                    const float *db_vec = database + dbi * dim;
                    float dist = this->distance_computer->compute_dist(const_cast<float *>(q_vec), const_cast<float *>(db_vec), dim, FLT_MAX);

                    if ((idx_t)pq.size() < k)
                    {
                        pq.emplace(dist, dbi); // equivalent to pq.push(make_pair(dist, dbi));
                    }
                    else if (dist < pq.top().first) // you have to change this condition to check if dist < the max distance you have in the priority queue.
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

        ~BruteForceSearch()
        {
            delete[] database;
        }
    };
} // namespace diNoLib

#endif // BRUTEFORCESEARCH_HPP