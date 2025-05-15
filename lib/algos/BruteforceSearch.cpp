#include "BruteforceSearch.hpp"

namespace diNoLib
{

    BruteForceSearch::BruteForceSearch(DistanceType distance_type)
        : SimilaritySearchAlgorithm(distance_type)
    {
    }

    void BruteForceSearch::setNumThreads(int num_threads)
    {
        int max_threads = omp_get_max_threads();

        if (num_threads > max_threads) 
        {
            std::cerr << "[Warning] " << num_threads 
                    << " threads exceeds max available " << max_threads << " Using the max threads available.\n";
            this->num_threads = max_threads;
        } 
        else if (num_threads < 1) 
        {
            std::cerr << "[Warning] Thread count must be >= 1. Using 1.\n";
            this->num_threads = 1;
        } 
        else 
        {
            this->num_threads = num_threads;
        }
    } 

    int BruteForceSearch::getNumThreads() const
    {
        return this->num_threads;
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
        #pragma omp parallel num_threads(num_threads)
        {
            #pragma omp for 
            for (idx_t qi = 0; qi < n_query; qi++)
            {   
                std::priority_queue<std::pair<float, idx_t>> pq;
                const float *q_vec = query + qi * dim;

                float bound = FLT_MAX;  // initialize bound to max float

                for (idx_t dbi = 0; dbi < n_database; ++dbi)
                {
                    const float *db_vec = database + dbi * dim;
                    float dist = this->distance_computer->compute_dist(const_cast<float *>(q_vec), 
                                                                        const_cast<float *>(db_vec), 
                                                                        dim, 
                                                                        bound);
                    if ((idx_t)pq.size() < k) // maintain max-heap
                    {
                        pq.emplace(dist, dbi); // equivalent to pq.push(make_pair(dist, dbi));
                    }
                    else if (dist < pq.top().first) 
                    {
                        pq.pop();
                        pq.emplace(dist, dbi);                         
                        bound = pq.top().first;
                    }

                }
                
                // store top-k results in reverse order
                for (idx_t j = k; j > 0; --j)
                {
                    D[qi * k + (j - 1)] = pq.top().first;
                    I[qi * k + (j - 1)] = pq.top().second;
                    pq.pop();
                }               
            }
        }
    }

    BruteForceSearch::~BruteForceSearch()
    {
        delete[] database;
    }
}