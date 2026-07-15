#include "Bruteforce.hpp"

namespace daisy
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

    void BruteForceSearch::buildIndex(DataSource *data_source)
    {
        this->dim = data_source->getDim();
        this->n_database = data_source->getTotalRecords();

        if (this->n_database == 0)
        {

            data_source->reset();
            idx_t count = 0;
            float *dummy = new float[this->dim];
            while (data_source->nextRecord(dummy))
            {
                count++;
            }
            delete[] dummy;
            this->n_database = count;
            data_source->reset();
        }

        this->database = new float[this->n_database * this->dim];
        float *record = new float[this->dim];
        idx_t idx = 0;
        while (data_source->nextRecord(record))
        {
            std::copy(record, record + this->dim, this->database + idx * this->dim);
            idx++;
        }
        delete[] record;
    }

    void BruteForceSearch::searchIndexL2Squared(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {

        if (!validateSearchParams(k, n_query))
        {
            return;
        }

#pragma omp parallel num_threads(num_threads)
        {
#pragma omp for
            for (idx_t qi = 0; qi < n_query; qi++)
            {
                std::priority_queue<std::pair<float, idx_t>> pq;
                const float *q_vec = query + qi * dim;

                float bound = FLT_MAX;

                for (idx_t dbi = 0; dbi < n_database; ++dbi)
                {
                    const float *db_vec = database + dbi * dim;
                    float dist = this->distance_computer->compute_dist(const_cast<float *>(q_vec),
                                                                       const_cast<float *>(db_vec),
                                                                       dim,
                                                                       bound);
                    if ((idx_t)pq.size() < k)
                    {
                        pq.emplace(dist, dbi);
                    }
                    else if (dist < pq.top().first)
                    {
                        pq.pop();
                        pq.emplace(dist, dbi);
                        bound = pq.top().first;
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
    }

    void BruteForceSearch::searchIndexDTW(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {

        if (!validateSearchParams(k, n_query))
        {
            return;
        }

#pragma omp parallel num_threads(num_threads)
        {
#pragma omp for
            for (idx_t qi = 0; qi < n_query; qi++)
            {
                std::priority_queue<std::pair<float, idx_t>> pq;
                const float *q_vec = query + qi * dim;

                float bound = FLT_MAX;

                for (idx_t dbi = 0; dbi < n_database; ++dbi)
                {
                    const float *db_vec = database + dbi * dim;
                    float dist = this->distance_computer->compute_dist(const_cast<float *>(q_vec),
                                                                       const_cast<float *>(db_vec),
                                                                       dim,
                                                                       bound);
                    if ((idx_t)pq.size() < k)
                    {
                        pq.emplace(dist, dbi);
                    }
                    else if (dist < pq.top().first)
                    {
                        pq.pop();
                        pq.emplace(dist, dbi);
                        bound = pq.top().first;
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
    }

    void BruteForceSearch::searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        if (this->distance_type == DistanceType::L2_SQUARED)
        {
            searchIndexL2Squared(query, n_query, k, I, D);
        }
        else if (this->distance_type == DistanceType::DTW)
        {
            searchIndexDTW(query, n_query, k, I, D);
        }
        else
        {
            std::cerr << "Error: Unsupported distance type for BruteForce index." << std::endl;
            exit(1);
        }
    }

    void BruteForceSearch::searchIndex(const float *query, idx_t n_query, const SearchConfig &config,
                                       std::vector<std::vector<idx_t>> &I,
                                       std::vector<std::vector<float>> &D)
    {
        if (config.type == QueryType::TOP_K)
        {
            SimilaritySearchAlgorithm::searchIndex(query, n_query, config, I, D);
            return;
        }

        if (database == nullptr)
        {
            std::cerr << "[Error] Index must be built before searching\n";
            return;
        }
        if (n_query == 0)
        {
            std::cerr << "[Error] n_query must be greater than 0\n";
            return;
        }

        I.resize(n_query);
        D.resize(n_query);

#pragma omp parallel num_threads(num_threads)
        {
#pragma omp for
            for (idx_t qi = 0; qi < n_query; qi++)
            {
                const float *q_vec = query + qi * dim;
                std::vector<std::pair<float, idx_t>> hits;

                for (idx_t dbi = 0; dbi < n_database; ++dbi)
                {
                    const float *db_vec = database + dbi * dim;
                    float dist = distance_computer->compute_dist(const_cast<float *>(q_vec),
                                                                 const_cast<float *>(db_vec),
                                                                 dim,
                                                                 config.r);
                    if (dist <= config.r)
                        hits.emplace_back(dist, dbi);
                }

                std::sort(hits.begin(), hits.end());

                I[qi].resize(hits.size());
                D[qi].resize(hits.size());
                for (size_t j = 0; j < hits.size(); j++)
                {
                    D[qi][j] = hits[j].first;
                    I[qi][j] = hits[j].second;
                }
            }
        }
    }

    BruteForceSearch::~BruteForceSearch()
    {
        delete[] database;
    }
}