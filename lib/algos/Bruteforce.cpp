#include "Bruteforce.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>

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
        if (data_source == nullptr)
            throw std::invalid_argument("BruteForceSearch::buildIndex received a null data source");

        const idx_t new_dim = data_source->getDim();
        idx_t new_n_database = data_source->getTotalRecords();

        if (new_dim == 0)
            throw std::invalid_argument("BruteForceSearch::buildIndex requires a positive dimension");

        if (new_n_database == 0)
        {
            data_source->reset();
            idx_t count = 0;
            std::vector<float> dummy(new_dim);
            while (data_source->nextRecord(dummy.data()))
            {
                count++;
            }
            new_n_database = count;
            data_source->reset();
        }

        if (new_n_database > std::numeric_limits<size_t>::max() / new_dim)
            throw std::length_error("BruteForceSearch database is too large");

        const idx_t new_capacity = std::max<idx_t>(new_n_database, 1);
        std::unique_ptr<float[]> new_database(
            new float[static_cast<size_t>(new_capacity) * static_cast<size_t>(new_dim)]);
        std::vector<float> record(new_dim);
        idx_t idx = 0;
        while (idx < new_n_database && data_source->nextRecord(record.data()))
        {
            std::copy(record.begin(), record.end(),
                      new_database.get() + static_cast<size_t>(idx) * new_dim);
            idx++;
        }

        delete[] this->database;
        this->database = new_database.release();
        this->dim = new_dim;
        this->n_database = idx;
        this->database_capacity = new_capacity;
    }

    void BruteForceSearch::reserveDatabase(idx_t required_capacity)
    {
        if (required_capacity <= this->database_capacity)
            return;

        idx_t new_capacity = std::max<idx_t>(this->database_capacity, 1);
        while (new_capacity < required_capacity)
        {
            if (new_capacity > std::numeric_limits<idx_t>::max() / 2)
            {
                new_capacity = required_capacity;
                break;
            }
            new_capacity *= 2;
        }

        if (new_capacity > std::numeric_limits<size_t>::max() / this->dim)
            throw std::length_error("BruteForceSearch database is too large");

        std::unique_ptr<float[]> grown_database(
            new float[static_cast<size_t>(new_capacity) * static_cast<size_t>(this->dim)]);
        std::copy_n(this->database,
                    static_cast<size_t>(this->n_database) * static_cast<size_t>(this->dim),
                    grown_database.get());

        delete[] this->database;
        this->database = grown_database.release();
        this->database_capacity = new_capacity;
    }

    void BruteForceSearch::insert(const float *series)
    {
        insertBatch(series, 1);
    }

    void BruteForceSearch::insertBatch(const float *data, idx_t n)
    {
        if (n == 0)
            return;
        if (this->database == nullptr || this->dim == 0)
            throw std::runtime_error("BruteForceSearch::insertBatch requires an initial buildIndex first");
        if (data == nullptr)
            throw std::invalid_argument("BruteForceSearch::insertBatch received null data");
        if (n > std::numeric_limits<idx_t>::max() - this->n_database)
            throw std::length_error("BruteForceSearch database size overflow");
        if (n > std::numeric_limits<size_t>::max() / this->dim)
            throw std::length_error("BruteForceSearch insert batch is too large");

        const size_t current_values =
            static_cast<size_t>(this->n_database) * static_cast<size_t>(this->dim);
        const size_t inserted_values = static_cast<size_t>(n) * static_cast<size_t>(this->dim);
        const uintptr_t database_begin = reinterpret_cast<uintptr_t>(this->database);
        const uintptr_t database_end = database_begin + current_values * sizeof(float);
        const uintptr_t data_address = reinterpret_cast<uintptr_t>(data);
        const bool aliases_database = data_address >= database_begin && data_address < database_end;
        size_t source_offset = 0;
        if (aliases_database)
        {
            const uintptr_t byte_offset = data_address - database_begin;
            if (byte_offset % sizeof(float) != 0)
                throw std::invalid_argument("BruteForceSearch::insertBatch received an unaligned database pointer");
            source_offset = static_cast<size_t>(byte_offset / sizeof(float));
            if (inserted_values > current_values - source_offset)
                throw std::invalid_argument("BruteForceSearch::insertBatch source exceeds the live database");
        }

        const idx_t required_capacity = this->n_database + n;
        reserveDatabase(required_capacity);
        const float *source = aliases_database ? this->database + source_offset : data;

        std::copy_n(source,
                    inserted_values,
                    this->database + static_cast<size_t>(this->n_database) * this->dim);
        this->n_database = required_capacity;
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
        const float abandon_bound = std::nextafter(config.r, FLT_MAX);

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
                                                                 abandon_bound);
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
        database = nullptr;
    }
}
