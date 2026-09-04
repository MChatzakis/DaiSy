#include "LbBruteforce.hpp"
#include "../isax/iSAXIndex.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>

namespace daisy
{

    LbBruteforce::LbBruteforce(DistanceType distance_type)
        : SimilaritySearchAlgorithm(distance_type)
    {
    }

    void LbBruteforce::setNumThreads(int num_threads)
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

    int LbBruteforce::getNumThreads() const
    {
        return this->num_threads;
    }

    void LbBruteforce::buildIndex(DataSource *data_source)
    {
        if (data_source == nullptr)
            throw std::invalid_argument("LbBruteforce::buildIndex received a null data source");

        this->dim = data_source->getDim();
        this->n_database = data_source->getTotalRecords();

        if (this->dim == 0)
            throw std::invalid_argument("LbBruteforce::buildIndex requires a positive dimension");

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

        this->database_capacity = std::max<idx_t>(this->n_database, 1);
        if (this->database_capacity > std::numeric_limits<size_t>::max() / this->dim)
            throw std::length_error("LbBruteforce database is too large");

        this->database = new float[static_cast<size_t>(this->database_capacity) * this->dim];
        float *record = new float[this->dim];
        idx_t idx = 0;
        while (data_source->nextRecord(record))
        {
            std::copy(record, record + this->dim, this->database + idx * this->dim);
            idx++;
        }
        delete[] record;

        this->index_settings = isax_index_settings_init("",
                                                        this->dim,
                                                        this->paa_segments,
                                                        this->sax_cardinality,
                                                        this->leaf_size,
                                                        this->min_leaf_size,
                                                        this->initial_lbl_size,
                                                        this->flush_limit,
                                                        this->initial_fbl_size,
                                                        this->total_loaded_leaves,
                                                        this->tight_bound,
                                                        0,
                                                        1,
                                                        1,
                                                        this->bp_mode);

        // Equi-depth breakpoints (no-op in Gaussian mode), installed as active.
        compute_equidepth_breakpoints(this->index_settings, this->database, this->n_database);
        activateBreakpoints();

        this->index = isax_index_init_inmemory(this->index_settings);

        this->db_sax_representations =
            (sax_type **)malloc(static_cast<size_t>(this->database_capacity) * sizeof(sax_type *));
        if (this->db_sax_representations == nullptr)
            throw std::bad_alloc();

#pragma omp parallel for num_threads(num_threads)
        for (idx_t dbi = 0; dbi < n_database; dbi++)
        {
            this->db_sax_representations[dbi] = (sax_type *)malloc(sizeof(sax_type) * this->index->settings->paa_segments);
            float *vi_vec = this->database + dbi * dim;

            if (!this->distance_computer->compute_sax_from_ts(vi_vec,
                                                              this->db_sax_representations[dbi],
                                                              this->index->settings->ts_values_per_paa_segment,
                                                              this->index->settings->paa_segments,
                                                              this->index->settings->sax_alphabet_cardinality,
                                                              this->index->settings->sax_bit_cardinality))
            {
                fprintf(stderr, "error: cannot insert record in index, since sax representation failed to be created");
                exit(EXIT_FAILURE);
            }
        }
    }

    void LbBruteforce::reserveDatabase(idx_t required_capacity)
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

        if (new_capacity > std::numeric_limits<size_t>::max() / this->dim ||
            new_capacity > std::numeric_limits<size_t>::max() / sizeof(sax_type *))
            throw std::length_error("LbBruteforce database is too large");

        std::unique_ptr<float[]> grown_database(
            new float[static_cast<size_t>(new_capacity) * static_cast<size_t>(this->dim)]);
        std::copy_n(this->database,
                    static_cast<size_t>(this->n_database) * static_cast<size_t>(this->dim),
                    grown_database.get());

        sax_type **grown_sax =
            (sax_type **)malloc(static_cast<size_t>(new_capacity) * sizeof(sax_type *));
        if (grown_sax == nullptr)
            throw std::bad_alloc();
        std::copy_n(this->db_sax_representations,
                    static_cast<size_t>(this->n_database), grown_sax);

        delete[] this->database;
        free(this->db_sax_representations);
        this->database = grown_database.release();
        this->db_sax_representations = grown_sax;
        this->database_capacity = new_capacity;
    }

    void LbBruteforce::insert(const float *series)
    {
        insertBatch(series, 1);
    }

    void LbBruteforce::insertBatch(const float *data, idx_t n)
    {
        if (n == 0)
            return;
        if (this->database == nullptr || this->index == nullptr ||
            this->index_settings == nullptr || this->dim == 0)
            throw std::runtime_error("LbBruteforce::insertBatch requires an initial buildIndex first");
        if (data == nullptr)
            throw std::invalid_argument("LbBruteforce::insertBatch received null data");
        if (n > std::numeric_limits<idx_t>::max() - this->n_database)
            throw std::length_error("LbBruteforce database size overflow");
        if (n > std::numeric_limits<size_t>::max() / this->dim)
            throw std::length_error("LbBruteforce insert batch is too large");

        const idx_t required_capacity = this->n_database + n;
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
                throw std::invalid_argument("LbBruteforce::insertBatch received an unaligned database pointer");
            source_offset = static_cast<size_t>(byte_offset / sizeof(float));
            if (inserted_values > current_values - source_offset)
                throw std::invalid_argument("LbBruteforce::insertBatch source exceeds the live database");
        }

        reserveDatabase(required_capacity);
        const float *source_data = aliases_database ? this->database + source_offset : data;

        activateBreakpoints();

        std::vector<sax_type *> new_sax_records;
        new_sax_records.reserve(static_cast<size_t>(n));
        try
        {
            for (idx_t i = 0; i < n; ++i)
            {
                sax_type *sax = (sax_type *)malloc(
                    sizeof(sax_type) * this->index->settings->paa_segments);
                if (sax == nullptr)
                    throw std::bad_alloc();

                const float *series = source_data + static_cast<size_t>(i) * this->dim;
                if (!this->distance_computer->compute_sax_from_ts(
                        series,
                        sax,
                        this->index->settings->ts_values_per_paa_segment,
                        this->index->settings->paa_segments,
                        this->index->settings->sax_alphabet_cardinality,
                        this->index->settings->sax_bit_cardinality))
                {
                    free(sax);
                    throw std::runtime_error("LbBruteforce::insertBatch failed to compute SAX representation");
                }
                new_sax_records.push_back(sax);
            }

            std::copy_n(source_data,
                        inserted_values,
                        this->database + static_cast<size_t>(this->n_database) * this->dim);
            for (idx_t i = 0; i < n; ++i)
                this->db_sax_representations[this->n_database + i] = new_sax_records[i];
            this->n_database = required_capacity;
        }
        catch (...)
        {
            for (sax_type *sax : new_sax_records)
                free(sax);
            throw;
        }
    }

    void LbBruteforce::searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
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
            std::cerr << "Error: Unsupported distance type for LbBruteforce index." << std::endl;
            exit(1);
        }
    }

    void LbBruteforce::searchIndexL2Squared(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        if (!validateSearchParams(k, n_query))
            return;
        activateBreakpoints();

#pragma omp parallel num_threads(num_threads)
        {
#pragma omp for
            for (idx_t qi = 0; qi < n_query; qi++)
            {
                std::priority_queue<std::pair<float, idx_t>> pq;
                const float *q_vec = query + qi * dim;
                float bound = FLT_MAX;

                ts_type *q_paa = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);
                if (q_paa == nullptr)
                {
                    fprintf(stderr, "Error: Failed to allocate memory for PAA representation\n");
                    continue;
                }

                this->distance_computer->compute_paa_from_ts(
                    q_vec, q_paa,
                    index->settings->paa_segments,
                    index->settings->ts_values_per_paa_segment);

                for (idx_t dbi = 0; dbi < n_database; ++dbi)
                {
                    const float *db_vec = database + dbi * dim;

                    float minimum_distance = this->distance_computer->compute_minidist_SIMD(
                        q_paa,
                        db_sax_representations[dbi],
                        (const int *)(index->settings->max_sax_cardinalities),
                        index->settings->sax_bit_cardinality,
                        index->settings->sax_alphabet_cardinality,
                        index->settings->paa_segments,
                        MINVAL,
                        MAXVAL,
                        index->settings->mindist_sqrt);
                    if (minimum_distance < bound)
                    {
                        float dist = this->distance_computer->compute_dist_SIMD(const_cast<float *>(q_vec),
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
                }

                free(q_paa);

                for (idx_t j = k; j > 0; --j)
                {
                    D[qi * k + (j - 1)] = pq.top().first;
                    I[qi * k + (j - 1)] = pq.top().second;
                    pq.pop();
                }
            }
        }
    }

    void LbBruteforce::searchIndexDTW(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        if (!validateSearchParams(k, n_query))
            return;
        activateBreakpoints();

#pragma omp parallel num_threads(num_threads)
        {
#pragma omp for
            for (idx_t qi = 0; qi < n_query; qi++)
            {
                std::priority_queue<std::pair<float, idx_t>> pq;
                const float *q_vec = query + qi * dim;
                float bound = FLT_MAX;

                float *lower_envelope = (float *)malloc(sizeof(float) * dim);
                float *upper_envelope = (float *)malloc(sizeof(float) * dim);
                ts_type *q_paa_upper = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);
                ts_type *q_paa_lower = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);

                if (!lower_envelope || !upper_envelope || !q_paa_upper || !q_paa_lower)
                {
                    fprintf(stderr, "Error: Failed to allocate memory for DTW computation\n");

                    if (lower_envelope)
                        free(lower_envelope);
                    if (upper_envelope)
                        free(upper_envelope);
                    if (q_paa_upper)
                        free(q_paa_upper);
                    if (q_paa_lower)
                        free(q_paa_lower);
                    continue;
                }

                int warping_window = static_cast<int>(dim * 0.1);
                lower_upper_lemire(const_cast<float *>(q_vec), dim, warping_window, lower_envelope, upper_envelope);

                this->distance_computer->compute_paa_from_ts(
                    upper_envelope, q_paa_upper,
                    index->settings->paa_segments,
                    index->settings->ts_values_per_paa_segment);
                this->distance_computer->compute_paa_from_ts(
                    lower_envelope, q_paa_lower,
                    index->settings->paa_segments,
                    index->settings->ts_values_per_paa_segment);

                for (idx_t dbi = 0; dbi < n_database; ++dbi)
                {
                    const float *db_vec = database + dbi * dim;

                    float minimum_distance = this->distance_computer->wrap_minidist_paa_to_isax_DTW(
                        q_paa_upper, q_paa_lower,
                        db_sax_representations[dbi],
                        (sax_type *)(index->settings->max_sax_cardinalities),
                        index->settings->sax_bit_cardinality,
                        index->settings->sax_alphabet_cardinality,
                        index->settings->paa_segments,
                        MINVAL,
                        MAXVAL,
                        index->settings->mindist_sqrt);

                    if (minimum_distance < bound)
                    {

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
                }

                for (idx_t j = k; j > 0; --j)
                {
                    D[qi * k + (j - 1)] = pq.top().first;
                    I[qi * k + (j - 1)] = pq.top().second;
                    pq.pop();
                }

                free(lower_envelope);
                free(upper_envelope);
                free(q_paa_upper);
                free(q_paa_lower);
            }
        }
    }

    void LbBruteforce::searchIndex(const float *query, idx_t n_query, const SearchConfig &config,
                                   std::vector<std::vector<idx_t>> &I,
                                   std::vector<std::vector<float>> &D)
    {
        if (config.type == QueryType::TOP_K) {
            SimilaritySearchAlgorithm::searchIndex(query, n_query, config, I, D);
            return;
        }

        if (database == nullptr || index == nullptr)
            throw std::runtime_error("LbBruteforce index must be built before searching");
        if (n_query == 0)
            throw std::invalid_argument("n_query must be greater than 0");
        activateBreakpoints();

        float r = config.r;
        I.assign(n_query, {});
        D.assign(n_query, {});

#pragma omp parallel num_threads(num_threads)
        {
#pragma omp for
            for (idx_t qi = 0; qi < n_query; qi++) {
                const float *q_vec = query + qi * dim;
                std::vector<std::pair<float, idx_t>> hits;

                ts_type *q_paa = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);
                if (q_paa == nullptr) {
                    fprintf(stderr, "Error: Failed to allocate memory for PAA representation\n");
                    continue;
                }

                this->distance_computer->compute_paa_from_ts(
                    q_vec, q_paa,
                    index->settings->paa_segments,
                    index->settings->ts_values_per_paa_segment);

                for (idx_t dbi = 0; dbi < n_database; ++dbi) {
                    float minimum_distance = this->distance_computer->compute_minidist_SIMD(
                        q_paa,
                        db_sax_representations[dbi],
                        (const int *)(index->settings->max_sax_cardinalities),
                        index->settings->sax_bit_cardinality,
                        index->settings->sax_alphabet_cardinality,
                        index->settings->paa_segments,
                        MINVAL,
                        MAXVAL,
                        index->settings->mindist_sqrt);

                    if (minimum_distance <= r) {
                        float dist = this->distance_computer->compute_dist_SIMD(
                            const_cast<float *>(q_vec),
                            const_cast<float *>(database + dbi * dim),
                            dim,
                            FLT_MAX);
                        if (dist <= r)
                            hits.emplace_back(dist, dbi);
                    }
                }

                free(q_paa);

                std::sort(hits.begin(), hits.end());
                I[qi].resize(hits.size());
                D[qi].resize(hits.size());
                for (size_t j = 0; j < hits.size(); ++j) {
                    D[qi][j] = hits[j].first;
                    I[qi][j] = hits[j].second;
                }
            }
        }
    }

    LbBruteforce::~LbBruteforce()
    {
        delete[] database;

        for (idx_t dbi = 0; dbi < n_database; ++dbi)
        {
            free(db_sax_representations[dbi]);
        }
        free(db_sax_representations);
        db_sax_representations = nullptr;

        if (this->index)
        {
            if (this->index->fbl)
            {
                destroy_fbl(this->index->fbl);
                this->index->fbl = nullptr;
            }
            free(this->index->answer);
            this->index->answer = nullptr;
            free(this->index);
            this->index = nullptr;
        }

        if (this->index_settings)
        {
            if (daisy_active_breakpoints == this->index_settings->breakpoints)
                set_active_breakpoints(nullptr, nullptr);
            free(this->index_settings->max_sax_cardinalities);
            free(this->index_settings->bit_masks);
            free(this->index_settings->breakpoints_owned);
            free(this->index_settings->breakpoints_max_owned);
            free(this->index_settings);
            this->index_settings = nullptr;
        }
    }
}
