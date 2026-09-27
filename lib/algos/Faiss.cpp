#include "Faiss.hpp"

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace daisy
{
    namespace
    {
        // FAISS labels are int64_t, DaiSy indices are unsigned long long. Both are
        // 64-bit, but the signedness differs, so conversions go through here.
        inline idx_t toDaisyIndex(faiss::idx_t label)
        {
            if (label < 0)
                throw std::runtime_error("FaissFlat: FAISS returned a missing result (-1)");
            return static_cast<idx_t>(label);
        }
    }

    FaissFlat::FaissFlat(DistanceType distance_type)
        : SimilaritySearchAlgorithm(distance_type)
    {
        if (distance_type != DistanceType::L2_SQUARED)
            throw std::invalid_argument(
                "FaissFlat supports L2_SQUARED only; FAISS has no DTW equivalent");
    }

    FaissFlat::~FaissFlat() = default;

    void FaissFlat::setNumThreads(int num_threads)
    {
        if (num_threads < 1)
            throw std::invalid_argument("FaissFlat::setNumThreads requires at least one thread");
        this->num_threads = num_threads;
    }

    void FaissFlat::refreshDatabaseView()
    {
        // Non-owning view into FAISS storage: for IndexFlatL2 the code size is
        // dim * sizeof(float), so the code array is exactly the vector array.
        // Never freed here; the FAISS index owns it.
        this->database = (faiss_index && faiss_index->ntotal > 0) ? faiss_index->get_xb() : nullptr;
        this->n_database = faiss_index ? static_cast<idx_t>(faiss_index->ntotal) : 0;
    }

    void FaissFlat::buildIndex(DataSource *data_source)
    {
        if (data_source == nullptr)
            throw std::invalid_argument("FaissFlat::buildIndex received a null data source");

        const idx_t new_dim = data_source->getDim();
        if (new_dim == 0)
            throw std::invalid_argument("FaissFlat::buildIndex requires a positive dimension");
        if (new_dim > static_cast<idx_t>(std::numeric_limits<int>::max()))
            throw std::length_error("FaissFlat: dimension exceeds what FAISS can index");

        idx_t new_n_database = data_source->getTotalRecords();
        if (new_n_database == 0)
        {
            data_source->reset();
            std::vector<float> dummy(new_dim);
            idx_t count = 0;
            while (data_source->nextRecord(dummy.data()))
                count++;
            new_n_database = count;
            data_source->reset();
        }

        if (new_n_database > static_cast<idx_t>(std::numeric_limits<faiss::idx_t>::max()))
            throw std::length_error("FaissFlat database is too large");

        auto new_index = std::make_unique<faiss::IndexFlatL2>(static_cast<int>(new_dim));

        // A contiguous in-memory source can be handed to FAISS in one call.
        if (const float *raw = data_source->rawPointer())
        {
            new_index->add(static_cast<faiss::idx_t>(new_n_database), raw);
        }
        else
        {
            // Stream the source in blocks so a large file-backed build does not
            // need a full second copy of the data alongside the FAISS storage.
            constexpr idx_t BLOCK = 8192;
            std::vector<float> block(static_cast<size_t>(BLOCK) * new_dim);
            idx_t loaded = 0;
            while (loaded < new_n_database)
            {
                idx_t in_block = 0;
                while (in_block < BLOCK && loaded + in_block < new_n_database &&
                       data_source->nextRecord(block.data() + static_cast<size_t>(in_block) * new_dim))
                    in_block++;

                if (in_block == 0)
                    break;

                new_index->add(static_cast<faiss::idx_t>(in_block), block.data());
                loaded += in_block;
            }
        }

        this->faiss_index = std::move(new_index);
        this->dim = new_dim;
        refreshDatabaseView();
    }

    void FaissFlat::insert(const float *series)
    {
        if (series == nullptr)
            throw std::invalid_argument("FaissFlat::insert received null data");
        insertBatch(series, 1);
    }

    void FaissFlat::insertBatch(const float *data, idx_t n)
    {
        if (n == 0)
            return;
        if (faiss_index == nullptr || this->dim == 0)
            throw std::runtime_error("FaissFlat::insertBatch requires an initial buildIndex first");
        if (data == nullptr)
            throw std::invalid_argument("FaissFlat::insertBatch received null data");
        if (n > std::numeric_limits<idx_t>::max() - this->n_database)
            throw std::length_error("FaissFlat database size overflow");
        if (this->n_database + n > static_cast<idx_t>(std::numeric_limits<faiss::idx_t>::max()))
            throw std::length_error("FaissFlat database is too large");

        // A caller may pass a slice of the live database, which FAISS would read
        // from while reallocating its own storage. Copy first in that case.
        const float *storage_begin = faiss_index->ntotal > 0 ? faiss_index->get_xb() : nullptr;
        const float *storage_end = storage_begin
                                       ? storage_begin + static_cast<size_t>(faiss_index->ntotal) * this->dim
                                       : nullptr;
        const bool aliases_storage = storage_begin && data >= storage_begin && data < storage_end;

        if (aliases_storage)
        {
            std::vector<float> owned(data, data + static_cast<size_t>(n) * this->dim);
            faiss_index->add(static_cast<faiss::idx_t>(n), owned.data());
        }
        else
        {
            faiss_index->add(static_cast<faiss::idx_t>(n), data);
        }

        refreshDatabaseView();
    }

    void FaissFlat::searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        if (!validateSearchParams(k, n_query))
            return;
        if (query == nullptr || I == nullptr || D == nullptr)
            throw std::invalid_argument("FaissFlat::searchIndex received a null buffer");

#ifdef _OPENMP
        omp_set_num_threads(this->num_threads);
#endif

        // FAISS writes int64_t labels; translate into DaiSy's unsigned indices.
        std::vector<faiss::idx_t> labels(static_cast<size_t>(n_query) * k);
        faiss_index->search(static_cast<faiss::idx_t>(n_query), query,
                            static_cast<faiss::idx_t>(k), D, labels.data());

        for (size_t i = 0; i < labels.size(); ++i)
            I[i] = toDaisyIndex(labels[i]);
    }

    void FaissFlat::searchIndex(const float *query, idx_t n_query, const SearchConfig &config,
                                std::vector<std::vector<idx_t>> &I,
                                std::vector<std::vector<float>> &D)
    {
        if (config.type == QueryType::TOP_K)
        {
            SimilaritySearchAlgorithm::searchIndex(query, n_query, config, I, D);
            return;
        }

        if (faiss_index == nullptr || this->database == nullptr)
        {
            std::cerr << "[Error] Index must be built before searching\n";
            return;
        }
        if (n_query == 0)
        {
            std::cerr << "[Error] n_query must be greater than 0\n";
            return;
        }
        if (query == nullptr)
            throw std::invalid_argument("FaissFlat::searchIndex received a null query buffer");

#ifdef _OPENMP
        omp_set_num_threads(this->num_threads);
#endif

        // DaiSy range search includes hits exactly at the radius (dist <= r), while
        // FAISS keeps only dist < radius. Widening by one float makes r itself
        // reachable; the explicit filter below then drops anything in the gap, so
        // the boundary matches the other algorithms exactly.
        const float faiss_radius = std::nextafter(config.r, std::numeric_limits<float>::max());

        faiss::RangeSearchResult result(static_cast<size_t>(n_query));
        faiss_index->range_search(static_cast<faiss::idx_t>(n_query), query, faiss_radius, &result);

        I.assign(n_query, {});
        D.assign(n_query, {});

        std::vector<std::pair<float, idx_t>> hits;
        for (idx_t qi = 0; qi < n_query; ++qi)
        {
            const size_t begin = result.lims[qi];
            const size_t end = result.lims[qi + 1];

            hits.clear();
            hits.reserve(end - begin);
            for (size_t j = begin; j < end; ++j)
            {
                if (result.distances[j] <= config.r)
                    hits.emplace_back(result.distances[j], toDaisyIndex(result.labels[j]));
            }

            // FAISS returns range hits unsorted; DaiSy reports them by distance.
            std::sort(hits.begin(), hits.end());

            I[qi].resize(hits.size());
            D[qi].resize(hits.size());
            for (size_t j = 0; j < hits.size(); ++j)
            {
                D[qi][j] = hits[j].first;
                I[qi][j] = hits[j].second;
            }
        }
    }

}
