#include "Coconut.hpp"

#include "../isax/SAX.hpp"
#include "../isax/iSAXTypes.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <queue>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace daisy
{

    Coconut::Coconut(DistanceType distance_type)
        : SimilaritySearchAlgorithm(distance_type)
    {
    }

    Coconut::~Coconut()
    {
        cleanupLeafFiles();
    }

    std::string Coconut::leafPath(int leaf_no) const
    {
        return index_dir_ + "/leaf." + std::to_string(leaf_no);
    }

    // Compare two sortable-SAX keys lexicographically over their bytes (COCONUT's order).
    static bool key_less(const std::vector<sax_type> &a, const std::vector<sax_type> &b)
    {
        for (size_t i = 0; i < a.size(); i++)
        {
            if (a[i] != b[i])
                return a[i] < b[i];
        }
        return false;
    }

    void Coconut::buildIndex(DataSource *data_source)
    {
        this->dim = data_source->getDim();
        this->n_database = data_source->getTotalRecords();
        if (this->n_database == 0)
        {
            data_source->reset();
            idx_t count = 0;
            float *dummy = new float[this->dim];
            while (data_source->nextRecord(dummy))
                count++;
            delete[] dummy;
            this->n_database = count;
            data_source->reset();
        }

        const int seg = this->paa_segments;
        const int bits = this->sax_cardinality;
        this->ts_values_per_segment_ = (int)this->dim / seg;
        this->sax_alphabet_cardinality_ = (int)std::pow(2, bits);
        this->mindist_sqrt_ = (float)this->dim / (float)seg;
        this->max_cardinalities_.assign(seg, (sax_type)bits);

        // Read the whole database into a temporary buffer (freed once leaves are on disk).
        std::vector<float> buffer((size_t)this->n_database * this->dim);
        {
            float *rec = new float[this->dim];
            idx_t idx = 0;
            while (data_source->nextRecord(rec) && idx < this->n_database)
            {
                std::copy(rec, rec + this->dim, buffer.data() + (size_t)idx * this->dim);
                idx++;
            }
            delete[] rec;
        }

        // 1. Compute SAX + sortable SAX for every series.
        records_.resize(this->n_database);
        for (idx_t i = 0; i < this->n_database; i++)
        {
            Record &r = records_[i];
            r.series_id = i;
            r.sax.resize(seg);
            r.key.resize(seg);
            sax_from_ts(buffer.data() + (size_t)i * this->dim, r.sax.data(),
                        this->ts_values_per_segment_, seg,
                        this->sax_alphabet_cardinality_, bits);
            sortable_sax_from_sax(r.sax.data(), r.key.data(), seg, bits);
        }

        // 2. Sort records by sortable-SAX key (bottom-up ordering).
        std::sort(records_.begin(), records_.end(),
                  [](const Record &a, const Record &b) { return key_less(a.key, b.key); });

        // 3. Write on-disk leaf files: consecutive sorted records, each = [inv-SAX][raw TS].
        static std::atomic<unsigned long> uid{0};
        this->index_dir_ = "/tmp/daisy_coconut_" + std::to_string((unsigned long)getpid()) +
                           "_" + std::to_string(uid.fetch_add(1));
        mkdir(this->index_dir_.c_str(), 0777);

        FILE *lf = nullptr;
        int cur_leaf = -1;
        for (size_t p = 0; p < records_.size(); p++)
        {
            int leaf_no = (int)(p / (size_t)this->leaf_capacity);
            if (leaf_no != cur_leaf)
            {
                if (lf) fclose(lf);
                lf = fopen(leafPath(leaf_no).c_str(), "wb");
                cur_leaf = leaf_no;
            }
            if (lf)
            {
                fwrite(records_[p].key.data(), sizeof(sax_type), seg, lf);
                fwrite(buffer.data() + (size_t)records_[p].series_id * this->dim,
                       sizeof(float), this->dim, lf);
            }
        }
        if (lf) fclose(lf);
        // buffer freed here: raw series now live on disk in the leaf files.

        this->next_id_ = this->n_database;  // streaming inserts get ids after the built set
    }

    void Coconut::insert(const float *series)
    {
        if (this->paa_segments <= 0 || this->ts_values_per_segment_ <= 0)
            throw std::runtime_error("Coconut::insert requires an initial buildIndex first");

        const int seg = this->paa_segments;
        Record r;
        r.series_id = this->next_id_++;
        r.sax.resize(seg);
        r.key.resize(seg);
        sax_from_ts(const_cast<float *>(series), r.sax.data(),
                    this->ts_values_per_segment_, seg,
                    this->sax_alphabet_cardinality_, this->sax_cardinality);
        sortable_sax_from_sax(r.sax.data(), r.key.data(), seg, this->sax_cardinality);

        buffer_records_.push_back(std::move(r));
        buffer_data_.insert(buffer_data_.end(), series, series + this->dim);
        this->n_database++;
    }

    void Coconut::insertBatch(const float *data, idx_t n)
    {
        buffer_data_.reserve(buffer_data_.size() + (size_t)n * this->dim);
        for (idx_t i = 0; i < n; i++)
            insert(data + (size_t)i * this->dim);
    }

    void Coconut::readSeriesAt(FILE *lf, int slot, float *out) const
    {
        const long record_bytes = (long)this->paa_segments * (long)sizeof(sax_type) +
                                   (long)this->dim * (long)sizeof(float);
        // seek past this slot's inv-SAX to its raw TS
        fseek(lf, (long)slot * record_bytes + (long)this->paa_segments * (long)sizeof(sax_type), SEEK_SET);
        size_t got = fread(out, sizeof(float), this->dim, lf);
        (void)got;
    }

    void Coconut::readSeriesFromLeaf(int leaf_no, int slot, float *out) const
    {
        FILE *lf = fopen(leafPath(leaf_no).c_str(), "rb");
        if (!lf)
            return;
        readSeriesAt(lf, slot, out);
        fclose(lf);
    }

    void Coconut::searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        const int seg = this->paa_segments;
        std::vector<float> paa(seg);
        std::vector<float> ts(this->dim);

        for (idx_t q = 0; q < n_query; q++)
        {
            const float *qts = query + (size_t)q * this->dim;
            paa_from_ts(qts, paa.data(), seg, this->ts_values_per_segment_);

            // k-BSF as a max-heap on distance (top = current worst of the best k).
            std::priority_queue<std::pair<float, idx_t>> bsf;

            for (size_t p = 0; p < records_.size(); p++)
            {
                float kth = (bsf.size() >= (size_t)k) ? bsf.top().first : FLT_MAX;

                float mindist = minidist_paa_to_isax(
                    paa.data(),
                    const_cast<sax_type *>(records_[p].sax.data()),
                    const_cast<sax_type *>(this->max_cardinalities_.data()),
                    (sax_type)this->sax_cardinality,
                    this->sax_alphabet_cardinality_,
                    seg, MINVAL, MAXVAL, this->mindist_sqrt_);

                if (mindist >= kth)
                    continue;  // lower bound already worse than the k-th best -> prune

                int leaf_no = (int)(p / (size_t)this->leaf_capacity);
                int slot = (int)(p % (size_t)this->leaf_capacity);
                readSeriesFromLeaf(leaf_no, slot, ts.data());

                float dist = ts_euclidean_distance_SIMD(const_cast<float *>(qts), ts.data(),
                                                        (int)this->dim, kth);
                if (bsf.size() < (size_t)k)
                    bsf.push({dist, records_[p].series_id});
                else if (dist < bsf.top().first)
                {
                    bsf.pop();
                    bsf.push({dist, records_[p].series_id});
                }
            }

            // Scan the in-memory streaming buffer (raw TS already in memory).
            for (size_t b = 0; b < buffer_records_.size(); b++)
            {
                float kth = (bsf.size() >= (size_t)k) ? bsf.top().first : FLT_MAX;
                float mindist = minidist_paa_to_isax(
                    paa.data(),
                    const_cast<sax_type *>(buffer_records_[b].sax.data()),
                    const_cast<sax_type *>(this->max_cardinalities_.data()),
                    (sax_type)this->sax_cardinality,
                    this->sax_alphabet_cardinality_,
                    seg, MINVAL, MAXVAL, this->mindist_sqrt_);
                if (mindist >= kth)
                    continue;
                float *bts = buffer_data_.data() + b * (size_t)this->dim;
                float dist = ts_euclidean_distance_SIMD(const_cast<float *>(qts), bts,
                                                        (int)this->dim, kth);
                if (bsf.size() < (size_t)k)
                    bsf.push({dist, buffer_records_[b].series_id});
                else if (dist < bsf.top().first)
                {
                    bsf.pop();
                    bsf.push({dist, buffer_records_[b].series_id});
                }
            }

            // Emit in ascending distance order.
            idx_t got = (idx_t)bsf.size();
            for (idx_t j = got; j > 0; j--)
            {
                D[q * k + (j - 1)] = bsf.top().first;
                I[q * k + (j - 1)] = bsf.top().second;
                bsf.pop();
            }
            for (idx_t j = got; j < k; j++)  // pad if fewer than k available
            {
                D[q * k + j] = FLT_MAX;
                I[q * k + j] = 0;
            }
        }
    }

    void Coconut::searchIndex(const float *query, idx_t n_query, const SearchConfig &config,
                              std::vector<std::vector<idx_t>> &I,
                              std::vector<std::vector<float>> &D)
    {
        if (config.type == QueryType::TOP_K)
        {
            SimilaritySearchAlgorithm::searchIndex(query, n_query, config, I, D);
            return;
        }
        if (this->distance_type != DistanceType::L2_SQUARED)
            throw std::runtime_error("Coconut range search only supports L2_SQUARED.");

        const int seg = this->paa_segments;
        const float r = config.r;
        // Early-abandon just past r: a partial sum can then never tie with r and be mistaken
        // for a hit, while every series actually within the radius is still summed in full.
        const float abandon_bound = std::nextafter(r, FLT_MAX);

        I.assign(n_query, {});
        D.assign(n_query, {});

#pragma omp parallel for num_threads(num_threads)
        for (idx_t q = 0; q < n_query; q++)
        {
            const float *qts = query + (size_t)q * this->dim;
            std::vector<float> paa(seg);
            std::vector<float> ts(this->dim);
            std::vector<std::pair<float, idx_t>> hits;

            paa_from_ts(qts, paa.data(), seg, this->ts_values_per_segment_);

            // Leaf at a time: a wide radius leaves many candidates per leaf, and opening the
            // leaf file once for all of them keeps the scan from degenerating into one
            // fopen/fclose per record the way a k-NN query's tight BSF never exposes.
            const size_t cap = (size_t)this->leaf_capacity;
            const size_t n_leaves = (records_.size() + cap - 1) / cap;
            std::vector<size_t> candidates;

            for (size_t leaf_no = 0; leaf_no < n_leaves; leaf_no++)
            {
                const size_t begin = leaf_no * cap;
                const size_t end = std::min(begin + cap, records_.size());

                candidates.clear();
                for (size_t p = begin; p < end; p++)
                {
                    float mindist = minidist_paa_to_isax(
                        paa.data(),
                        const_cast<sax_type *>(records_[p].sax.data()),
                        const_cast<sax_type *>(this->max_cardinalities_.data()),
                        (sax_type)this->sax_cardinality,
                        this->sax_alphabet_cardinality_,
                        seg, MINVAL, MAXVAL, this->mindist_sqrt_);

                    if (mindist <= r)  // lower bound outside the radius -> prune
                        candidates.push_back(p);
                }

                if (candidates.empty())
                    continue;

                FILE *lf = fopen(leafPath((int)leaf_no).c_str(), "rb");
                if (!lf)
                    continue;
                for (size_t p : candidates)
                {
                    readSeriesAt(lf, (int)(p - begin), ts.data());
                    float dist = ts_euclidean_distance_SIMD(const_cast<float *>(qts), ts.data(),
                                                            (int)this->dim, abandon_bound);
                    if (dist <= r)
                        hits.emplace_back(dist, records_[p].series_id);
                }
                fclose(lf);
            }

            // Scan the in-memory streaming buffer (raw TS already in memory).
            for (size_t b = 0; b < buffer_records_.size(); b++)
            {
                float mindist = minidist_paa_to_isax(
                    paa.data(),
                    const_cast<sax_type *>(buffer_records_[b].sax.data()),
                    const_cast<sax_type *>(this->max_cardinalities_.data()),
                    (sax_type)this->sax_cardinality,
                    this->sax_alphabet_cardinality_,
                    seg, MINVAL, MAXVAL, this->mindist_sqrt_);

                if (mindist > r)
                    continue;

                float *bts = buffer_data_.data() + b * (size_t)this->dim;
                float dist = ts_euclidean_distance_SIMD(const_cast<float *>(qts), bts,
                                                        (int)this->dim, abandon_bound);
                if (dist <= r)
                    hits.emplace_back(dist, buffer_records_[b].series_id);
            }

            std::sort(hits.begin(), hits.end());
            I[q].resize(hits.size());
            D[q].resize(hits.size());
            for (size_t j = 0; j < hits.size(); j++)
            {
                D[q][j] = hits[j].first;
                I[q][j] = hits[j].second;
            }
        }
    }

    void Coconut::cleanupLeafFiles()
    {
        if (index_dir_.empty())
            return;
        int nleaves = (int)((records_.size() + leaf_capacity - 1) / (size_t)std::max(1, leaf_capacity));
        for (int l = 0; l < nleaves; l++)
            remove(leafPath(l).c_str());
        rmdir(index_dir_.c_str());
        index_dir_.clear();
    }

}
