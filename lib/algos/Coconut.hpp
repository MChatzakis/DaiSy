#ifndef COCONUT_HPP
#define COCONUT_HPP

#include "SimilaritySearchAlgorithm.hpp"

#include "../isax/iSAXTypes.hpp"

#include <cstdio>
#include <string>
#include <vector>
#include <cfloat>

namespace daisy
{

    // COCONUT (Kondylakis et al., VLDB'18): a "sortable SAX" key (bit-interleaved SAX) orders
    // series so the index can be built bottom-up by sorting and extended by streaming inserts.
    // buildIndex sorts records onto on-disk leaves ([inv-SAX][raw TS] each); searchIndex does
    // exact kNN or range search via SAX MINDIST + L2 refinement, parallel across queries.
    // Top-k starts with an ordered approximate search around the query's sortable-SAX key,
    // followed by exact SAX-bound refinement. Each query caches a complete leaf privately.
    // The sorted in-memory key directory provides ordered lookup without Berkeley DB.
    // Follow-ups: out-of-core key directory/sort, LSM merges, equi-depth breakpoints.
    class Coconut : public SimilaritySearchAlgorithm
    {
    public:
        Coconut(DistanceType distance_type);

        void setNumThreads(int num_threads) override { this->num_threads = num_threads; }
        int getNumThreads() const { return num_threads; }

        int getPaaSegments() const { return paa_segments; }
        void setPaaSegments(int v) { paa_segments = v; }
        int getSaxCardinality() const { return sax_cardinality; }
        void setSaxCardinality(int v) { sax_cardinality = v; }
        int getLeafSize() const { return leaf_capacity; }
        void setLeafSize(int v) { leaf_capacity = v; }

        using SimilaritySearchAlgorithm::buildIndex;
        void buildIndex(DataSource *data_source) override;
        void buildIndex(const std::string &filename, idx_t dim, idx_t n_database = 0) override
        {
            throw std::runtime_error("Coconut requires in-memory data. Use buildIndex(database, n_database, dim).");
        }

        void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) override;

        // Range search: every series within radius r of the query, in ascending distance order.
        void searchIndex(const float *query, idx_t n_query, const SearchConfig &config,
                         std::vector<std::vector<idx_t>> &I,
                         std::vector<std::vector<float>> &D) override;

        // Streaming: add series to a live index (needs buildIndex first). New series go into
        // an in-memory buffer that queries scan alongside the on-disk index.
        void insert(const float *series) override;
        void insertBatch(const float *data, idx_t n) override;

        ~Coconut() override;

    private:
        // After sorting, record p lives in leaf p/leaf_capacity at slot p%leaf_capacity.
        struct Record
        {
            std::vector<sax_type> sax;      // SAX word (for MINDIST)
            std::vector<sax_type> key;      // sortable SAX (defines the order)
            idx_t series_id;                // original index in the input
        };

        std::vector<Record> records_;       // main index, sorted by sortable-SAX key (TS on disk)
        std::string index_dir_;             // unique per-instance dir for leaf files
        int leaf_capacity = 1000;           // records per leaf file

        // Streaming buffer: inserted series kept in memory (summaries + raw TS), scanned by queries.
        std::vector<Record> buffer_records_;
        std::vector<float> buffer_data_;    // buffer_records_.size() * dim
        idx_t next_id_ = 0;

        int ts_values_per_segment_ = 0;     // dim / paa_segments
        int sax_alphabet_cardinality_ = 0;  // 2^sax_cardinality
        float mindist_sqrt_ = 0.0f;         // (dim / paa_segments), the MINDIST scale factor
        std::vector<sax_type> max_cardinalities_;  // all = sax_cardinality

        std::string leafPath(int leaf_no) const;
        bool readLeaf(size_t leaf_no, std::vector<unsigned char> &page) const;
        void copySeriesFromLeaf(const std::vector<unsigned char> &page, size_t slot, float *out) const;
        void cleanupLeafFiles();
    };

}

#endif
