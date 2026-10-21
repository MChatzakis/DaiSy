#ifndef COCONUT_HPP
#define COCONUT_HPP
#include "SimilaritySearchAlgorithm.hpp"
#include <memory>
#include <string>

namespace daisy
{
    struct CoconutSearchStats
    {
        idx_t summaries_examined = 0, leaf_reads = 0, raw_bytes_read = 0;
        idx_t runs_searched = 0, runs_skipped = 0;
        idx_t candidate_records = 0, unique_candidate_records = 0;
        idx_t distance_evaluations = 0, seed_evaluations = 0, raw_records_read = 0, seed_raw_records_read = 0;
        idx_t peak_scratch_bytes = 0;
        double approximate_seconds = 0, bounds_seconds = 0, refinement_seconds = 0;
    };

    // Paper-based Coconut-LSM-Full: bounded RAM memtable, resident summaries,
    // sorted disk runs, binary compaction and file-backed key directories.
    // Writes and queries are not concurrent; the index is temporary, not a
    // recoverable database. Gaussian SAX never normalizes the input vectors.
    class Coconut : public SimilaritySearchAlgorithm
    {
    public:
        explicit Coconut(DistanceType type);
        ~Coconut() override;
        Coconut(const Coconut &) = delete;
        Coconut &operator=(const Coconut &) = delete;
        void setNumThreads(int n) override { num_threads = n; }
        int getNumThreads() const { return num_threads; }
        int getPaaSegments() const { return paa_segments; }
        void setPaaSegments(int n);
        int getSaxCardinality() const { return sax_cardinality; }
        void setSaxCardinality(int n);
        int getLeafSize() const { return leaf_capacity_; }
        void setLeafSize(int n);
        idx_t getBufferCapacity() const { return buffer_capacity_; }
        void setBufferCapacity(idx_t records);
        void setIndexDirectory(const std::string &parent);
        std::string getIndexDirectory() const;
        idx_t getBufferedRecords() const;
        idx_t getDiskRecords() const;
        idx_t getDiskBytes() const;
        idx_t getRunCount() const;
        idx_t getFlushCount() const;
        idx_t getMergeCount() const;
        idx_t getPeakBufferedRecords() const;
        void setQueryBatchSize(idx_t queries);
        idx_t getQueryBatchSize() const { return query_batch_size_; }
        void setAsyncMerges(bool enabled);
        bool getAsyncMerges() const { return async_merges_; }
        void waitForMerges();
        idx_t getResidentSummaryBytes() const;
        CoconutSearchStats getLastSearchStats() const { return last_stats_; }
        // BTP recency is expressed as a count of most recently inserted vectors.
        void searchIndexRecent(const float *query, idx_t n_query, idx_t k, idx_t window,
                               idx_t *I, float *D);
        void setNormalized(bool normalized) override;
        using SimilaritySearchAlgorithm::buildIndex;
        void buildIndex(DataSource *source) override;
        void searchIndex(const float *query, idx_t n_query, idx_t k, idx_t *I, float *D) override;
        void searchIndex(const float *query, idx_t n_query, const SearchConfig &config,
                         std::vector<std::vector<idx_t>> &I, std::vector<std::vector<float>> &D) override;
        void insert(const float *series) override;
        void insertBatch(const float *data, idx_t n) override;
        void flush();
    private:
        struct DiskState;
        std::unique_ptr<DiskState> state_;
        int leaf_capacity_ = 1000;
        idx_t buffer_capacity_ = 1000000;
        std::string storage_parent_;
        bool async_merges_ = true;
        idx_t query_batch_size_ = 32;
        CoconutSearchStats last_stats_;
        void searchTopK(const float *query, idx_t n_query, idx_t k, idx_t first_id, idx_t *I, float *D);
        void requireMutableConfig() const;
        void requireBuilt() const;
    };
}
#endif
