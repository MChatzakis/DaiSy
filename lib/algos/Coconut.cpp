#include "Coconut.hpp"
#include "CoconutBounds.hpp"
#include "../isax/SAX.hpp"
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <numeric>
#include <queue>
#include <utility>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace daisy
{
namespace
{
    size_t checkedSize(idx_t n, size_t width)
    {
        if (width && n > std::numeric_limits<size_t>::max() / width)
            throw std::runtime_error("Coconut index size overflow");
        return (size_t)n * width;
    }
    void checkFinite(const float *data, idx_t n, idx_t dim)
    {
        if (n && !data) throw std::runtime_error("Coconut vector data is null");
        for (size_t i = 0, count = checkedSize(n, (size_t)dim); i < count; ++i)
            if (!std::isfinite(data[i])) throw std::runtime_error("Coconut requires finite vectors");
    }
    float squaredL2(const float *a, const float *b, idx_t dim, float bound)
    {
        // The shared SIMD kernel consumes full groups of eight. Handle a tail
        // locally rather than reading into the next vector for other dimensions.
        const int full = (int)dim / 8 * 8;
        float distance = full ? ts_euclidean_distance_SIMD(const_cast<float *>(a), const_cast<float *>(b), full, bound) : 0;
        for (idx_t d = full; d < dim && distance <= bound; ++d)
        { float delta = a[d] - b[d]; distance += delta * delta; }
        return distance;
    }
    struct File
    {
        FILE *p = nullptr;
        std::vector<char> buffer;
        File(const std::string &path, const char *mode) : buffer(1024 * 1024)
        {
            p = fopen(path.c_str(), mode);
            if (!p) throw std::runtime_error("Coconut failed to open index leaf/run: " + path);
            setvbuf(p, buffer.data(), _IOFBF, buffer.size());
        }
        ~File() { if (p) fclose(p); }
        File(const File &) = delete;
        void write(const void *data, size_t bytes)
        {
            if (fwrite(data, 1, bytes, p) != bytes) throw std::runtime_error("Coconut failed to write index leaf/run");
        }
        void read(void *data, size_t bytes)
        {
            if (fread(data, 1, bytes, p) != bytes) throw std::runtime_error("Coconut failed to read index leaf/run");
        }
        void finish()
        {
            if (fflush(p) || fsync(fileno(p))) throw std::runtime_error("Coconut failed to persist index leaf/run");
            FILE *closed = p; p = nullptr;
            if (fclose(closed)) throw std::runtime_error("Coconut failed to close index leaf/run");
        }
    };
    // Fixed-size directory records: [sortable key][SAX word][stable ID].
    // The directory is mmap-backed and reclaimable, not a vector of O(N)
    // heap allocations. Raw vectors are in a separate corresponding run file.
    using Clock = std::chrono::steady_clock;
    double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
    struct Run
    {
        idx_t n = 0, first_id = 0, end_id = 0, leaf_count = 0, leaf_base = 0, large_leaves = 0;
        int segments = 0;
        static constexpr idx_t fanout = 128;
        std::vector<sax_type> summaries;
        // Bottom-up packed B+tree: each level stores one maximum key per child.
        std::vector<std::vector<unsigned char>> tree;
        size_t stride = 0, raw_stride = 0;
        unsigned level = 0;
        std::string meta_path, raw_path;
        int meta_fd = -1, raw_fd = -1;
        const unsigned char *map = nullptr;
        ~Run()
        {
            if (map) munmap(const_cast<unsigned char *>(map), (size_t)n * stride);
            if (meta_fd >= 0) close(meta_fd);
            if (raw_fd >= 0) close(raw_fd);
            if (!meta_path.empty()) unlink(meta_path.c_str());
            if (!raw_path.empty()) unlink(raw_path.c_str());
        }
        const unsigned char *record(idx_t pos) const { return map + (size_t)pos * stride; }
        idx_t id(idx_t pos, int seg) const
        {
            idx_t result; memcpy(&result, record(pos) + 2 * seg, sizeof(result)); return result;
        }
        void validate() const
        {
            struct stat metadata, raw;
            if (fstat(meta_fd, &metadata) || fstat(raw_fd, &raw) ||
                (uint64_t)metadata.st_size != checkedSize(n, stride) ||
                (uint64_t)raw.st_size != checkedSize(n, raw_stride))
                throw std::runtime_error("Coconut truncated or unreadable index leaf/run");
        }
        idx_t leafBegin(idx_t leaf) const { return leaf * leaf_base + std::min(leaf, large_leaves); }
        idx_t leafOf(idx_t pos) const
        {
            idx_t wide = (leaf_base + 1) * large_leaves;
            return pos < wide ? pos / (leaf_base + 1) : large_leaves + (pos - wide) / leaf_base;
        }
        idx_t locate(const sax_type *key) const
        {
            idx_t selected = 0;
            for (size_t l = tree.size(); l; --l)
            {
                const auto &keys = tree[l - 1];
                idx_t begin = selected * fanout, end = std::min<idx_t>(begin + fanout, keys.size() / segments);
                idx_t lo = begin, hi = end;
                while (lo < hi)
                {
                    idx_t mid = lo + (hi - lo) / 2;
                    if (memcmp(keys.data() + (size_t)mid * segments, key, segments) < 0) lo = mid + 1;
                    else hi = mid;
                }
                selected = std::min(lo, end - 1);
            }
            idx_t lo = leafBegin(selected), hi = leafBegin(selected + 1);
            while (lo < hi)
            {
                idx_t mid = lo + (hi - lo) / 2;
                if (memcmp(record(mid), key, segments) < 0) lo = mid + 1; else hi = mid;
            }
            return lo;
        }
        void open(int seg, int leaf_size)
        {
            meta_fd = ::open(meta_path.c_str(), O_RDONLY); raw_fd = ::open(raw_path.c_str(), O_RDONLY);
            if (meta_fd < 0 || raw_fd < 0) throw std::runtime_error("Coconut failed to open index leaf/run");
            validate();
            void *p = mmap(nullptr, checkedSize(n, stride), PROT_READ, MAP_SHARED, meta_fd, 0);
            if (p == MAP_FAILED) throw std::runtime_error("Coconut cannot map its on-disk key directory");
            map = static_cast<const unsigned char *>(p);
            segments = seg;
            // Algorithm 7 scans a packed, resident SAX snapshot, not mapped keys/IDs.
            summaries.resize(checkedSize(n, (size_t)seg));
            for (idx_t i = 0; i < n; ++i)
                memcpy(summaries.data() + (size_t)i * seg, record(i) + seg, seg);
            leaf_count = 1 + (n - 1) / (idx_t)leaf_size;
            leaf_base = n / leaf_count; large_leaves = n % leaf_count;
            tree.emplace_back(checkedSize(leaf_count, (size_t)seg));
            for (idx_t i = 0; i < leaf_count; ++i)
                memcpy(tree[0].data() + (size_t)i * seg, record(leafBegin(i + 1) - 1), seg);
            while (tree.back().size() / seg > fanout)
            {
                const auto &lower = tree.back(); idx_t children = lower.size() / seg;
                std::vector<unsigned char> upper(checkedSize(1 + (children - 1) / fanout, (size_t)seg));
                for (idx_t i = 0; i < upper.size() / seg; ++i)
                    memcpy(upper.data() + (size_t)i * seg, lower.data() + (size_t)(std::min(children, (i + 1) * fanout) - 1) * seg, seg);
                tree.push_back(std::move(upper));
            }
        }
        void leaf(idx_t begin, idx_t count, std::vector<float> &page) const
        {
            size_t bytes = checkedSize(count, raw_stride), got = 0;
            page.resize(bytes / sizeof(float));
            while (got < bytes)
            {
                ssize_t result = pread(raw_fd, reinterpret_cast<char *>(page.data()) + got,
                    bytes - got, (off_t)checkedSize(begin, raw_stride) + got);
                if (result < 0 && errno == EINTR) continue;
                if (result <= 0) throw std::runtime_error("Coconut failed to read index leaf/run");
                got += (size_t)result;
            }
        }
    };
}
    struct Coconut::DiskState
    {
        idx_t dim, capacity, count = 0, buffer_first = 0;
        int seg, bits, values, alphabet, leaf_size;
        size_t stride;
        std::string directory;
        std::atomic<unsigned> serial{0};
        idx_t flushes = 0, peak_buffered = 0;
        std::atomic<idx_t> merges{0};
        bool bulk_loading = true, async_merges = true, stopping = false, merging = false;
        mutable std::mutex mutex;
        std::condition_variable changed;
        std::thread merger;
        std::exception_ptr merge_error;
        std::vector<float> data;
        std::vector<sax_type> sax, keys, cardinalities;
        std::vector<std::shared_ptr<Run>> runs;
        DiskState(idx_t d, int s, int b, int leaf, idx_t cap, const std::string &parent, bool async)
            : dim(d), capacity(cap), seg(s), bits(b), values((int)d / s), alphabet(1 << b),
              leaf_size(leaf), stride(2 * (size_t)s + sizeof(idx_t)), cardinalities(s, (sax_type)b)
        {
            std::string root = parent.empty() ? std::filesystem::temp_directory_path().string() : parent;
            std::filesystem::create_directories(root);
            std::string pattern = root + "/daisy_coconut_" + std::to_string((unsigned long)getpid()) + "_XXXXXX";
            std::vector<char> name(pattern.begin(), pattern.end()); name.push_back(0);
            if (!mkdtemp(name.data())) throw std::runtime_error("Coconut cannot create its disk directory");
            directory = name.data(); async_merges = async;
        }
        ~DiskState()
        {
            { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
            changed.notify_all(); if (merger.joinable()) merger.join();
            runs.clear();
            std::error_code ignored; std::filesystem::remove_all(directory, ignored);
        }
        idx_t buffered() const { return data.size() / (size_t)dim; }
        std::shared_ptr<Run> newRun(idx_t n, unsigned level)
        {
            auto run = std::make_shared<Run>(); run->n = n; run->level = level;
            run->stride = stride; run->raw_stride = checkedSize(dim, sizeof(float));
            std::string prefix = directory + "/run." + std::to_string(serial++);
            run->meta_path = prefix + ".meta"; run->raw_path = prefix + ".raw"; return run;
        }
        std::shared_ptr<Run> merge(const std::vector<std::shared_ptr<Run>> &inputs, unsigned level)
        {
            idx_t total = 0;
            for (const auto &run : inputs) { run->validate(); total += run->n; }
            auto result = newRun(total, level);
            result->first_id = inputs.front()->first_id; result->end_id = inputs.front()->end_id;
            for (const auto &run : inputs)
            { result->first_id = std::min(result->first_id, run->first_id); result->end_id = std::max(result->end_id, run->end_id); }
            File meta(result->meta_path, "wb"), raw(result->raw_path, "wb");
            std::vector<std::unique_ptr<File>> readers;
            for (const auto &run : inputs) readers.emplace_back(new File(run->raw_path, "rb"));
            std::vector<idx_t> positions(inputs.size(), 0);
            auto later = [&](size_t a, size_t b) {
                int cmp = memcmp(inputs[a]->record(positions[a]), inputs[b]->record(positions[b]), seg);
                return cmp ? cmp > 0 : inputs[a]->id(positions[a], seg) > inputs[b]->id(positions[b], seg);
            };
            std::priority_queue<size_t, std::vector<size_t>, decltype(later)> heap(later);
            for (size_t i = 0; i < inputs.size(); ++i) heap.push(i);
            std::vector<float> ts(dim);
            while (!heap.empty())
            {
                size_t i = heap.top(); heap.pop();
                meta.write(inputs[i]->record(positions[i]), stride);
                readers[i]->read(ts.data(), checkedSize(dim, sizeof(float)));
                raw.write(ts.data(), checkedSize(dim, sizeof(float)));
                if (++positions[i] < inputs[i]->n) heap.push(i);
            }
            meta.finish(); raw.finish(); result->open(seg, leaf_size); ++merges; return result;
        }
        std::pair<size_t, size_t> mergePair() const
        {
            // Only adjacent temporal intervals merge; this preserves BTP recency.
            for (size_t i = 0; i + 1 < runs.size(); ++i)
                if (runs[i]->level == runs[i + 1]->level)
                    return {i, i + 1};
            return {runs.size(), runs.size()};
        }
        void compactLoop()
        {
            std::unique_lock<std::mutex> lock(mutex);
            while (true)
            {
                changed.wait(lock, [&] { return stopping || mergePair().first < runs.size(); });
                if (stopping) return;
                auto pair = mergePair();
                auto a = runs[pair.first], b = runs[pair.second]; merging = true;
                lock.unlock();
                std::shared_ptr<Run> replacement;
                try { replacement = merge({a, b}, a->level + 1); }
                catch (...)
                {
                    lock.lock(); merge_error = std::current_exception(); merging = false;
                    changed.notify_all(); return;
                }
                lock.lock();
                // Atomically publish a complete replacement, retaining old files
                // while any query snapshot still refers to the two input runs.
                auto it = std::find(runs.begin(), runs.end(), a);
                auto pos = it - runs.begin();
                runs.erase(it, it + 2); runs.insert(runs.begin() + pos, std::move(replacement));
                a.reset(); b.reset(); merging = false; changed.notify_all();
            }
        }
        void waitForMerges()
        {
            if (bulk_loading) return;
            std::unique_lock<std::mutex> lock(mutex);
            changed.wait(lock, [&] { return merge_error || (!merging && mergePair().first == runs.size()); });
            if (merge_error) std::rethrow_exception(merge_error);
        }
        std::vector<std::shared_ptr<Run>> snapshot() const
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (merge_error) std::rethrow_exception(merge_error);
            return runs;
        }
        void publish(std::shared_ptr<Run> run)
        {
            { std::lock_guard<std::mutex> lock(mutex); runs.push_back(std::move(run)); }
            if (bulk_loading) return;
            changed.notify_all();
            if (!async_merges) { waitForMerges(); return; }
            // Bound pending runs even when the writer outruns disk compaction.
            unsigned levels = 1; idx_t chunks = count / capacity;
            while (chunks > 1) { chunks /= 2; ++levels; }
            std::unique_lock<std::mutex> lock(mutex);
            changed.wait(lock, [&] { return merge_error || runs.size() <= 2 * (size_t)(levels + 1); });
            if (merge_error) std::rethrow_exception(merge_error);
        }
        void flush()
        {
            idx_t n = buffered(); if (!n) return;
            std::vector<idx_t> order(n); std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](idx_t a, idx_t b) {
                int cmp = memcmp(keys.data() + (size_t)a * seg, keys.data() + (size_t)b * seg, seg);
                return cmp ? cmp < 0 : a < b;
            });
            auto run = newRun(n, 0); run->first_id = buffer_first; run->end_id = buffer_first + n;
            File meta(run->meta_path, "wb"), raw(run->raw_path, "wb");
            for (idx_t slot : order)
            {
                idx_t id = buffer_first + slot;
                meta.write(keys.data() + (size_t)slot * seg, seg);
                meta.write(sax.data() + (size_t)slot * seg, seg); meta.write(&id, sizeof(id));
                raw.write(data.data() + checkedSize(slot, (size_t)dim), checkedSize(dim, sizeof(float)));
            }
            meta.finish(); raw.finish(); run->open(seg, leaf_size);
            publish(std::move(run));
            data.clear(); sax.clear(); keys.clear(); buffer_first = count; ++flushes;
        }
        void append(const float *input, idx_t n, int threads)
        {
            if (n > std::numeric_limits<idx_t>::max() - count) throw std::runtime_error("Coconut record count overflow");
            snapshot(); // Surface any failed background merge before accepting more records.
            while (n)
            {
                if (buffered() == capacity) flush();
                idx_t first = buffered(), take = std::min(n, capacity - first);
                if (!first) buffer_first = count;
                size_t raw_size = checkedSize(first + take, (size_t)dim);
                size_t key_size = checkedSize(first + take, (size_t)seg);
                // Explicit reserve avoids vector's geometric growth exceeding
                // the declared memtable capacity after many small inserts.
                if (data.capacity() < raw_size) data.reserve(raw_size);
                if (sax.capacity() < key_size) sax.reserve(key_size);
                if (keys.capacity() < key_size) keys.reserve(key_size);
                data.resize(raw_size); sax.resize(key_size); keys.resize(key_size);
                memcpy(data.data() + checkedSize(first, (size_t)dim), input, checkedSize(take, checkedSize(dim, sizeof(float))));
#pragma omp parallel num_threads(std::max(1, threads))
                {
                    std::vector<float> paa(seg);
#pragma omp for schedule(static)
                    for (idx_t i = first; i < first + take; ++i)
                    {
                        paa_from_ts(data.data() + checkedSize(i, (size_t)dim), paa.data(), seg, values);
                        sax_from_paa(paa.data(), sax.data() + (size_t)i * seg, seg, alphabet, bits);
                        sortable_sax_from_sax(sax.data() + (size_t)i * seg, keys.data() + (size_t)i * seg, seg, bits);
                    }
                }
                count += take; n -= take; input += checkedSize(take, (size_t)dim);
                peak_buffered = std::max(peak_buffered, buffered());
                if (buffered() == capacity) flush();
            }
        }
        void consolidateBuild()
        {
            // Algorithm 6: sort initial chunks once, then merge them; do not run
            // incremental binary compactions repeatedly during initial loading.
            unsigned level = 0; idx_t size = capacity;
            while (size < count && size <= std::numeric_limits<idx_t>::max() / 2) { size *= 2; ++level; }
            constexpr size_t max_fanin = 64;
            while (runs.size() > 1)
            {
                std::vector<std::shared_ptr<Run>> next;
                for (size_t i = 0; i < runs.size(); i += max_fanin)
                {
                    size_t end = std::min(runs.size(), i + max_fanin);
                    std::vector<std::shared_ptr<Run>> group(runs.begin() + i, runs.begin() + end);
                    next.push_back(group.size() > 1 ? merge(group, level) : group[0]);
                }
                runs = std::move(next);
            }
            if (!runs.empty()) runs[0]->level = level;
            bulk_loading = false;
            merger = std::thread([this] { compactLoop(); });
        }
        float bound(const float *paa, const sax_type *word) const
        {
            return minidist_paa_to_isax(const_cast<float *>(paa), const_cast<sax_type *>(word),
                const_cast<sax_type *>(cardinalities.data()), (sax_type)bits, alphabet,
                seg, MINVAL, MAXVAL, (float)values);
        }
        void validateRuns() const { for (const auto &run : snapshot()) run->validate(); }
    };

    Coconut::Coconut(DistanceType type) : SimilaritySearchAlgorithm(type) {}
    Coconut::~Coconut() = default;
    void Coconut::requireMutableConfig() const
    { if (state_) throw std::runtime_error("Coconut layout settings cannot change after buildIndex"); }
    void Coconut::requireBuilt() const
    { if (!state_) throw std::runtime_error("Coconut requires an initial buildIndex first"); }
    void Coconut::setPaaSegments(int n) { requireMutableConfig(); paa_segments = n; }
    void Coconut::setSaxCardinality(int n) { requireMutableConfig(); sax_cardinality = n; }
    void Coconut::setLeafSize(int n) { requireMutableConfig(); leaf_capacity_ = n; }
    void Coconut::setBufferCapacity(idx_t n) { requireMutableConfig(); buffer_capacity_ = n; }
    void Coconut::setQueryBatchSize(idx_t queries)
    {
        if (!queries || queries > 64) throw std::runtime_error("Coconut query batch size must be between 1 and 64");
        query_batch_size_ = queries;
    }
    void Coconut::setAsyncMerges(bool enabled) { requireMutableConfig(); async_merges_ = enabled; }
    void Coconut::waitForMerges() { requireBuilt(); state_->waitForMerges(); }
    idx_t Coconut::getResidentSummaryBytes() const
    {
        idx_t bytes = 0; if (state_) for (const auto &run : state_->snapshot()) bytes += run->summaries.size();
        return bytes;
    }
    void Coconut::setIndexDirectory(const std::string &p) { requireMutableConfig(); storage_parent_ = p; }
    void Coconut::setNormalized(bool normalized)
    {
        if (!normalized) throw std::runtime_error("Coconut supports Gaussian breakpoints only; input is never normalized");
        SimilaritySearchAlgorithm::setNormalized(true);
    }
    std::string Coconut::getIndexDirectory() const { return state_ ? state_->directory : std::string(); }
    idx_t Coconut::getBufferedRecords() const { return state_ ? state_->buffered() : 0; }
    idx_t Coconut::getDiskRecords() const { return n_database - getBufferedRecords(); }
    idx_t Coconut::getDiskBytes() const
    { return state_ ? getDiskRecords() * (state_->stride + state_->dim * sizeof(float)) : 0; }
    idx_t Coconut::getRunCount() const { return state_ ? state_->snapshot().size() : 0; }
    idx_t Coconut::getFlushCount() const { return state_ ? state_->flushes : 0; }
    idx_t Coconut::getMergeCount() const { return state_ ? state_->merges.load() : 0; }
    idx_t Coconut::getPeakBufferedRecords() const { return state_ ? state_->peak_buffered : 0; }

    void Coconut::buildIndex(DataSource *source)
    {
        if (!source) throw std::runtime_error("Coconut datasource is null");
        idx_t d = source->getDim();
        if (!d || d > (idx_t)std::numeric_limits<int>::max() || paa_segments <= 0 || d % paa_segments ||
            sax_cardinality < 1 || sax_cardinality > 8 || leaf_capacity_ <= 0 || !buffer_capacity_)
            throw std::runtime_error("Coconut requires positive leaf/PAA sizes, buffer capacity, 1-8 SAX bits, and a dimension divisible by PAA size");
        checkedSize(buffer_capacity_, checkedSize(d, sizeof(float)));
        auto fresh = std::make_unique<DiskState>(d, paa_segments, sax_cardinality, leaf_capacity_, buffer_capacity_, storage_parent_, async_merges_);
        set_active_breakpoints(nullptr, nullptr); source->reset();
        idx_t expected = source->getTotalRecords(); const float *raw = source->rawPointer();
        if (raw)
            for (idx_t start = 0; start < expected; start += buffer_capacity_)
            {
                idx_t take = std::min(buffer_capacity_, expected - start);
                checkFinite(raw + checkedSize(start, (size_t)d), take, d);
                fresh->append(raw + checkedSize(start, (size_t)d), take, num_threads);
            }
        else
        {
            std::vector<float> chunk(checkedSize(buffer_capacity_, (size_t)d));
            while (!expected || fresh->count < expected)
            {
                idx_t got = 0, limit = expected ? std::min(buffer_capacity_, expected - fresh->count) : buffer_capacity_;
                while (got < limit && source->nextRecord(chunk.data() + checkedSize(got, (size_t)d))) ++got;
                if (!got) break;
                checkFinite(chunk.data(), got, d); fresh->append(chunk.data(), got, num_threads);
                if (got < limit) break;
            }
            if (expected && fresh->count != expected)
                throw std::runtime_error("Coconut datasource ended before all records were read");
        }
        fresh->flush(); fresh->consolidateBuild();
        dim = d; n_database = fresh->count; state_ = std::move(fresh);
    }
    void Coconut::insert(const float *data) { insertBatch(data, 1); }
    void Coconut::insertBatch(const float *data, idx_t n)
    {
        requireBuilt(); checkFinite(data, n, dim); set_active_breakpoints(nullptr, nullptr);
        try { state_->append(data, n, num_threads); }
        catch (...) { n_database = state_->count; throw; }
        n_database = state_->count;
    }
    void Coconut::flush() { requireBuilt(); state_->flush(); }

    void Coconut::searchIndex(const float *query, idx_t n_query, idx_t k, idx_t *I, float *D)
    { searchTopK(query, n_query, k, 0, I, D); }

    void Coconut::searchIndexRecent(const float *query, idx_t n_query, idx_t k, idx_t window, idx_t *I, float *D)
    {
        requireBuilt();
        if (!window) throw std::runtime_error("Coconut recent window must be positive");
        searchTopK(query, n_query, k, window < n_database ? n_database - window : 0, I, D);
    }

    void Coconut::searchTopK(const float *query, idx_t n_query, idx_t k, idx_t first_id, idx_t *I, float *D)
    {
        requireBuilt();
        if (distance_type != DistanceType::L2_SQUARED) throw std::runtime_error("Coconut top-k search only supports L2_SQUARED");
        if (!k) throw std::runtime_error("Coconut top-k search requires k > 0");
        checkFinite(query, n_query, dim); set_active_breakpoints(nullptr, nullptr);
        const auto &s = *state_; auto runs = s.snapshot();
        for (const auto &run : runs) run->validate();
        last_stats_ = {};
        // Batch SIMS keeps Coconut's seed -> SAX bounds -> exact refinement.
        // Queries share bounded ordered reads, never their heaps or pruning bound.
        struct alignas(64) Work
        {
            const float *query = nullptr;
            std::vector<float> paa;
            coconut_detail::QueryBounds lookup;
            std::vector<sax_type> key;
            std::vector<std::pair<idx_t, idx_t>> windows;
            std::priority_queue<std::pair<float, idx_t>> best;
            idx_t candidates = 0, evaluations = 0, seeds = 0;
        };
        const idx_t max_raw_bytes = 8 * 1024 * 1024;
        const size_t raw_stride = checkedSize(dim, sizeof(float));
        const idx_t block_capacity = std::min<idx_t>(32768, std::max<idx_t>(1, max_raw_bytes / raw_stride));
        for (idx_t base = 0; base < n_query; base += query_batch_size_)
        {
            idx_t count = std::min(query_batch_size_, n_query - base);
            const int workers = std::max(1, num_threads);
            std::vector<Work> work(count);
            std::vector<float> page;
            page.reserve(checkedSize(block_capacity, (size_t)dim));
            auto refine = [&](Work &w, const float *ts, idx_t id, bool seed) {
                if (id < first_id) return;
                ++w.evaluations; if (seed) ++w.seeds;
                float kth = w.best.size() >= k ? w.best.top().first : FLT_MAX;
                float distance = squaredL2(w.query, ts, dim, std::nextafter(kth, FLT_MAX));
                std::pair<float, idx_t> candidate(distance, id);
                if (w.best.size() < k) w.best.push(candidate);
                else if (candidate < w.best.top()) { w.best.pop(); w.best.push(candidate); }
            };
            auto read = [&](const std::shared_ptr<Run> &run, idx_t begin, idx_t records, bool seed = false) {
                run->leaf(begin, records, page);
                ++last_stats_.leaf_reads; last_stats_.raw_bytes_read += records * raw_stride;
                last_stats_.raw_records_read += records;
                if (seed) last_stats_.seed_raw_records_read += records;
            };
            auto started = Clock::now();
            for (idx_t q = 0; q < count; ++q)
            {
                auto &w = work[q]; w.query = query + checkedSize(base + q, (size_t)dim);
                w.paa.resize(s.seg); w.key.resize(s.seg); w.windows.resize(runs.size());
                std::vector<sax_type> word(s.seg);
                paa_from_ts(const_cast<float *>(w.query), w.paa.data(), s.seg, s.values);
                w.lookup.reset(w.paa.data(), s.seg, s.bits, (float)s.values);
                sax_from_paa(w.paa.data(), word.data(), s.seg, s.alphabet, s.bits);
                sortable_sax_from_sax(word.data(), w.key.data(), s.seg, s.bits);
                for (size_t r = 0; r < runs.size(); ++r)
                {
                    const auto &run = runs[r];
                    if (run->end_id <= first_id) { ++last_stats_.runs_skipped; continue; }
                    ++last_stats_.runs_searched;
                    idx_t target = run->locate(w.key.data());
                    idx_t window = std::min(run->n, std::max<idx_t>(s.leaf_size, k));
                    idx_t begin = std::min(target > window / 2 ? target - window / 2 : 0, run->n - window);
                    w.windows[r] = {begin, begin + window};
                    for (idx_t first = begin; first < begin + window;)
                    {
                        idx_t end = std::min(begin + window, std::min(first + block_capacity, run->leafBegin(run->leafOf(first) + 1)));
                        read(run, first, end - first, true);
                        for (idx_t pos = first; pos < end; ++pos)
                            refine(w, page.data() + checkedSize(pos - first, (size_t)dim), run->id(pos, s.seg), true);
                        first = end;
                    }
                }
                for (idx_t b = 0; b < s.buffered(); ++b)
                    refine(w, s.data.data() + checkedSize(b, (size_t)dim), s.buffer_first + b, true);
            }
            last_stats_.approximate_seconds += elapsed(started);
            // Scratch is independent of N: one raw block plus query-count x block
            // bounds/masks. SAX summaries remain the only resident O(N) search data.
            std::vector<float> bounds(checkedSize(count, (size_t)block_capacity));
            std::vector<unsigned char> candidate(bounds.size()), active(block_capacity);
            page.reserve(checkedSize(block_capacity, (size_t)dim));
            size_t lookup_bytes = 0; for (const auto &w : work) lookup_bytes += w.lookup.bytes();
            last_stats_.peak_scratch_bytes = std::max<idx_t>(last_stats_.peak_scratch_bytes,
                bounds.size() * sizeof(float) + candidate.size() + active.size() + page.capacity() * sizeof(float) + lookup_bytes);
            for (size_t r = 0; r < runs.size(); ++r)
            {
                const auto &run = runs[r]; if (run->end_id <= first_id) continue;
                last_stats_.summaries_examined += count * run->n;
                const bool boundary_run = run->first_id < first_id;
                for (idx_t begin = 0; begin < run->n; begin += block_capacity)
                {
                    idx_t records = std::min(block_capacity, run->n - begin);
                    started = Clock::now();
                    std::vector<float> threshold(count);
                    for (idx_t q = 0; q < count; ++q)
                        threshold[q] = work[q].best.size() >= k ? work[q].best.top().first : FLT_MAX;
#pragma omp parallel for collapse(2) num_threads(workers) schedule(static)
                    for (idx_t q = 0; q < count; ++q)
                        for (idx_t i = 0; i < records; ++i)
                        {
                            const auto &w = work[q]; size_t row = (size_t)q * block_capacity;
                            idx_t pos = begin + i;
                            float bound = w.lookup(run->summaries.data() + (size_t)pos * s.seg, threshold[q]);
                            bounds[row + i] = bound;
                            bool survives = bound <= threshold[q] && !(pos >= w.windows[r].first && pos < w.windows[r].second);
                            if (survives && boundary_run) survives = run->id(pos, s.seg) >= first_id;
                            candidate[row + i] = survives;
                        }
                    for (idx_t q = 0; q < count; ++q)
                        for (idx_t i = 0; i < records; ++i)
                            work[q].candidates += candidate[(size_t)q * block_capacity + i];
                    last_stats_.bounds_seconds += elapsed(started);
                    started = Clock::now();
                    std::fill_n(active.data(), records, 0);
                    for (idx_t q = 0; q < count; ++q)
                    {
                        const auto *mask = candidate.data() + (size_t)q * block_capacity;
                        for (idx_t i = 0; i < records; ++i) active[i] |= mask[i];
                    }
                    idx_t active_count = 0;
                    for (idx_t i = 0; i < records; ++i) active_count += active[i] != 0;
                    last_stats_.unique_candidate_records += active_count;
                    // Dense blocks trade at most 2x unique-candidate bytes for
                    // a single sequential read; sparse blocks retain short gaps.
                    size_t max_gap = active_count * 2 >= records ? records * raw_stride : 4096;
                    // Sparse positions are coalesced only across <=4 KiB gaps.
                    // Dense candidate blocks become one bounded read reused by
                    // every query in this tile. No unconditional raw full scan.
                    idx_t cursor = 0;
                    while (cursor < records)
                    {
                        while (cursor < records && !active[cursor]) ++cursor;
                        if (cursor == records) break;
                        idx_t first = cursor, last = cursor++;
                        while (cursor < records)
                        {
                            if (active[cursor]) last = cursor;
                            else if ((cursor - last) * raw_stride > max_gap) break;
                            ++cursor;
                        }
                        idx_t end = last + 1;
                        read(run, begin + first, end - first);
                        std::exception_ptr error;
#pragma omp parallel for num_threads(workers) if(count > 1) schedule(static)
                        for (idx_t q = 0; q < count; ++q)
                        {
                            try
                            {
                                auto &w = work[q]; size_t row = (size_t)q * block_capacity;
                                for (idx_t i = first; i < end; ++i)
                                {
                                    if (!candidate[row + i]) continue;
                                    float kth = w.best.size() >= k ? w.best.top().first : FLT_MAX;
                                    if (bounds[row + i] > kth) continue;
                                    refine(w, page.data() + checkedSize(i - first, (size_t)dim), run->id(begin + i, s.seg), false);
                                }
                            }
                            catch (...)
                            {
#pragma omp critical(coconut_batch_query_error)
                                { if (!error) error = std::current_exception(); }
                            }
                        }
                        if (error) std::rethrow_exception(error);
                    }
                    last_stats_.refinement_seconds += elapsed(started);
                }
            }
            for (idx_t q = 0; q < count; ++q)
            {
                auto &w = work[q]; last_stats_.candidate_records += w.candidates;
                last_stats_.distance_evaluations += w.evaluations; last_stats_.seed_evaluations += w.seeds;
                idx_t got = w.best.size(), offset = (base + q) * k;
                for (idx_t j = got; j; --j)
                { D[offset + j - 1] = w.best.top().first; I[offset + j - 1] = w.best.top().second; w.best.pop(); }
                for (idx_t j = got; j < k; ++j) { I[offset + j] = 0; D[offset + j] = FLT_MAX; }
            }
        }
    }
    void Coconut::searchIndex(const float *query, idx_t n_query, const SearchConfig &config,
                              std::vector<std::vector<idx_t>> &I, std::vector<std::vector<float>> &D)
    {
        if (config.type == QueryType::TOP_K)
        { SimilaritySearchAlgorithm::searchIndex(query, n_query, config, I, D); return; }
        requireBuilt();
        if (distance_type != DistanceType::L2_SQUARED) throw std::runtime_error("Coconut range search only supports L2_SQUARED");
        if (std::isnan(config.r) || config.r < 0) throw std::runtime_error("Coconut radius must be nonnegative");
        checkFinite(query, n_query, dim); state_->validateRuns(); set_active_breakpoints(nullptr, nullptr);
        I.assign(n_query, {}); D.assign(n_query, {});
        const auto &s = *state_; auto runs = s.snapshot(); std::exception_ptr error;
        int workers = (int)std::max<idx_t>(1, std::min<idx_t>(std::max(1, num_threads), n_query));
#pragma omp parallel for num_threads(workers) schedule(dynamic, 1)
        for (idx_t q = 0; q < n_query; ++q)
        {
            try
            {
                const float *qts = query + checkedSize(q, (size_t)dim);
                std::vector<float> paa(s.seg), page; paa_from_ts(const_cast<float *>(qts), paa.data(), s.seg, s.values);
                std::vector<std::pair<float, idx_t>> hits;
                auto refine = [&](const float *ts, idx_t id) {
                    float distance = squaredL2(qts, ts, dim, std::nextafter(config.r, FLT_MAX));
                    if (distance <= config.r) hits.emplace_back(distance, id);
                };
                for (const auto &run : runs)
                    for (idx_t leaf = 0; leaf < run->leaf_count; ++leaf)
                    {
                        idx_t begin = run->leafBegin(leaf), end = run->leafBegin(leaf + 1); std::vector<idx_t> candidates;
                        for (idx_t p = begin; p < end; ++p)
                            if (s.bound(paa.data(), run->summaries.data() + (size_t)p * s.seg) <= config.r) candidates.push_back(p);
                        if (candidates.empty()) continue;
                        run->leaf(begin, end - begin, page);
                        for (idx_t p : candidates)
                            refine(page.data() + checkedSize(p - begin, (size_t)dim), run->id(p, s.seg));
                    }
                for (idx_t b = 0; b < s.buffered(); ++b)
                    if (s.bound(paa.data(), s.sax.data() + (size_t)b * s.seg) <= config.r)
                        refine(s.data.data() + checkedSize(b, (size_t)dim), s.buffer_first + b);
                std::sort(hits.begin(), hits.end());
                for (const auto &hit : hits) { D[q].push_back(hit.first); I[q].push_back(hit.second); }
            }
            catch (...)
            {
#pragma omp critical(coconut_query_error)
                { if (!error) error = std::current_exception(); }
            }
        }
        if (error) std::rethrow_exception(error);
    }
}
