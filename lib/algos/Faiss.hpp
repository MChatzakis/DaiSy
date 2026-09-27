#ifndef DAISY_FAISS_HPP
#define DAISY_FAISS_HPP

#include "SimilaritySearchAlgorithm.hpp"

#include <memory>

// Declared rather than included so consumers of daisy.hpp do not need the FAISS
// headers on their include path. The concrete type is only needed in Faiss.cpp.
namespace faiss
{
    struct IndexFlatL2;
}

namespace daisy
{

    /**
     * Exact similarity search backed by faiss::IndexFlatL2.
     *
     * This is DaiSy's reference baseline rather than a new index: a flat FAISS
     * index answers every query with a full SIMD/BLAS scan, so results are exact
     * and no structure has to be maintained across updates.
     *
     * That makes its streaming behaviour the cheapest an insert can be.
     * faiss::IndexFlatL2::add() appends into the flat code storage with no
     * training step and no rebuild, and the index stays exact afterwards, so
     * unlike LbBruteforce, Messi and Coconut there are no breakpoints, centroids
     * or codebooks frozen at build time that could drift from the live data.
     *
     * Only L2_SQUARED is supported. FAISS METRIC_L2 reports squared Euclidean
     * distances, which is what DaiSy's L2_SQUARED already means, so distances
     * pass through unscaled. DTW has no FAISS equivalent and is rejected.
     *
     * As with the other streaming algorithms, inserts are not concurrent with
     * queries: callers must serialize updates against searches.
     */
    class FaissFlat : public SimilaritySearchAlgorithm
    {
    private:
        std::unique_ptr<faiss::IndexFlatL2> faiss_index;

        // FAISS owns the vectors inside its own storage, which moves when it
        // grows. The inherited `database` pointer is kept as a non-owning view of
        // that storage and refreshed after every build and insert.
        void refreshDatabaseView();

    public:
        explicit FaissFlat(DistanceType distance_type);

        void setNumThreads(int num_threads) override;
        int getNumThreads() const { return this->num_threads; }

        using SimilaritySearchAlgorithm::buildIndex;

        void buildIndex(DataSource *data_source) override;

        // Appends into the live FAISS index. No retraining, and the index stays exact.
        void insert(const float *series) override;
        void insertBatch(const float *data, idx_t n) override;

        void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) override;

        void searchIndex(const float *query, idx_t n_query, const SearchConfig &config,
                         std::vector<std::vector<idx_t>> &I,
                         std::vector<std::vector<float>> &D) override;

        // Defined in the .cpp, where faiss::IndexFlatL2 is a complete type.
        ~FaissFlat() override;
    };

}

#endif
