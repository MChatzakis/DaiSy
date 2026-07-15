#pragma once

#include "SimilaritySearchAlgorithm.hpp"
#include "../isax/iSAXTypes.hpp"

#include <vector>

namespace daisy {

// Optional constructor parameters. Defaults follow the DumpyOS paper.
struct DumpyOSConfig {
    int   leaf_size           = 10000;
    int   paa_segments        = 16;
    int   sax_bit_cardinality = 8;
    float alpha               = 0.2f;
    float fill_lower          = 0.5f;   // f_low  in the paper
    float fill_upper          = 3.0f;   // f_high in the paper
};

// A node in the DumpyOS multi-ary tree. Adapted from FADASNode in the reference impl.
// Internal nodes keep chosen_segs and children, leaves keep entries.
struct DumpyOSNode {
    std::vector<int>          levels;       // bits_cardinality[] per segment
    std::vector<int>          sax_word;     // SAX word at current bit depth (needed for LB)
    std::vector<int>          chosen_segs;  // chosen segments: empty , leaf
    std::vector<DumpyOSNode*> children;     // 2^|chosen_segs| entries (may be nullptr)
    std::vector<idx_t>        entries;      // series indices (leaf only)
    int n = 0;
};

// Data-adaptive multi-ary iSAX index for exact similarity search.
class DumpyOS : public SimilaritySearchAlgorithm {
public:
    // Default constructor. Uses DumpyOSConfig defaults.
    DumpyOS(DistanceType distance_type);
    // Explicit constructor.
    DumpyOS(DistanceType distance_type, const DumpyOSConfig& config);

    using SimilaritySearchAlgorithm::buildIndex;

    // Set the Sakoe-Chiba band radius for DTW.
    void setWarpingWindow(int w) { warping_window = w; }

    // Build the DumpyOS tree from an in-memory data source.
    void buildIndex(DataSource* data_source) override;
    // Top-k search using the configured distance type.
    void searchIndex(const float* query, idx_t n_query, idx_t k,
                     idx_t* I, float* D) override;
    // Range and top-k combined entry point. Reads mode and radius from config.
    void searchIndex(const float *query, idx_t n_query, const SearchConfig &config,
                     std::vector<std::vector<idx_t>> &I,
                     std::vector<std::vector<float>> &D) override;

    ~DumpyOS() override;

private:
    DumpyOSConfig config_;
    DumpyOSNode*  root_          = nullptr;
    sax_type*     sax_table_     = nullptr;  // [n_database * paa_segments]
    bool          owns_database_ = false;

    // Pick the segments to split on for this node using the DumpyOS heuristic.
    void determineSegments_(DumpyOSNode* node);
    // Materialise the children of a node from the chosen segments.
    void splitNode_(DumpyOSNode* node);
    // Recursively free the tree.
    void destroyTree_(DumpyOSNode* node);
};

} 
