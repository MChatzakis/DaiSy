#pragma once

#include "SimilaritySearchAlgorithm.hpp"
#include "../isax/iSAXTypes.hpp"

#include <vector>

namespace daisy {

struct DumpyOSConfig {
    int   leaf_size           = 10000;
    int   paa_segments        = 16;
    int   sax_bit_cardinality = 8;
    float alpha               = 0.2f;
    float fill_lower          = 0.5f;   // f_low  in the paper
    float fill_upper          = 3.0f;   // f_high in the paper
};

// Adapted from FADASNode in DumpyOS
struct DumpyOSNode {
    std::vector<int>          levels;       // bits_cardinality[] per segment
    std::vector<int>          sax_word;     // SAX word at current bit depth (needed for LB)
    std::vector<int>          chosen_segs;  // chosen segments: empty , leaf
    std::vector<DumpyOSNode*> children;     // 2^|chosen_segs| entries (may be nullptr)
    std::vector<idx_t>        entries;      // series indices (leaf only)
    int n = 0;
};

class DumpyOS : public SimilaritySearchAlgorithm {
public:
    DumpyOS(DistanceType distance_type);
    DumpyOS(DistanceType distance_type, const DumpyOSConfig& config);

    using SimilaritySearchAlgorithm::buildIndex;

    void setWarpingWindow(int w) { warping_window = w; }

    void buildIndex(DataSource* data_source) override;
    void searchIndex(const float* query, idx_t n_query, idx_t k,
                     idx_t* I, float* D) override;
    void searchIndex(const float *query, idx_t n_query, const SearchConfig &config,
                     std::vector<std::vector<idx_t>> &I,
                     std::vector<std::vector<float>> &D) override;

    ~DumpyOS() override;

private:
    DumpyOSConfig config_;
    DumpyOSNode*  root_          = nullptr;
    sax_type*     sax_table_     = nullptr;  // [n_database * paa_segments]
    bool          owns_database_ = false;

    void determineSegments_(DumpyOSNode* node);
    void splitNode_(DumpyOSNode* node);
    void destroyTree_(DumpyOSNode* node);
};

} 
