#ifndef DS_TREE_NODE_HPP
#define DS_TREE_NODE_HPP

#include <vector>
#include <cfloat>
#include <cstdint>
#include <pthread.h>

namespace daisy {

struct SegmentSketch {
    float indicators[4] = {-FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX};
    int num_indicators = 4;
};

struct node_split_policy {
    int split_from = 0;
    int split_to = 0;
    int indicator_split_idx = 0;
    float indicator_split_val = 0.0f;
};

struct HerculesNode {
    std::vector<int> node_points;
    std::vector<int> hs_node_points;
    std::vector<SegmentSketch> node_segment_sketches;
    std::vector<SegmentSketch> hs_node_segment_sketches;

    node_split_policy *split_policy = nullptr;

    HerculesNode *left_child  = nullptr;
    HerculesNode *right_child = nullptr;
    HerculesNode *parent      = nullptr;

    std::vector<int> series_indices;

    unsigned int node_size = 0;
    unsigned int level     = 0;
    bool is_leaf  = true;
    bool is_left  = false;

    uint64_t file_pos = 0;

    pthread_mutex_t lock_data;
    pthread_mutex_t lock_split;

    HerculesNode() {
        pthread_mutex_init(&lock_data,  nullptr);
        pthread_mutex_init(&lock_split, nullptr);
    }
    ~HerculesNode() {
        pthread_mutex_destroy(&lock_data);
        pthread_mutex_destroy(&lock_split);
        delete split_policy;
    }
};

} // namespace daisy
#endif // DS_TREE_NODE_HPP
