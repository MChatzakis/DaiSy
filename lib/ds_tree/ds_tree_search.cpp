#include "ds_tree_search.hpp"
#include "ds_tree_index.hpp"

#include <cmath>
#include <cfloat>
#include <algorithm>

namespace daisy
{

    float l2sq(const float *a, const float *b, int dim, float bound)
    {
        float sum = 0;
        for (int i = 0; i < dim; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
            if (sum >= bound) return sum;
        }
        return sum;
    }

    void knn_bounded_insert(std::vector<HerculesKnnResult> &knn, idx_t k,
                            idx_t series_idx, float dist)
    {
        knn[k - 1].distance   = dist;
        knn[k - 1].series_idx = series_idx;

        for (idx_t i = 1; i < k; i++) {
            idx_t j = i;
            while (j > 0 && knn[j - 1].distance > knn[j].distance) {
                std::swap(knn[j], knn[j - 1]);
                --j;
            }
        }
    }

    float calculate_node_min_distance(HerculesNode *node, const float *query)
    {
        float sum = 0;
        int num_points = (int)node->node_points.size();

        std::vector<float> mean_ps(num_points);
        std::vector<float> stdev_ps(num_points);

        for (int i = 0; i < num_points; i++)
            calc_mean_stdev(const_cast<float *>(query),
                            get_segment_start(node->node_points, i),
                            get_segment_end(node->node_points, i),
                            &mean_ps[i], &stdev_ps[i]);

        for (int i = 0; i < num_points; i++) {
            float temp_dist = 0, temp = 0;

            if ((stdev_ps[i] - node->node_segment_sketches[i].indicators[2]) *
                (stdev_ps[i] - node->node_segment_sketches[i].indicators[3]) > 0) {
                temp = fminf(fabsf(stdev_ps[i] - node->node_segment_sketches[i].indicators[2]),
                             fabsf(stdev_ps[i] - node->node_segment_sketches[i].indicators[3]));
                temp_dist += temp * temp;
            }

            if ((mean_ps[i] - node->node_segment_sketches[i].indicators[0]) *
                (mean_ps[i] - node->node_segment_sketches[i].indicators[1]) > 0) {
                temp = fminf(fabsf(mean_ps[i] - node->node_segment_sketches[i].indicators[0]),
                             fabsf(mean_ps[i] - node->node_segment_sketches[i].indicators[1]));
                temp_dist += temp * temp;
            }

            sum += temp_dist * (float)get_segment_length(node->node_points, i);
        }

        return sum;
    }

    void approximate_knn_search(const float *query, int dim,
                                 HerculesPQ &pq, FILE *raw_file,
                                 std::vector<float> &ts_buf,
                                 std::vector<HerculesKnnResult> &knn,
                                 idx_t k, int max_leaves)
    {
        int loaded_leaves = 0;
        while (!pq.empty()) {
            std::pair<float, HerculesNode *> top = pq.top();
            pq.pop();
            float lb       = top.first;
            HerculesNode *node = top.second;

            float kth_bsf = knn[k - 1].distance;
            if (lb > kth_bsf) break;

            if (node->is_leaf) {
                ts_buf.resize((size_t)node->node_size * dim);
                fseek(raw_file, (long)node->file_pos * dim * (long)sizeof(float), SEEK_SET);
                fread(ts_buf.data(), sizeof(float), (size_t)node->node_size * dim, raw_file);

                for (unsigned int idx = 0; idx < node->node_size; idx++) {
                    kth_bsf = knn[k - 1].distance;
                    float dist = l2sq(query, ts_buf.data() + (size_t)idx * dim, dim, kth_bsf);
                    if (dist < kth_bsf)
                        knn_bounded_insert(knn, k, (idx_t)node->series_indices[idx], dist);
                }

                ++loaded_leaves;
                if (loaded_leaves >= max_leaves) break;
            } else {
                kth_bsf = knn[k - 1].distance;

                float child_lb = calculate_node_min_distance(node->left_child, query);
                if (child_lb < kth_bsf)
                    pq.push(std::make_pair(child_lb, node->left_child));

                child_lb = calculate_node_min_distance(node->right_child, query);
                if (child_lb < kth_bsf)
                    pq.push(std::make_pair(child_lb, node->right_child));
            }
        }
    }

} // namespace daisy
