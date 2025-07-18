#include "DistanceComputer.hpp"
#include "../isax/SAX.hpp"
#include "../isax/iSAXSearch.hpp"

namespace diNoLib
{
    void DistanceComputer::init_distance_map()
    {
        distance_map[DistanceType::L2_SQUARED] = &DistanceComputer::l2_dist;
        distance_map[DistanceType::DTW] = &DistanceComputer::dtw_dist_method;
    }

    float DistanceComputer::l2_dist(float *t, float *s, int dim, float bound)
    {
        if (dim % 8 == 0)
        {
            return l2_dist_SIMD(t, s, dim, bound);
        }
        else
        {
            return l2_dist_naive(t, s, dim, bound);
        }
    }

    float DistanceComputer::l2_dist_SIMD(float *t, float *s, int dim, float bound)
    {
        float distance = 0;
        int i = 0;
        float distancef[8];

        __m256 v_t, v_s, v_d, distancev;
        while (dim > 0 && distance < bound)
        {
            v_t = _mm256_loadu_ps(&t[i]);
            v_s = _mm256_loadu_ps(&s[i]);

            v_d = _mm256_sub_ps(v_t, v_s);

            v_d = _mm256_mul_ps(v_d, v_d);
            dim -= 8;

            i = i + 8;
            distancev = _mm256_hadd_ps(v_d, v_d);
            distancev = _mm256_hadd_ps(distancev, distancev);
            _mm256_storeu_ps(distancef, distancev);
            distance += distancef[0] + distancef[4];
        }

        // if (apply_sqrt)
        //{
        // distance = sqrtf(distance);
        //}

        return distance;
    }

    float DistanceComputer::l2_dist_naive(float *t, float *s, int dim, float bound)
    {
        float distance = 0;
        for (int i = 0; i < dim; i++)
        {
            distance += (t[i] - s[i]) * (t[i] - s[i]);
            if (distance > bound)
            {
                return distance;
            }
        }

        return distance;
    }

    float DistanceComputer::compute_dist(float *t, float *s, int dim, float bound)
    {
        return (this->*distance_map[distance_type])(t, s, dim, bound);
    }

    float DistanceComputer::compute_minidist_SIMD(const ts_type *q_paa,
                                                  const sax_type *db_sax,
                                                  const int *max_sax_cardinalities,
                                                  int sax_bit_cardinality,
                                                  int sax_alphabet_cardinality,
                                                  int paa_segments,
                                                  float minval,
                                                  float maxval,
                                                  bool mindist_sqrt)
    {
        return minidist_paa_to_isax_rawa_SIMD(
            const_cast<float *>(q_paa),
            const_cast<unsigned char *>(db_sax),
            const_cast<unsigned char *>(reinterpret_cast<const unsigned char *>(max_sax_cardinalities)),
            sax_bit_cardinality,
            sax_alphabet_cardinality,
            paa_segments,
            minval,
            maxval,
            mindist_sqrt);
    }

    void DistanceComputer::compute_paa_from_ts(const float *ts,
                                               ts_type *paa,
                                               int paa_segments,
                                               int ts_values_per_segment)
    {
        paa_from_ts(
            // const_cast<float*>(ts),
            ts,
            paa,
            paa_segments,
            ts_values_per_segment);
    }

    bool DistanceComputer::compute_sax_from_ts(const float *ts,
                                               sax_type *sax,
                                               int ts_values_per_paa_segment,
                                               int paa_segments,
                                               int sax_alphabet_cardinality,
                                               int sax_bit_cardinality)
    {
        return sax_from_ts(
                   const_cast<float *>(ts),
                   sax,
                   ts_values_per_paa_segment,
                   paa_segments,
                   sax_alphabet_cardinality,
                   sax_bit_cardinality) == SUCCESS;
    }

    float DistanceComputer::compute_dist_SIMD(float *t, float *s, int dim, float bound)
    {
        return l2_dist_SIMD(t, s, dim, bound);
    }

    ////// Wrapper Methods's definition for SAX.hpp functions //////
    float DistanceComputer::wrap_minidist_paa_to_isax(float *paa, sax_type *sax,
                                                      sax_type *sax_cardinalities,
                                                      sax_type max_bit_cardinality,
                                                      int max_cardinality,
                                                      int number_of_segments,
                                                      int min_val,
                                                      int max_val,
                                                      float ratio_sqrt)
    {
        return minidist_paa_to_isax(paa, sax, sax_cardinalities,
                                    max_bit_cardinality, max_cardinality,
                                    number_of_segments, min_val, max_val,
                                    ratio_sqrt);
    }

    float DistanceComputer::wrap_minidist_paa_to_isax_raw_SIMD(float *paa, sax_type *sax,
                                                               sax_type *sax_cardinalities,
                                                               sax_type max_bit_cardinality,
                                                               int max_cardinality,
                                                               int number_of_segments,
                                                               int min_val,
                                                               int max_val,
                                                               float ratio_sqrt)
    {
        return minidist_paa_to_isax_raw_SIMD(paa, sax, sax_cardinalities,
                                             max_bit_cardinality, max_cardinality,
                                             number_of_segments, min_val, max_val,
                                             ratio_sqrt);
    }

    float DistanceComputer::wrap_ts_euclidean_distance(ts_type *t, ts_type *s, int size, float bound)
    {
        return ts_euclidean_distance(t, s, size, bound);
    }

    float DistanceComputer::wrap_ts_euclidean_distance_SIMD(ts_type *t, ts_type *s, int size, float bound)
    {
        return ts_euclidean_distance_SIMD(t, s, size, bound);
    }

    float DistanceComputer::wrap_minidist_paa_to_isax_rawa_SIMD(float *paa, sax_type *sax,
                                                                sax_type *sax_cardinalities,
                                                                sax_type max_bit_cardinality,
                                                                int max_cardinality,
                                                                int number_of_segments,
                                                                int min_val,
                                                                int max_val,
                                                                float ratio_sqrt)
    {
        return minidist_paa_to_isax_rawa_SIMD(paa, sax, sax_cardinalities,
                                              max_bit_cardinality, max_cardinality,
                                              number_of_segments, min_val, max_val,
                                              ratio_sqrt);
    }

    float DistanceComputer::wrap_minidist_paa_to_isax_raw_DTW_SIMD(float *paaU, float *paaL, sax_type *sax,
                                                                   sax_type *sax_cardinalities,
                                                                   sax_type max_bit_cardinality,
                                                                   int max_cardinality,
                                                                   int number_of_segments,
                                                                   int min_val,
                                                                   int max_val,
                                                                   float ratio_sqrt)
    {
        return minidist_paa_to_isax_raw_DTW_SIMD(paaU, paaL, sax, sax_cardinalities,
                                                 max_bit_cardinality, max_cardinality,
                                                 number_of_segments, min_val, max_val,
                                                 ratio_sqrt);
    }

    float DistanceComputer::wrap_lb_keogh_data_bound(float *qo, float *tu, float *tl, float *cb, int len, float bsf)
    {
        return lb_keogh_data_bound(qo, tu, tl, cb, len, bsf);
    }

    float DistanceComputer::wrap_dtwsimdPruned(float *A, float *B, float *cb, int m, int r, float bsf, float *tSum, float *pCost, float *rDist)
    {
        return dtwsimdPruned(A, B, cb, m, r, bsf, tSum, pCost, rDist);
    }

    float DistanceComputer::wrap_minidist_paa_to_isax_DTW(float *paaU, float *paaL, sax_type *sax,
                                                          sax_type *sax_cardinalities,
                                                          sax_type max_bit_cardinality,
                                                          int max_cardinality,
                                                          int number_of_segments,
                                                          int min_val,
                                                          int max_val,
                                                          float ratio_sqrt)
    {
        return minidist_paa_to_isax_DTW(paaU, paaL, sax, sax_cardinalities,
                                        max_bit_cardinality, max_cardinality,
                                        number_of_segments, min_val, max_val,
                                        ratio_sqrt);
    }

    // DTW distance methods implementation
    float DistanceComputer::dtw_dist_method(float *t, float *s, int dim, float bound)
    {
        // For DTW, we need to decide between SIMD and naive implementation
        // Since DTW is more complex than L2, we'll use the SIMD version when available
        return dtw_dist_SIMD(t, s, dim, bound);
    }

    float DistanceComputer::dtw_dist_SIMD(float *t, float *s, int dim, float bound)
    {
        // Use the optimized DTW SIMD implementation
        // We need to allocate memory for the cost buffer and other DTW-specific variables
        float *cb = (float *)calloc(dim, sizeof(float));
        float *tSum = (float *)calloc(dim, sizeof(float));
        float *pCost = (float *)calloc(dim, sizeof(float));
        float *rDist = (float *)calloc(dim, sizeof(float));

        // Calculate warp window (typically 10% of sequence length, but can be adjusted)
        int warp_window = (int)(0.1 * dim);
        if (warp_window < 1)
            warp_window = 1;

        float distance = dtwsimdPruned(t, s, cb, dim, warp_window, bound, tSum, pCost, rDist);

        // Clean up allocated memory
        free(cb);
        free(tSum);
        free(pCost);
        free(rDist);

        return distance;
    }

    float DistanceComputer::dtw_dist_naive(float *t, float *s, int dim, float bound)
    {
        // Use the basic DTW implementation
        float *cb = (float *)calloc(dim, sizeof(float));

        // Calculate warp window (typically 10% of sequence length)
        int warp_window = (int)(0.1 * dim);
        if (warp_window < 1)
            warp_window = 1;

        float distance = dtw(t, s, cb, dim, warp_window, bound);

        free(cb);
        return distance;
    }
} // namespace diNoLib