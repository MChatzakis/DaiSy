#include "DistanceComputer.hpp"
#include "../lib/isax/SAX.hpp"

namespace diNoLib
{
    void DistanceComputer::init_distance_map()
    {
        distance_map[DistanceType::L2_SQUARED] = &DistanceComputer::l2_dist;
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
            const_cast<float*>(q_paa),
            const_cast<unsigned char*>(db_sax),
            const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(max_sax_cardinalities)),
            sax_bit_cardinality,
            sax_alphabet_cardinality,
            paa_segments,
            minval,
            maxval,
            mindist_sqrt
        );
    }

    void DistanceComputer::compute_paa_from_ts(const float *ts,
                                            ts_type *paa,
                                            int paa_segments,
                                            int ts_values_per_segment)
    {
        paa_from_ts(
            const_cast<float*>(ts),
            paa,
            paa_segments,
            ts_values_per_segment
        );
    }

    bool DistanceComputer::compute_sax_from_ts(const float *ts,
                                            sax_type *sax,
                                            int ts_values_per_paa_segment,
                                            int paa_segments,
                                            int sax_alphabet_cardinality,
                                            int sax_bit_cardinality)
    {
        return sax_from_ts(
            const_cast<float*>(ts),
            sax,
            ts_values_per_paa_segment,
            paa_segments,
            sax_alphabet_cardinality,
            sax_bit_cardinality
        ) == SUCCESS;
    }    

    float DistanceComputer::compute_dist_SIMD(float *t, float *s, int dim, float bound)
    {
        return l2_dist_SIMD(t, s, dim, bound);
    }
}