#include "DistanceComputer.hpp"

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

}