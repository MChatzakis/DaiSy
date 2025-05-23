#include "SAX.hpp"

#include <math.h>
#include <float.h>

#include "immintrin.h"

namespace diNoLib
{

    int compare(const void *a, const void *b)
    {
        float *c = (float *)b - 1;
        if (*(float *)a > *(float *)c && *(float *)a <= *(float *)b)
        {
            // printf("Found %lf between %lf and %lf\n",*(float*)a,*(float*)c,*(float*)b);
            return 0;
        }
        else if (*(float *)a <= *(float *)c)
        {
            return -1;
        }
        else
        {
            return 1;
        }
    }

    enum response sax_from_ts(ts_type *ts_in, sax_type *sax_out, int ts_values_per_segment, int segments, int cardinality, int bit_cardinality)
    {
        // Create PAA representation
        float *paa = (float *)malloc(sizeof(float) * segments);
        if (paa == NULL)
        {
            fprintf(stderr, "error: could not allocate memory for PAA representation.\n");
            return FAILURE;
        }

        int s, i;
        for (s = 0; s < segments; s++)
        {
            paa[s] = 0;
            for (i = 0; i < ts_values_per_segment; i++)
            {
                paa[s] += ts_in[(s * ts_values_per_segment) + i];
            }
            paa[s] /= ts_values_per_segment;
            // #ifdef DEBUG
            // printf("%d: %lf\n", s, paa[s]);
            // #endif
        }

        // Convert PAA to SAX
        // Note: Each cardinality has cardinality - 1 break points if c is cardinality
        //       the breakpoints can be found in the following array positions:
        //       FROM (c - 1) * (c - 2) / 2
        //       TO   (c - 1) * (c - 2) / 2 + c - 1
        int offset = ((cardinality - 1) * (cardinality - 2)) / 2;
        // printf("FROM %lf TO %lf\n", sax_breakpoints[offset], sax_breakpoints[offset + cardinality - 2]);

        int si;
        for (si = 0; si < segments; si++)
        {
            sax_out[si] = 0;

            // First object = sax_breakpoints[offset]
            // Last object = sax_breakpoints[offset + cardinality - 2]
            // Size of sub-array = cardinality - 1

            float *res = (float *)bsearch(&paa[si], &sax_breakpoints[offset], cardinality - 1,
                                          sizeof(ts_type), compare);
            if (res != NULL)
                sax_out[si] = (int)(res - &sax_breakpoints[offset]);
            else if (paa[si] > 0)
                sax_out[si] = cardinality - 1;
        }

        // sax_print(sax_out, segments, cardinality);
        free(paa);
        return SUCCESS;
    }

    enum response paa_from_ts(ts_type *ts_in, ts_type *paa_out, int segments, int ts_values_per_segment)
    {
        int s, i;
        for (s = 0; s < segments; s++)
        {
            paa_out[s] = 0;
            for (i = 0; i < ts_values_per_segment; i++)
            {
                paa_out[s] += ts_in[(s * ts_values_per_segment) + i];
            }
            paa_out[s] /= ts_values_per_segment;
        }
        return SUCCESS;
    }

    enum response sax_from_paa(ts_type *paa, sax_type *sax, int segments, int cardinality, int bit_cardinality)
    {
        int offset = ((cardinality - 1) * (cardinality - 2)) / 2;
        // printf("FROM %lf TO %lf\n", sax_breakpoints[offset], sax_breakpoints[offset + cardinality - 2]);

        int si;
        for (si = 0; si < segments; si++)
        {
            sax[si] = 0;

            // First object = sax_breakpoints[offset]
            // Last object = sax_breakpoints[offset + cardinality - 2]
            // Size of sub-array = cardinality - 1

            float *res = (float *)bsearch(&paa[si], &sax_breakpoints[offset], cardinality - 1,
                                          sizeof(ts_type), compare);
            if (res != NULL)
                sax[si] = (int)(res - &sax_breakpoints[offset]);
            else if (paa[si] > 0)
                sax[si] = cardinality - 1;
        }

        return SUCCESS;
    }

    float minidist_paa_to_isax(float *paa, sax_type *sax,
                               sax_type *sax_cardinalities,
                               sax_type max_bit_cardinality,
                               int max_cardinality,
                               int number_of_segments,
                               int min_val,
                               int max_val,
                               float ratio_sqrt)
    {

        float distance = 0;
        // TODO: Store offset in index settings. and pass index settings as parameter.

        int offset = ((max_cardinality - 1) * (max_cardinality - 2)) / 2;

        // For each sax record find the break point
        int i;
        for (i = 0; i < number_of_segments; i++)
        {

            sax_type c_c = sax_cardinalities[i];

            sax_type c_m = max_bit_cardinality;
            sax_type v = sax[i];
            // sax_print(&v, 1, c_m);

            sax_type region_lower = (v << (c_m - c_c));
            sax_type region_upper = (~((int)MAXFLOAT << (c_m - c_c)) | region_lower);
            // printf("[%d, %d] %d -- %d\n", sax[i], c_c, region_lower, region_upper);
            float breakpoint_lower = 0; // <-- TODO: calculate breakpoints.
            float breakpoint_upper = 0; // <-- - || -

            if (region_lower == 0)
            {
                breakpoint_lower = min_val;
            }
            else
            {
                breakpoint_lower = sax_breakpoints[offset + region_lower - 1];
            }
            if (region_upper == max_cardinality - 1)
            {
                breakpoint_upper = max_val;
            }
            else
            {
                breakpoint_upper = sax_breakpoints[offset + region_upper];
            }
            // printf("\n%d.%d is from %d to %d, %lf - %lf\n", v, c_c, region_lower, region_upper,
            //        breakpoint_lower, breakpoint_upper);

            // printf("FROM: \n");
            // sax_print(&region_lower, 1, c_m);
            // printf("TO: \n");
            // sax_print(&region_upper, 1, c_m);

            // printf ("\n---------\n");

            if (breakpoint_lower > paa[i])
            {
                distance += pow(breakpoint_lower - paa[i], 2);
            }
            else if (breakpoint_upper < paa[i])
            {
                distance += pow(breakpoint_upper - paa[i], 2);
            }
            //        else {
            //            printf("%lf is between: %lf and %lf\n", paa[i], breakpoint_lower, breakpoint_upper);
            //        }
        }

        // distance = ratio_sqrt * sqrtf(distance);
        distance = ratio_sqrt * distance;
        return distance;
    }

    float minidist_paa_to_isax_raw_SIMD(float *paa, sax_type *sax,
                                        sax_type *sax_cardinalities,
                                        sax_type max_bit_cardinality,
                                        int max_cardinality,
                                        int number_of_segments,
                                        int min_val,
                                        int max_val,
                                        float ratio_sqrt)
    {

        int region_upper[16], region_lower[16];
        float distancef[16];
        int offset = ((max_cardinality - 1) * (max_cardinality - 2)) / 2;

        __m256i vectorsignbit = _mm256_set1_epi32(0xffffffff);

        //__m256i c_cv_0 = _mm256_set_epi32 ( sax_cardinalities[7] , sax_cardinalities[6] ,sax_cardinalities[5] ,sax_cardinalities[4] , sax_cardinalities[3] ,sax_cardinalities[2] ,sax_cardinalities[1],sax_cardinalities[0]);
        //__m256i c_cv_1 = _mm256_set_epi32 ( sax_cardinalities[15], sax_cardinalities[14],sax_cardinalities[13],sax_cardinalities[12], sax_cardinalities[11],sax_cardinalities[10],sax_cardinalities[9],sax_cardinalities[8]);
        __m128i sax_cardinalitiesv8 = _mm_lddqu_si128((const __m128i *)sax_cardinalities);
        __m256i sax_cardinalitiesv16 = _mm256_cvtepu8_epi16(sax_cardinalitiesv8);
        __m128i sax_cardinalitiesv16_0 = _mm256_extractf128_si256(sax_cardinalitiesv16, 0);
        __m128i sax_cardinalitiesv16_1 = _mm256_extractf128_si256(sax_cardinalitiesv16, 1);
        __m256i c_cv_0 = _mm256_cvtepu16_epi32(sax_cardinalitiesv16_0);
        __m256i c_cv_1 = _mm256_cvtepu16_epi32(sax_cardinalitiesv16_1);

        //__m256i v_0    = _mm256_set_epi32 (sax[7],sax[6],sax[5],sax[4],sax[3],sax[2],sax[1],sax[0]);
        //__m256i v_1    = _mm256_set_epi32 (sax[15],sax[14],sax[13],sax[12],sax[11],sax[10],sax[9],sax[8]);
        __m128i saxv8 = _mm_lddqu_si128((const __m128i *)sax);
        __m256i saxv16 = _mm256_cvtepu8_epi16(saxv8);
        __m128i saxv16_0 = _mm256_extractf128_si256(saxv16, 0);
        __m128i saxv16_1 = _mm256_extractf128_si256(saxv16, 1);
        __m256i v_0 = _mm256_cvtepu16_epi32(saxv16_0);
        __m256i v_1 = _mm256_cvtepu16_epi32(saxv16_1);

        __m256i c_m = _mm256_set1_epi32(max_bit_cardinality);
        __m256i cm_ccv_0 = _mm256_sub_epi32(c_m, c_cv_0);
        __m256i cm_ccv_1 = _mm256_sub_epi32(c_m, c_cv_1);

        //__m256i _mm256_set_epi32 (int e7, int e6, int e5, int e4, int e3, int e2, int e1, int e0)
        //  __m256i _mm256_set1_epi32 (int a)
        __m256i region_lowerv_0 = _mm256_srlv_epi32(v_0, cm_ccv_0);
        __m256i region_lowerv_1 = _mm256_srlv_epi32(v_1, cm_ccv_1);
        region_lowerv_0 = _mm256_sllv_epi32(region_lowerv_0, cm_ccv_0);
        region_lowerv_1 = _mm256_sllv_epi32(region_lowerv_1, cm_ccv_1);

        __m256i v1 = _mm256_andnot_si256(_mm256_setzero_si256(), vectorsignbit);

        __m256i region_upperv_0 = _mm256_sllv_epi32(v1, cm_ccv_0);
        __m256i region_upperv_1 = _mm256_sllv_epi32(v1, cm_ccv_1);
        region_upperv_0 = _mm256_andnot_si256(region_upperv_0, vectorsignbit);
        region_upperv_1 = _mm256_andnot_si256(region_upperv_1, vectorsignbit);

        region_upperv_0 = _mm256_or_si256(region_upperv_0, region_lowerv_0);
        region_upperv_1 = _mm256_or_si256(region_upperv_1, region_lowerv_1);

        _mm256_storeu_si256((__m256i_u *)&(region_lower[0]), region_lowerv_0);
        _mm256_storeu_si256((__m256i_u *)&(region_lower[8]), region_lowerv_1);
        _mm256_storeu_si256((__m256i_u *)&(region_upper[0]), region_upperv_0);
        _mm256_storeu_si256((__m256i_u *)&(region_upper[8]), region_upperv_1);

        // lower

        __m256i lower_juge_zerov_0 = _mm256_cmpeq_epi32(region_lowerv_0, _mm256_setzero_si256());
        __m256i lower_juge_zerov_1 = _mm256_cmpeq_epi32(region_lowerv_1, _mm256_setzero_si256());

        __m256i lower_juge_nzerov_0 = _mm256_andnot_si256(lower_juge_zerov_0, vectorsignbit);
        __m256i lower_juge_nzerov_1 = _mm256_andnot_si256(lower_juge_zerov_1, vectorsignbit);

        __m256 minvalv = _mm256_set1_ps(min_val);

        //__m256 lsax_breakpoints_shiftv_0 _mm256_i32gather_ps (sax_breakpoints, __m256i vindex, const int scale)
        __m256 lsax_breakpoints_shiftv_0 = _mm256_set_ps(sax_breakpoints[offset + region_lower[7] - 1],
                                                         sax_breakpoints[offset + region_lower[6] - 1],
                                                         sax_breakpoints[offset + region_lower[5] - 1],
                                                         sax_breakpoints[offset + region_lower[4] - 1],
                                                         sax_breakpoints[offset + region_lower[3] - 1],
                                                         sax_breakpoints[offset + region_lower[2] - 1],
                                                         sax_breakpoints[offset + region_lower[1] - 1],
                                                         sax_breakpoints[offset + region_lower[0] - 1]);
        __m256 lsax_breakpoints_shiftv_1 = _mm256_set_ps(sax_breakpoints[offset + region_lower[15] - 1],
                                                         sax_breakpoints[offset + region_lower[14] - 1],
                                                         sax_breakpoints[offset + region_lower[13] - 1],
                                                         sax_breakpoints[offset + region_lower[12] - 1],
                                                         sax_breakpoints[offset + region_lower[11] - 1],
                                                         sax_breakpoints[offset + region_lower[10] - 1],
                                                         sax_breakpoints[offset + region_lower[9] - 1],
                                                         sax_breakpoints[offset + region_lower[8] - 1]);

        __m256 breakpoint_lowerv_0 = (__m256)_mm256_or_si256(_mm256_and_si256(lower_juge_zerov_0, (__m256i)minvalv), _mm256_and_si256(lower_juge_nzerov_0, (__m256i)lsax_breakpoints_shiftv_0));
        __m256 breakpoint_lowerv_1 = (__m256)_mm256_or_si256(_mm256_and_si256(lower_juge_zerov_1, (__m256i)minvalv), _mm256_and_si256(lower_juge_nzerov_1, (__m256i)lsax_breakpoints_shiftv_1));

        // uper
        __m256 usax_breakpoints_shiftv_0 = _mm256_set_ps(sax_breakpoints[offset + region_upper[7]],
                                                         sax_breakpoints[offset + region_upper[6]],
                                                         sax_breakpoints[offset + region_upper[5]],
                                                         sax_breakpoints[offset + region_upper[4]],
                                                         sax_breakpoints[offset + region_upper[3]],
                                                         sax_breakpoints[offset + region_upper[2]],
                                                         sax_breakpoints[offset + region_upper[1]],
                                                         sax_breakpoints[offset + region_upper[0]]);
        __m256 usax_breakpoints_shiftv_1 = _mm256_set_ps(sax_breakpoints[offset + region_upper[15]],
                                                         sax_breakpoints[offset + region_upper[14]],
                                                         sax_breakpoints[offset + region_upper[13]],
                                                         sax_breakpoints[offset + region_upper[12]],
                                                         sax_breakpoints[offset + region_upper[11]],
                                                         sax_breakpoints[offset + region_upper[10]],
                                                         sax_breakpoints[offset + region_upper[9]],
                                                         sax_breakpoints[offset + region_upper[8]]);

        __m256i upper_juge_maxv_0 = _mm256_cmpeq_epi32(region_upperv_0, _mm256_set1_epi32(max_cardinality - 1));
        __m256i upper_juge_maxv_1 = _mm256_cmpeq_epi32(region_upperv_1, _mm256_set1_epi32(max_cardinality - 1));

        __m256i upper_juge_nmaxv_0 = _mm256_andnot_si256(upper_juge_maxv_0, vectorsignbit);
        __m256i upper_juge_nmaxv_1 = _mm256_andnot_si256(upper_juge_maxv_1, vectorsignbit);

        __m256 breakpoint_upperv_0 = (__m256)_mm256_or_si256(_mm256_and_si256(upper_juge_maxv_0, (__m256i)_mm256_set1_ps(max_val)), _mm256_and_si256(upper_juge_nmaxv_0, (__m256i)usax_breakpoints_shiftv_0));
        __m256 breakpoint_upperv_1 = (__m256)_mm256_or_si256(_mm256_and_si256(upper_juge_maxv_1, (__m256i)_mm256_set1_ps(max_val)), _mm256_and_si256(upper_juge_nmaxv_1, (__m256i)usax_breakpoints_shiftv_1));

        // dis
        __m256 paav_0, paav_1;

        paav_0 = _mm256_loadu_ps(paa);
        paav_1 = _mm256_loadu_ps(&(paa[8]));

        __m256 dis_juge_upv_0 = _mm256_cmp_ps(breakpoint_lowerv_0, paav_0, _CMP_GT_OS);
        __m256 dis_juge_upv_1 = _mm256_cmp_ps(breakpoint_lowerv_1, paav_1, _CMP_GT_OS);

        __m256 dis_juge_lov_0 = (__m256)_mm256_and_si256((__m256i)_mm256_cmp_ps(breakpoint_lowerv_0, paav_0, _CMP_NGT_US), (__m256i)_mm256_cmp_ps(breakpoint_upperv_0, paav_0, _CMP_LT_OS));
        __m256 dis_juge_lov_1 = (__m256)_mm256_and_si256((__m256i)_mm256_cmp_ps(breakpoint_lowerv_1, paav_1, _CMP_NGT_US), (__m256i)_mm256_cmp_ps(breakpoint_upperv_1, paav_1, _CMP_LT_OS));

        __m256 dis_juge_elv_0 = (__m256)_mm256_andnot_si256(_mm256_or_si256((__m256i)dis_juge_upv_0, (__m256i)dis_juge_lov_0), vectorsignbit);
        __m256 dis_juge_elv_1 = (__m256)_mm256_andnot_si256(_mm256_or_si256((__m256i)dis_juge_upv_1, (__m256i)dis_juge_lov_1), vectorsignbit);

        __m256 dis_lowv_0 = _mm256_mul_ps(_mm256_sub_ps(breakpoint_lowerv_0, paav_0), _mm256_sub_ps(breakpoint_lowerv_0, paav_0));
        __m256 dis_lowv_1 = _mm256_mul_ps(_mm256_sub_ps(breakpoint_lowerv_1, paav_1), _mm256_sub_ps(breakpoint_lowerv_1, paav_1));
        __m256 dis_uppv_0 = _mm256_mul_ps(_mm256_sub_ps(breakpoint_upperv_0, paav_0), _mm256_sub_ps(breakpoint_upperv_0, paav_0));
        __m256 dis_uppv_1 = _mm256_mul_ps(_mm256_sub_ps(breakpoint_upperv_1, paav_1), _mm256_sub_ps(breakpoint_upperv_1, paav_1));

        __m256 distancev_0 = (__m256)_mm256_or_si256(_mm256_or_si256(_mm256_and_si256((__m256i)dis_juge_upv_0, (__m256i)dis_lowv_0), _mm256_and_si256((__m256i)dis_juge_lov_0, (__m256i)dis_uppv_0)), _mm256_and_si256((__m256i)dis_juge_elv_0, (__m256i)_mm256_set1_ps(0.0)));
        __m256 distancev_1 = (__m256)_mm256_or_si256(_mm256_or_si256(_mm256_and_si256((__m256i)dis_juge_upv_1, (__m256i)dis_lowv_1), _mm256_and_si256((__m256i)dis_juge_lov_1, (__m256i)dis_uppv_1)), _mm256_and_si256((__m256i)dis_juge_elv_1, (__m256i)_mm256_set1_ps(0.0)));

        __m256 distancev = _mm256_add_ps(distancev_0, distancev_1);
        __m256 distancev2 = _mm256_hadd_ps(distancev, distancev);
        __m256 distancevf = _mm256_hadd_ps(distancev2, distancev2);

        _mm256_storeu_ps(distancef, distancevf);
        //_mm256_storeu_ps (&checkvalue[8] ,distancev_1);

        return (distancef[0] + distancef[4]) * ratio_sqrt;
    }

    float ts_euclidean_distance(ts_type *t, ts_type *s, int size, float bound)
    {
        float distance = 0;
        while (size > 0 && distance < bound)
        {
            size--;
            distance += (t[size] - s[size]) * (t[size] - s[size]);
        }

        //    distance = sqrtf(distance);

        return distance;
    }
    
    float ts_euclidean_distance_SIMD(ts_type *t, ts_type *s, int size, float bound)
    {
        float distance = 0;
        int i = 0;
        float distancef[8];

        __m256 v_t, v_s, v_d, distancev;
        while (size > 0 && distance < bound)
        {
            v_t = _mm256_loadu_ps(&t[i]);
            v_s = _mm256_loadu_ps(&s[i]);

            v_d = _mm256_sub_ps(v_t, v_s);

            v_d = _mm256_mul_ps(v_d, v_d);
            size -= 8;

            i = i + 8;
            distancev = _mm256_hadd_ps(v_d, v_d);
            distancev = _mm256_hadd_ps(distancev, distancev);
            _mm256_storeu_ps(distancef, distancev);
            distance += distancef[0] + distancef[4];
        }

        //    distance = sqrtf(distance);

        return distance;
    }

}