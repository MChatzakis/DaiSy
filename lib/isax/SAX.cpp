#include <math.h>
#include <float.h>

#include "SAX.hpp"
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

    float minidist_paa_to_isax_rawa_SIMD(float *paa, sax_type *sax,
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
        __m256i vloweroffset = _mm256_set1_epi32(offset - 1);
        __m256i vupperoffset = _mm256_set1_epi32(offset);

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

        __m256i region_lowerv_0_offset = _mm256_add_epi32(region_lowerv_0, vloweroffset);
        __m256i region_lowerv_1_offset = _mm256_add_epi32(region_lowerv_1, vloweroffset);
        __m256i region_upperv_0_offset = _mm256_add_epi32(region_upperv_0, vupperoffset);
        __m256i region_upperv_1_offset = _mm256_add_epi32(region_upperv_1, vupperoffset);
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

        __m256 lsax_breakpoints_shiftv_0 = _mm256_i32gather_ps(sax_breakpoints, region_lowerv_0_offset, 4);
        //__m256 lsax_breakpoints_shiftv_0= _mm256_set_ps (sax_breakpoints[region_lower[7]],
        // sax_breakpoints[region_lower[6]],
        // sax_breakpoints[region_lower[5]],
        // sax_breakpoints[region_lower[4]],
        // sax_breakpoints[region_lower[3]],
        // sax_breakpoints[region_lower[2]],
        // sax_breakpoints[region_lower[1]],
        // sax_breakpoints[region_lower[0]]);
        __m256 lsax_breakpoints_shiftv_1 = _mm256_i32gather_ps(sax_breakpoints, region_lowerv_1_offset, 4);
        //__m256 lsax_breakpoints_shiftv_1= _mm256_set_ps (sax_breakpoints[region_lower[15]],
        // sax_breakpoints[region_lower[14]],
        // sax_breakpoints[region_lower[13]],
        // sax_breakpoints[region_lower[12]],
        // sax_breakpoints[region_lower[11]],
        // sax_breakpoints[region_lower[10]],
        // sax_breakpoints[region_lower[9]],
        // sax_breakpoints[region_lower[8]]);

        __m256 breakpoint_lowerv_0 = (__m256)_mm256_or_si256(_mm256_and_si256(lower_juge_zerov_0, (__m256i)minvalv), _mm256_and_si256(lower_juge_nzerov_0, (__m256i)lsax_breakpoints_shiftv_0));
        __m256 breakpoint_lowerv_1 = (__m256)_mm256_or_si256(_mm256_and_si256(lower_juge_zerov_1, (__m256i)minvalv), _mm256_and_si256(lower_juge_nzerov_1, (__m256i)lsax_breakpoints_shiftv_1));

        // uper
        __m256 usax_breakpoints_shiftv_0 = _mm256_i32gather_ps(sax_breakpoints, region_upperv_0_offset, 4);
        //__m256 usax_breakpoints_shiftv_0= _mm256_set_ps (sax_breakpoints[region_upper[7]],
        // sax_breakpoints[region_upper[6]],
        // sax_breakpoints[region_upper[5]],
        // sax_breakpoints[region_upper[4]],
        // sax_breakpoints[region_upper[3]],
        // sax_breakpoints[region_upper[2]],
        // sax_breakpoints[region_upper[1]],
        // sax_breakpoints[region_upper[0]]);
        __m256 usax_breakpoints_shiftv_1 = _mm256_i32gather_ps(sax_breakpoints, region_upperv_1_offset, 4);
        //__m256 usax_breakpoints_shiftv_1= _mm256_set_ps (sax_breakpoints[region_upper[15]],
        // sax_breakpoints[region_upper[14]],
        // sax_breakpoints[region_upper[13]],
        // sax_breakpoints[region_upper[12]],
        // sax_breakpoints[region_upper[11]],
        // sax_breakpoints[region_upper[10]],
        // sax_breakpoints[region_upper[9]],
        // sax_breakpoints[region_upper[8]]);

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

        __m256 dis_juge_lov_0 = _mm256_cmp_ps(breakpoint_upperv_0, paav_0, _CMP_LT_OS);
        __m256 dis_juge_lov_1 = _mm256_cmp_ps(breakpoint_upperv_1, paav_1, _CMP_LT_OS);

        // __m256 dis_juge_lov_0=(__m256)_mm256_and_si256 ((__m256i)_mm256_cmp_ps (breakpoint_lowerv_0, paav_0, _CMP_NGT_US),(__m256i)_mm256_cmp_ps (breakpoint_upperv_0, paav_0, _CMP_LT_OS))  ;
        // __m256 dis_juge_lov_1=(__m256)_mm256_and_si256 ((__m256i)_mm256_cmp_ps (breakpoint_lowerv_1, paav_1, _CMP_NGT_US),(__m256i)_mm256_cmp_ps (breakpoint_upperv_1, paav_1, _CMP_LT_OS));

        __m256 dis_juge_elv_0 = (__m256)_mm256_andnot_si256(_mm256_or_si256((__m256i)dis_juge_upv_0, (__m256i)dis_juge_lov_0), vectorsignbit);
        __m256 dis_juge_elv_1 = (__m256)_mm256_andnot_si256(_mm256_or_si256((__m256i)dis_juge_upv_1, (__m256i)dis_juge_lov_1), vectorsignbit);

        __m256 dis_lowv_0 = _mm256_sub_ps(breakpoint_lowerv_0, paav_0);
        __m256 dis_lowv_1 = _mm256_sub_ps(breakpoint_lowerv_1, paav_1);
        __m256 dis_uppv_0 = _mm256_sub_ps(breakpoint_upperv_0, paav_0);
        __m256 dis_uppv_1 = _mm256_sub_ps(breakpoint_upperv_1, paav_1);

        __m256 distancev_0 = (__m256)_mm256_or_si256(_mm256_or_si256(_mm256_and_si256((__m256i)dis_juge_upv_0, (__m256i)dis_lowv_0), _mm256_and_si256((__m256i)dis_juge_lov_0, (__m256i)dis_uppv_0)), _mm256_and_si256((__m256i)dis_juge_elv_0, (__m256i)_mm256_set1_ps(0.0)));
        __m256 distancev_1 = (__m256)_mm256_or_si256(_mm256_or_si256(_mm256_and_si256((__m256i)dis_juge_upv_1, (__m256i)dis_lowv_1), _mm256_and_si256((__m256i)dis_juge_lov_1, (__m256i)dis_uppv_1)), _mm256_and_si256((__m256i)dis_juge_elv_1, (__m256i)_mm256_set1_ps(0.0)));

        __m256 distancesum_0 = _mm256_dp_ps(distancev_0, distancev_0, 0xff);
        __m256 distancesum_1 = _mm256_dp_ps(distancev_1, distancev_1, 0xff);
        __m256 distancevf = _mm256_add_ps(distancesum_0, distancesum_1);
        //__m256 distancev2 = _mm256_hadd_ps (distancev, distancev);
        //__m256 distancevf = _mm256_hadd_ps (distancev2, distancev2);
        //__m256 _mm256_dp_ps (__m256 a, __m256 b, const int imm8);

        _mm256_storeu_ps(distancef, distancevf);
        //_mm256_storeu_ps (&checkvalue[8] ,distancev_1);

        return (distancef[0] + distancef[4]) * ratio_sqrt;
    }

    float minidist_paa_to_isax_raw_DTW_SIMD(float *paaU, float *paaL, sax_type *sax,
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
        __m256i vloweroffset = _mm256_set1_epi32(offset - 1);
        __m256i vupperoffset = _mm256_set1_epi32(offset);

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

        __m256i region_lowerv_0_offset = _mm256_add_epi32(region_lowerv_0, vloweroffset);
        __m256i region_lowerv_1_offset = _mm256_add_epi32(region_lowerv_1, vloweroffset);
        __m256i region_upperv_0_offset = _mm256_add_epi32(region_upperv_0, vupperoffset);
        __m256i region_upperv_1_offset = _mm256_add_epi32(region_upperv_1, vupperoffset);

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

        __m256 lsax_breakpoints_shiftv_0 = _mm256_i32gather_ps(sax_breakpoints, region_lowerv_0_offset, 4);
        //__m256 lsax_breakpoints_shiftv_0= _mm256_set_ps (sax_breakpoints[region_lower[7]],
        // sax_breakpoints[region_lower[6]],
        // sax_breakpoints[region_lower[5]],
        // sax_breakpoints[region_lower[4]],
        // sax_breakpoints[region_lower[3]],
        // sax_breakpoints[region_lower[2]],
        // sax_breakpoints[region_lower[1]],
        // sax_breakpoints[region_lower[0]]);
        __m256 lsax_breakpoints_shiftv_1 = _mm256_i32gather_ps(sax_breakpoints, region_lowerv_1_offset, 4);
        //__m256 lsax_breakpoints_shiftv_1= _mm256_set_ps (sax_breakpoints[region_lower[15]],
        // sax_breakpoints[region_lower[14]],
        // sax_breakpoints[region_lower[13]],
        // sax_breakpoints[region_lower[12]],
        // sax_breakpoints[region_lower[11]],
        // sax_breakpoints[region_lower[10]],
        // sax_breakpoints[region_lower[9]],
        // sax_breakpoints[region_lower[8]]);

        __m256 breakpoint_lowerv_0 = (__m256)_mm256_or_si256(_mm256_and_si256(lower_juge_zerov_0, (__m256i)minvalv), _mm256_and_si256(lower_juge_nzerov_0, (__m256i)lsax_breakpoints_shiftv_0));
        __m256 breakpoint_lowerv_1 = (__m256)_mm256_or_si256(_mm256_and_si256(lower_juge_zerov_1, (__m256i)minvalv), _mm256_and_si256(lower_juge_nzerov_1, (__m256i)lsax_breakpoints_shiftv_1));

        // uper
        __m256 usax_breakpoints_shiftv_0 = _mm256_i32gather_ps(sax_breakpoints, region_upperv_0_offset, 4);
        //__m256 usax_breakpoints_shiftv_0= _mm256_set_ps (sax_breakpoints[region_upper[7]],
        // sax_breakpoints[region_upper[6]],
        // sax_breakpoints[region_upper[5]],
        // sax_breakpoints[region_upper[4]],
        // sax_breakpoints[region_upper[3]],
        // sax_breakpoints[region_upper[2]],
        // sax_breakpoints[region_upper[1]],
        // sax_breakpoints[region_upper[0]]);
        __m256 usax_breakpoints_shiftv_1 = _mm256_i32gather_ps(sax_breakpoints, region_upperv_1_offset, 4);
        //__m256 usax_breakpoints_shiftv_1= _mm256_set_ps (sax_breakpoints[region_upper[15]],
        // sax_breakpoints[region_upper[14]],
        // sax_breakpoints[region_upper[13]],
        // sax_breakpoints[region_upper[12]],
        // sax_breakpoints[region_upper[11]],
        // sax_breakpoints[region_upper[10]],
        // sax_breakpoints[region_upper[9]],
        // sax_breakpoints[region_upper[8]]);

        __m256i upper_juge_maxv_0 = _mm256_cmpeq_epi32(region_upperv_0, _mm256_set1_epi32(max_cardinality - 1));
        __m256i upper_juge_maxv_1 = _mm256_cmpeq_epi32(region_upperv_1, _mm256_set1_epi32(max_cardinality - 1));

        __m256i upper_juge_nmaxv_0 = _mm256_andnot_si256(upper_juge_maxv_0, vectorsignbit);
        __m256i upper_juge_nmaxv_1 = _mm256_andnot_si256(upper_juge_maxv_1, vectorsignbit);

        __m256 breakpoint_upperv_0 = (__m256)_mm256_or_si256(_mm256_and_si256(upper_juge_maxv_0, (__m256i)_mm256_set1_ps(max_val)), _mm256_and_si256(upper_juge_nmaxv_0, (__m256i)usax_breakpoints_shiftv_0));
        __m256 breakpoint_upperv_1 = (__m256)_mm256_or_si256(_mm256_and_si256(upper_juge_maxv_1, (__m256i)_mm256_set1_ps(max_val)), _mm256_and_si256(upper_juge_nmaxv_1, (__m256i)usax_breakpoints_shiftv_1));

        // dis
        __m256 paaUv_0, paaUv_1, paaLv_0, paaLv_1;

        /// paav_0 =_mm256_loadu_ps (paa);
        // paav_1 =_mm256_loadu_ps (&(paa[8]));
        paaUv_0 = _mm256_loadu_ps(paaU);
        paaUv_1 = _mm256_loadu_ps(&(paaU[8]));
        paaLv_0 = _mm256_loadu_ps(paaL);
        paaLv_1 = _mm256_loadu_ps(&(paaL[8]));

        __m256 dis_juge_upv_0 = _mm256_cmp_ps(breakpoint_lowerv_0, paaUv_0, _CMP_GT_OS);
        __m256 dis_juge_upv_1 = _mm256_cmp_ps(breakpoint_lowerv_1, paaUv_1, _CMP_GT_OS);

        __m256 dis_juge_lov_0 = _mm256_cmp_ps(breakpoint_upperv_0, paaLv_0, _CMP_LT_OS);
        __m256 dis_juge_lov_1 = _mm256_cmp_ps(breakpoint_upperv_1, paaLv_1, _CMP_LT_OS);

        //__m256 dis_juge_lov_0=(__m256)_mm256_and_si256 ((__m256i)_mm256_cmp_ps (breakpoint_lowerv_0, paaUv_0, _CMP_NGT_US),(__m256i)_mm256_cmp_ps (breakpoint_upperv_0, paav_0, _CMP_LT_OS))  ;
        //__m256 dis_juge_lov_1=(__m256)_mm256_and_si256 ((__m256i)_mm256_cmp_ps (breakpoint_lowerv_1, paaUv_1, _CMP_NGT_US),(__m256i)_mm256_cmp_ps (breakpoint_upperv_1, paav_1, _CMP_LT_OS));

        __m256 dis_juge_elv_0 = (__m256)_mm256_andnot_si256(_mm256_or_si256((__m256i)dis_juge_upv_0, (__m256i)dis_juge_lov_0), vectorsignbit);
        __m256 dis_juge_elv_1 = (__m256)_mm256_andnot_si256(_mm256_or_si256((__m256i)dis_juge_upv_1, (__m256i)dis_juge_lov_1), vectorsignbit);

        __m256 dis_lowv_0 = _mm256_sub_ps(breakpoint_lowerv_0, paaUv_0);
        __m256 dis_lowv_1 = _mm256_sub_ps(breakpoint_lowerv_1, paaUv_1);
        __m256 dis_uppv_0 = _mm256_sub_ps(breakpoint_upperv_0, paaLv_0);
        __m256 dis_uppv_1 = _mm256_sub_ps(breakpoint_upperv_1, paaLv_1);

        __m256 distancev_0 = (__m256)_mm256_or_si256(_mm256_or_si256(_mm256_and_si256((__m256i)dis_juge_upv_0, (__m256i)dis_lowv_0), _mm256_and_si256((__m256i)dis_juge_lov_0, (__m256i)dis_uppv_0)), _mm256_and_si256((__m256i)dis_juge_elv_0, (__m256i)_mm256_set1_ps(0.0)));
        __m256 distancev_1 = (__m256)_mm256_or_si256(_mm256_or_si256(_mm256_and_si256((__m256i)dis_juge_upv_1, (__m256i)dis_lowv_1), _mm256_and_si256((__m256i)dis_juge_lov_1, (__m256i)dis_uppv_1)), _mm256_and_si256((__m256i)dis_juge_elv_1, (__m256i)_mm256_set1_ps(0.0)));

        __m256 distancesum_0 = _mm256_dp_ps(distancev_0, distancev_0, 0xff);
        __m256 distancesum_1 = _mm256_dp_ps(distancev_1, distancev_1, 0xff);
        __m256 distancevf = _mm256_add_ps(distancesum_0, distancesum_1);
        //__m256 distancev2 = _mm256_hadd_ps (distancev, distancev);
        //__m256 distancevf = _mm256_hadd_ps (distancev2, distancev2);
        //__m256 _mm256_dp_ps (__m256 a, __m256 b, const int imm8);

        _mm256_storeu_ps(distancef, distancevf);
        //_mm256_storeu_ps (&checkvalue[8] ,distancev_1);

        return (distancef[0] + distancef[4]) * ratio_sqrt;
    }

    float lb_keogh_data_bound(float *qo, float *tu, float *tl, float *cb, int len, float bsf)
    {
        float lb = 0;
        float uu = 0, ll = 0, d = 0;
        int i = 0;

        int len1 = (len / 8) * 8;
        __m256 tu256, tl256, cb256, Q, calc1, calc2;
        __m128 temp1, temp2;
        float *cbtmp = (float *)malloc(sizeof(float) * 8);

        for (i = 0; i < len1 && lb < bsf; i += 8)
        {
            Q = _mm256_loadu_ps(&qo[i]);
            tu256 = _mm256_loadu_ps(&tu[i]);
            tl256 = _mm256_loadu_ps(&tl[i]);
            // tu256 = _mm_setr_ps(tu[order[i]],tu[order[i+1]],tu[order[i+2]],tu[order[i+3]]);
            // tl256 = _mm_setr_ps(tl[order[i]],tl[order[i+1]],tl[order[i+2]],tl[order[i+3]]);
            calc1 = _mm256_min_ps(Q, tu256);
            calc1 = _mm256_sub_ps(Q, calc1);

            calc2 = _mm256_max_ps(Q, tl256);
            calc2 = _mm256_sub_ps(calc2, Q);
            calc1 = _mm256_add_ps(calc1, calc2);

            calc1 = _mm256_mul_ps(calc1, calc1);

            _mm256_storeu_ps(cbtmp, calc1);

            calc1 = _mm256_hadd_ps(calc1, calc1);
            calc1 = _mm256_hadd_ps(calc1, calc1);
            temp1 = _mm256_extractf128_ps(calc1, 1);
            temp2 = _mm_add_ss(_mm256_castps256_ps128(calc1), temp1);
            lb += _mm_cvtss_f32(temp2);

            cb[i] = cbtmp[0];
            cb[i + 1] = cbtmp[1];
            cb[i + 2] = cbtmp[2];
            cb[i + 3] = cbtmp[3];
            cb[i + 4] = cbtmp[4];
            cb[i + 5] = cbtmp[5];
            cb[i + 6] = cbtmp[6];
            cb[i + 7] = cbtmp[7];
        }

        for (; i < len && lb < bsf; i++)
        {
            uu = tu[i];
            ll = tl[i];
            d = 0;
            if (qo[i] > uu)
            {
                d = dist(qo[i], uu);
            }
            else if (qo[i] < ll)
            {
                d = dist(qo[i], ll);
            }
            lb += d;
            cb[i] = d;
        }

        free(cbtmp);
        return lb;
    }

    float dtwsimdPruned(float *A, float *B, float *cb, int m, int r, float bsf, float *tSum, float *pCost, float *rDist)
    {
        int length = 2 * r + 1;
        // SIMD register
        //__m256 a256, b256;

        int start, end;
        float minCost = 0.0f;
        // the first line
        for (int k = 0; k <= r; k++)
        {
            rDist[k] = (A[0] - B[k]) * (A[0] - B[k]);
        }

        tSum[0] = rDist[0];
        for (int ij = 1; ij <= r; ij++)
            tSum[ij] = tSum[ij - 1] + rDist[ij];

        pCost[0] = tSum[0];
        for (int ij = 1; ij <= r; ij++)
        {
            pCost[ij] = min(tSum[ij - 1], tSum[ij]);
        }
        pCost[r + 1] = tSum[r];

        for (int i = 1; i < m - 1; i++)
        {
            start = max(0, i - r);
            end = min(m - 1, i + r);

            for (int k = start; k <= end; k++)
            {
                rDist[k - start] = (A[i] - B[k]) * (A[i] - B[k]);
            }

            for (int k = start; k <= end; k++)
            {
                tSum[k - start] = pCost[k - start] + rDist[k - start];
            }

            minCost = tSum[0];
            for (int k = start + 1; k <= end; k++)
            {
                if (tSum[k - 1 - start] < pCost[k - start])
                {
                    tSum[k - start] = tSum[k - 1 - start] + rDist[k - start];
                }
                if (tSum[k - start] < minCost)
                    minCost = tSum[k - start];
            }
            if (i + r < m - 1 && minCost + cb[i + r + 1] >= bsf)
            {
                return minCost + cb[i + r + 1];
            }

            if ((end - start + 1) < length && start == 0)
            {
                pCost[start - start] = tSum[start - start];
                for (int ij = start + 1; ij <= end; ij++)
                {
                    pCost[ij - start] = min(tSum[ij - 1 - start], tSum[ij - start]);
                }
                pCost[end + 1 - start] = tSum[end - start];
            }
            else
            {
                for (int ij = start + 1; ij <= end; ij++)
                {
                    pCost[ij - 1 - start] = min(tSum[ij - 1 - start], tSum[ij - start]);
                }
                pCost[end - start] = tSum[end - start];
            }
        }

        // the last line
        start = m - 1 - r;
        end = m - 1;

        for (int k = start; k <= end; k++)
        {
            rDist[k - start] = (A[m - 1] - B[k]) * (A[m - 1] - B[k]);
        }

        for (int k = start; k <= end; k++)
        {
            tSum[k - start] = pCost[k - start] + rDist[k - start];
        }
        for (int k = start + 1; k <= end; k++)
        {
            if (tSum[k - 1 - start] < pCost[k - start])
            {
                tSum[k - start] = tSum[k - 1 - start] + rDist[k - start];
            }
        }
        float ret = tSum[r];
        return ret;
    }

    float minidist_paa_to_isax_DTW(float *paaU, float *paaL, sax_type *sax,
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

            if (breakpoint_lower > paaU[i])
            {
                distance += pow(breakpoint_lower - paaU[i], 2);
            }
            else if (breakpoint_upper < paaL[i])
            {
                distance += pow(breakpoint_upper - paaL[i], 2);
            }
            //        else {
            //            printf("%lf is between: %lf and %lf\n", paa[i], breakpoint_lower, breakpoint_upper);
            //        }
        }

        // distance = ratio_sqrt * sqrtf(distance);
        distance = ratio_sqrt * distance;
        return distance;
    }
}