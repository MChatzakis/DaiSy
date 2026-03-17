#include "Sing.hpp"
#include "Messi.hpp"
#include "../isax/iSAXIndex.hpp"
#include "../isax/iSAXSearch.hpp"
#include "../isax/iSAXTypes.hpp"
#include "../isax/iSAXPqueue.hpp"
#include "../isax/SAX.hpp"
#include "../isax/SAXBreakpoints.hpp"
#include "../utils/SIMD.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <pthread.h>
#include <vector>
#include <unordered_set>
#include <sys/time.h>
#if SING_CUDA_ENABLED
#include "singlib.hpp"
#include <cuda_runtime.h>
#endif

namespace daisy
{

    static long added_tree_node = 0;

    void approximate_topk_SING(ts_type *ts, ts_type *paa, isax_index *index, pqueue_bsf *pq_bsf, float *rawfile);
    void *multigapworker(void *arg);
    void *exact_knn_SING_worker(void *arg);
#if SING_CUDA_ENABLED
    
#else
    void LBDfloatstreamGPU(sax_type *gsaxarray, float *positionmap, ts_type *paa, float *gqts, float bsf,
                           unsigned long size, float *gpositionmap, int paa_segments, float mindist_sqrt);
#endif
    first_buffer_layer2 *initialize_simrec(int initial_buffer_size, int number_of_buffers,
                                           int max_total_buffers_size, isax_index *index);
    void *index_creation_worker2_inmemory(void *arg);

    void pass_tree_node_m(isax_node *node, isax_index *index, pthread_mutex_t *lock_queue,
                          unsigned long int *currentposition, sax_type *saxarray, sax_type *sortsaxarray,
                          float *lbdarray);


#if DAISY_SIMD_AVAILABLE
    float minidist_paa_to_isax_raw_SING_SIMD(float *paa, sax_type *sax, sax_type *sax_cardinalities,
                                             sax_type max_bit_cardinality, int max_cardinality,
                                             int number_of_segments, int min_val, int max_val, float ratio_sqrt)
    {
        int region_upper[16], region_lower[16];
        float distancef[16];
        int offset = 0;

        __m256i vectorsignbit = _mm256_set1_epi32(0xffffffff);
        __m256i vloweroffset = _mm256_set1_epi32(offset);
        __m256i vupperoffset = _mm256_set1_epi32(offset + 1);

        __m128i sax_cardinalitiesv8 = _mm_lddqu_si128((const __m128i *)sax_cardinalities);
        __m256i sax_cardinalitiesv16 = _mm256_cvtepu8_epi16(sax_cardinalitiesv8);
        __m128i sax_cardinalitiesv16_0 = _mm256_extractf128_si256(sax_cardinalitiesv16, 0);
        __m128i sax_cardinalitiesv16_1 = _mm256_extractf128_si256(sax_cardinalitiesv16, 1);
        __m256i c_cv_0 = _mm256_cvtepu16_epi32(sax_cardinalitiesv16_0);
        __m256i c_cv_1 = _mm256_cvtepu16_epi32(sax_cardinalitiesv16_1);

        __m128i saxv8 = _mm_lddqu_si128((const __m128i *)sax);
        __m256i saxv16 = _mm256_cvtepu8_epi16(saxv8);
        __m128i saxv16_0 = _mm256_extractf128_si256(saxv16, 0);
        __m128i saxv16_1 = _mm256_extractf128_si256(saxv16, 1);
        __m256i v_0 = _mm256_cvtepu16_epi32(saxv16_0);
        __m256i v_1 = _mm256_cvtepu16_epi32(saxv16_1);

        __m256i c_m = _mm256_set1_epi32(max_bit_cardinality);
        __m256i cm_ccv_0 = _mm256_sub_epi32(c_m, c_cv_0);
        __m256i cm_ccv_1 = _mm256_sub_epi32(c_m, c_cv_1);

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

        region_lowerv_0 = _mm256_add_epi32(region_lowerv_0, vloweroffset);
        region_lowerv_1 = _mm256_add_epi32(region_lowerv_1, vloweroffset);
        region_upperv_0 = _mm256_add_epi32(region_upperv_0, vupperoffset);
        region_upperv_1 = _mm256_add_epi32(region_upperv_1, vupperoffset);
        _mm256_storeu_si256((__m256i *)&region_lower[0], region_lowerv_0);
        _mm256_storeu_si256((__m256i *)&region_lower[8], region_lowerv_1);
        _mm256_storeu_si256((__m256i *)&region_upper[0], region_upperv_0);
        _mm256_storeu_si256((__m256i *)&region_upper[8], region_upperv_1);

        __m256i lower_juge_zerov_0 = _mm256_cmpeq_epi32(region_lowerv_0, _mm256_setzero_si256());
        __m256i lower_juge_zerov_1 = _mm256_cmpeq_epi32(region_lowerv_1, _mm256_setzero_si256());
        __m256i lower_juge_nzerov_0 = _mm256_andnot_si256(lower_juge_zerov_0, vectorsignbit);
        __m256i lower_juge_nzerov_1 = _mm256_andnot_si256(lower_juge_zerov_1, vectorsignbit);
        __m256 minvalv = _mm256_set1_ps((float)min_val);

        __m256 lsax_breakpoints_shiftv_0 = _mm256_i32gather_ps(sax_breakpointsnew3, region_lowerv_0, 4);
        __m256 lsax_breakpoints_shiftv_1 = _mm256_i32gather_ps(sax_breakpointsnew3, region_lowerv_1, 4);
        __m256 breakpoint_lowerv_0 = (__m256)_mm256_castsi256_ps(_mm256_or_si256(
            _mm256_and_si256(lower_juge_zerov_0, _mm256_castps_si256(minvalv)),
            _mm256_and_si256(lower_juge_nzerov_0, _mm256_castps_si256(lsax_breakpoints_shiftv_0))));
        __m256 breakpoint_lowerv_1 = (__m256)_mm256_castsi256_ps(_mm256_or_si256(
            _mm256_and_si256(lower_juge_zerov_1, _mm256_castps_si256(minvalv)),
            _mm256_and_si256(lower_juge_nzerov_1, _mm256_castps_si256(lsax_breakpoints_shiftv_1))));

        __m256 usax_breakpoints_shiftv_0 = _mm256_i32gather_ps(sax_breakpointsnew3, region_upperv_0, 4);
        __m256 usax_breakpoints_shiftv_1 = _mm256_i32gather_ps(sax_breakpointsnew3, region_upperv_1, 4);
        __m256i upper_juge_maxv_0 = _mm256_cmpeq_epi32(region_upperv_0, _mm256_set1_epi32(max_cardinality - 1));
        __m256i upper_juge_maxv_1 = _mm256_cmpeq_epi32(region_upperv_1, _mm256_set1_epi32(max_cardinality - 1));
        __m256i upper_juge_nmaxv_0 = _mm256_andnot_si256(upper_juge_maxv_0, vectorsignbit);
        __m256i upper_juge_nmaxv_1 = _mm256_andnot_si256(upper_juge_maxv_1, vectorsignbit);
        __m256 maxvalv = _mm256_set1_ps((float)max_val);
        __m256 breakpoint_upperv_0 = (__m256)_mm256_castsi256_ps(_mm256_or_si256(
            _mm256_and_si256(upper_juge_maxv_0, _mm256_castps_si256(maxvalv)),
            _mm256_and_si256(upper_juge_nmaxv_0, _mm256_castps_si256(usax_breakpoints_shiftv_0))));
        __m256 breakpoint_upperv_1 = (__m256)_mm256_castsi256_ps(_mm256_or_si256(
            _mm256_and_si256(upper_juge_maxv_1, _mm256_castps_si256(maxvalv)),
            _mm256_and_si256(upper_juge_nmaxv_1, _mm256_castps_si256(usax_breakpoints_shiftv_1))));

        __m256 paav_0 = _mm256_loadu_ps(paa);
        __m256 paav_1 = _mm256_loadu_ps(&paa[8]);

        __m256 dis_juge_upv_0 = _mm256_cmp_ps(breakpoint_lowerv_0, paav_0, _CMP_GT_OS);
        __m256 dis_juge_upv_1 = _mm256_cmp_ps(breakpoint_lowerv_1, paav_1, _CMP_GT_OS);
        __m256 dis_juge_lov_0 = _mm256_cmp_ps(breakpoint_upperv_0, paav_0, _CMP_LT_OS);
        __m256 dis_juge_lov_1 = _mm256_cmp_ps(breakpoint_upperv_1, paav_1, _CMP_LT_OS);
        __m256 dis_juge_elv_0 = (__m256)_mm256_castsi256_ps(_mm256_andnot_si256(
            _mm256_or_si256(_mm256_castps_si256(dis_juge_upv_0), _mm256_castps_si256(dis_juge_lov_0)), vectorsignbit));
        __m256 dis_juge_elv_1 = (__m256)_mm256_castsi256_ps(_mm256_andnot_si256(
            _mm256_or_si256(_mm256_castps_si256(dis_juge_upv_1), _mm256_castps_si256(dis_juge_lov_1)), vectorsignbit));

        __m256 dis_lowv_0 = _mm256_sub_ps(breakpoint_lowerv_0, paav_0);
        __m256 dis_lowv_1 = _mm256_sub_ps(breakpoint_lowerv_1, paav_1);
        __m256 dis_uppv_0 = _mm256_sub_ps(breakpoint_upperv_0, paav_0);
        __m256 dis_uppv_1 = _mm256_sub_ps(breakpoint_upperv_1, paav_1);

        __m256 distancev_0 = (__m256)_mm256_castsi256_ps(_mm256_or_si256(_mm256_or_si256(
                                                                             _mm256_and_si256(_mm256_castps_si256(dis_juge_upv_0), _mm256_castps_si256(dis_lowv_0)),
                                                                             _mm256_and_si256(_mm256_castps_si256(dis_juge_lov_0), _mm256_castps_si256(dis_uppv_0))),
                                                                         _mm256_and_si256(_mm256_castps_si256(dis_juge_elv_0), _mm256_castps_si256(_mm256_set1_ps(0.0f)))));
        __m256 distancev_1 = (__m256)_mm256_castsi256_ps(_mm256_or_si256(_mm256_or_si256(
                                                                             _mm256_and_si256(_mm256_castps_si256(dis_juge_upv_1), _mm256_castps_si256(dis_lowv_1)),
                                                                             _mm256_and_si256(_mm256_castps_si256(dis_juge_lov_1), _mm256_castps_si256(dis_uppv_1))),
                                                                         _mm256_and_si256(_mm256_castps_si256(dis_juge_elv_1), _mm256_castps_si256(_mm256_set1_ps(0.0f)))));

        __m256 distancesum_0 = _mm256_dp_ps(distancev_0, distancev_0, 0xff);
        __m256 distancesum_1 = _mm256_dp_ps(distancev_1, distancev_1, 0xff);
        __m256 distancevf = _mm256_add_ps(distancesum_0, distancesum_1);
        _mm256_storeu_ps(distancef, distancevf);

        return (distancef[0] + distancef[4]) * ratio_sqrt;
    }
#else
    float minidist_paa_to_isax_raw_SING_SIMD(float *paa, sax_type *sax, sax_type *sax_cardinalities,
                                             sax_type max_bit_cardinality, int max_cardinality,
                                             int number_of_segments, int min_val, int max_val, float ratio_sqrt)
    {
        THROW_SIMD_NOT_AVAILABLE("minidist_paa_to_isax_raw_SING_SIMD");
    }
#endif

    void approximate_topk_SING(ts_type *ts, ts_type *paa, isax_index *index, pqueue_bsf *pq_bsf, float *rawfile)
    {
        sax_type *sax = (sax_type *)malloc(sizeof(sax_type) * (size_t)index->settings->paa_segments);
        sax_from_paa(paa, sax, index->settings->paa_segments,
                     index->settings->sax_alphabet_cardinality,
                     index->settings->sax_bit_cardinality);

        root_mask_type root_mask = 0;
        CREATE_MASK(root_mask, index, sax);

        if ((&((first_buffer_layer2 *)(index->fbl))->soft_buffers[(int)root_mask])->initialized)
        {
            isax_node *node = (&((first_buffer_layer2 *)(index->fbl))->soft_buffers[(int)root_mask])->node;
            while (!node->is_leaf)
            {
                if (node->split_data == NULL)
                    break;
                int location = index->settings->sax_bit_cardinality - 1 -
                               node->split_data->split_mask[node->split_data->splitpoint];
                root_mask_type mask = index->settings->bit_masks[location];

                if (sax[node->split_data->splitpoint] & mask)
                {
                    node = node->right_child;
                }
                else
                {
                    node = node->left_child;
                }
            }
            calculate_node_topk_inmemory(index, node, ts, pq_bsf, rawfile);
        }
        
        free(sax);
    }

    float nodedistance(float *paa, sax_type *sax,
                       sax_type *sax_cardinalities,
                       sax_type max_bit_cardinality,
                       int max_cardinality,
                       int number_of_segments,
                       int min_val,
                       int max_val,
                       float ratio_sqrt)
    {
        (void)max_bit_cardinality;
        (void)max_cardinality;
        (void)min_val;
        (void)max_val;
        float distance = 0;
        int i;
        for (i = 0; i < number_of_segments; i++)
        {
            sax_type c_c = sax[i];
            if ((c_c == 0 && paa[i] > 0.0f))
            {
                distance += paa[i] * paa[i];
            }
            else if (c_c != 0 && paa[i] < 0.0f)
            {
                distance += paa[i] * paa[i];
            }
        }
        distance = ratio_sqrt * distance;
        return distance;
    }

#ifndef MAXFLOAT
#define MAXFLOAT FLT_MAX
#endif

    float minidist_paa_to_isax_Breakpoly(float *paa, sax_type *sax, sax_type *sax_cardinalities,
                                         sax_type max_bit_cardinality, int max_cardinality, int number_of_segments,
                                         int min_val, int max_val, float ratio_sqrt)
    {
        float distance = 0;
        int offset = ((max_cardinality - 1) * (max_cardinality - 2)) / 2;
        (void)offset;
        int i;
        for (i = 0; i < number_of_segments; i++)
        {
            sax_type c_c = sax_cardinalities[i];
            sax_type c_m = max_bit_cardinality;
            sax_type v = sax[i];
            sax_type region_lower = (sax_type)(v << (c_m - c_c));
            
            sax_type mask = ((c_m - c_c) >= (int)(sizeof(sax_type) * 8)) ? (sax_type)-1 : ((sax_type)1 << (c_m - c_c)) - 1;
            sax_type region_upper = region_lower | mask;
            float breakpoint_lower = 0;
            float breakpoint_upper = 0;

            if (region_lower == 0)
            {
                breakpoint_lower = (float)min_val;
            }
            else
            {
                breakpoint_lower = sax_breakpointsnew3[region_lower];
            }
            if (region_upper == (sax_type)(max_cardinality - 1))
            {
                breakpoint_upper = (float)max_val;
            }
            else
            {
                breakpoint_upper = sax_breakpointsnew3[region_upper + 1];
            }

            if (breakpoint_lower > paa[i])
            {
                distance += (float)pow((double)(breakpoint_lower - paa[i]), 2.0);
            }
            else if (breakpoint_upper < paa[i])
            {
                distance += (float)pow((double)(breakpoint_upper - paa[i]), 2.0);
            }
        }
        distance = ratio_sqrt * distance;
        return distance;
    }

    void *multigapworker(void *gapworkerdata)
    {
        gap_workerdata *gd = (gap_workerdata *)gapworkerdata;
        int amountnode = gd->amountnode;
        isax_node **nodelist = gd->nodelist;
        isax_index *index = gd->index;
        ts_type *paa = gd->paa;
        float bsf = gd->bsf;
        unsigned long *offsetarray = gd->offsetarray;
        bool *activechunk = gd->activechunk;
        bool *activenode = gd->activenode;
        int chunknumber = gd->chunknumber;
        (void)amountnode;

        for (int current_root_node_number = gd->workerstartnode; current_root_node_number < gd->workerstopnode; current_root_node_number++)
        {
            isax_node *node = nodelist[current_root_node_number];
            if (node == nullptr || node->isax_values == nullptr || node->isax_cardinalities == nullptr)
            {
                activenode[current_root_node_number] = false;
                continue;
            }
            float distance = nodedistance(paa, node->isax_values,
                                          node->isax_cardinalities,
                                          (sax_type)index->settings->sax_bit_cardinality,
                                          index->settings->sax_alphabet_cardinality,
                                          index->settings->paa_segments,
                                          MINVAL, MAXVAL,
                                          index->settings->mindist_sqrt);
            if (distance < bsf)
            {
                activechunk[(int)(offsetarray[current_root_node_number] / (index->sax_cache_size / chunknumber))] = true;
                activechunk[(int)(offsetarray[current_root_node_number + 1] / (index->sax_cache_size / chunknumber))] = true;
                activenode[current_root_node_number] = true;
            }
            else
            {
                activenode[current_root_node_number] = false;
            }
        }
        pthread_exit(NULL);
    }

    void pass_tree_node_m(isax_node *node, isax_index *index, pthread_mutex_t *lock_queue,
                          unsigned long int *currentposition, sax_type *saxarray, sax_type *sortsaxarray,
                          float *lbdarray)
    {
        (void)lock_queue;
        (void)saxarray;
        if (node == nullptr)
            return;
        if (node->is_leaf)
        {
            if (node->buffer == nullptr || node->buffer->partial_sax_buffer == nullptr || node->buffer->partial_buffer_size <= 0)
                return;
            unsigned long int arrayposition = __sync_fetch_and_add(currentposition, (unsigned long)node->buffer->partial_buffer_size);
            node->buffer->lbdarray = &lbdarray[arrayposition];
            node->buffer->arrayposition = arrayposition + (unsigned long)node->buffer->partial_buffer_size;
            const int paa_segments = index->settings->paa_segments;
            const size_t sax_byte_size = (size_t)index->settings->sax_byte_size;
            for (int i = 0; i < node->buffer->partial_buffer_size; i++)
            {
                std::memcpy((void *)&sortsaxarray[(arrayposition + (unsigned long)i) * paa_segments],
                            node->buffer->partial_sax_buffer[i], sax_byte_size);
            }
        }
        else
        {
            if (node->left_child != nullptr && node->left_child->isax_cardinalities != nullptr)
                pass_tree_node_m(node->left_child, index, lock_queue, currentposition, saxarray, sortsaxarray, lbdarray);
            if (node->right_child != nullptr && node->right_child->isax_cardinalities != nullptr)
                pass_tree_node_m(node->right_child, index, lock_queue, currentposition, saxarray, sortsaxarray, lbdarray);
        }
    }

    float nodedistancedtw(float *paa, float *paaU, float *paaL, sax_type *sax,
                          sax_type *sax_cardinalities,
                          sax_type max_bit_cardinality,
                          int max_cardinality,
                          int number_of_segments,
                          int min_val,
                          int max_val,
                          float ratio_sqrt)
    {
        (void)sax_cardinalities;
        (void)max_bit_cardinality;
        (void)max_cardinality;
        (void)min_val;
        (void)max_val;
        float distance = 0;
        int i;
        for (i = 0; i < number_of_segments; i++)
        {
            sax_type c_c = sax[i];
            if ((c_c == 0 && paaL[i] > 0.0f))
            {
                distance += paaL[i] * paaL[i];
            }
            else if (c_c != 0 && paaU[i] < 0.0f)
            {
                distance += paaU[i] * paaU[i];
            }
        }
        distance = ratio_sqrt * distance;
        (void)paa;
        return distance;
    }

    void insert_tree_node_m_hybridpqueueBreakpolydtw(float *paaU, float *paaL, isax_node *node, isax_index *index,
                                                     float bsf, pqueue_t **pq, pthread_mutex_t *lock_queue,
                                                     int *tnumber, int n_pqueue)
    {
        float distance = minidist_paa_to_isax_DTWbp(paaU, paaL, node->isax_values,
                                                    node->isax_cardinalities,
                                                    (sax_type)index->settings->sax_bit_cardinality,
                                                    index->settings->sax_alphabet_cardinality,
                                                    index->settings->paa_segments,
                                                    MINVAL, MAXVAL,
                                                    index->settings->mindist_sqrt);
        if (distance < bsf)
        {
            if (node->is_leaf)
            {
                query_result *mindist_result = (query_result *)malloc(sizeof(query_result));
                mindist_result->node = node;
                mindist_result->distance = distance;
                pthread_mutex_lock(&lock_queue[*tnumber]);
                pqueue_insert(pq[*tnumber], mindist_result);
                pthread_mutex_unlock(&lock_queue[*tnumber]);
                *tnumber = (*tnumber + 1) % n_pqueue;
                added_tree_node++;
            }
            else
            {
                if (node->left_child != nullptr && node->left_child->isax_cardinalities != nullptr)
                {
                    insert_tree_node_m_hybridpqueueBreakpolydtw(paaU, paaL, node->left_child, index, bsf, pq, lock_queue, tnumber, n_pqueue);
                }
                if (node->right_child != nullptr && node->right_child->isax_cardinalities != nullptr)
                {
                    insert_tree_node_m_hybridpqueueBreakpolydtw(paaU, paaL, node->right_child, index, bsf, pq, lock_queue, tnumber, n_pqueue);
                }
            }
        }
    }

    float minidist_paa_to_isax_DTWbp(float *paaU, float *paaL, sax_type *sax,
                                     sax_type *sax_cardinalities,
                                     sax_type max_bit_cardinality,
                                     int max_cardinality,
                                     int number_of_segments,
                                     int min_val,
                                     int max_val,
                                     float ratio_sqrt)
    {
        (void)max_bit_cardinality;
        (void)max_cardinality;
        (void)min_val;
        (void)max_val;
        float distance = 0;
        int offset = ((max_cardinality - 1) * (max_cardinality - 2)) / 2;
        (void)offset;
        int i;
        for (i = 0; i < number_of_segments; i++)
        {
            sax_type c_c = sax_cardinalities[i];
            sax_type c_m = max_bit_cardinality;
            sax_type v = sax[i];
            sax_type region_lower = (sax_type)(v << (c_m - c_c));
            sax_type mask = (sax_type)((1u << (c_m - c_c)) - 1u);
            sax_type region_upper = (sax_type)((~((sax_type)0 << (c_m - c_c))) | region_lower);
            float breakpoint_lower = 0;
            float breakpoint_upper = 0;

            if (region_lower == 0)
            {
                breakpoint_lower = (float)min_val;
            }
            else
            {
                breakpoint_lower = sax_breakpointsnew3[region_lower - 1];
            }
            if (region_upper == (sax_type)(max_cardinality - 1))
            {
                breakpoint_upper = (float)max_val;
            }
            else
            {
                breakpoint_upper = sax_breakpointsnew3[region_upper];
            }

            (void)mask;

            if (breakpoint_lower > paaU[i])
            {
                distance += (float)pow((double)(breakpoint_lower - paaU[i]), 2.0);
            }
            else if (breakpoint_upper < paaL[i])
            {
                distance += (float)pow((double)(breakpoint_upper - paaL[i]), 2.0);
            }
        }
        distance = ratio_sqrt * distance;
        return distance;
    }

    float calculate_node_distance_SING_short_dtw(isax_index *index, isax_node *node, ts_type *query, float *uo,
                                                 float *lo, ts_type *paa, ts_type *paaU, ts_type *paaL, float bsf,
                                                 float shortrate, int warpWind, float *rawfile)
    {
        (void)paa;
        (void)paaU;
        (void)paaL;
        if (node->buffer == nullptr)
            return bsf;
        float distmin, dist;
        int k;
        float *cb = (float *)malloc(sizeof(float) * (size_t)index->settings->timeseries_size);
        float *cb1 = (float *)malloc(sizeof(float) * (size_t)index->settings->timeseries_size);
        int length = 2 * warpWind + 1;
        float *tSum = (float *)malloc(sizeof(float) * (size_t)length);
        float *pCost = (float *)malloc(sizeof(float) * (size_t)length);
        float *rDist = (float *)malloc(sizeof(float) * (size_t)length);
        if (cb == nullptr || cb1 == nullptr || tSum == nullptr || pCost == nullptr || rDist == nullptr)
        {
            free(cb);
            free(cb1);
            free(tSum);
            free(pCost);
            free(rDist);
            return bsf;
        }
        short bsfshort = (short)(bsf * shortrate);
        for (int i = 0; i < node->buffer->partial_buffer_size; i++)
        {
            if (node->buffer->lbdarrayshort == nullptr)
                continue;
            if (node->buffer->lbdarrayshort[i] < bsfshort)
            {
                if (node->buffer->partial_position_buffer == nullptr || node->buffer->partial_position_buffer[i] == nullptr)
                    continue;
                distmin = lb_keogh_data_bound(&(rawfile[*node->buffer->partial_position_buffer[i]]), uo, lo, cb1,
                                              index->settings->timeseries_size, bsf);
                if (distmin < bsf)
                {
                    cb[index->settings->timeseries_size - 1] = cb1[index->settings->timeseries_size - 1];
                    for (k = index->settings->timeseries_size - 2; k >= 0; k--)
                        cb[k] = cb[k + 1] + cb1[k];

                    dist = dtwsimdPruned(query, &(rawfile[*node->buffer->partial_position_buffer[i]]), cb,
                                         index->settings->timeseries_size, warpWind, bsf, tSum, pCost, rDist);
                    if (dist < bsf)
                        bsf = dist;
                }
            }
        }
        free(tSum);
        free(pCost);
        free(rDist);
        free(cb);
        free(cb1);
        return bsf;
    }

    float calculate_node_DTW2_inmemorysing(isax_index *index, isax_node *node, ts_type *query, float *uo,
                                           float *lo, ts_type *paa, ts_type *paaU, ts_type *paaL, float bsf,
                                           int warpWind, float *rawfile, idx_t *best_position)
    {
        (void)paa;
        if (node->buffer == nullptr)
            return bsf;
        float distmin, dist;
        int k;
        float *cb = (float *)malloc(sizeof(float) * (size_t)index->settings->timeseries_size);
        float *cb1 = (float *)malloc(sizeof(float) * (size_t)index->settings->timeseries_size);
        int length = 2 * warpWind + 1;
        float *tSum = (float *)malloc(sizeof(float) * (size_t)length);
        float *pCost = (float *)malloc(sizeof(float) * (size_t)length);
        float *rDist = (float *)malloc(sizeof(float) * (size_t)length);
        if (cb == nullptr || cb1 == nullptr || tSum == nullptr || pCost == nullptr || rDist == nullptr)
        {
            free(cb);
            free(cb1);
            free(tSum);
            free(pCost);
            free(rDist);
            return bsf;
        }
        for (int i = 0; i < node->buffer->partial_buffer_size; i++)
        {
            if (node->buffer->partial_position_buffer == nullptr || node->buffer->partial_position_buffer[i] == nullptr)
                continue;

            distmin = minidist_paa_to_isax_raw_DTW_SING_SIMD(paaU, paaL, node->buffer->partial_sax_buffer[i],
                                                             index->settings->max_sax_cardinalities,
                                                             index->settings->sax_bit_cardinality,
                                                             index->settings->sax_alphabet_cardinality,
                                                             index->settings->paa_segments, MINVAL, MAXVAL,
                                                             index->settings->mindist_sqrt);

            if (distmin < bsf)
            {
                distmin = lb_keogh_data_bound(&(rawfile[*node->buffer->partial_position_buffer[i]]), uo, lo, cb1,
                                              index->settings->timeseries_size, bsf);
                if (distmin < bsf)
                {
                    cb[index->settings->timeseries_size - 1] = cb1[index->settings->timeseries_size - 1];
                    for (k = index->settings->timeseries_size - 2; k >= 0; k--)
                        cb[k] = cb[k + 1] + cb1[k];

                    dist = dtwsimdPruned(query, &(rawfile[*node->buffer->partial_position_buffer[i]]), cb,
                                         index->settings->timeseries_size, warpWind, bsf, tSum, pCost, rDist);
                    if (dist < bsf)
                    {
                        bsf = dist;
                        if (best_position != nullptr)
                        {
                            *best_position = (idx_t)(*node->buffer->partial_position_buffer[i] / index->settings->timeseries_size);
                        }
                    }
                }
            }
        }
        free(tSum);
        free(pCost);
        free(rDist);
        free(cb);
        free(cb1);
        return bsf;
    }

    void *uniworkershort_v4dtw(void *uniworkerdata)
    {
        int amountnode = ((uni_workerdata *)uniworkerdata)->amountnode;
        isax_node **nodelist = ((uni_workerdata *)uniworkerdata)->nodelist;
        isax_index *index = ((uni_workerdata *)uniworkerdata)->index;
        ts_type *paa = ((uni_workerdata *)uniworkerdata)->paa;
        ts_type *paaU = ((uni_workerdata *)uniworkerdata)->paaU;
        ts_type *paaL = ((uni_workerdata *)uniworkerdata)->paaL;
        ts_type *uo = ((uni_workerdata *)uniworkerdata)->uo;
        ts_type *lo = ((uni_workerdata *)uniworkerdata)->lo;
        int warpWind = ((uni_workerdata *)uniworkerdata)->warpWind;

        float bsf = ((uni_workerdata *)uniworkerdata)->bsf;
        unsigned long *offsetarray = ((uni_workerdata *)uniworkerdata)->offsetarray;
        bool *activechunk = ((uni_workerdata *)uniworkerdata)->activechunk;
        bool *activenode = ((uni_workerdata *)uniworkerdata)->activenode;
        int current_root_node_number;
        int chunknumber = ((uni_workerdata *)uniworkerdata)->chunknumber;
        int *chunkcounter = ((uni_workerdata *)uniworkerdata)->chunkcounter;
        (void)amountnode;
        (void)chunkcounter;

        isax_node *current_root_node;
        query_result *n;
        ts_type *ts = ((uni_workerdata *)uniworkerdata)->ts;
        pqueue_t *pq = ((uni_workerdata *)uniworkerdata)->pq;
        (void)pq;
        query_result *do_not_remove = ((uni_workerdata *)uniworkerdata)->bsf_result;
        float minimum_distance = ((uni_workerdata *)uniworkerdata)->minimum_distance;
        int limit = ((uni_workerdata *)uniworkerdata)->limit;
        int checks = 0;
        bool finished = true;
        int tight_bound = index->settings->tight_bound;
        int aggressive_check = index->settings->aggressive_check;
        (void)do_not_remove;
        (void)limit;
        (void)tight_bound;
        (void)aggressive_check;
        query_result *bsf_result = (((uni_workerdata *)uniworkerdata)->bsf_result);
        float bsfdisntance = bsf;
        float distance;
        int calculate_node = 0, calculate_node_quque = 0;
        (void)calculate_node;
        (void)calculate_node_quque;
        int startqueuenumber = ((uni_workerdata *)uniworkerdata)->startqueuenumber;
        int tnumber = startqueuenumber;
        short *lbdmap = ((uni_workerdata *)uniworkerdata)->lbdmapshort;
        (void)lbdmap;
        int localqueuecounter = 0;
        (void)localqueuecounter;
        pthread_mutex_t *alllock = ((uni_workerdata *)uniworkerdata)->alllock;
        int n_pqueue = ((uni_workerdata *)uniworkerdata)->n_pqueue;
        int *allqueuelabel = ((uni_workerdata *)uniworkerdata)->allqueuelabel;

        for (current_root_node_number = ((uni_workerdata *)uniworkerdata)->workerstartnode;
             current_root_node_number < ((uni_workerdata *)uniworkerdata)->workerstopnode;
             current_root_node_number++)
        {
            isax_node *node = nodelist[current_root_node_number];
            distance = nodedistancedtw(paa, paaU, paaL, node->isax_values,
                                       node->isax_cardinalities,
                                       (sax_type)index->settings->sax_bit_cardinality,
                                       index->settings->sax_alphabet_cardinality,
                                       index->settings->paa_segments,
                                       MINVAL, MAXVAL,
                                       index->settings->mindist_sqrt);
            if (distance < bsf)
            {
                activechunk[(int)(offsetarray[current_root_node_number] / (index->sax_cache_size / chunknumber))] = true;
                activechunk[(int)(offsetarray[current_root_node_number + 1] / (index->sax_cache_size / chunknumber))] = true;
                activenode[current_root_node_number] = true;
            }
            else
            {
                activenode[current_root_node_number] = false;
            }
        }

        pthread_barrier_wait(((uni_workerdata *)uniworkerdata)->lock_barrierfirst);

        current_root_node_number = 0;
        while (1)
        {
            current_root_node_number = __sync_fetch_and_add(((uni_workerdata *)uniworkerdata)->node_counter, 1);
            if (current_root_node_number >= ((uni_workerdata *)uniworkerdata)->amountnode)
                break;
            current_root_node = ((uni_workerdata *)uniworkerdata)->nodelist[current_root_node_number];
            if (activenode[current_root_node_number] == true)
            {
                insert_tree_node_m_hybridpqueueBreakpolydtw(paaU, paaL, current_root_node, index, bsfdisntance,
                                                            ((uni_workerdata *)uniworkerdata)->allpq,
                                                            ((uni_workerdata *)uniworkerdata)->alllock,
                                                            &tnumber, n_pqueue);
            }
        }
        pthread_barrier_wait(((uni_workerdata *)uniworkerdata)->lock_barrier);

        int queuenumber;
        int offset = startqueuenumber;
        while (1)
        {
            finished = true;
            for (int i = 0; i < n_pqueue; i++)
            {
                queuenumber = (i + offset) % n_pqueue;
                if (allqueuelabel[queuenumber] == 1)
                {
                    finished = false;
                    bsfdisntance = bsf_result->distance;
                    pthread_mutex_lock(&alllock[queuenumber]);
                    n = (query_result *)pqueue_pop(((uni_workerdata *)uniworkerdata)->allpq[queuenumber]);
                    if (n == nullptr)
                    {
                        allqueuelabel[queuenumber] = 0;
                        pthread_mutex_unlock(&alllock[queuenumber]);
                        continue;
                    }
                    else
                    {
                        pthread_mutex_unlock(&alllock[queuenumber]);
                    }

                    if (n->distance > bsfdisntance || n->distance > minimum_distance)
                    {
                        allqueuelabel[queuenumber] = 0;
                        free(n);
                        continue;
                    }
                    else
                    {
                        if (n->node->is_leaf)
                        {
                            checks++;
                            float *rawfile = ((uni_workerdata *)uniworkerdata)->rawfile;
                            if (n->node->buffer->arrayposition < *((uni_workerdata *)uniworkerdata)->gpuoffset)
                            {
                                distance = calculate_node_distance_SING_short_dtw(index, n->node, ts, uo, lo,
                                                                                  paa, paaU, paaL,
                                                                                  bsfdisntance,
                                                                                  ((uni_workerdata *)uniworkerdata)->shortrate,
                                                                                  ((uni_workerdata *)uniworkerdata)->warpWind,
                                                                                  rawfile);
                            }
                            else
                            {
                                distance = calculate_node_DTW2_inmemorysing(index, n->node, ts, uo, lo,
                                                                            paa, paaU, paaL,
                                                                            bsfdisntance, warpWind,
                                                                            rawfile, nullptr);
                            }
                            if (distance < bsfdisntance)
                            {
                                pthread_rwlock_wrlock(((uni_workerdata *)uniworkerdata)->lock_bsf);
                                if (distance < bsf_result->distance)
                                {
                                    bsf_result->distance = distance;
                                    bsf_result->node = n->node;
                                }
                                pthread_rwlock_unlock(((uni_workerdata *)uniworkerdata)->lock_bsf);
                            }
                        }
                    }
                    free(n);
                }
                else if (allqueuelabel[queuenumber] == 0)
                {
                    pthread_mutex_lock(&(((uni_workerdata *)uniworkerdata)->alllock[queuenumber]));
                    n = (query_result *)pqueue_pop(((uni_workerdata *)uniworkerdata)->allpq[queuenumber]);
                    pthread_mutex_unlock(&(((uni_workerdata *)uniworkerdata)->alllock[queuenumber]));
                    if (n != nullptr)
                    {
                        free(n);
                    }
                    else
                    {
                        allqueuelabel[queuenumber] = 2;
                    }
                }
            }
            if (finished)
            {
                break;
            }
        }

        (void)checks;
        pthread_exit(nullptr);
    }

    void pass_tree_node_m_short(isax_node *node, isax_index *index, pthread_mutex_t *lock_queue,
                                unsigned long int *currentposition, sax_type *saxarray, sax_type *sortsaxarray,
                                short *lbdarray)
    {
        (void)lock_queue;
        (void)saxarray;
        if (node == nullptr)
            return;
        if (node->is_leaf)
        {
            if (node->buffer == nullptr || node->buffer->partial_sax_buffer == nullptr || node->buffer->partial_buffer_size <= 0)
                return;
            unsigned long int arrayposition = __sync_fetch_and_add(currentposition, (unsigned long)node->buffer->partial_buffer_size);
            node->buffer->lbdarrayshort = &lbdarray[arrayposition];
            node->buffer->arrayposition = arrayposition + (unsigned long)node->buffer->partial_buffer_size;
            const int paa_segments = index->settings->paa_segments;
            for (int i = 0; i < node->buffer->partial_buffer_size; i++)
            {
                for (int j = 0; j < paa_segments; j++)
                {
                    sortsaxarray[(arrayposition + (unsigned long)i) * (unsigned long)paa_segments + (unsigned long)j] =
                        (sax_type)node->buffer->partial_sax_buffer[i][j];
                }
            }
        }
        else
        {
            if (node->left_child != nullptr && node->left_child->isax_cardinalities != nullptr)
                pass_tree_node_m_short(node->left_child, index, lock_queue, currentposition, saxarray, sortsaxarray, lbdarray);
            if (node->right_child != nullptr && node->right_child->isax_cardinalities != nullptr)
                pass_tree_node_m_short(node->right_child, index, lock_queue, currentposition, saxarray, sortsaxarray, lbdarray);
        }
    }

    float calculate_node_DTW_inmemory(isax_index *index, isax_node *node, ts_type *query, float bsf, int warpWind, float *rawfile)
    {
        if (node->buffer == nullptr)
            return bsf;
        float *cb = (float *)calloc((size_t)index->settings->timeseries_size, sizeof(float));
        if (cb == nullptr)
            return bsf;
        int i;
        for (i = 0; i < node->buffer->full_buffer_size; i++)
        {
            float dist = ts_euclidean_distance(query, node->buffer->full_ts_buffer[i],
                                              index->settings->timeseries_size, bsf);
            if (dist < bsf)
                bsf = dist;
        }
        for (i = 0; i < node->buffer->tmp_full_buffer_size; i++)
        {
            float dist = ts_euclidean_distance(query, node->buffer->tmp_full_ts_buffer[i],
                                               index->settings->timeseries_size, bsf);
            if (dist < bsf)
                bsf = dist;
        }
        if (rawfile != nullptr && node->buffer->partial_position_buffer != nullptr)
        {
            for (i = 0; i < node->buffer->partial_buffer_size; i++)
            {
                if (node->buffer->partial_position_buffer[i] == nullptr)
                    continue;
                float dist = dtw(query, &(rawfile[*node->buffer->partial_position_buffer[i]]), cb,
                                index->settings->timeseries_size, warpWind, bsf);
                if (dist < bsf)
                    bsf = dist;
            }
        }
        free(cb);
        return bsf;
    }

    query_result approximate_DTW_inmemory_sing(ts_type *ts, ts_type *paa, isax_index *index, int warpWind, float *rawfile)
    {
        query_result result;
        result.node = nullptr;
        result.distance = FLT_MAX;
        result.pqueue_position = 0;
        result.pq_bsf = nullptr;
        result.total_time = 0.0f;

        sax_type *sax = (sax_type *)malloc(sizeof(sax_type) * (size_t)index->settings->paa_segments);
        if (sax == nullptr)
            return result;
        sax_from_paa(paa, sax, index->settings->paa_segments,
                    index->settings->sax_alphabet_cardinality,
                    index->settings->sax_bit_cardinality);

        root_mask_type root_mask = 0;
        CREATE_MASK(root_mask, index, sax);

        if ((&((first_buffer_layer2 *)(index->fbl))->soft_buffers[(int)root_mask])->initialized)
        {
            isax_node *node = (&((first_buffer_layer2 *)(index->fbl))->soft_buffers[(int)root_mask])->node;
            while (!node->is_leaf)
            {
                int location = index->settings->sax_bit_cardinality - 1 -
                              node->split_data->split_mask[node->split_data->splitpoint];
                root_mask_type mask = index->settings->bit_masks[location];
                if (sax[node->split_data->splitpoint] & mask)
                    node = node->right_child;
                else
                    node = node->left_child;
            }
            result.distance = calculate_node_DTW_inmemory(index, node, ts, FLT_MAX, warpWind, rawfile);
            result.node = node;
        }

        free(sax);
        return result;
    }

    void insert_tree_node_m_hybridpqueueBreakpolyroot(float *paa, isax_node *node, isax_index *index, float bsf,
                                                      pqueue_t **pq, pthread_mutex_t *lock_queue, int *tnumber, int n_pqueue)
    {
        if (node == nullptr)
            return;
        if (node->is_leaf)
        {
            float distance = minidist_paa_to_isax_Breakpoly(paa, node->isax_values,
                                                            node->isax_cardinalities,
                                                            (sax_type)index->settings->sax_bit_cardinality,
                                                            index->settings->sax_alphabet_cardinality,
                                                            index->settings->paa_segments,
                                                            MINVAL, MAXVAL,
                                                            index->settings->mindist_sqrt);
            query_result *mindist_result = (query_result *)malloc(sizeof(query_result));
            mindist_result->node = node;
            mindist_result->distance = distance;
            pthread_mutex_lock(&lock_queue[*tnumber]);
            pqueue_insert(pq[*tnumber], mindist_result);
            pthread_mutex_unlock(&lock_queue[*tnumber]);
            *tnumber = (*tnumber + 1) % n_pqueue;
            added_tree_node++;
        }
        else
        {
            if (node->left_child != nullptr && node->left_child->isax_cardinalities != nullptr)
                insert_tree_node_m_hybridpqueueBreakpolyroot(paa, node->left_child, index, bsf, pq, lock_queue, tnumber, n_pqueue);
            if (node->right_child != nullptr && node->right_child->isax_cardinalities != nullptr)
                insert_tree_node_m_hybridpqueueBreakpolyroot(paa, node->right_child, index, bsf, pq, lock_queue, tnumber, n_pqueue);
        }
    }

    float nodedistance(float *paa, sax_type *sax,
                       sax_type *sax_cardinalities,
                       sax_type max_bit_cardinality,
                       int max_cardinality,
                       int number_of_segments,
                       int min_val,
                       int max_val,
                       float ratio_sqrt);

    void calculate_node_topk_SING(isax_index *index, isax_node *node, ts_type *query, ts_type *paa,
                                  pqueue_bsf *pq_bsf, pthread_rwlock_t *lock_queue, float *rawfile)
    {
        if (node == nullptr || node->buffer == nullptr)
            return;
        for (int i = 0; i < node->buffer->partial_buffer_size; i++)
        {
            if (node->buffer->partial_position_buffer == nullptr || node->buffer->partial_position_buffer[i] == nullptr)
                continue;
            if (node->buffer->lbdarray != nullptr && node->buffer->lbdarray[i] > pq_bsf->knn[pq_bsf->k - 1])
                continue;
            float dist = ts_euclidean_distance_SIMD(query, &(rawfile[*node->buffer->partial_position_buffer[i]]),
                                                    index->settings->timeseries_size, pq_bsf->knn[pq_bsf->k - 1]);
            if (dist <= pq_bsf->knn[pq_bsf->k - 1])
            {
                pthread_rwlock_wrlock(lock_queue);
                pqueue_bsf_insert(pq_bsf, dist, (long int)(*node->buffer->partial_position_buffer[i] / index->settings->timeseries_size), node);
                pthread_rwlock_unlock(lock_queue);
            }
        }
    }

    void calculate_node_cal_topk_inmemory(isax_index *index, isax_node *node, ts_type *query, ts_type *paa,
                                          pqueue_bsf *pq_bsf, pthread_rwlock_t *lock_queue, float *rawfile)
    {
        if (node == nullptr || node->buffer == nullptr)
            return;

        int i;
        for (i = 0; i < node->buffer->full_buffer_size; i++)
        {
            float dist = ts_euclidean_distance_SIMD(query, node->buffer->full_ts_buffer[i],
                                                    index->settings->timeseries_size, pq_bsf->knn[pq_bsf->k - 1]);
            if (dist <= pq_bsf->knn[pq_bsf->k - 1])
            {
                long int pos = 0;
                if (node->buffer->full_position_buffer != nullptr &&
                    node->buffer->full_position_buffer[i] != nullptr)
                    pos = (long int)(*node->buffer->full_position_buffer[i] / index->settings->timeseries_size);
                pthread_rwlock_wrlock(lock_queue);
                pqueue_bsf_insert(pq_bsf, dist, pos, node);
                pthread_rwlock_unlock(lock_queue);
            }
        }

        for (i = 0; i < node->buffer->tmp_full_buffer_size; i++)
        {
            float dist = ts_euclidean_distance_SIMD(query, node->buffer->tmp_full_ts_buffer[i],
                                                    index->settings->timeseries_size, pq_bsf->knn[pq_bsf->k - 1]);
            if (dist <= pq_bsf->knn[pq_bsf->k - 1])
            {
                long int pos = 0;
                if (node->buffer->tmp_full_position_buffer != nullptr &&
                    node->buffer->tmp_full_position_buffer[i] != nullptr)
                    pos = (long int)(*node->buffer->tmp_full_position_buffer[i] / index->settings->timeseries_size);
                pthread_rwlock_wrlock(lock_queue);
                pqueue_bsf_insert(pq_bsf, dist, pos, node);
                pthread_rwlock_unlock(lock_queue);
            }
        }

        for (i = 0; i < node->buffer->partial_buffer_size; i++)
        {
            float distmin = minidist_paa_to_isax_raw_SING_SIMD(paa, node->buffer->partial_sax_buffer[i],
                                                               index->settings->max_sax_cardinalities,
                                                               index->settings->sax_bit_cardinality,
                                                               index->settings->sax_alphabet_cardinality,
                                                               index->settings->paa_segments, MINVAL, MAXVAL,
                                                               index->settings->mindist_sqrt);
            if (distmin <= pq_bsf->knn[pq_bsf->k - 1])
            {
                float dist = ts_euclidean_distance_SIMD(query, &(rawfile[*node->buffer->partial_position_buffer[i]]),
                                                        index->settings->timeseries_size, pq_bsf->knn[pq_bsf->k - 1]);
                if (dist <= pq_bsf->knn[pq_bsf->k - 1])
                {
                    pthread_rwlock_wrlock(lock_queue);
                    pqueue_bsf_insert(pq_bsf, dist, (long int)(*node->buffer->partial_position_buffer[i] / index->settings->timeseries_size), node);
                    pthread_rwlock_unlock(lock_queue);
                }
            }
        }
    }

    query_result exact_DTW_SING_new(ts_type *ts, ts_type *paa, ts_type *paaU, ts_type *paaL,
                                    isax_index *index, float minimum_distance, int min_checked_leaves,
                                    float *qts, float *gqts, float *gqtsU, float *gqtsL,
                                    sax_type *gsaxarray, short *positionmap, short *gpositionmap,
                                    float *gdictionary, node_list *nodelist,
                                    unsigned long int *offsetarray, int **arrangenodearray,
                                    int *arrangenodearraynumber, int warpWind,
                                    int N_PQUEUE, int maxquerythread, int chunknumber,
                                    float *rawfile) 
    {   
        
        //query_result approximate_result = approximate_search_inmemory_messi(ts, paa, index);
            //printf("this is check point 02\n");

        query_result approximate_result = approximate_DTW_inmemory_sing(ts, paa, index, warpWind, rawfile);
        //printf("this is check point 1\n");
        int tight_bound = index->settings->tight_bound;
        int aggressive_check = index->settings->aggressive_check;
        
        bool labelvalue=false;
        // Early termination...
        if (approximate_result.distance == 0) {
            return approximate_result;
        }
        float *lowerLemire= (float *)malloc(sizeof(float)*index->settings->timeseries_size);
        float *upperLemire= (float *)malloc(sizeof(float)*index->settings->timeseries_size);
        lower_upper_lemire(ts,index->settings->timeseries_size,warpWind,lowerLemire,upperLemire);

        if(approximate_result.distance == FLT_MAX || min_checked_leaves > 1) {
            //approximate_result = refine_answer_inmemory(ts, paa, index, approximate_result, minimum_distance, min_checked_leaves);
        }
        query_result bsf_result = approximate_result;
        pqueue_t **allpq=(pqueue_t**)malloc(sizeof(pqueue_t*)*N_PQUEUE);


        pthread_mutex_t ququelock[N_PQUEUE];
        int queuelabel[N_PQUEUE];

        query_result *do_not_remove = &approximate_result;

        //printf("this is check point 2\n");

        if(approximate_result.node != NULL) {
            // Insert approximate result in heap.
            //pqueue_insert(pq, &approximate_result);
            //GOOD: if(approximate_result.node->filename != NULL)
            //GOOD: printf("POPS: %.5lf\t", approximate_result.distance);
        }
        // Insert all root nodes in heap.
        isax_node *current_root_node = index->first_node;

        pthread_t threadid[maxquerythread],threadid2[maxquerythread];
        SING_workerdata workerdata[maxquerythread];
        pthread_mutex_t lock_queue=PTHREAD_MUTEX_INITIALIZER,lock_current_root_node=PTHREAD_MUTEX_INITIALIZER;
        pthread_rwlock_t lock_bsf=PTHREAD_RWLOCK_INITIALIZER;
        pthread_barrier_t lock_barrier,lock_barrierfirst;
        pthread_barrier_init(&lock_barrier, NULL, maxquerythread+1);
        pthread_barrier_init(&lock_barrierfirst, NULL, maxquerythread+1);

        int queueoffset[N_PQUEUE];
        unsigned long int gpuoffset=0;
        int loopnumber = chunknumber;
        for (int i = 0; i < N_PQUEUE; i++)
        {
                    allpq[i]=pqueue_init(index->settings->root_nodes_size/N_PQUEUE,
                                   cmp_pri, get_pri, set_pri, get_pos, set_pos);
                    pthread_mutex_init(&ququelock[i], NULL);
                    queuelabel[i]=1;
        }

        bool *activechunk=(bool*)malloc(sizeof(bool)*(chunknumber+1));
        int *chunkcounter=(int*)malloc(sizeof(int)*(chunknumber));
        int *chunkfinishcounter=(int*)malloc(sizeof(int)*(chunknumber));

        for (int i = 0; i < chunknumber; i++)
        {
            activechunk[i]=false;
            chunkcounter[i]=0;
            chunkfinishcounter[i]=0;
        }
        activechunk[loopnumber]=true;
        gap_workerdata gapworkerdata[maxquerythread];
        int startnode=nodelist->node_amount, stopnode=0,nodecounter=0,nodecounter2=nodelist->node_amount-1;//, *gapstartnode,*gapstopnode
        bool *activenode=(bool*)malloc(sizeof(bool)*nodelist->node_amount);
        uni_workerdata uniworkerdata[maxquerythread];
        int node_counter=0;
        unsigned long gpuoffsetnum=0;
        float shortrate=5000.0f/approximate_result.distance;
        //printf("the shortrate is %f\n",shortrate);
        for (int i = 0; i < maxquerythread; i++)
        {
            uniworkerdata[i].nodelist = nodelist->nlist;
            uniworkerdata[i].amountnode = nodelist->node_amount;
            uniworkerdata[i].startnode = &startnode;
            uniworkerdata[i].stopnode = &stopnode; // *gapstartnode,*gapstopnode;
            uniworkerdata[i].chunkcounter = chunkcounter;
            uniworkerdata[i].chunkfinishcounter = chunkfinishcounter;
            uniworkerdata[i].nodecounter = &nodecounter;
            uniworkerdata[i].arrangenodearray = arrangenodearray;
            uniworkerdata[i].arrangenodearraynumber = arrangenodearraynumber;
            uniworkerdata[i].nodecounter2 = &nodecounter2;
            uniworkerdata[i].index = index;
            uniworkerdata[i].bsf = approximate_result.distance;
            uniworkerdata[i].paa = paa;
            uniworkerdata[i].paaU = paaU;
            uniworkerdata[i].paaL = paaL;
            uniworkerdata[i].uo = upperLemire;
            uniworkerdata[i].lo = lowerLemire;
            uniworkerdata[i].lockposition = &lock_queue;
            uniworkerdata[i].offsetarray = offsetarray;
            uniworkerdata[i].activechunk = activechunk;
            uniworkerdata[i].chunknumber = chunknumber;
            uniworkerdata[i].activenode = activenode;
            uniworkerdata[i].workerstartnode = (i)*nodelist->node_amount / maxquerythread;
            uniworkerdata[i].workerstopnode = (i + 1) * nodelist->node_amount / maxquerythread;
            uniworkerdata[i].shortrate = shortrate;
            uniworkerdata[i].ts = ts;
            uniworkerdata[i].lock_queue = &lock_queue;
            uniworkerdata[i].lock_current_root_node = &lock_current_root_node;
            uniworkerdata[i].lock_bsf = &lock_bsf;
            uniworkerdata[i].minimum_distance = minimum_distance;
            uniworkerdata[i].node_counter = &node_counter;
            uniworkerdata[i].pq = allpq[i];
            uniworkerdata[i].bsf_result = &bsf_result;
            uniworkerdata[i].lock_barrier = &lock_barrier;
            uniworkerdata[i].lock_barrierfirst = &lock_barrierfirst;
            uniworkerdata[i].warpWind = warpWind;

            uniworkerdata[i].alllock = ququelock;
            uniworkerdata[i].allqueuelabel = queuelabel;
            uniworkerdata[i].allpq = allpq;
            uniworkerdata[i].startqueuenumber = (i % N_PQUEUE);
            uniworkerdata[i].lbdmapshort = positionmap;
            uniworkerdata[i].labelvalue = &labelvalue;
            uniworkerdata[i].gpuoffset = &gpuoffset;
            uniworkerdata[i].activenode = activenode;
            uniworkerdata[i].rawfile = rawfile;
            uniworkerdata[i].n_pqueue = N_PQUEUE;
        }
        uniworkerdata[maxquerythread-1].workerstopnode=nodelist->node_amount;
        //printf("this is check point 3\n");

        for (int i = 0; i < maxquerythread; i++)
        {

            pthread_create(&(threadid2[i]),NULL,uniworkershort_v4dtw,(void*)&(uniworkerdata[i]));
        }
       // cudaStream_t *streams=LBDfloatstreamGPUdevidebegin( qts, gqts,datastartnumber,datasizerun,loopnumber);
        pthread_barrier_wait(&lock_barrierfirst);
        if(activechunk[loopnumber])
            activechunk[loopnumber-1]=true;
        //printf("this is check point 4\n");

            //printf("this is check point 5\n");
        LBDshortstreamGPUinsidedynamicratev2dtw(gsaxarray, positionmap, paaU, paaL, gqtsU, gqtsL, approximate_result.distance,index->sax_cache_size,gpositionmap,index->settings->paa_segments,index->settings->mindist_sqrt, &gpuoffset,shortrate,loopnumber,activechunk);

            pthread_barrier_wait(&lock_barrier);
        //printf("this is check point 6\n");

            //for (int i = 0; i < loopnumber; i++)
            {
                //if(activechunk[i])
                {
                 //LBDshortstreamGPUinsidedynamicrate(&gsaxarray[i*index->sax_cache_size/loopnumber*index->settings->paa_segments], &positionmap[i*index->sax_cache_size/loopnumber], paa, gqts, approximate_result.distance,index->sax_cache_size/loopnumber,&gpositionmap[i*index->sax_cache_size/loopnumber],index->settings->paa_segments,index->settings->mindist_sqrt, &gpuoffsetnum,shortrate);
                 //LBDcalculationnumber+=index->sax_cache_size/loopnumber;
                }
               // __sync_fetch_and_add(&gpuoffset,index->sax_cache_size/loopnumber);
                //gpuoffset=(i+1)*index->sax_cache_size/loopnumber;
            }
            //LBDfloatstreamGPU(gsaxarray, positionmap, paa, gqts, approximate_result.distance,index->sax_cache_size,gpositionmap, gdictionary);
            
        
        for (int i = 0; i < maxquerythread; i++)
        {
            pthread_join(threadid2[i],NULL); 
        }

        
        //printf("this is check point 7\n");










        

        // Free the nodes that where not popped.
        // Free the priority queue.
        pthread_barrier_destroy(&lock_barrier);

        //pqueue_free(pq);
        for (int i = 0; i < N_PQUEUE; i++)
        {
            pqueue_free(allpq[i]);
        }
        free(allpq);
        free(activechunk);
        free(chunkcounter);
        free(chunkfinishcounter);
        //free(rfdata);
        //LBDcalculationnumber=(datasizerun-datastartnumber)*index->sax_cache_size/loopnumber;
        //printf("the number of node added is \t%ld\t\t and real distance calculation is \t%ld\n ",LBDcalculationnumber,RDcalculationnumber);
        //printf("the number of lower bound distance calculation is \t%ld\t\t and the delete node is \t%ld\n ",LBDcalculationnumber,RDcalculationnumber);

        return bsf_result;

        // Free the nodes that where not popped.

    }

    void *exact_knn_SING_worker(void *arg)
    {
        SING_workerdata *wd = (SING_workerdata *)arg;
        isax_index *index = wd->index;
        ts_type *paa = wd->paa;
        ts_type *ts = wd->ts;
        pqueue_bsf *pq_bsf = wd->pq_bsf;
        float minimum_distance = wd->minimum_distance;
        bool *activenode = wd->activenode;
        int startqueuenumber = wd->startqueuenumber;
        int tnumber = startqueuenumber;
        pthread_mutex_t *alllock = wd->alllock;
        int n_pqueue = wd->n_pqueue;
        float *rawfile = wd->rawfile;

        float bsfdistance = pq_bsf->knn[pq_bsf->k - 1];
        query_result *n;
        isax_node *current_root_node;
        bool finished = true;
        int current_root_node_number;

        while (1)
        {
            current_root_node_number = __sync_fetch_and_add(wd->node_counter, 1);
            if (current_root_node_number >= wd->amountnode)
                break;
            current_root_node = wd->nodelist[current_root_node_number];
            if (activenode[current_root_node_number] == true)
                insert_tree_node_m_hybridpqueueBreakpolyroot(paa, current_root_node, index, bsfdistance,
                                                             wd->allpq, wd->alllock, &tnumber, n_pqueue);
        }
        pthread_barrier_wait(wd->lock_barrier);

        int offset = startqueuenumber;
        while (1)
        {
            finished = true;
            for (int i = 0; i < n_pqueue; i++)
            {
                int queuenumber = (i + offset) % n_pqueue;
                if (wd->allqueuelabel[queuenumber] == 1)
                {
                    finished = false;
                    bsfdistance = pq_bsf->knn[pq_bsf->k - 1];
                    pthread_mutex_lock(&alllock[queuenumber]);
                    n = (query_result *)pqueue_pop(wd->allpq[queuenumber]);
                    if (n == NULL)
                    {
                        wd->allqueuelabel[queuenumber] = 0;
                        pthread_mutex_unlock(&alllock[queuenumber]);
                        continue;
                    }
                    pthread_mutex_unlock(&alllock[queuenumber]);

                    if (!n->node->is_leaf &&
                        (n->distance > bsfdistance || n->distance > minimum_distance))
                    {
                        wd->allqueuelabel[queuenumber] = 0;
                        free(n);
                        continue;
                    }
                    if (n->node->is_leaf)
                    {
                        if (n->node->buffer != nullptr && wd->gpuoffset != nullptr &&
                            n->node->buffer->arrayposition < *wd->gpuoffset)
                            calculate_node_topk_SING(index, n->node, ts, paa, pq_bsf, wd->lock_bsf, rawfile);
                        else
                            calculate_node_cal_topk_inmemory(index, n->node, ts, paa, pq_bsf, wd->lock_bsf, rawfile);
                    }
                    free(n);
                }
                else if (wd->allqueuelabel[queuenumber] == 0)
                {
                    pthread_mutex_lock(&alllock[queuenumber]);
                    n = (query_result *)pqueue_pop(wd->allpq[queuenumber]);
                    pthread_mutex_unlock(&alllock[queuenumber]);
                    if (n != NULL)
                        free(n);
                    else
                        wd->allqueuelabel[queuenumber] = 2;
                }
            }
            if (finished)
                break;
        }
        pthread_exit(NULL);
        return nullptr;
    }
#if !SING_CUDA_ENABLED
    void LBDfloatstreamGPU(sax_type *gsaxarray, float *positionmap, ts_type *paa, float *gqts, float bsf,
                           unsigned long size, float *gpositionmap, int paa_segments, float mindist_sqrt)
    {
        (void)gsaxarray;
        (void)positionmap;
        (void)paa;
        (void)gqts;
        (void)bsf;
        (void)size;
        (void)gpositionmap;
        (void)paa_segments;
        (void)mindist_sqrt;
        
    }
#endif

    first_buffer_layer2 *initialize_simrec(int initial_buffer_size, int number_of_buffers,
                                           int max_total_buffers_size, isax_index *index)
    {
        first_buffer_layer2 *fbl = (first_buffer_layer2 *)malloc(sizeof(first_buffer_layer2));

        fbl->max_total_size = max_total_buffers_size;
        fbl->initial_buffer_size = initial_buffer_size;
        fbl->number_of_buffers = number_of_buffers;

        fbl->soft_buffers = (fbl_soft_buffer2 *)malloc(sizeof(fbl_soft_buffer2) * (size_t)number_of_buffers);
        fbl->current_record_index = 0;
        fbl->current_record = NULL;
        fbl->hard_buffer = NULL;

        for (int i = 0; i < number_of_buffers; i++)
        {
            fbl->soft_buffers[i].initialized = 0;
            fbl->soft_buffers[i].max_buffer_size = 0;
            fbl->soft_buffers[i].buffer_size = 0;
            fbl->soft_buffers[i].node = NULL;
            fbl->soft_buffers[i].sax_records = NULL;
            fbl->soft_buffers[i].pos_records = NULL;
        }
        return fbl;
    }

    void *index_creation_worker2_inmemory(void *transferdata)
    {
        sax_type *sax = (sax_type *)malloc(sizeof(sax_type) * ((buffer_data_inmemory *)transferdata)->index->settings->paa_segments);
        unsigned long start_number = ((buffer_data_inmemory *)transferdata)->start_number;
        unsigned long stop_number = ((buffer_data_inmemory *)transferdata)->stop_number;
        file_position_type *pos = (file_position_type *)malloc(sizeof(file_position_type));
        isax_index *index = ((buffer_data_inmemory *)transferdata)->index;
        ts_type *ts = (ts_type *)malloc(sizeof(ts_type) * (size_t)index->settings->timeseries_size);
        int paa_segments = ((buffer_data_inmemory *)transferdata)->index->settings->paa_segments;

        (void)start_number;
        (void)stop_number;
        (void)paa_segments;

        float *raw_file = ((buffer_data_inmemory *)transferdata)->ts;
        (void)raw_file;

        free(pos);
        free(sax);
        free(ts);

        int j, c = 1, k;
        (void)c;
        isax_node_record *r = (isax_node_record *)malloc(sizeof(isax_node_record));

        int worker_id = ((buffer_data_inmemory *)transferdata)->workernumber;
        (void)worker_id;

        while (1)
        {
            j = __sync_fetch_and_add(((buffer_data_inmemory *)transferdata)->node_counter, 1);
            if (j >= index->fbl->number_of_buffers)
                break;

            fbl_soft_buffer2 *current_fbl_node = &((first_buffer_layer2 *)(index->fbl))->soft_buffers[j];
            if (!current_fbl_node->initialized)
                continue;

            if (current_fbl_node->buffer_size > 0)
            {
                for (k = 0; k < current_fbl_node->buffer_size; k++)
                {
                    r->sax = &(index->sax_cache[current_fbl_node->pos_records[k] / index->settings->timeseries_size * index->settings->paa_segments]);
                    r->position = &current_fbl_node->pos_records[k];
                    r->insertion_mode = (insertion_mode)(NO_TMP | PARTIAL);
                    add_record_to_node(index, current_fbl_node->node, r, 1);
                }
                flush_subtree_leaf_buffers_inmemory(index, current_fbl_node->node);
            }
        }
        free(r);
        return nullptr;
    }

    Sing::Sing(DistanceType distance_type)
        : SimilaritySearchAlgorithm(distance_type)
    {
    }

    void Sing::setNumThreads(int num_threads)
    {
        int max_threads = omp_get_max_threads();
        int n = num_threads;

        if (num_threads > max_threads)
        {
            std::cerr << "[Warning] " << num_threads
                      << " threads exceeds max available " << max_threads << ". Using the max threads available.\n";
            n = max_threads;
        }
        else if (num_threads < 1)
        {
            std::cerr << "[Warning] Thread count must be >= 1. Using 1.\n";
            n = 1;
        }

        this->num_threads = n;
        
        this->search_workers = n;
    }

    int Sing::getNumThreads() const
    {
        return this->num_threads;
    }

    void Sing::buildIndex(DataSource *data_source)
    {
        this->dim = data_source->getDim();
        this->n_database = data_source->getTotalRecords();

        if (this->n_database == 0)
        {
            data_source->reset();
            idx_t count = 0;
            float *dummy = new float[this->dim];
            while (data_source->nextRecord(dummy))
            {
                count++;
            }
            delete[] dummy;
            this->n_database = count;
            data_source->reset();
        }

        data_source->reset();
        this->database = new float[this->n_database * this->dim];
        float *record = new float[this->dim];
        idx_t ri = 0;
        while (data_source->nextRecord(record))
        {
            std::copy(record, record + this->dim, this->database + ri * this->dim);
            ri++;
        }
        delete[] record;

        const char *index_path = "";                                               
        this->index_settings = isax_index_settings_init(index_path,                
                                                        this->dim,                 
                                                        this->paa_segments,        
                                                        this->sax_cardinality,     
                                                        this->leaf_size,           
                                                        this->min_leaf_size,       
                                                        this->initial_lbl_size,    
                                                        this->flush_limit,         
                                                        this->initial_fbl_size,    
                                                        this->total_loaded_leaves, 
                                                        this->tight_bound,         
                                                        this->aggressive_check,    
                                                        1, 1);                     

        this->index = isax_index_init_inmemory(this->index_settings);
        this->index->sax_cache_size = this->n_database;
        index_creation_gpu(this->database, this->n_database, this->index);
    }

    void Sing::index_creation_gpu(float *dataset, idx_t dataset_size, isax_index *idx)
    {
        ts_type *rawfile = (ts_type *)dataset;
        long int ts_num = (long int)dataset_size;
        int maxquerythread = this->index_workers;

        // build-index debug logging removed (handled at demo level)
        idx->sax_file = NULL;

        long int ts_loaded = 0;
        (void)ts_loaded;
        int i;
        int node_counter = 0;
        pthread_t *threadid = (pthread_t *)malloc(sizeof(pthread_t) * (size_t)maxquerythread);
        buffer_data_inmemory *input_data = (buffer_data_inmemory *)malloc(sizeof(buffer_data_inmemory) * (size_t)maxquerythread);

        idx->sax_cache = (sax_type *)malloc(sizeof(sax_type) * (size_t)(idx->settings->paa_segments * ts_num));

        if (idx->settings->raw_filename != nullptr)
        {
            free(idx->settings->raw_filename);
            idx->settings->raw_filename = nullptr;
        }
        idx->settings->raw_filename = (char *)malloc(256);
        if (idx->settings->raw_filename != nullptr)
            strcpy(idx->settings->raw_filename, "inmemory");

        pthread_mutex_t lock_record = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_t lockfbl = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_t lock_index = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_t lock_firstnode = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_t lock_disk = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_t *lockcbl = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * LOCK_SIZE);
        pthread_barrier_t lock_barrier1;
        pthread_barrier_t lock_barrier2;
        pthread_barrier_init(&lock_barrier1, NULL, maxquerythread);
        pthread_barrier_init(&lock_barrier2, NULL, maxquerythread);

        destroy_fbl(idx->fbl);
        idx->fbl = (first_buffer_layer *)initialize_simrec(
            idx->settings->initial_fbl_buffer_size,
            (int)pow(2, idx->settings->paa_segments),
            idx->settings->max_total_buffer_size + DISK_BUFFER_SIZE * (PROGRESS_CALCULATE_THREAD_NUMBER - 1),
            idx);

        if (idx->fbl == nullptr)
        {
            fprintf(stderr, "Sing::index_creation_gpu: initialize_simrec failed\n");
            throw std::runtime_error("Sing::index_creation_gpu: initialize_simrec failed");
        }

        for (i = 0; i < LOCK_SIZE; i++)
            pthread_mutex_init(&lockcbl[i], NULL);

#pragma omp parallel for num_threads(maxquerythread)
        for (long int j = 0; j < ts_num; j++)
        {
            sax_type *sax = &(idx->sax_cache[j * idx->settings->paa_segments]);
            if (sax_from_ts(&(rawfile[j * idx->settings->timeseries_size]), sax,
                            idx->settings->ts_values_per_paa_segment,
                            idx->settings->paa_segments, idx->settings->sax_alphabet_cardinality,
                            idx->settings->sax_bit_cardinality) == SUCCESS)
            {
                root_mask_type first_bit_mask = 0x00;
                CREATE_MASK(first_bit_mask, idx, sax);
                fbl_soft_buffer2 *current_buffer = &((first_buffer_layer2 *)(idx->fbl))->soft_buffers[(int)first_bit_mask];
                __sync_fetch_and_add(&(current_buffer->max_buffer_size), 1);
            }
        }

#pragma omp parallel for num_threads(maxquerythread)
        for (i = 0; i < ((first_buffer_layer2 *)(idx->fbl))->number_of_buffers; i++)
        {
            fbl_soft_buffer2 *current_buffer = &((first_buffer_layer2 *)(idx->fbl))->soft_buffers[i];
            if (current_buffer->max_buffer_size != 0)
            {
                current_buffer->initialized = 1;
                current_buffer->sax_records = (sax_type *)malloc(sizeof(sax_type) * idx->settings->paa_segments * (size_t)current_buffer->max_buffer_size);
                current_buffer->pos_records = (file_position_type *)malloc(sizeof(file_position_type) * (size_t)current_buffer->max_buffer_size);
                current_buffer->node = isax_root_node_init((root_mask_type)i, idx->settings->initial_leaf_buffer_size);
                current_buffer->node->is_leaf = 1;
            }
        }

#pragma omp parallel for num_threads(maxquerythread)
        for (long int j = 0; j < ts_num; j++)
        {
            root_mask_type first_bit_mask = 0x00;
            sax_type *sax = &(idx->sax_cache[j * idx->settings->paa_segments]);
            CREATE_MASK(first_bit_mask, idx, sax);
            fbl_soft_buffer2 *current_buffer = &((first_buffer_layer2 *)(idx->fbl))->soft_buffers[(int)first_bit_mask];
            int buffersize = __sync_fetch_and_add(&(current_buffer->buffer_size), 1);
            memcpy(&current_buffer->sax_records[buffersize * idx->settings->paa_segments], sax, sizeof(sax_type) * (size_t)idx->settings->paa_segments);
            current_buffer->pos_records[buffersize] = (file_position_type)(j * idx->settings->timeseries_size);
        }

        for (i = 0; i < maxquerythread; i++)
        {
            input_data[i].index = idx;
            input_data[i].lock_fbl = &lockfbl;
            input_data[i].lock_record = &lock_record;
            input_data[i].lock_cbl = lockcbl;
            input_data[i].lock_firstnode = &lock_firstnode;
            input_data[i].lock_index = &lock_index;
            input_data[i].lock_nodeconter = nullptr;
            input_data[i].lock_disk = &lock_disk;
            input_data[i].ts = rawfile;
            input_data[i].workernumber = i;
            input_data[i].total_workernumber = maxquerythread;
            input_data[i].start_number = i * (int)(ts_num / maxquerythread);
            input_data[i].stop_number = (i + 1) * (int)(ts_num / maxquerythread);
            input_data[i].node_counter = &node_counter;
            input_data[i].lock_barrier1 = &lock_barrier1;
            input_data[i].lock_barrier2 = &lock_barrier2;
            input_data[i].shared_start_number = nullptr;
            input_data[i].read_block_length = this->read_block_length;
            input_data[i].finished = false;
            input_data[i].nodeid = nullptr;
        }
        input_data[maxquerythread - 1].start_number = (maxquerythread - 1) * (int)(ts_num / maxquerythread);
        input_data[maxquerythread - 1].stop_number = (int)ts_num;

        for (i = 0; i < maxquerythread; i++)
            pthread_create(&(threadid[i]), NULL, index_creation_worker2_inmemory, (void *)&(input_data[i]));
        for (i = 0; i < maxquerythread; i++)
            pthread_join(threadid[i], NULL);

        idx->sax_cache_size = (unsigned long)ts_num;

        free(lockcbl);
        free(threadid);
        free(input_data);
        pthread_barrier_destroy(&lock_barrier1);
        pthread_barrier_destroy(&lock_barrier2);
    }

    void Sing::searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        if (this->distance_type == DistanceType::L2_SQUARED && this->index != nullptr)
        {
            searchIndexL2Square(query, n_query, k, I, D);
            return;
        }
        if (this->distance_type == DistanceType::DTW)
        {
            if (k != 1)
            {
                fprintf(stderr, "[Sing][DTW] Warning: only k = 1 is supported for DTW in this version; proceeding with k = 1.\n");
            }
            searchIndex_DTW(query, n_query, 1, I, D);
            return;
        }

#pragma omp parallel num_threads(num_threads)
        {
#pragma omp for
            for (idx_t qi = 0; qi < n_query; qi++)
            {
                std::priority_queue<std::pair<float, idx_t>> pq;
                const float *q_vec = query + qi * this->dim;
                float bound = FLT_MAX;
                for (idx_t dbi = 0; dbi < this->n_database; ++dbi)
                {
                    const float *db_vec = this->database + dbi * this->dim;
                    float dist = this->distance_computer->compute_dist(const_cast<float *>(q_vec),
                                                                       const_cast<float *>(db_vec),
                                                                       this->dim, bound);
                    if ((idx_t)pq.size() < k)
                        pq.emplace(dist, dbi);
                    else if (dist < pq.top().first)
                    {
                        pq.pop();
                        pq.emplace(dist, dbi);
                        bound = pq.top().first;
                    }
                }
                for (idx_t j = k; j > 0; --j)
                {
                    D[qi * k + (j - 1)] = pq.top().first;
                    I[qi * k + (j - 1)] = pq.top().second;
                    pq.pop();
                }
            }
        }
    }

    void Sing::searchIndex_DTW(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        isax_index *index = this->index;
        first_buffer_layer2 *fbl2 = (first_buffer_layer2 *)index->fbl;
        if (fbl2 == nullptr || fbl2->soft_buffers == nullptr)
        {
            fprintf(stderr, "[Sing searchIndex_DTW] fbl not available, running brute-force.\n");
#pragma omp parallel num_threads(num_threads)
            {
#pragma omp for
                for (idx_t qi = 0; qi < n_query; qi++)
                {
                    std::priority_queue<std::pair<float, idx_t>> pq;
                    const float *q_vec = query + qi * this->dim;
                    float bound = FLT_MAX;
                    for (idx_t dbi = 0; dbi < this->n_database; ++dbi)
                    {
                        const float *db_vec = this->database + dbi * this->dim;
                        float dist = this->distance_computer->compute_dist(const_cast<float *>(q_vec),
                                                                           const_cast<float *>(db_vec),
                                                                           this->dim, bound);
                        if ((idx_t)pq.size() < k)
                            pq.emplace(dist, dbi);
                        else if (dist < pq.top().first)
                        {
                            pq.pop();
                            pq.emplace(dist, dbi);
                            bound = pq.top().first;
                        }
                    }
                    for (idx_t j = k; j > 0; --j)
                    {
                        D[qi * k + (j - 1)] = pq.top().first;
                        I[qi * k + (j - 1)] = pq.top().second;
                        pq.pop();
                    }
                }
            }
            return;
        }

        node_list nodelist;
        int root_nodes = index->settings->root_nodes_size;
        nodelist.nlist = (isax_node **)malloc(sizeof(isax_node *) * (size_t)root_nodes);
        nodelist.node_amount = 0;
        for (int i = 0; i < root_nodes; i++)
        {
            fbl_soft_buffer2 *current_buffer = &fbl2->soft_buffers[i];
            if (current_buffer->initialized == 1 && current_buffer->node != nullptr)
            {
                nodelist.nlist[nodelist.node_amount++] = current_buffer->node;
            }
        }
        if (nodelist.node_amount > 0)
            index->first_node = nodelist.nlist[0];

        const int paa_segments = index->settings->paa_segments;
        const int ts_values_per_paa = index->settings->ts_values_per_paa_segment;
        ts_type *paa = (ts_type *)malloc(sizeof(ts_type) * (size_t)paa_segments);
        ts_type *paaU = (ts_type *)malloc(sizeof(ts_type) * (size_t)paa_segments);
        ts_type *paaL = (ts_type *)malloc(sizeof(ts_type) * (size_t)paa_segments);
        ts_type *upperLemire = (ts_type *)malloc(sizeof(ts_type) * (size_t)index->settings->timeseries_size);
        ts_type *lowerLemire = (ts_type *)malloc(sizeof(ts_type) * (size_t)index->settings->timeseries_size);

        short *posbitmap = nullptr;
        short *gposbitmap = nullptr;
        float *qts = nullptr;
        float *gqts = nullptr;
        float *gqtsU = nullptr;
        float *gqtsL = nullptr;
        sax_type *gsaxarray = nullptr;

#if SING_CUDA_ENABLED
        initialdevice();
        posbitmap = initialposbitmapshort(posbitmap, (unsigned long)index->sax_cache_size);
        gposbitmap = initialgposbitmapshort(gposbitmap, (unsigned long)index->sax_cache_size);
        qts = (float *)malloc(sizeof(float) * (size_t)paa_segments);
        gqts = initialgqts(gqts);
        gqtsU = initialgqts(gqtsU);
        gqtsL = initialgqts(gqtsL);
        sax_type *gsaxarray_d = (sax_type *)initialgsaxarray(nullptr, (unsigned long)index->sax_cache_size);
#else
        posbitmap = nullptr;
        gposbitmap = nullptr;
        qts = (float *)malloc(sizeof(float) * (size_t)paa_segments);
        gqts = nullptr;
        gqtsU = nullptr;
        gqtsL = nullptr;
        sax_type *gsaxarray_d = nullptr;
#endif

        unsigned long int *offsetarray = (unsigned long int *)malloc(sizeof(unsigned long int) * (size_t)(nodelist.node_amount + 1));
        sax_type *saxarrarysort = (sax_type *)malloc(sizeof(sax_type) * (size_t)index->sax_cache_size * (size_t)paa_segments);
        unsigned long int currentposition = 0;
        for (int i = 0; i < nodelist.node_amount; i++)
        {
            offsetarray[i] = currentposition;
            pass_tree_node_m_short(nodelist.nlist[i], index, NULL, &currentposition, index->sax_cache, saxarrarysort, posbitmap);
        }
        offsetarray[nodelist.node_amount] = (unsigned long int)index->sax_cache_size;

        if (currentposition > 0)
        {
#if SING_CUDA_ENABLED
            gpumemcpy(gsaxarray_d, saxarrarysort, (unsigned long)index->sax_cache_size);
            gsaxarray = gsaxarray_d;
#else
            gsaxarray = saxarrarysort;
#endif
        }
        else
        {
            for (int i = 0; i < nodelist.node_amount; i++)
                offsetarray[i] = (unsigned long int)((unsigned long long)index->sax_cache_size * i / nodelist.node_amount);
            offsetarray[nodelist.node_amount] = (unsigned long int)index->sax_cache_size;
#if SING_CUDA_ENABLED
            gpumemcpy(gsaxarray_d, index->sax_cache, (unsigned long)index->sax_cache_size);
            gsaxarray = gsaxarray_d;
#else
            gsaxarray = index->sax_cache;
#endif
            free(saxarrarysort);
            saxarrarysort = nullptr;
        }

        int chunknumber = 20;
        int *arrangenodearraynumber = (int *)malloc(sizeof(int) * (size_t)chunknumber);
        int *arrangernodearraynumbertemp = (int *)malloc(sizeof(int) * (size_t)chunknumber);
        int **arrangenodearray = (int **)malloc(sizeof(int *) * (size_t)chunknumber);
        for (int i = 0; i < chunknumber; i++)
        {
            arrangenodearraynumber[i] = 0;
            arrangernodearraynumbertemp[i] = 0;
        }

        for (int i = 0; i < nodelist.node_amount - 1; i++)
        {
            for (int j = (int)(offsetarray[i] / (index->sax_cache_size / chunknumber));
                 j <= (int)(offsetarray[i + 1] / (index->sax_cache_size / chunknumber)); j++)
            {
                arrangenodearraynumber[j]++;
            }
        }
        arrangenodearraynumber[chunknumber - 1]++;

        for (int i = 0; i < chunknumber; i++)
        {
            arrangenodearray[i] = (int *)malloc(sizeof(int) * (size_t)arrangenodearraynumber[i]);
        }

        for (int i = 0; i < nodelist.node_amount - 1; i++)
        {
            for (int j = (int)(offsetarray[i] / (index->sax_cache_size / chunknumber));
                 j <= (int)(offsetarray[i + 1] / (index->sax_cache_size / chunknumber)); j++)
            {
                arrangenodearray[j][arrangernodearraynumbertemp[j]] = i;
                arrangernodearraynumbertemp[j]++;
            }
        }
        arrangenodearray[chunknumber - 1][arrangernodearraynumbertemp[chunknumber - 1]] = nodelist.node_amount - 1;

        int N_PQUEUE = this->n_pqueue;
        int maxquerythread = this->search_workers;
        float minimum_distance = this->minimum_distance;
        int min_checked_leaves = this->min_checked_leaves;

        for (idx_t q_loaded = 0; q_loaded < n_query; q_loaded++)
        {
            ts_type *ts = (ts_type *)(query + q_loaded * this->dim);
            lower_upper_lemire(ts, index->settings->timeseries_size, this->warping_window, lowerLemire, upperLemire);
            paa_from_ts(upperLemire, paaU, paa_segments, ts_values_per_paa);
            paa_from_ts(lowerLemire, paaL, paa_segments, ts_values_per_paa);
            paa_from_ts(ts, paa, paa_segments, ts_values_per_paa);
            std::memcpy(qts, paa, sizeof(float) * (size_t)paa_segments);

       
            int chunknumber = 20;  // same value used in L2 search

            query_result result = exact_DTW_SING_new(ts, paa, paaU, paaL, index,
                                                     minimum_distance, min_checked_leaves,
                                                     qts, gqts, gqtsU, gqtsL,
                                                     gsaxarray, posbitmap, gposbitmap,
                                                     nullptr, &nodelist, offsetarray,
                                                     arrangenodearray, arrangenodearraynumber,
                                                     this->warping_window,
                                                     this->n_pqueue, this->search_workers, chunknumber,
                                                     this->database);

            // Use the final DTW result to recover the exact database index (k = 1).
            idx_t best_pos = 0;
            float best_dist = result.distance;
            if (result.node != nullptr)
            {
                best_dist = calculate_node_DTW2_inmemorysing(index, result.node, ts, upperLemire, lowerLemire,
                                                             paa, paaU, paaL, result.distance,
                                                             this->warping_window, this->database, &best_pos);
            }
            D[q_loaded * k] = best_dist;
            I[q_loaded * k] = best_pos;
        }

        free(arrangenodearraynumber);
        free(arrangernodearraynumbertemp);
        for (int i = 0; i < chunknumber; i++)
            free(arrangenodearray[i]);
        free(arrangenodearray);
        free(saxarrarysort);
        free(offsetarray);
        free(paa);
        free(paaU);
        free(paaL);
        free(upperLemire);
        free(lowerLemire);
        free(qts);
        free(nodelist.nlist);

#if SING_CUDA_ENABLED
        GPUfree(gsaxarray);
        GPUfree(gposbitmap);
        GPUfree(gqts);
        GPUfree(gqtsU);
        GPUfree(gqtsL);
        cudaFreeHost(posbitmap);
#endif
    }

    void Sing::searchIndexL2Square(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        isax_index *index = this->index;
        first_buffer_layer2 *fbl2 = (first_buffer_layer2 *)index->fbl;
        if (fbl2 == nullptr || fbl2->soft_buffers == nullptr)
        {
            fprintf(stderr, "[Sing searchIndexL2Square] fbl not available, running brute-force.\n");
#pragma omp parallel num_threads(num_threads)
            {
#pragma omp for
                for (idx_t qi = 0; qi < n_query; qi++)
                {
                    std::priority_queue<std::pair<float, idx_t>> pq;
                    const float *q_vec = query + qi * this->dim;
                    float bound = FLT_MAX;
                    for (idx_t dbi = 0; dbi < this->n_database; ++dbi)
                    {
                        const float *db_vec = this->database + dbi * this->dim;
                        float dist = this->distance_computer->compute_dist(const_cast<float *>(q_vec),
                                                                           const_cast<float *>(db_vec),
                                                                           this->dim, bound);
                        if ((idx_t)pq.size() < k)
                            pq.emplace(dist, dbi);
                        else if (dist < pq.top().first)
                        {
                            pq.pop();
                            pq.emplace(dist, dbi);
                            bound = pq.top().first;
                        }
                    }
                    for (idx_t j = k; j > 0; --j)
                    {
                        D[qi * k + (j - 1)] = pq.top().first;
                        I[qi * k + (j - 1)] = pq.top().second;
                        pq.pop();
                    }
                }
            }
            return;
        }

        node_list nodelist;
        int max_nodes = fbl2->number_of_buffers;
        nodelist.nlist = (isax_node **)malloc(sizeof(isax_node *) * (size_t)max_nodes);
        nodelist.node_amount = 0;
        for (int i = 0; i < max_nodes; i++)
        {
            fbl_soft_buffer2 *sb = &fbl2->soft_buffers[i];
            if (sb->initialized && sb->node != nullptr)
            {
                nodelist.nlist[nodelist.node_amount++] = sb->node;
            }
        }
        if (nodelist.node_amount > 0)
            index->first_node = nodelist.nlist[0];

        const int paa_segments = index->settings->paa_segments;
        const int ts_values_per_paa = index->settings->ts_values_per_paa_segment;
        ts_type *paa = (ts_type *)malloc(sizeof(ts_type) * (size_t)paa_segments);
        float *positionmap;
        float *gpositionmap;
        float *gqts;
        sax_type *gsaxarray;
#if SING_CUDA_ENABLED
        initialdevice();
        positionmap = (float *)initialposbitmapfloat(nullptr, (unsigned long)index->sax_cache_size);
        gpositionmap = (float *)initialgposbitmapfloat(nullptr, (unsigned long)index->sax_cache_size);
        gqts = (float *)initialgqts(nullptr);
        sax_type *gsaxarray_d = (sax_type *)initialgsaxarray(nullptr, (unsigned long)index->sax_cache_size);
#else
        positionmap = (float *)malloc(sizeof(float) * (unsigned long)index->sax_cache_size);
        gpositionmap = (float *)malloc(sizeof(float) * (unsigned long)index->sax_cache_size);
        gqts = (float *)malloc(sizeof(float) * (size_t)index->settings->timeseries_size);
#endif
        unsigned long int *offsetarray = (unsigned long int *)malloc(sizeof(unsigned long int) * (size_t)(nodelist.node_amount + 1));
        sax_type *saxarrarysort = (sax_type *)malloc(sizeof(sax_type) * (size_t)index->sax_cache_size * (size_t)paa_segments);
        unsigned long int currentposition = 0;
        for (int i = 0; i < nodelist.node_amount; i++)
        {
            offsetarray[i] = currentposition;
            pass_tree_node_m(nodelist.nlist[i], index, NULL, &currentposition, index->sax_cache, saxarrarysort, positionmap);
        }
        offsetarray[nodelist.node_amount] = currentposition;
        
        if (currentposition > 0)
        {
#if SING_CUDA_ENABLED
            gpumemcpy((singlib_sax_t *)gsaxarray_d, (const singlib_sax_t *)saxarrarysort, (unsigned long)index->sax_cache_size);
            gsaxarray = gsaxarray_d;
#else
            gsaxarray = saxarrarysort;
#endif
        }
        else
        {
            for (int i = 0; i < nodelist.node_amount; i++)
                offsetarray[i] = (unsigned long int)((unsigned long long)index->sax_cache_size * i / nodelist.node_amount);
            offsetarray[nodelist.node_amount] = (unsigned long int)index->sax_cache_size;
#if SING_CUDA_ENABLED
            gpumemcpy((singlib_sax_t *)gsaxarray_d, (const singlib_sax_t *)index->sax_cache, (unsigned long)index->sax_cache_size);
            gsaxarray = gsaxarray_d;
#else
            gsaxarray = index->sax_cache;
#endif
            free(saxarrarysort);
            saxarrarysort = nullptr;
        }

        int N_PQUEUE = this->n_pqueue;
        int maxquerythread = this->search_workers;
        float minimum_distance = this->minimum_distance;
        int min_checked_leaves = this->min_checked_leaves;

        for (idx_t q_loaded = 0; q_loaded < n_query; q_loaded++)
        {
            ts_type *ts = (ts_type *)(query + q_loaded * this->dim);
            paa_from_ts(ts, paa, paa_segments, ts_values_per_paa);
#if !SING_CUDA_ENABLED
            std::memcpy(gqts, ts, sizeof(float) * (size_t)index->settings->timeseries_size);
#endif

            pqueue_bsf *pq_bsf = pqueue_bsf_init((int)k);
            approximate_topk_SING(ts, paa, index, pq_bsf, this->database);
            int tight_bound = index->settings->tight_bound;
            int aggressive_check = index->settings->aggressive_check;
            (void)tight_bound;
            (void)aggressive_check;

            bool labelvalue = false;
            if (pq_bsf->knn[k - 1] == FLT_MAX || min_checked_leaves > 1)
            {
                refine_topk_answer_inmemory(ts, paa, index, pq_bsf, minimum_distance, min_checked_leaves, this->database);
            }
            pqueue_t **allpq = (pqueue_t **)malloc(sizeof(pqueue_t *) * (size_t)N_PQUEUE);
            pthread_mutex_t *ququelock = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * (size_t)N_PQUEUE);
            int *queuelabel = (int *)malloc(sizeof(int) * (size_t)N_PQUEUE);

            isax_node *current_root_node = index->first_node;
            (void)current_root_node;

            pthread_t *threadid = (pthread_t *)malloc(sizeof(pthread_t) * (size_t)maxquerythread);
            pthread_t *threadid2 = (pthread_t *)malloc(sizeof(pthread_t) * (size_t)maxquerythread);
            SING_workerdata *workerdata = (SING_workerdata *)malloc(sizeof(SING_workerdata) * (size_t)maxquerythread);
            pthread_mutex_t lock_queue = PTHREAD_MUTEX_INITIALIZER, lock_current_root_node = PTHREAD_MUTEX_INITIALIZER;
            pthread_rwlock_t lock_bsf = PTHREAD_RWLOCK_INITIALIZER;
            pthread_barrier_t lock_barrier;
            pthread_barrier_init(&lock_barrier, NULL, (unsigned int)maxquerythread);
            int *queueoffset = (int *)malloc(sizeof(int) * (size_t)N_PQUEUE);
            (void)queueoffset;
            unsigned long int gpuoffset = 0;
            int loopnumber = 20;
            for (int i = 0; i < N_PQUEUE; i++)
            {
                allpq[i] = pqueue_init(index->settings->root_nodes_size / N_PQUEUE,
                                       cmp_pri, get_pri, set_pri, get_pos, set_pos);
                pthread_mutex_init(&ququelock[i], NULL);
                queuelabel[i] = 1;
            }

            bool *activechunk = (bool *)malloc(sizeof(bool) * (size_t)(loopnumber + 1));
            for (int i = 0; i < loopnumber + 1; i++)
            {
                activechunk[i] = false;
            }
            gap_workerdata *gapworkerdata = (gap_workerdata *)malloc(sizeof(gap_workerdata) * (size_t)maxquerythread);
            int startnode = nodelist.node_amount, stopnode = 0, nodecounter = 0, nodecounter2 = nodelist.node_amount - 1;
            bool *activenode = (bool *)malloc(sizeof(bool) * (size_t)nodelist.node_amount);
            for (int i = 0; i < maxquerythread; i++)
            {
                gapworkerdata[i].nodelist = nodelist.nlist;
                gapworkerdata[i].amountnode = nodelist.node_amount;
                gapworkerdata[i].startnode = &startnode;
                gapworkerdata[i].stopnode = &stopnode;
                gapworkerdata[i].nodecounter = &nodecounter;
                gapworkerdata[i].nodecounter2 = &nodecounter2;
                gapworkerdata[i].index = index;
                gapworkerdata[i].bsf = pq_bsf->knn[pq_bsf->k - 1];
                gapworkerdata[i].paa = paa;
                gapworkerdata[i].lockposition = &lock_queue;
                gapworkerdata[i].offsetarray = (unsigned long *)offsetarray;
                gapworkerdata[i].activechunk = activechunk;
                gapworkerdata[i].chunknumber = loopnumber;
                gapworkerdata[i].activenode = activenode;
                gapworkerdata[i].workerstartnode = (i)*nodelist.node_amount / loopnumber;
                gapworkerdata[i].workerstopnode = (i + 1) * nodelist.node_amount / loopnumber;
            }
            gapworkerdata[maxquerythread - 1].workerstopnode = nodelist.node_amount;

            for (int i = 0; i < maxquerythread; i++)
            {
                pthread_create(&threadid2[i], NULL, multigapworker, (void *)&gapworkerdata[i]);
            }
            for (int i = 0; i < maxquerythread; i++)
            {
                pthread_join(threadid2[i], NULL);
            }
            if (activechunk[loopnumber])
                activechunk[loopnumber - 1] = true;

            int node_counter = 0;

            for (int i = 0; i < maxquerythread; i++)
            {
                workerdata[i].paa = paa;
                workerdata[i].ts = ts;
                workerdata[i].lock_queue = &lock_queue;
                workerdata[i].lock_current_root_node = &lock_current_root_node;
                workerdata[i].lock_bsf = &lock_bsf;
                workerdata[i].nodelist = nodelist.nlist;
                workerdata[i].amountnode = nodelist.node_amount;
                workerdata[i].index = index;
                workerdata[i].minimum_distance = minimum_distance;
                workerdata[i].node_counter = &node_counter;
                workerdata[i].pq = allpq[i];
                workerdata[i].lock_barrier = &lock_barrier;
                workerdata[i].alllock = ququelock;
                workerdata[i].allqueuelabel = queuelabel;
                workerdata[i].allpq = allpq;
                workerdata[i].startqueuenumber = (i % N_PQUEUE);
                workerdata[i].lbdmap = positionmap;
                workerdata[i].labelvalue = &labelvalue;
                workerdata[i].gpuoffset = &gpuoffset;
                workerdata[i].activenode = activenode;
                workerdata[i].pq_bsf = pq_bsf;
                workerdata[i].n_pqueue = N_PQUEUE;
                workerdata[i].rawfile = this->database;
            }

            for (int i = 0; i < maxquerythread; i++)
            {
                pthread_create(&threadid[i], NULL, exact_knn_SING_worker, (void *)&workerdata[i]);
            }
            for (int i = 0; i < loopnumber; i++)
            {
                if (activechunk[i])
                {
                    LBDfloatstreamGPU(&gsaxarray[i * index->sax_cache_size / loopnumber * index->settings->paa_segments], &positionmap[i * index->sax_cache_size / loopnumber], paa, gqts, pq_bsf->knn[pq_bsf->k - 1], (unsigned long)(index->sax_cache_size / loopnumber), &gpositionmap[i * index->sax_cache_size / loopnumber], index->settings->paa_segments, index->settings->mindist_sqrt);
                }
                gpuoffset = (unsigned long int)((i + 1) * index->sax_cache_size / loopnumber);
            }

            for (int i = 0; i < maxquerythread; i++)
            {
                pthread_join(threadid[i], NULL);
            }
            pthread_barrier_destroy(&lock_barrier);
            for (int i = 0; i < N_PQUEUE; i++)
            {
                pqueue_free(allpq[i]);
            }
            free(allpq);

            pqueue_bsf result = *pq_bsf;

            std::vector<std::pair<float, long>> pairs;
            pairs.reserve(static_cast<size_t>(k));
            for (idx_t ik = 0; ik < k; ik++)
            {
                if (result.position[ik] >= 0 && result.knn[ik] < FLT_MAX * 0.99f)
                    pairs.emplace_back(result.knn[ik], result.position[ik]);
            }
            std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b)
                      {
                if (a.first != b.first) return a.first < b.first;
                return a.second < b.second; });
            std::unordered_set<long> seen_pos;
            std::vector<std::pair<float, long>> uniq;
            uniq.reserve(pairs.size());
            for (const auto &p : pairs)
            {
                if (seen_pos.insert(p.second).second)
                    uniq.push_back(p);
            }
            long last_pos = 0;
            float last_dist = 0.0f;
            for (idx_t ik = 0; ik < k; ik++)
            {
                if (ik < static_cast<idx_t>(uniq.size()))
                {
                    last_dist = uniq[static_cast<size_t>(ik)].first;
                    last_pos = uniq[static_cast<size_t>(ik)].second;
                }
                I[q_loaded * k + ik] = static_cast<idx_t>(last_pos >= 0 ? last_pos : 0);
                D[q_loaded * k + ik] = last_dist;
            }

            pqueue_bsf_free(pq_bsf);
            free(ququelock);
            free(queuelabel);
            free(threadid);
            free(threadid2);
            free(workerdata);
            free(gapworkerdata);
            free(activechunk);
            free(activenode);
            free(queueoffset);
        }

        free(saxarrarysort);
        free(offsetarray);
#if SING_CUDA_ENABLED
        GPUfree(gsaxarray);
        GPUfree(gpositionmap);
        GPUfree(gqts);
        cudaFreeHost(positionmap);
#else
        free(gpositionmap);
        free(positionmap);
        free(gqts);
#endif
        free(nodelist.nlist);
        free(paa);
    }

    void Sing::printBuildIndexDebug() const
    {
        if (index == nullptr)
        {
            fprintf(stderr, "[Sing printBuildIndexDebug] index is NULL (buildIndex not called?)\n");
            return;
        }
        fprintf(stderr, "--- Sing index debug ---\n");
        fprintf(stderr, "  sax_cache_size   = %lu\n", (unsigned long)index->sax_cache_size);
        fprintf(stderr, "  paa_segments     = %d\n", index->settings->paa_segments);
        fprintf(stderr, "  timeseries_size  = %d\n", index->settings->timeseries_size);
        if (index->fbl == nullptr)
        {
            fprintf(stderr, "  fbl             = NULL\n");
            return;
        }
        first_buffer_layer2 *fbl2 = (first_buffer_layer2 *)index->fbl;
        fprintf(stderr, "  fbl number_of_buffers = %d\n", fbl2->number_of_buffers);
        int init_count = 0;
        unsigned long total_records = 0;
        for (int i = 0; i < fbl2->number_of_buffers; i++)
        {
            fbl_soft_buffer2 *sb = &fbl2->soft_buffers[i];
            if (sb->initialized)
            {
                init_count++;
                total_records += (unsigned long)sb->buffer_size;
            }
        }
        fprintf(stderr, "  buffers initialized = %d / %d\n", init_count, fbl2->number_of_buffers);
        fprintf(stderr, "  total records in FBL = %lu\n", total_records);
        fprintf(stderr, "--- end ---\n");
    }

    Sing::~Sing()
    {
        delete[] database;

        if (index != nullptr)
        {
            if (index->sax_cache != nullptr)
                free(index->sax_cache);
            if (index->answer != nullptr)
                free(index->answer);
            if (index->fbl != nullptr)
                destroy_fbl2((first_buffer_layer2 *)index->fbl);
            if (index->sax_file != nullptr)
                fclose(index->sax_file);
            free(index);
            index = nullptr;
        }
        if (index_settings != nullptr)
        {
            if (index_settings->bit_masks != nullptr)
                free(index_settings->bit_masks);
            if (index_settings->max_sax_cardinalities != nullptr)
                free(index_settings->max_sax_cardinalities);
            free(index_settings);
            index_settings = nullptr;
        }
    }
}
