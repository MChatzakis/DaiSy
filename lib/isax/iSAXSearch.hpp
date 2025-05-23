#ifndef ISAXSEARCH_HPP
#define ISAXSEARCH_HPP

#include <stdlib.h>
#include <stdio.h>

#include "SAX.hpp"
#include "iSAXTypes.hpp"
#include "iSAXIndex.hpp"
#include "iSAXPqueue.hpp"

namespace diNoLib
{
    void calculate_node_topk_inmemory(isax_index *index, isax_node *node, ts_type *query, pqueue_bsf *pq_bsf, float *rawfile);

    void approximate_topk_inmemory(ts_type *ts, ts_type *paa, isax_index *index, pqueue_bsf *pq_bsf, float *rawfile);

    void refine_topk_answer_inmemory(ts_type *ts, ts_type *paa, isax_index *index, pqueue_bsf *pq_bsf, float minimum_distance, int limit, float *rawfile);

    float dtw(float *A, float *B, float *cb, int m, int r, float bsf);

    void calculate_node_DTWknn_inmemory(isax_index *index, isax_node *node, ts_type *query, int warpWind, pqueue_bsf *pq_bsf, float *rawfile);

    void approximate_DTWtopk_inmemory(ts_type *ts, ts_type *paa, isax_index *index, int warpWind, pqueue_bsf *pq_bsf, float *rawfile);

} // namespace diNoLib

#endif // ISAXSEARCH_HPP