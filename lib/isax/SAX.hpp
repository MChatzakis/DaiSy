#ifndef SAX_HPP
#define SAX_HPP

#include "./iSAXTypes.hpp"
#include "./SAXBreakpoints.hpp"

#include <stdlib.h>
#include <stdio.h>

namespace daisy
{
    // Process-global active breakpoints (triangular / flat-max). Default to the Gaussian
    // tables; an index installs its own via set_active_breakpoints() before build/search.
    //
    // Being process-global, these are only correct while a single index is live: two indices
    // with different (equi-depth) tables would read each other's. They are being retired. The
    // lower-bound helpers below now take the tables explicitly, and a null argument means
    // "fall back to the global" so un-migrated callers keep working. Query paths that pass
    // their own tables need no set_active_breakpoints() call at all.
    extern const float *daisy_active_breakpoints;
    extern const float *daisy_active_breakpoints_max;

    // NULL args restore the Gaussian defaults.
    void set_active_breakpoints(const float *breakpoints, const float *breakpoints_max);

    int compare(const void *a, const void *b);

    enum response sax_from_ts(ts_type *ts_in, sax_type *sax_out, int ts_values_per_segment, int segments, int cardinality, int bit_cardinality);
    enum response paa_from_ts(const ts_type *ts_in, ts_type *paa_out, int segments, int ts_values_per_segment);
    enum response sax_from_paa(ts_type *paa, sax_type *sax, int segments, int cardinality, int bit_cardinality);

    // COCONUT "sortable SAX": bit-interleave a SAX word so byte order preserves SAX
    // neighbourhood (enables bottom-up build + ordered inserts). out/sax are `segments` bytes.
    void sortable_sax_from_sax(const sax_type *sax, sax_type *out, int segments, int bit_cardinality);

    float minidist_paa_to_isax(float *paa, sax_type *sax,
                               sax_type *sax_cardinalities,
                               sax_type max_bit_cardinality,
                               int max_cardinality,
                               int number_of_segments,
                               int min_val,
                               int max_val,
                               float ratio_sqrt,
                               const float *bp = nullptr);

    float minidist_paa_to_isax_raw_SIMD(float *paa, sax_type *sax,
                                        sax_type *sax_cardinalities,
                                        sax_type max_bit_cardinality,
                                        int max_cardinality,
                                        int number_of_segments,
                                        int min_val,
                                        int max_val,
                                        float ratio_sqrt,
                                        const float *bp = nullptr);

    float ts_euclidean_distance(ts_type *t, ts_type *s, int size, float bound);

    float ts_euclidean_distance_SIMD(ts_type *t, ts_type *s, int size, float bound);

    float minidist_paa_to_isax_rawa_SIMD(float *paa, sax_type *sax,
                                         sax_type *sax_cardinalities,
                                         sax_type max_bit_cardinality,
                                         int max_cardinality,
                                         int number_of_segments,
                                         int min_val,
                                         int max_val,
                                         float ratio_sqrt,
                                         const float *bp = nullptr);

    float minidist_paa_to_isax_raw_DTW_SING_SIMD(float *paaU, float *paaL, sax_type *sax,
                                                 sax_type *sax_cardinalities,
                                                 sax_type max_bit_cardinality,
                                                 int max_cardinality,
                                                 int number_of_segments,
                                                 int min_val,
                                                 int max_val,
                                                 float ratio_sqrt,
                                                 const float *bp_max = nullptr);

    float minidist_paa_to_isax_raw_DTW_SIMD(float *paaU, float *paaL, sax_type *sax,
                                            sax_type *sax_cardinalities,
                                            sax_type max_bit_cardinality,
                                            int max_cardinality,
                                            int number_of_segments,
                                            int min_val,
                                            int max_val,
                                            float ratio_sqrt,
                                            const float *bp = nullptr);

    float lb_keogh_data_bound(float *qo, float *tu, float *tl, float *cb, int len, float bsf);
    
    float dtwsimdPruned(float *A, float *B, float *cb, int m, int r, float bsf, float *tSum, float *pCost, float *rDist);

    float minidist_paa_to_isax_DTW(float *paaU, float *paaL, sax_type *sax,
                                   sax_type *sax_cardinalities,
                                   sax_type max_bit_cardinality,
                                   int max_cardinality,
                                   int number_of_segments,
                                   int min_val,
                                   int max_val,
                                   float ratio_sqrt,
                                   const float *bp = nullptr);

}

#endif