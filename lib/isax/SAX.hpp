#ifndef SAX_HPP
#define SAX_HPP

#include "./iSAXTypes.hpp"
#include "./SAXBreakpoints.hpp"

#include <stdlib.h>
#include <stdio.h>

namespace diNoLib
{
    int compare(const void *a, const void *b);

    enum response sax_from_ts(ts_type *ts_in, sax_type *sax_out, int ts_values_per_segment, int segments, int cardinality, int bit_cardinality);
    enum response paa_from_ts(ts_type *ts_in, ts_type *paa_out, int segments, int ts_values_per_segment);
    enum response sax_from_paa(ts_type *paa, sax_type *sax, int segments, int cardinality, int bit_cardinality);

    float minidist_paa_to_isax(float *paa, sax_type *sax,
                               sax_type *sax_cardinalities,
                               sax_type max_bit_cardinality,
                               int max_cardinality,
                               int number_of_segments,
                               int min_val,
                               int max_val,
                               float ratio_sqrt);

    float minidist_paa_to_isax_raw_SIMD(float *paa, sax_type *sax,
                                        sax_type *sax_cardinalities,
                                        sax_type max_bit_cardinality,
                                        int max_cardinality,
                                        int number_of_segments,
                                        int min_val,
                                        int max_val,
                                        float ratio_sqrt);
    
    float ts_euclidean_distance(ts_type *t, ts_type *s, int size, float bound);
    
    float ts_euclidean_distance_SIMD(ts_type *t, ts_type *s, int size, float bound);

}

#endif