#ifndef SAX_HPP
#define SAX_HPP

#include "./iSAXTypes.hpp"
#include "./SAXBreakpoints.hpp"

#include <stdlib.h>
#include <stdio.h>


namespace diNoLib
{

    int compare(const void *a, const void *b);
    
    enum response sax_from_ts(ts_type *ts_in, sax_type *sax_out, int ts_values_per_segment,
                              int segments, int cardinality, int bit_cardinality);
}
#endif