#include "SAX.hpp"

namespace diNoLib
{

    /**
        This is used for converting to sax
        */
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

    enum response sax_from_ts(ts_type *ts_in, sax_type *sax_out, int ts_values_per_segment,
                              int segments, int cardinality, int bit_cardinality)
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

}