#ifndef ISAXTYPES_HPP
#define ISAXTYPES_HPP

#include <algorithm>

namespace diNoLib
{

#define DISK_BUFFER_SIZE 8192
#define PROGRESS_CALCULATE_THREAD_NUMBER 12
#define LOCK_SIZE 256
#define CREATE_MASK(mask, index, sax_array)                                                            \
    int mask__i;                                                                                       \
    for (mask__i = 0; mask__i < index->settings->paa_segments; mask__i++)                              \
        if (index->settings->bit_masks[index->settings->sax_bit_cardinality - 1] & sax_array[mask__i]) \
            mask |= index->settings->bit_masks[index->settings->paa_segments - mask__i - 1];
#define BUFFER_REALLOCATION_RATE 2
#define PAGE_SIZE 4096

#define MINVAL -2000000
#define MAXVAL 2000000


    enum response
    {
        OUT_OF_MEMORY_FAILURE,
        FAILURE,
        SUCCESS
    };

    enum insertion_mode
    {
        PARTIAL = 1,
        TMP = 2,
        FULL = 4,
        NO_TMP = 8
    };

    typedef unsigned char sax_type;
    typedef unsigned long long root_mask_type;
    typedef unsigned long long file_position_type;
    typedef unsigned long long file_position_type;
    typedef float ts_type;

    enum node_cleaning_mode {DO_NOT_INCLUDE_CHILDREN = 0, INCLUDE_CHILDREN = 1};
    enum buffer_cleaning_mode {FULL_CLEAN, TMP_ONLY_CLEAN, TMP_AND_TS_CLEAN};

    // #define dist(x,y) ((x-y)*(x-y))
    // #define max(x,y) ((x)>(y)?(x):(y))
    // #define min(x, y) (x < y ? x : y)

    template <typename T>
    inline T max(T a, T b) {
        return std::max(a, b);
    }

    template <typename T>
    inline T min(T a, T b) {
        return std::min(a, b);
    }
    
    inline float dist(float x, float y) { return (x - y) * (x - y); }

}

#endif