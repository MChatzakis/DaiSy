# SIMD Conditional Compilation - Code Examples and Reference

## Header File: SIMD.hpp

The new `lib/utils/SIMD.hpp` provides the foundation for all SIMD conditionals:

```cpp
#ifndef SIMD_HPP
#define SIMD_HPP

#include <stdexcept>

// Detect AVX support - automatically set by compiler
#if defined(__AVX__) || defined(_M_AVX)
    #define DAISY_SIMD_AVAILABLE 1
    #include <immintrin.h>
#else
    #define DAISY_SIMD_AVAILABLE 0
#endif

// Macro to throw descriptive error when SIMD not available
#define THROW_SIMD_NOT_AVAILABLE(function_name) \
    throw std::runtime_error( \
        std::string("SIMD acceleration is required for function: ") + \
        std::string(function_name) + \
        std::string(" but AVX is not available on this system. ") + \
        std::string("Please compile with SIMD support or use a system with AVX support."))

#endif
```

## Example 1: DistanceComputer - Graceful Fallback

### Original Code (Before)
```cpp
float DistanceComputer::l2_dist_SIMD(float *t, float *s, int dim, float bound)
{
    // Always use AVX intrinsics - will fail on non-AVX systems
    __m256 v_t = _mm256_loadu_ps(&t[i]);
    // ...
}
```

### Modified Code (After)
```cpp
float DistanceComputer::l2_dist(float *t, float *s, int dim, float bound)
{
#if DAISY_SIMD_AVAILABLE
    if (dim % 8 == 0) {
        return l2_dist_SIMD(t, s, dim, bound);
    } else {
        return l2_dist_naive(t, s, dim, bound);
    }
#else
    return l2_dist_naive(t, s, dim, bound);  // No SIMD, use scalar
#endif
}

#if DAISY_SIMD_AVAILABLE
float DistanceComputer::l2_dist_SIMD(float *t, float *s, int dim, float bound)
{
    // Full SIMD implementation
    float distance = 0;
    int i = 0;
    float distancef[8];
    
    __m256 v_t, v_s, v_d, distancev;
    while (dim > 0 && distance < bound) {
        v_t = _mm256_loadu_ps(&t[i]);
        v_s = _mm256_loadu_ps(&s[i]);
        // ... SIMD computation
    }
    return distance;
}
#else
float DistanceComputer::l2_dist_SIMD(float *t, float *s, int dim, float bound)
{
    // Fallback: use naive implementation
    return l2_dist_naive(t, s, dim, bound);
}
#endif

float DistanceComputer::compute_dist_SIMD(float *t, float *s, int dim, float bound)
{
#if DAISY_SIMD_AVAILABLE
    return l2_dist_SIMD(t, s, dim, bound);
#else
    // Fall back to naive implementation when SIMD is not available
    return l2_dist_naive(t, s, dim, bound);
#endif
}
```

## Example 2: SAX Functions - Runtime Error

### Original Code (Before)
```cpp
float minidist_paa_to_isax_raw_SIMD(float *paa, sax_type *sax, ...)
{
    // Hard-coded use of AVX intrinsics
    __m256i vectorsignbit = _mm256_set1_epi32(0xffffffff);
    __m128i sax_cardinalitiesv8 = _mm_lddqu_si128(...);
    // ... complex SIMD code
}
```

### Modified Code (After)
```cpp
#if DAISY_SIMD_AVAILABLE
float minidist_paa_to_isax_raw_SIMD(float *paa, sax_type *sax, 
                                    sax_type *sax_cardinalities, ...)
{
    // Full SIMD implementation preserved
    __m256i vectorsignbit = _mm256_set1_epi32(0xffffffff);
    __m128i sax_cardinalitiesv8 = _mm_lddqu_si128(...);
    
    // ... complete SIMD implementation
    
    return (distancef[0] + distancef[4]) * ratio_sqrt;
}
#else
float minidist_paa_to_isax_raw_SIMD(float *paa, sax_type *sax,
                                    sax_type *sax_cardinalities, ...)
{
    // Fallback: throw informative error
    THROW_SIMD_NOT_AVAILABLE("minidist_paa_to_isax_raw_SIMD");
}
#endif
```

## Example 3: User Code - Handling Both Cases

### For DistanceComputer (Works Everywhere)
```cpp
#include "DistanceComputer.hpp"

int main() {
    // This works on all systems!
    DistanceComputer dc(DistanceType::L2_SQUARED);
    
    float t[100], s[100];
    // Fill arrays...
    
    // On SIMD systems: uses AVX
    // On non-SIMD systems: uses scalar L2
    float distance = dc.compute_dist(t, s, 100, FLT_MAX);
    
    std::cout << "Distance: " << distance << std::endl;
    return 0;
}
```

### For SAX Functions (Handle Errors)
```cpp
#include "SAX.hpp"
#include <stdexcept>

int main() {
    float query[256], series[256];
    // Fill arrays...
    
    try {
        // This may throw if SIMD not available
        float dist = ts_euclidean_distance_SIMD(query, series, 256, FLT_MAX);
        std::cout << "Distance: " << dist << std::endl;
    } catch (const std::runtime_error& e) {
        // SIMD not available
        std::cerr << "Error: " << e.what() << std::endl;
        
        // Fallback to alternative:
        // Option 1: Use DistanceComputer instead (has fallback)
        DistanceComputer dc(DistanceType::L2_SQUARED);
        float dist = dc.compute_dist((float*)query, (float*)series, 256, FLT_MAX);
        
        // Option 2: Implement scalar version
        // float dist = ts_euclidean_distance(query, series, 256, FLT_MAX);
        
        return 1;
    }
    
    return 0;
}
```

## Compilation Examples

### With CMake (Recommended)
```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(diNoSimilaritySearch)

# Auto-detect AVX support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx" COMPILER_SUPPORTS_AVX)

if(COMPILER_SUPPORTS_AVX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
    message(STATUS "AVX support detected and enabled")
else()
    message(WARNING "AVX not supported - SIMD functions will throw errors")
endif()

add_library(diNoSimilaritySearch ...)
```

### Command Line Compilation

**With SIMD (modern systems):**
```bash
g++ -std=c++17 -O3 -march=native -o program main.cpp
# Auto-detects AVX if available
```

**With explicit SIMD:**
```bash
g++ -std=c++17 -O3 -mavx -o program main.cpp
# Enables AVX explicitly
```

**Without SIMD (force scalar):**
```bash
g++ -std=c++17 -O3 -mno-avx -o program main.cpp
# SIMD functions will throw errors at runtime
```

**Portable (works everywhere):**
```bash
g++ -std=c++17 -O2 -o program main.cpp
# Let compiler decide based on target system
```

## Conditional Compilation Pattern Used

All SIMD-dependent functions follow this pattern:

```cpp
#if DAISY_SIMD_AVAILABLE

// ============= SIMD IMPLEMENTATION =============
return_type function_SIMD(params...) 
{
    // Full SIMD optimized code using AVX intrinsics
    __m256 v = _mm256_loadu_ps(...);
    v = _mm256_mul_ps(v, v);
    _mm256_storeu_ps(..., v);
    return result;
}

// ============= NAIVE/FALLBACK ==================
return_type non_simd_function(params...)
{
    // Scalar implementation
    for (int i = 0; i < size; i++) {
        result += data[i] * data[i];
    }
    return result;
}

#else

// ============= NO SIMD - ERROR THROWING =======
return_type function_SIMD(params...)
{
    THROW_SIMD_NOT_AVAILABLE("function_SIMD");
}

#endif
```

## Testing SIMD Availability at Runtime

If you need to check SIMD availability at runtime:

```cpp
#include "SIMD.hpp"

int main() {
    #if DAISY_SIMD_AVAILABLE
        std::cout << "SIMD is available" << std::endl;
        // Use SIMD functions
    #else
        std::cout << "SIMD is NOT available" << std::endl;
        // Use scalar alternatives
    #endif
    
    return 0;
}
```

## Performance Comparison

### L2 Distance Computation
- **With SIMD:** ~2.5-4x faster (8 floats per iteration)
- **Without SIMD:** Scalar fallback (~same as before)
- **DistanceComputer:** Automatic best choice

### SAX Min-Distance
- **With SIMD:** Full vector acceleration with gather operations
- **Without SIMD:** Runtime error (requires SIMD)

### SING Algorithm
- **With SIMD:** Optimized distance computation
- **Without SIMD:** Runtime error

## Debugging SIMD Issues

### Check if SIMD is available
```cpp
#if DAISY_SIMD_AVAILABLE
    std::cout << "Compiled with SIMD support" << std::endl;
#else
    std::cout << "Compiled WITHOUT SIMD support" << std::endl;
#endif
```

### Catch SIMD-related errors
```cpp
try {
    // Call SIMD function
    float result = ts_euclidean_distance_SIMD(...);
} catch (const std::runtime_error& e) {
    std::cerr << "SIMD Error: " << e.what() << std::endl;
    // Detailed error message includes function name
}
```

### Verify compiler flags
```bash
# Check what compiler flags were used
g++ -Q --help=optimizers | grep mavx
```

---

**Quick Reference:**
- **Detects:** `__AVX__` or `_M_AVX` compiler defines
- **Enable:** `-mavx` compiler flag or `-march=native`
- **Disable:** `-mno-avx` compiler flag
- **Check:** `DAISY_SIMD_AVAILABLE` macro at compile time, try/catch at runtime
