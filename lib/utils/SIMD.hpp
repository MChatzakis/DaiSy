#ifndef SIMD_HPP
#define SIMD_HPP

/**
 * @file SIMD.hpp
 * @brief SIMD availability detection and configuration
 *
 * This header provides macros to detect SIMD support and safely include
 * SIMD-specific headers only when available. It enables conditional compilation
 * of SIMD-optimized functions.
 */

#include <stdexcept>
#include <string>

// Detect AVX2 support. The guarded code uses AVX2 intrinsics (_mm256_srlv_epi32,
// _mm256_cvtepu8_epi16 and friends), so testing for plain __AVX__ turned a CPU
// with AVX but no AVX2 into a compile error instead of the scalar fallback.
#if defined(__AVX2__)
    #define DAISY_SIMD_AVAILABLE 1
    #include <immintrin.h>
#else
    #define DAISY_SIMD_AVAILABLE 0
#endif

/**
 * @brief Guard for SIMD-specific code blocks
 * Usage in conditional compilation:
 *   #if DAISY_SIMD_AVAILABLE
 *   // SIMD code here
 *   #else
 *   // Fallback code here
 *   #endif
 */

/**
 * @brief Macro to throw a runtime error when SIMD is required but not available
 * @param function_name Name of the function requiring SIMD
 */
#define THROW_SIMD_NOT_AVAILABLE(function_name) \
    throw std::runtime_error( \
        std::string("SIMD acceleration is required for function: ") + \
        std::string(function_name) + \
        std::string(" but AVX is not available on this system. ") + \
        std::string("Please compile with SIMD support or use a system with AVX support."))

#endif // SIMD_HPP
