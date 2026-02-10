# SIMD Conditional Compilation Implementation

## Overview
This document describes the conditional compilation guards added to handle SIMD (AVX) support across the diNoSimilaritySearch library.

## Files Created

### `lib/utils/SIMD.hpp`
A new header file that provides:
- Detection of AVX support via `DAISY_SIMD_AVAILABLE` macro
- Conditional inclusion of `<immintrin.h>` only when AVX is available
- `THROW_SIMD_NOT_AVAILABLE(function_name)` macro for runtime error handling

## Files Modified

### 1. `lib/distance_computers/DistanceComputer.hpp`
- **Change**: Replaced `#include "immintrin.h"` with `#include "../utils/SIMD.hpp"`
- **Impact**: Conditional inclusion of SIMD headers, fallback to naive implementations when SIMD is unavailable

### 2. `lib/distance_computers/DistanceComputer.cpp`
- **Changes**:
  - Wrapped `l2_dist_SIMD ()` implementation with `#if DAISY_SIMD_AVAILABLE`
  - Added fallback implementation (`l2_dist_naive()`) when SIMD not available
  - Updated `l2_dist()` to conditionally use SIMD or naive implementation
  - Updated `compute_dist_SIMD()` to fallback to naive when SIMD unavailable

- **Strategy**: For DistanceComputer, naive implementations are used as fallbacks, ensuring graceful degradation

### 3. `lib/isax/SAX.cpp`
- **Changes**:
  - Replaced `#include "immintrin.h"` with `#include "../utils/SIMD.hpp"`
  - Wrapped the following SIMD functions:
    - `minidist_paa_to_isax_raw_SIMD()` - Throws runtime error when SIMD unavailable
    - `ts_euclidean_distance_SIMD()` - Throws runtime error when SIMD unavailable
    - `minidist_paa_to_isax_rawa_SIMD()` - Throws runtime error when SIMD unavailable

- **Strategy**: These complex SAX functions require SIMD optimization for correctness and throw runtime errors when not available

- **Functions Still Requiring Wrapping** (TODO):
  - `minidist_paa_to_isax_raw_DTW_SIMD()` - DTW variant with SIMD
  - `lb_keogh_data_bound()` - Uses SIMD internally
  - `dtwsimdPruned()` - DTW pruning with SIMD optimizations

### 4. `lib/algos/Sing.cpp`
- **Changes**:
  - Replaced `#include <immintrin.h>` with `#include "../utils/SIMD.hpp"`
  - Wrapped `minidist_paa_to_isax_raw_SING_SIMD()` with `#if DAISY_SIMD_AVAILABLE`
  - Added error-throwing fallback when SIMD unavailable

- **Strategy**: Throws runtime error when SIMD acceleration required but unavailable

## Compilation Options

### With SIMD Support (Default on AVX-capable systems)
```bash
# Compiler will automatically detect AVX support
cmake ..
make
```

### Without SIMD Support (Force)
```bash
# Use -mavx or compiler-specific flags to enable/disable
# For systems without AVX, SIMD_AVAILABLE will be 0
cmake ..
make
```

## Runtime Behavior

### When SIMD is Available
- All SIMD-optimized functions are used directly
- Maximum performance with AVX instructions

### When SIMD is NOT Available
- **DistanceComputer**: Falls back to naive scalar implementations seamlessly
- **SAX/Sing Algorithms**: Throws `std::runtime_error` with descriptive message indicating which function requires SIMD

## Example Error Message
```
SIMD acceleration is required for function: ts_euclidean_distance_SIMD but AVX is not available on this system. Please compile with SIMD support or use a system with AVX support.
```

## Architecture of Conditional Compilation

Each SIMD-dependent code block follows this pattern:

```cpp
#if DAISY_SIMD_AVAILABLE
    // Full SIMD implementation using AVX intrinsics
    float function_SIMD(...) { ... }
#else
    // Fallback: Either naive implementation or error-throwing stub
    float function_SIMD(...) {
        THROW_SIMD_NOT_AVAILABLE("function_SIMD");
    }
#endif
```

## Testing Recommendations

1. **Test with SIMD Available**:
   - Run on system with AVX support
   - Verify fast execution paths are used

2. **Test without SIMD**:
   - Compile with `-mno-avx` flag
   - Verify appropriate error messages are thrown
   - Test DistanceComputer fallback implementations

3. **Cross-Platform Testing**:
   - Test on x86_64 with AVX support
   - Test on older x86_64 without AVX
   - Monitor performance impact

## Integration with Build System

The CMakeLists.txt should be updated to:
1. Detect AVX support automatically
2. Set appropriate compiler flags `-mavx` if available
3. Document SIMD support in the build output

## Future Enhancements

1. **Wrap remaining SAX.cpp functions**:
   - `minidist_paa_to_isax_raw_DTW_SIMD()`
   - `lb_keogh_data_bound()`
   - `dtwsimdPruned()`

2. **Support for other SIMD extensions**:
   - SSE4.2 fallback
   - NEON (ARM)
   - SVE (ARMv8)

3. **Runtime SIMD detection**:
   - Check CPU capabilities at runtime
   - Select optimal code path dynamically

4. **Performance monitoring**:
   - Add benchmarks comparing SIMD vs scalar implementations
   - Measure performance impact of fallbacks

## Compatibility Matrix

| System | AVX Support | Status |
|--------|-------------|--------|
| Modern x86_64 (Sandy Bridge+) | Yes | SIMD enabled |
| Older x86_64 | No | Scalar fallbacks |
| Legacy systems | No | Error thrown for complex SIMD functions |

## Notes

- The header file `SIMD.hpp` is self-contained and can be reused for other SIMD-dependent modules
- All changes maintain backward compatibility
- No breaking changes to public APIs
