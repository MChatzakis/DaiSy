# SIMD Conditional Compilation - Implementation Summary

## Overview
Successfully implemented conditional compilation guards throughout the diNoSimilaritySearch library to support systems with and without AVX SIMD capabilities.

## Changes Made

### 1. New Files Created

#### `lib/utils/SIMD.hpp`
- Central header file for all SIMD-related macros and configuration
- Detects AVX support via `__AVX__` or `_M_AVX` compiler defines
- Defines `DAISY_SIMD_AVAILABLE` macro (1 if AVX available, 0 otherwise)
- Provides `THROW_SIMD_NOT_AVAILABLE(func_name)` macro for error handling
- Conditionally includes `<immintrin.h>` only when SIMD is available

#### `SIMD_CONDITIONAL_COMPILATION.md`
- Comprehensive documentation of SIMD changes
- Testing recommendations
- Integration guidelines
- Future enhancement suggestions

### 2. Header Files Modified

#### `lib/distance_computers/DistanceComputer.hpp`
**Changes:**
- Line 4: Replaced `#include "immintrin.h"` with `#include "../utils/SIMD.hpp"`

**Impact:** 
- Conditional inclusion of SIMD headers
- No breaking changes to public APIs
- All method signatures remain unchanged

### 3. Implementation Files Modified

#### `lib/distance_computers/DistanceComputer.cpp`
**Changes:**
- Wrapped `l2_dist_SIMD()` function with `#if DAISY_SIMD_AVAILABLE`
- Added conditional logic in `l2_dist()` to choose between SIMD and naive
- Updated `compute_dist_SIMD()` to fallback gracefully when SIMD unavailable

**Key Functions Modified:**
1. `float l2_dist(float *t, float *s, int dim, float bound)`
   - Conditionally calls SIMD or naive implementation
   
2. `float l2_dist_SIMD(float *t, float *s, int dim, float bound)`
   - Wrapped with `#if DAISY_SIMD_AVAILABLE` blocks
   - Full SIMD implementation preserved when available
   - Function still exists (non-SIMD version calls naive)
   
3. `float compute_dist_SIMD(float *t, float *s, int dim, float bound)`
   - Falls back to naive implementation when SIMD unavailable

**Strategy:** Graceful degradation - scalar naive implementations used as fallback

---

#### `lib/isax/SAX.cpp`
**Changes:**
- Line 5: Replaced `#include "immintrin.h"` with `#include "../utils/SIMD.hpp"`
- Wrapped 4 major SIMD distance computation functions

**Functions Wrapped with `#if DAISY_SIMD_AVAILABLE` ... `#else` ... `#endif`:**

1. **minidist_paa_to_isax_raw_SIMD()** (lines ~176-323)
   - Complex PAA-to-iSAX minimum distance with SIMD optimizations
   - Throws `std::runtime_error` when SIMD unavailable
   
2. **ts_euclidean_distance_SIMD()** (lines ~346-361)
   - SIMD-accelerated Euclidean distance computation
   - Throws error when SIMD unavailable
   
3. **minidist_paa_to_isax_rawa_SIMD()** (lines ~385-537)
   - Enhanced PAA-to-iSAX distance with gather operations
   - Throws error when SIMD unavailable
   
4. **minidist_paa_to_isax_raw_DTW_SIMD()** (lines ~528-698)
   - DTW variant with SIMD optimizations
   - Throws error when SIMD unavailable
   
5. **lb_keogh_data_bound()** (lines ~717-780)
   - Lower-bound Keogh computation with AVX acceleration
   - Throws error when SIMD unavailable

**Note on dtwsimdPruned():**
- Doesn't use SIMD intrinsics directly (scalar DTW implementation)
- Not wrapped with conditional guards (works in both scenarios)

**Strategy:** Runtime error throwing - complex algorithms require SIMD optimizations

---

#### `lib/algos/Sing.cpp`
**Changes:**
- Line 9: Replaced `#include <immintrin.h>` with `#include "../utils/SIMD.hpp"`
- Wrapped SING-specific SIMD function

**Functions Wrapped:**

1. **minidist_paa_to_isax_raw_SING_SIMD()** (lines ~47-161)
   - SING algorithm's SIMD-optimized distance calculation
   - Throws error when SIMD unavailable

**Strategy:** Runtime error throwing for SING algorithm

---

## Compilation Behavior

### Automatic Detection (Recommended)
```bash
# System with AVX support
gcc -mavx main.cpp  # or let compiler auto-detect
# Result: DAISY_SIMD_AVAILABLE = 1, uses full SIMD

# System without AVX
gcc main.cpp  # or explicitly -mno-avx
# Result: DAISY_SIMD_AVAILABLE = 0, uses fallbacks/errors
```

### Manual Control
```bash
# Force SIMD compilation
g++ -D__AVX__ -mavx code.cpp

# Force non-SIMD (disable AVX detection)
g++ -U__AVX__ code.cpp
```

## Runtime Behavior Summary

| Case | DistanceComputer | SAX Algorithms | SING Algorithm |
|------|------------------|---|---|
| SIMD Available | Uses SIMD implementations | Uses SIMD functions | Uses SIMD function |
| SIMD Unavailable | Falls back to naive L2 | Throws runtime error | Throws runtime error |
| Performance | Best case: SIMD; Fallback: scalar L2 | Requires SIMD | Requires SIMD |

## Error Messages Generated

When functions requiring SIMD are called without AVX support:
```
std::runtime_error: SIMD acceleration is required for function: 
[function_name] but AVX is not available on this system. 
Please compile with SIMD support or use a system with AVX support.
```

Affected functions throwing errors:
- `minidist_paa_to_isax_raw_SIMD`
- `ts_euclidean_distance_SIMD`
- `minidist_paa_to_isax_rawa_SIMD`
- `minidist_paa_to_isax_raw_DTW_SIMD`
- `lb_keogh_data_bound`
- `minidist_paa_to_isax_raw_SING_SIMD`

## Testing Checklist

- [ ] Compile with AVX support (`-mavx` flag)
  - [ ] Verify SIMD functions are used
  - [ ] Benchmark performance
  
- [ ] Compile without AVX support (`-mno-avx` flag)
  - [ ] DistanceComputer falls back to naive L2
  - [ ] Verify error messages when calling SAX/SING functions
  
- [ ] Cross-platform testing
  - [ ] x86_64 with AVX (Sandy Bridge or newer)
  - [ ] x86_64 without AVX (Pentium 4, older systems)
  - [ ] ARM systems (if ported)
  
- [ ] Error handling
  - [ ] Catch and handle runtime_error from SIMD-dependent functions
  - [ ] Verify meaningful error messages
  
- [ ] Integration
  - [ ] Rebuild with CMake
  - [ ] Run existing test suite
  - [ ] Verify binary compatibility

## Migration Guide for Users

### For Applications Using DistanceComputer
**Good news:** No changes needed! Your code will automatically use SIMD when available and fall back to naive implementations otherwise.

```cpp
// This works the same way on all systems
DistanceComputer dc(DistanceType::L2_SQUARED);
float dist = dc.compute_dist(t, s, dim, bound); // Works everywhere!
```

### For Applications Using SAX/SING Functions Directly
**Warning:** These functions now require SIMD support. Handle errors gracefully:

```cpp
try {
    float mindist = ts_euclidean_distance_SIMD(query, series, size, bound);
    // Process result
} catch (const std::runtime_error& e) {
    // Fallback to non-SIMD alternatives or report error
    std::cerr << "SIMD not available: " << e.what() << std::endl;
    // Use DistanceComputer instead, which has fallbacks
}
```

## Files Modified Summary

| File | Lines Changed | Type of Change |
|------|---|---|
| `lib/utils/SIMD.hpp` | New file | Creation |
| `lib/distance_computers/DistanceComputer.hpp` | 1 line | Include replacement |
| `lib/distance_computers/DistanceComputer.cpp` | ~20 lines | Conditional guards |
| `lib/isax/SAX.cpp` | ~150 lines | Function wrapping |
| `lib/algos/Sing.cpp` | ~15 lines | Include + function wrapping |
| `SIMD_CONDITIONAL_COMPILATION.md` | New file | Documentation |
| `SIMD_Implementation_Summary.md` | New file | This document |

## Next Steps

1. **Build Testing**
   ```bash
   # Test with SIMD
   cmake -DCMAKE_CXX_FLAGS="-march=native" ..
   make
   
   # Test without SIMD
   cmake -DCMAKE_CXX_FLAGS="-mno-avx" ..
   make
   ```

2. **Update CMakeLists.txt** (Recommended but optional)
   - Add AVX detection
   - Set appropriate compiler flags
   - Document SIMD support in build output

3. **Integration Testing**
   - Run full test suite
   - Benchmark performance with/without SIMD
   - Update documentation

## Benefits Achieved

✅ **Portability:** Code runs on systems with and without AVX  
✅ **Graceful Degradation:** DistanceComputer works everywhere  
✅ **Clear Errors:** SAX/SING functions give informative messages  
✅ **Performance:** SIMD optimizations used when available  
✅ **Maintainability:** Centralized SIMD configuration in SIMD.hpp  
✅ **Backward Compatible:** No API changes, existing code still works

---

**Implementation Date:** February 10, 2026  
**Status:** Complete and ready for testing
