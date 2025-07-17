# DTW Support in BruteForce Algorithm

This implementation adds Dynamic Time Warping (DTW) distance support to the BruteForce similarity search algorithm in the diNo-lib library.

## What was added

### Files Modified

1. **`lib/algos/Bruteforce.hpp`**

   - Added private methods `searchIndexL2Squared()` and `searchIndexDTW()`
   - These separate the logic for different distance types

2. **`lib/algos/Bruteforce.cpp`**
   - Refactored `searchIndex()` to dispatch to distance-specific methods
   - Added `searchIndexL2Squared()` - contains the original L2² logic
   - Added `searchIndexDTW()` - handles DTW distance computation
   - Both methods use the same search structure (priority queue, multi-threading) but delegate distance calculation to the `DistanceComputer`

### Files Created

3. **`demos/demo_bruteforce_dtw.cpp`**

   - Demonstration of DTW usage with BruteForce
   - Shows side-by-side comparison of DTW vs L2² results

4. **`tests/test_bruteforce_dtw_integration.cpp`**
   - Integration tests validating DTW functionality
   - Tests basic functionality, result differences, and multi-threading

## Usage

### Basic Usage

```cpp
#include "lib/algos/Bruteforce.hpp"

// Create BruteForce search with DTW distance
diNoLib::BruteForceSearch bf_search(diNoLib::DistanceType::DTW);

// Build index with your data
bf_search.buildIndex(database, n_database, dim);

// Search for k nearest neighbors
diNoLib::idx_t *I = new diNoLib::idx_t[n_query * k];
float *D = new float[n_query * k];
bf_search.searchIndex(query, n_query, k, I, D);
```

### Comparison with L2²

```cpp
// DTW search
diNoLib::BruteForceSearch bf_dtw(diNoLib::DistanceType::DTW);
bf_dtw.buildIndex(database, n_database, dim);
bf_dtw.searchIndex(query, n_query, k, I_dtw, D_dtw);

// L2² search
diNoLib::BruteForceSearch bf_l2(diNoLib::DistanceType::L2_SQUARED);
bf_l2.buildIndex(database, n_database, dim);
bf_l2.searchIndex(query, n_query, k, I_l2, D_l2);
```

## Key Features

1. **Distance-agnostic API**: The same interface works for both DTW and L2² distances
2. **Multi-threading support**: DTW computations are parallelized using OpenMP
3. **Optimized DTW implementation**: Uses the existing optimized DTW distance computer with SIMD acceleration
4. **Consistent with Messi pattern**: Follows the same architectural pattern used in the Messi algorithm

## Implementation Details

The DTW implementation:

- Uses the existing `DistanceComputer` class which handles DTW distance calculation
- Supports warping windows (default 10% of sequence length)
- Uses the same priority queue-based k-NN search as L2²
- Maintains thread safety through OpenMP parallel regions
- Early termination with bounding for efficiency

## Testing

To test the DTW functionality:

```bash
# Build the project
cd build && make -j4

# Run the DTW demo
./demos/demo_bruteforce_dtw

# Run integration tests
./tests/test_bruteforce_dtw_integration
```

## Performance Notes

- DTW is computationally more expensive than L2² due to dynamic programming matrix computation
- The implementation uses optimized SIMD DTW when possible
- Warping window constraint reduces computational complexity
- Multi-threading helps parallelize the computation across multiple queries

## Compatibility

This implementation is fully backward compatible:

- Existing L2² code continues to work unchanged
- No API breaking changes
- Same performance characteristics for L2² searches
